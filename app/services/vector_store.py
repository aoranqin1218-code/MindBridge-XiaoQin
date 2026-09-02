from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from app.core.config import Settings
from app.models.entities import KnowledgeChunk


# 主检索方案标签：向量 + BM25 混合召回 + 本地重排（Chroma 可用时的完整方案，供状态展示与日志用）
PRIMARY_RETRIEVAL_LABEL = "Chroma vector + BM25 hybrid + local reranker"
# 降级检索方案标签：仅本地 BM25 + 融合分数重排（Chroma/向量不可用时的兜底方案）
FALLBACK_RETRIEVAL_LABEL = "local BM25 + hybrid_score reranker"


# 向量库不可用异常：缺 OPENAI_API_KEY、缺 chromadb 依赖、向量数量不匹配等场景抛出，调用方可据此降级到 BM25
class VectorStoreUnavailable(RuntimeError):
    pass


# 向量检索的单条命中：由 ChromaKnowledgeStore.query() 返回，供 _retrieve_vector 转成 SearchResult 后进入融合
@dataclass
class VectorSearchHit:
    chunk_id: int | None                            # 命中片段在数据库中的主键；Chroma 查不到对应行时为 None
    source: str                                     # 来源文档名/标识
    source_index: int                               # 片段在来源文档内的序号（用于邻居扩展和排序）
    content: str                                    # 命中片段文本内容
    score: float                                    # 向量相似度分数


class ChromaKnowledgeStore:
    """主 RAG 路径：OpenAI text-embedding-3-small 生成的向量，存储在 Chroma 中并用于查询。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.can_embed = False
        self.error = ""
        if not settings.knowledge_vector_enabled:
            self.error = "Chroma 向量库未启用"
            return
        
        if not settings.openai_api_key:
            if settings.knowledge_vector_required:
                raise VectorStoreUnavailable("缺少 OPENAI_API_KEY，无法启用 Chroma + text-embedding-3-small 主检索方案")
            self.error = f"缺少 OPENAI_API_KEY，Chroma + text-embedding-3-small 不可用，已回退到{FALLBACK_RETRIEVAL_LABEL}"
            return
        
        try:
            import chromadb
        except ImportError as exc:
            if settings.knowledge_vector_required:
                raise VectorStoreUnavailable("缺少 chromadb 依赖，无法启用 Chroma + text-embedding-3-small 主检索方案") from exc
            self.error = f"缺少 chromadb 依赖，Chroma + text-embedding-3-small 不可用，已回退到{FALLBACK_RETRIEVAL_LABEL}"
            return

        # 准备 Chroma 持久化存储：
        # 把配置的相对路径解析成绝对路径 → 目录不存在则创建 → 用该目录建持久化客户端（数据落在磁盘，重启不丢）
        persist_dir = self._resolve_path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=str(persist_dir))                       # 连上持久化的 Chroma

        # collection 是 chromadb SDK 的 Collection 类实例
        self.collection = self.client.get_or_create_collection(                              # 拿/建一个 collection     
            name=settings.chroma_collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": settings.openai_embedding_model},
        )
        self.can_embed = settings.knowledge_vector_enabled

    # 把片段+向量以 Chroma 需要的格式（ids/documents/metadatas）写入向量库（存在则更新、不存在则插入）
    # 成功后打快照，返回写入条数
    def upsert_chunks(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> int:

        rows = [chunk for chunk in chunks if chunk.id is not None and chunk.content.strip()]
        if not rows:
            return 0

        # 调用方保证 embeddings 和 chunks 顺序一致、长度一致
        ids = [self._id(chunk.id) for chunk in rows]
        documents = [chunk.content for chunk in rows]
        metadatas = [
            {"db_id": int(chunk.id), "source": chunk.source, "source_index": int(chunk.source_index)}
            for chunk in rows
        ]

        # 存在更新、不存在插入
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

        # 把 Chroma 持久化目录整体复制一份，生成一个带时间戳的备份快照 - 每次写入即备份
        self.snapshot()
        return len(rows)

    # 全量同步：先删掉 Chroma 里 DB 已不存在的旧 id（残留），再更新全部片段向量，保证向量库与数据库完全一致
    def sync_chunks(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> int:

        valid_ids = {self._id(int(chunk.id)) for chunk in chunks if chunk.id is not None}
        current_ids = set(self.collection.get().get("ids", []))

        # set 定义了差集操作符 -（Python 内置）：A - B = "在 A 中但不在 B 中的元素"。这和数学里的集合差集完全一致
        # Chroma 有、DB 没有的 = 残留/陈旧 id（比如 DB 删除了某文档，但 Chroma 里的向量没删干净）
        # 如果DB有，Chroma没有 -> 空集合 [] -> 删除被跳过，直接进 upsert_chunks，会把 DB 里多的自动插入 Chroma——缺失被补上
        stale_ids = sorted(current_ids - valid_ids)

        if stale_ids:
            self.collection.delete(ids=stale_ids)
        return self.upsert_chunks(chunks, embeddings)

    # 检查 Chroma 里的向量 id 集合与传入片段的 id 集合是否完全一致（判断索引是否与 DB 对齐）
    def has_exact_chunk_ids(self, chunks: list[KnowledgeChunk]) -> bool:
        valid_ids = {self._id(int(chunk.id)) for chunk in chunks if chunk.id is not None}

        # self.collection.get() — Chroma 的 get() 方法，取集合里所有向量记录（不查相似度，直接按存储取）
        # .get("ids", []) — 从返回的 dict 里取 "ids" 键（Chroma 返回 {"ids": [...], "metadatas": [...], ...}）
        current_ids = set(self.collection.get().get("ids", []))

        # id 对齐检查关心的是"元素集合一致"，不是"顺序一致"——所以用 set 最合适
        return current_ids == valid_ids 

    def delete_source(self, source: str) -> None:
        if not self.can_embed:
            return
        self.collection.delete(where={"source": source})

    # 按查询向量到 Chroma 查 top_k 条最相似的片段：调 collection.query，把返回的文本/元数据/距离组装成 VectorSearchHit 列表（距离转相似度分）
    def query(self, query_embedding: list[float], top_k: int) -> list[VectorSearchHit]:
        result = self.collection.query(                                     # Chroma SDK 的内置方法 - 用 HNSW 索引做近似最近邻（ANN）搜索
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],                # 每条结果要带哪些字段
        )

        # 返回结构固定是"每个查询一组"，即使你只传 1 个，外层还是套着。项目里只查 1 个，所以固定 [0]。
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0] 

        hits = []

        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(
                VectorSearchHit(
                    chunk_id=int(metadata["db_id"]) if metadata.get("db_id") is not None else None,
                    source=str(metadata.get("source", "")),
                    source_index=int(metadata.get("source_index", 0)),
                    content=document or "",
                    score=1.0 / (1.0 + max(0.0, distance)),
                )
            )
        return hits

    # 把一批文本向量化（公开入口）：先检查向量能力，再调 _embed 调 OpenAI 接口，返回每段文本对应的向量列表
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.can_embed:
            raise VectorStoreUnavailable(self.error or "Chroma + text-embedding-3-small 主检索方案不可用")
        return self._embed(texts)

    # 把 Chroma 持久化目录整体复制成"时间戳命名的快照"备份（写入后调用，损坏可恢复），并修剪只留最近 N 个；不可用时返回 None
    def snapshot(self) -> str | None:
        if not self.can_embed:
            return None
        if not self.persist_dir.exists():                                       # Chroma 还没建目录 → 返回 None
            return None
        snapshot_root = self._resolve_path(self.settings.chroma_snapshot_dir)   # 快照存放根目录（配置）
        snapshot_root.mkdir(parents=True, exist_ok=True)                        # 不存在则创建

        # 生成时间戳命名的目标目录
        destination = snapshot_root / datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")

        # 把 persist_dir（Chroma 数据所在，含 HNSW 索引文件）完整复制到 destination（新快照目录）
        shutil.copytree(self.persist_dir, destination)
        # 清理过旧的快照
        self._prune_snapshots(snapshot_root)
        
        return str(destination)

    def count(self) -> int:
        if not self.can_embed:
            return 0
        # Chroma 的 Collection 类有 count() 方法，返回集合里有多少条向量记录
        return int(self.collection.count())                                 

    # 调 OpenAI embeddings 接口把文本向量化的真实实现：按 index 对齐返回向量，数量不匹配或含空向量则抛 VectorStoreUnavailable
    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.settings.openai_embedding_model,
            "input": [text if text.strip() else " " for text in texts],
        }

        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}

        response = httpx.post(
            f"{self.settings.openai_base_url}/embeddings",
            headers=headers,
            json=payload,
            timeout=self.settings.embedding_timeout_seconds,
        )

        response.raise_for_status()

        # response.json() 是 httpx 响应对象的方法 , 内部包装了一层json.loads()
        # 因为后面要按位置取向量，index 顺序必须和输入顺序对上
        rows = sorted(response.json().get("data", []), key=lambda item: item.get("index", 0))

        # 按排序后的顺序取 embedding
        embeddings = [row.get("embedding") for row in rows]

        if len(embeddings) != len(texts) or any(not embedding for embedding in embeddings):
            raise VectorStoreUnavailable("OpenAI embeddings 接口返回向量数量不匹配")
        return [[float(value) for value in embedding] for embedding in embeddings]

    # 把配置里的路径解析为绝对路径：本身是绝对路径直接用，相对路径基于项目根目录拼接（配置里常用相对路径，运行时代码要绝对路径）
    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.settings.project_root / path

    # 清理过旧的快照：按时间倒序排，只保留最近 keep 个（默认配置），更早的整个目录删掉；防止快照无限累积占磁盘
    def _prune_snapshots(self, snapshot_root: Path) -> None:
        keep = max(1, self.settings.chroma_snapshot_keep)
        snapshots = sorted([path for path in snapshot_root.iterdir() if path.is_dir()], reverse=True)
        for stale in snapshots[keep:]:
            shutil.rmtree(stale, ignore_errors=True)

    # 把 MySQL 的整数主键转成 Chroma 里的字符串 id
    def _id(self, chunk_id: int) -> str:
        return f"knowledge-chunk-{chunk_id}"


ChromaKnowledgeVectorStore = ChromaKnowledgeStore
