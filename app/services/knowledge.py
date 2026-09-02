from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Hashable

from pypdf import PdfReader
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.entities import KnowledgeChunk
from app.services.vector_store import FALLBACK_RETRIEVAL_LABEL, PRIMARY_RETRIEVAL_LABEL, ChromaKnowledgeStore


logger = logging.getLogger(__name__)


# 一条检索命中的知识片段：BM25 检索、向量检索、邻居扩展、rerank 的通用产物，也是最终交给 Agent 的知识单元
@dataclass
class SearchResult:
    chunk_id: int | None                            # 知识片段在数据库中的主键；向量检索命中但库里查不到时为 None
    source: str                                     # 来源文档名/标识（如 risk-policy.md），展示和溯源用
    content: str                                    # 片段文本内容，最终拼进回复 prompt 的知识段落
    score: float                                    # 相关性分数：单一检索时是原始分，融合阶段是加权后的最终分


# 融合阶段的中间候选：把"同一片段"的向量分与 BM25 分合并记录，再加权求和出最终 score
# 注意它是可变对象（非 frozen）：vector_score/bm25_score 会被 max() 累积更新后参与融合
@dataclass
class RetrievalCandidate:
    result: SearchResult                            # 命中的知识片段本体（含最终 score）
    vector_score: float = 0.0                       # 归一化后的向量检索分数，融合时按权重累加
    bm25_score: float = 0.0                         # 归一化后的 BM25 分数，融合时按权重累加


class KnowledgeService:
    # 注入数据库会话与配置，并初始化 Chroma 向量库（知识检索的向量侧）
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.vector_store = ChromaKnowledgeStore(settings)

    # 统计知识库片段总数（数据库行数），供状态展示
    def count(self) -> int:
        return self.db.query(KnowledgeChunk).count()

    # 幂等入库：内容与库里已存片段一致则复用返回条数，否则重新 ingest（避免重复灌入）
    def ensure_source(self, source: str, content: str) -> int:
        chunks = chunk_text(content, self.settings.knowledge_chunk_size, self.settings.knowledge_chunk_overlap)
        existing = [
            chunk.content
            for chunk in self.db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.source == source)
            .order_by(KnowledgeChunk.source_index.asc())
            .all()
        ]
        if existing == chunks:
            return len(existing)
        return self.ingest(source, content)

    # 返回知识服务运行状态快照（检索顺序/向量可用性/各项配置），供管理接口展示
    def status(self) -> dict:
        vector_chunks = None
        vector_error = getattr(self.vector_store, "error", "")
        if self.vector_store.can_embed:
            try:
                vector_chunks = self.vector_store.count()
            except Exception as exc:
                vector_error = f"{type(exc).__name__}: {exc}"
        return {
            "retrievalOrder": [
                PRIMARY_RETRIEVAL_LABEL,
                f"{FALLBACK_RETRIEVAL_LABEL} when OPENAI_API_KEY/chromadb/vector call is unavailable",
            ],
            "primaryRetrieval": PRIMARY_RETRIEVAL_LABEL,
            "fallbackRetrieval": FALLBACK_RETRIEVAL_LABEL,
            "databaseChunks": self.count(),
            "vectorEnabled": self.settings.knowledge_vector_enabled,
            "vectorAvailable": self.vector_store.can_embed,
            "vectorRequired": self.settings.knowledge_vector_required,
            "embeddingModel": self.settings.openai_embedding_model,
            "vectorChunks": vector_chunks,
            "chromaPersistDir": self.settings.chroma_persist_dir,
            "chromaCollectionName": self.settings.chroma_collection_name,
            "chromaSnapshotDir": self.settings.chroma_snapshot_dir,
            "candidateK": self.settings.knowledge_candidate_k,
            "hybridVectorWeight": self.settings.knowledge_hybrid_vector_weight,
            "hybridBm25Weight": self.settings.knowledge_hybrid_bm25_weight,
            "rerankEnabled": self.settings.knowledge_rerank_enabled,
            "vectorError": vector_error,
        }

    # 从数据库全量重建向量索引（数据入库/升级后把 DB 片段整体同步到 Chroma），返回同步条数
    def rebuild_vector_index(self) -> int:
        if not self.vector_store.can_embed:
            raise RuntimeError(getattr(self.vector_store, "error", "") or "Chroma 向量库不可用")
        rows = self.db.query(KnowledgeChunk).order_by(KnowledgeChunk.source.asc(), KnowledgeChunk.source_index.asc()).all()
        self._sync_vector_chunks(rows)
        self.db.commit()
        return len(rows)

    # 给 Chroma 持久化目录打快照备份，返回快照路径（向量索引的灾备）
    def backup_vector_index(self) -> str:
        if not self.vector_store.can_embed:
            raise RuntimeError(getattr(self.vector_store, "error", "") or "Chroma 向量库不可用")
        snapshot = self.vector_store.snapshot()
        if snapshot is None:
            raise RuntimeError("Chroma 持久化目录不存在，无法生成快照")
        return snapshot

    # 把一段文本切分成片段后整体入库（删旧来源 + 写库 + 建向量索引），返回片段数
    def ingest(self, source: str, content: str) -> int:
        chunks = chunk_text(content, self.settings.knowledge_chunk_size, self.settings.knowledge_chunk_overlap)
        self._delete_vector_source(source)
        self.db.query(KnowledgeChunk).filter(KnowledgeChunk.source == source).delete()
        rows = []
        for index, chunk in enumerate(chunks):
            row = KnowledgeChunk(source=source, source_index=index, content=chunk)
            self.db.add(row)
            rows.append(row)
        self.db.flush()
        self._index_vector_chunks(rows)
        self.db.commit()
        return len(chunks)

    # 从文件名+字节流入库：PDF 走解析成文本，其余按 UTF-8 读取，再统一交给 ingest
    def ingest_file(self, filename: str, data: bytes) -> int:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            text = extract_pdf(data)
        else:
            text = data.decode("utf-8", errors="ignore")
        return self.ingest(filename, text)

    # 知识检索主入口：向量+BM25 混合召回 → 融合排序 → rerank → 邻居扩展，返回 top_k 条结果
    # 输入的query是经过ContextAgent._rewrite_query → LLM 改写的文本
    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        top_k = top_k or self.settings.knowledge_top_k                                  # 4

        # 候选池越大，两路的重合空间越充分，融合越准确。
        candidate_k = self._candidate_k(top_k)                                          # 召回的候选池 - 16

        # 把 MySQL knowledge_chunks 表里的所有片段一次全查出来
        chunks = self.db.query(KnowledgeChunk).all()

        # 主检索现在使用混合召回：语义向量候选 + BM25 关键词候选，随后再做确定性的本地重排
        vector_results = self._retrieve_vector(query, candidate_k)
        bm25_results = self._retrieve_bm25(query, candidate_k, chunks)

        ranked = self._fuse_and_rerank(query, vector_results, bm25_results, top_k)
        if ranked:
            return self._expand_best(ranked, top_k)
        return []

    # 纯关键词检索：对全部片段算 BM25 分数，取分数>0 的 top_k（向量不可用时的兜底通道）
    def _retrieve_bm25(self, query: str, top_k: int, chunks: list[KnowledgeChunk] | None = None) -> list[SearchResult]:
        chunks = chunks if chunks is not None else self.db.query(KnowledgeChunk).all()

        # {chunk_id: score} 字典——key 是 MySQL 片段主键，value 是 BM25 关键词相关度分
        scores = bm25_scores(query, chunks)                             

        ranked = [
            SearchResult(chunk.id, chunk.source, chunk.content, scores.get(chunk.id, 0.0))
            for chunk in chunks
            if chunk.id is not None and scores.get(chunk.id, 0.0) > 0
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]

    # 混合融合：把向量分与 BM25 分各自归一化后，按权重加权求和出最终分并排序；最后交给 rerank
    def _fuse_and_rerank(
        self,
        query: str,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:

        # Python 的 dict 用 hash 表实现：存 key 时算 hash，找 key 时也用 hash 定位
        candidates: dict[Hashable, RetrievalCandidate] = {}

        vector_scores = {result_key(item): item.score for item in vector_results if item.score > 0}
        bm25_scores_by_key = {result_key(item): item.score for item in bm25_results if item.score > 0}

        # 为什么必须归一化：
        # 向量分数和 BM25 分数的量纲不同（一个接近 1、一个能到几），直接加权会失衡——归一化后两路都落同一刻度，才能公平加权。
        normalized_vector = normalize_scores(vector_scores)
        normalized_bm25 = normalize_scores(bm25_scores_by_key)

        for item in [*vector_results, *bm25_results]:
            key = result_key(item)
            candidate = candidates.get(key)
            if candidate is None:
                candidate = RetrievalCandidate(result=item)                 # 建一个新候选，装下这个结果
                candidates[key] = candidate                                 # 放进合并表

            # 无论新建还是已存在，都把归一化后的向量分/BM25 分，用 max 更新进候选 
            candidate.vector_score = max(candidate.vector_score, normalized_vector.get(key, 0.0))
            candidate.bm25_score = max(candidate.bm25_score, normalized_bm25.get(key, 0.0))

        if not candidates:
            return []

        # BM25 永远会有结果——它是纯本地计算（MySQL 全表 + 词频），只要知识库非空，BM25 总能返回。
        # 而向量路可能没结果（Chroma 不可用、嵌入失败、向量查询异常）——所以只有向量路需要这个判断
        vector_weight = max(0.0, self.settings.knowledge_hybrid_vector_weight) if vector_results else 0.0
        bm25_weight = max(0.0, self.settings.knowledge_hybrid_bm25_weight)

        if vector_weight == 0.0 and bm25_weight == 0.0:
            bm25_weight = 1.0
        total_weight = vector_weight + bm25_weight

        fused = []                                                              # 混合分版本的结果列表

        for candidate in candidates.values():
            score = (
                candidate.vector_score * vector_weight
                + candidate.bm25_score * bm25_weight
            ) / total_weight

            fused.append(replace_score(candidate.result, score))

        fused.sort(key=lambda item: item.score, reverse=True)
        fused = fused[:self._candidate_k(top_k)]                                # 融合后：截到候选池大小

        return self._rerank(query, fused, top_k)                                # 交给重排（里面再截 top_k）

    # 可选的重排：开启时用更精细的打分（rerank_score）替换原始分再排序；关闭则直接截断 top_k
    def _rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not self.settings.knowledge_rerank_enabled:
            return candidates[:top_k]
        reranked = [
            replace_score(item, rerank_score(query, item.content, item.score))
            for item in candidates
        ]

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]

    # 候选集条数下限：至少取 max(top_k, 配置的 candidate_k)，给融合/扩展阶段留足余量
    def _candidate_k(self, top_k: int) -> int:
        return max(top_k, self.settings.knowledge_candidate_k)

    # 向量召回：向量库可用时先确保索引存在，再把查询词向量化后查向量库，返回 top_k 条命中；失败降级为空
    def _retrieve_vector(self, query: str, top_k: int) -> list[SearchResult]:

        # can_embed 是"初始化时用一次检查定的开关"
        # False 表示"Chroma 这条主路不可用"，True 表示"配置齐了，向量检索可用"
        if not self.vector_store.can_embed:
            return []
        
        try:
            self._ensure_vector_index()                                             # 用前自检：索引和 DB 对账
            query_embedding = self.vector_store.embed_texts([query])[0]             # 查询文本 - 向量化
            hits = self.vector_store.query(query_embedding, top_k)                  # 去 Chroma 查相似
        except Exception as exc:
            self._handle_vector_error("retrieve", exc)
            return []
        
        results = []
        for hit in hits:
            # 把完整片段从 MySQL 捞出来——因为 Chroma 里的 content 只是副本，MySQL 才是权威
            chunk = self.db.get(KnowledgeChunk, hit.chunk_id) if hit.chunk_id is not None else None
            results.append(
                SearchResult(
                    chunk.id if chunk is not None else hit.chunk_id,
                    chunk.source if chunk is not None else hit.source,
                    chunk.content if chunk is not None else hit.content,
                    hit.score,
                )
            )
        return results

    # "用向量前先确保 Chroma 和 MySQL 对齐"的自检 - 保证向量检索查到的索引一定是最新的
    # 向量索引完整性检查：数量一致、全部有向量、chunk id 精确匹配则跳过，否则重建同步
    def _ensure_vector_index(self) -> None:

        # 查 knowledge_chunks 全部片段，按"来源 → 序号"排序（source 升序，同源内 source_index 升序）
        # 排序是为了后面和 Chroma 里的顺序对齐比较
        rows = self.db.query(KnowledgeChunk).order_by(KnowledgeChunk.source.asc(), KnowledgeChunk.source_index.asc()).all()
        if not rows:
            return
        if (
            self.vector_store.count() == len(rows)                              # 条件A：数量一致
            and all(row.embedding_json for row in rows)                         # 条件B：每个片段都有向量缓存
            and self.vector_store.has_exact_chunk_ids(rows)                     # 条件C：Chroma 里的 id 和 DB 完全一致
        ):  
            return

        # 如果向量数据库和数据库没有对齐
        self._sync_vector_chunks(rows)                                          # 对全部片段批量向量化、整体写进 Chroma

        # 提交数据库事务（_sync_vector_chunks 里会写 embedding_json 到 MySQL，commit 让它落库）
        self.db.commit()                                                        

    # 删除某来源在向量库中的向量（覆盖入库前清理旧数据，防残留）
    def _delete_vector_source(self, source: str) -> None:
        if not self.vector_store.can_embed:
            return
        try:
            self.vector_store.delete_source(source)
        except Exception as exc:
            self._handle_vector_error("delete_source", exc)

    # 给新增片段批量向量化、缓存进 embedding_json 并 upsert 到向量库（入库时调用）
    def _index_vector_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        if not chunks or not self.vector_store.can_embed:
            return
        try:
            embeddings = self._embeddings_for_chunks(chunks)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding_json = json.dumps(embedding, separators=(",", ":"))
            self.vector_store.upsert_chunks(chunks, embeddings)
        except Exception as exc:
            self._handle_vector_error("index", exc)

    # 全量同步：批量向量化后整体写进向量库（重建索引时调用，与 _index_vector_chunks 相似但覆盖全部）
    def _sync_vector_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        if not chunks or not self.vector_store.can_embed:
            return
        
        try:
            embeddings = self._embeddings_for_chunks(chunks)

            for chunk, embedding in zip(chunks, embeddings):
                # separators=(",", ":") = "项之间用逗号、键值之间用冒号，都不要空格"——这就是最紧凑的 JSON 形式
                chunk.embedding_json = json.dumps(embedding, separators=(",", ":"))
            
            self.vector_store.sync_chunks(chunks, embeddings)
        except Exception as exc:
            self._handle_vector_error("sync", exc)

    # 批量取向量：已缓存的（embedding_json）直接用，缺的现算并补齐；最终数量必须与片段数一致，否则报错
    def _embeddings_for_chunks(self, chunks: list[KnowledgeChunk]) -> list[list[float]]:
        embeddings: list[list[float] | None] = []
        missing_indexes = []
        missing_texts = []

        for index, chunk in enumerate(chunks):
            # 这里必须无条件 append，因为 embeddings 列表的索引和 chunks 是对齐的
            embedding = parse_embedding(chunk.embedding_json)
            embeddings.append(embedding)
            if embedding is None:
                missing_indexes.append(index)
                missing_texts.append(chunk.content)
            
        if missing_texts:
            new_embeddings = self.vector_store.embed_texts(missing_texts)

            # 位置和向量一一配对
            # zip(列表A, 列表B) = 把 A 和 B 按位置一对一配对，每次迭代出一个 (A[i], B[i]) 元组
            for index, embedding in zip(missing_indexes, new_embeddings):
                embeddings[index] = embedding
        
        resolved = [embedding for embedding in embeddings if embedding is not None]
        if len(resolved) != len(chunks):
            raise ValueError("Embedding response count did not match knowledge chunks.")
        return resolved

    # 向量环节失败的统一处理：配置要求向量则抛错，否则记日志降级（检索走 BM25 兜底）
    def _handle_vector_error(self, action: str, exc: Exception) -> None:
        if self.settings.knowledge_vector_required:
            raise exc
        logger.warning(
            "%s %s failed; falling back to %s: %s",
            PRIMARY_RETRIEVAL_LABEL,
            action,
            FALLBACK_RETRIEVAL_LABEL,
            exc,
        )

    # 对排名第一的结果做邻居扩展，其余结果去重后填满 top_k（让首条命中带出上下文）
    # 文档被切成小块（chunk_text），检索命中的往往只是其中一小块。但模型回答时需要上下文——只给命中的一小块，信息不完整
    def _expand_best(self, ranked: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not ranked:
            return []

        # 只扩展第一名，是因为第一名的答案通常最关键——它最可能正是问题的答案，
        # 给它补上下文收益最大；其余片段保持原样，避免重复和超长
        best = ranked[0]

        expanded = self._expand(best)

        results = [expanded]

        # 把第一名扩展好之后，再把其余片段按原排名加进来，一直加到凑满 top_k 条为止
        for item in ranked[1:]:
            if item.chunk_id != expanded.chunk_id and len(results) < top_k:
                results.append(item)
        return results

    # 把单片段连同同一来源的前后邻居拼成一条更长结果（给命中补足上下文）
    def _expand(self, result: SearchResult) -> SearchResult:
        if result.chunk_id is None:
            return result
        chunk = self.db.get(KnowledgeChunk, result.chunk_id)
        if chunk is None:
            return result
        neighbors = (
            self.db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.source == chunk.source)
            .filter(KnowledgeChunk.source_index >= max(0, chunk.source_index - 1))
            .filter(KnowledgeChunk.source_index <= chunk.source_index + 1)
            .order_by(KnowledgeChunk.source_index.asc())
            .all()
        )
        return SearchResult(chunk.id, chunk.source, "\n\n".join(item.content for item in neighbors), result.score)


def chunk_text(content: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", content or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


def hybrid_score(query: str, content: str) -> float:
    return token_cosine(query, content) * 0.75 + keyword_score(query, content) * 0.25


# BM25 关键词打分：对每个片段按"查询词在片段中出现的频率"打分，返回 {chunk_id: score}，只保留分数>0 的片段。
# 算法三要素（每个查询词累加）：
#   1) TF（词频）：词在当前片段出现越多越相关，但非线性增长（k1 抑制高频词，默认 1.5）
#   2) IDF（逆文档频率）：词在越少片段中出现越有区分度，加分越多；到处都有的词（如"的"）几乎不加分
#   3) 长度归一化：片段越长越"稀释"词频，用 b=0.75 平衡长文档，防止长文靠词多占便宜
# score = sum( idf * (tf*(k1+1)) / (tf + k1*(1-b + b*len/avg_len)) )；查询词多次出现还有 log 加成。
# 返回分数越高 = 该片段和查询的关键词匹配越强；与向量召回（语义）互补，共同构成混合检索的关键词路。
def bm25_scores(query: str, chunks: list[KnowledgeChunk]) -> dict[int, float]:
    query_terms = counts(tokenize(query))
    if not query_terms or not chunks:
        return {}

    documents = []
    doc_freqs: dict[str, int] = {}
    for chunk in chunks:
        if chunk.id is None:
            continue
        token_counts = counts(tokenize(chunk.content))
        documents.append((chunk.id, token_counts, sum(token_counts.values())))
        for term in token_counts:
            doc_freqs[term] = doc_freqs.get(term, 0) + 1

    total_docs = len(documents)
    if total_docs == 0:
        return {}
    average_length = sum(length for _, _, length in documents) / total_docs or 1.0
    k1 = 1.5
    b = 0.75
    scores: dict[int, float] = {}

    for chunk_id, token_counts, doc_length in documents:
        score = 0.0
        length_norm = k1 * (1.0 - b + b * doc_length / average_length)
        for term, query_frequency in query_terms.items():
            term_frequency = token_counts.get(term, 0)
            if term_frequency == 0:
                continue
            doc_frequency = doc_freqs.get(term, 0)
            idf = math.log(1.0 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            query_boost = 1.0 + math.log(query_frequency)
            score += idf * query_boost * (term_frequency * (k1 + 1.0)) / (term_frequency + length_norm)
        if score > 0:
            scores[chunk_id] = score
    return scores


# 本地重排打分：
# 在融合分 base_score 基础上叠加三个文本层面信号（关键词相似、查询词覆盖、短语命中），按权重加权成最终重排分
# 重排的意义不是推翻融合排序，而是在融合分基础上微调——让字面匹配更精准的结果稍微靠前
def rerank_score(query: str, content: str, base_score: float) -> float:
    lexical = hybrid_score(query, content)                                          # 关键词相似度
    coverage = query_token_coverage(query, content)                                 # 查询词覆盖比例
    phrase = phrase_score(query, content)                                           # 整个查询短语是否出现
    return base_score * 0.55 + lexical * 0.25 + coverage * 0.15 + phrase * 0.05


def query_token_coverage(query: str, content: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    content_tokens = set(tokenize(content))
    return len(query_tokens & content_tokens) / len(query_tokens)


def phrase_score(query: str, content: str) -> float:
    normalized_query = compact_text(query)
    if not normalized_query:
        return 0.0
    normalized_content = compact_text(content)
    if normalized_query in normalized_content:
        return 1.0
    return keyword_score(query, content)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


# 最小-最大归一化：把一列的分数缩放到 [0,1]（(score-min)/(max-min)），让向量分和 BM25 分量纲一致后再加权融合；非正分给 0，全零/等值时特殊处理
def normalize_scores(scores: dict[Hashable, float]) -> dict[Hashable, float]:
    positives = [score for score in scores.values() if score > 0]
    if not positives:
        return {key: 0.0 for key in scores}
    
    lowest = min(positives)
    highest = max(positives)
    if math.isclose(lowest, highest):
        return {key: 1.0 if score > 0 else 0.0 for key, score in scores.items()}
    return {
        key: (score - lowest) / (highest - lowest) if score > 0 else 0.0
        for key, score in scores.items()
    }


# 生成"唯一标识一个片段"的哈希 key：有 chunk_id 用主键，没有（如向量兜底结果）退回"来源+内容"元组；融合时按它把同一片段的两路分数合并
def result_key(result: SearchResult) -> Hashable:
    return result.chunk_id if result.chunk_id is not None else (result.source, result.content)
 

def replace_score(result: SearchResult, score: float) -> SearchResult:
    return SearchResult(result.chunk_id, result.source, result.content, score)


# 把数据库缓存的 embedding_json 字符串解析回数值向量列表；空值/非法 JSON/非数值列表都返回 None，调用方再决定现算补齐
def parse_embedding(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data:
        return None
    if not all(isinstance(item, (int, float)) for item in data):
        return None
    return [float(item) for item in data]


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
    grams = words[:]
    compact = "".join(ch for ch in text.lower() if "\u4e00" <= ch <= "\u9fff")
    grams.extend(compact[i:i + 2] for i in range(max(0, len(compact) - 1)))
    return [item for item in grams if item.strip()]


def token_cosine(left: str, right: str) -> float:
    left_counts = counts(tokenize(left))
    right_counts = counts(tokenize(right))
    if not left_counts or not right_counts:
        return 0.0
    dot = sum(value * right_counts.get(key, 0) for key, value in left_counts.items())
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    return 0.0 if left_norm == 0 or right_norm == 0 else dot / (left_norm * right_norm)


def keyword_score(query: str, content: str) -> float:
    terms = [term for term in re.split(r"[\s，。！？、；：,.!?;:]+", query.lower()) if len(term) >= 2]
    if not terms:
        return 0.0
    lower = content.lower()
    matched = sum(1 for term in terms if term in lower)
    return min(1.0, matched / len(terms))


def counts(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def extract_pdf(data: bytes) -> str:
    from io import BytesIO

    reader = PdfReader(BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
