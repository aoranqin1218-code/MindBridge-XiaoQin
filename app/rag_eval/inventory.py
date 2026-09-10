from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from app.core.config import Settings
from app.rag_eval.dataset import CorpusEvidence, compute_corpus_fingerprints, normalize_corpus_content
from app.services.knowledge import chunk_text


class EvidenceInventoryError(ValueError):
    """知识分块清单无法根据给定来源或分块参数生成时抛出。"""


# 按线上相同的分块规则读取内置 Markdown，并产出稳定排序的证据集合
def load_bundled_corpus_evidence(
    knowledge_dir: str | Path,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[CorpusEvidence, ...]:
    _validate_chunking(chunk_size, chunk_overlap)
    root = Path(knowledge_dir)
    files = sorted(root.glob("*.md"), key=lambda path: path.name)
    if not files:
        raise EvidenceInventoryError(f"知识目录中没有 Markdown 文件：{root}")

    evidence: list[CorpusEvidence] = []
    for file_path in files:
        chunks = chunk_text(
            file_path.read_text(encoding="utf-8"),
            chunk_size,
            chunk_overlap,
        )
        if not chunks:
            raise EvidenceInventoryError(f"知识文件没有可用正文：{file_path}")
        evidence.extend(
            CorpusEvidence(
                source=file_path.name,
                source_index=source_index,
                content=content,
            )
            for source_index, content in enumerate(chunks)
        )
    return tuple(evidence)


# 将证据集合渲染为供 Gold Set 人工映射使用的 Markdown 清单
def render_evidence_inventory(
    evidence: Iterable[CorpusEvidence],
    chunk_size: int,
    chunk_overlap: int,
    preview_chars: int = 240,
) -> str:
    _validate_chunking(chunk_size, chunk_overlap)
    if isinstance(preview_chars, bool) or not isinstance(preview_chars, int) or preview_chars <= 0:
        raise EvidenceInventoryError("preview_chars 必须是正整数")

    records = sorted(evidence, key=lambda item: item.evidence_key)
    fingerprints = compute_corpus_fingerprints(records)
    source_counts = Counter(item.source for item in records)
    lines = [
        "# MindBridge RAG 证据清单",
        "",
        "> 此文件根据 `app/knowledge/*.md` 自动生成，只用于人工制作 Gold Set；它本身不是标准答案，也不参与指标计算。",
        "",
        "## 当前知识库快照",
        "",
        f"- 来源文件数：{len(source_counts)}",
        f"- 证据分块数：{fingerprints.evidence_count}",
        f"- 分块长度：{chunk_size}",
        f"- 重叠长度：{chunk_overlap}",
        f"- 结构指纹：`{fingerprints.structure_fingerprint}`",
        f"- 内容指纹：`{fingerprints.content_fingerprint}`",
        "",
        "## 分块目录",
        "",
        "| 来源 | sourceIndex | EvidenceKey | 正文字符数 | 短预览 |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for item in records:
        normalized_content = normalize_corpus_content(item.content)
        preview = _preview(normalized_content, preview_chars)
        evidence_key = f"({item.source!r}, {item.source_index})"
        lines.append(
            "| "
            f"`{_escape_table_text(item.source)}` | {item.source_index} | "
            f"`{_escape_table_text(evidence_key)}` | {len(normalized_content)} | "
            f"{_escape_table_text(preview)} |"
        )
    lines.append("")
    return "\n".join(lines)


# 把渲染后的证据清单以 UTF-8 写入指定位置
def write_evidence_inventory(
    output_path: str | Path,
    evidence: Iterable[CorpusEvidence],
    chunk_size: int,
    chunk_overlap: int,
    preview_chars: int = 240,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_evidence_inventory(
            evidence,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preview_chars=preview_chars,
        ),
        encoding="utf-8",
    )
    return destination


# 校验分块长度与重叠长度可以形成稳定、向前推进的窗口
def _validate_chunking(chunk_size: int, chunk_overlap: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise EvidenceInventoryError("chunk_size 必须是正整数")
    if (
        isinstance(chunk_overlap, bool)
        or not isinstance(chunk_overlap, int)
        or chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        raise EvidenceInventoryError("chunk_overlap 必须是小于 chunk_size 的非负整数")


# 截取单行短预览，并用省略号明确标记被截断的正文
def _preview(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    return f"{content[:limit]}…"


# 转义 Markdown 表格中的竖线，避免正文破坏表格列结构
def _escape_table_text(value: str) -> str:
    return value.replace("|", "\\|")


# 解析证据清单生成命令的参数
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="生成 MindBridge RAG 人工标注证据清单")
    parser.add_argument("--knowledge-dir", type=Path, default=project_root / "app" / "knowledge")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "app" / "rag_eval" / "mindbridge-rag-evidence-inventory.md",
    )
    parser.add_argument("--preview-chars", type=int, default=240)
    return parser.parse_args(argv)


# 使用当前项目分块配置生成内置知识库的人工标注清单
def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = Settings()
    evidence = load_bundled_corpus_evidence(
        args.knowledge_dir,
        settings.knowledge_chunk_size,
        settings.knowledge_chunk_overlap,
    )
    destination = write_evidence_inventory(
        args.output,
        evidence,
        chunk_size=settings.knowledge_chunk_size,
        chunk_overlap=settings.knowledge_chunk_overlap,
        preview_chars=args.preview_chars,
    )
    print(f"已生成证据清单：{destination}（{len(evidence)} 个分块）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
