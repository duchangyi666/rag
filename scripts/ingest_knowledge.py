"""将本地 Markdown/MinerU 文档增量写入配置的 Milvus 金融知识库。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_qa.core.document_process import DocumentProcessor
from rag_qa.core.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sources",
        nargs="+",
        help="Markdown 文件、普通 Markdown 目录或 MinerU 输出目录",
    )
    parser.add_argument("--parent-size", type=int, default=1200)
    parser.add_argument("--child-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    sources = [
        str((PROJECT_ROOT / source).resolve())
        if not Path(source).is_absolute() else source
        for source in args.sources
    ]
    processor = DocumentProcessor(
        parent_size=args.parent_size,
        child_size=args.child_size,
        overlap=args.overlap,
    )
    documents = processor.process(sources)
    if not documents:
        raise RuntimeError("没有生成任何可入库的文档块")

    store = VectorStore()
    store.create_collection()
    inserted = store.insert(documents)
    stats = store.get_collection_stats()

    if inserted != len(documents):
        raise RuntimeError(
            f"Milvus 写入不完整: 计划 {len(documents)}，成功 {inserted}"
        )

    print("=" * 60)
    print("FinRAG 知识库增量入库完成")
    print(f"源路径: {len(sources)}")
    print(f"写入块: {inserted}")
    print(f"Collection 总行数: {stats['count']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
