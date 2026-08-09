"""FinRAG 运行环境与数据只读检查。"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.config import config


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main() -> int:
    checks: list[bool] = []

    required_packages = [
        "fastapi", "uvicorn", "pymysql", "redis", "jieba", "rank_bm25",
        "torch", "transformers", "openai", "pymilvus", "milvus_model",
        "langchain_core", "langchain_text_splitters", "sentence_transformers",
    ]
    missing = [
        name for name in required_packages
        if importlib.util.find_spec(name) is None
    ]
    checks.append(report(
        "Python 依赖",
        not missing,
        "完整" if not missing else f"缺少 {', '.join(missing)}",
    ))

    model_paths = [
        Path(config.BERT_QUERY_CLASSIFIER),
        Path(config.embedding_model_path),
        Path(config.reranker_model_path),
    ]
    absent_models = [str(path) for path in model_paths if not path.is_dir()]
    checks.append(report(
        "本地模型",
        not absent_models,
        "3 个模型目录完整" if not absent_models else f"缺少 {absent_models}",
    ))

    csv_path = PROJECT_ROOT / "data" / "finance_qa_dataset.csv"
    faq_count = 0
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            faq_count = sum(1 for _ in csv.DictReader(file))
    checks.append(report("FAQ CSV", faq_count > 0, f"{faq_count} 条"))

    md_files = list((PROJECT_ROOT / "data").rglob("*.md"))
    empty_md = [path for path in md_files if path.stat().st_size == 0]
    checks.append(report(
        "知识文档",
        bool(md_files) and not empty_md,
        f"{len(md_files)} 个 Markdown，空文件 {len(empty_md)}",
    ))

    try:
        from mysql_qa.db.mysql_client import get_mysql_client

        mysql_rows = get_mysql_client().get_all_questions()
        checks.append(report("MySQL FAQ", len(mysql_rows) > 0, f"{len(mysql_rows)} 条"))
    except Exception as exc:
        checks.append(report("MySQL FAQ", False, str(exc)))

    try:
        from mysql_qa.cache.redis_client import RedisClient

        redis_client = RedisClient()
        question_count = redis_client.client.llen("all_questions")
        tokenizer_exists = bool(redis_client.client.exists("tokenizer_questions"))
        checks.append(report(
            "Redis FAQ",
            question_count > 0 and tokenizer_exists,
            f"问题 {question_count}，分词缓存 {'存在' if tokenizer_exists else '缺失'}",
        ))
    except Exception as exc:
        checks.append(report("Redis FAQ", False, str(exc)))

    try:
        from pymilvus import MilvusClient

        client = MilvusClient(
            uri=f"http://{config.milvus_host}:{config.milvus_port}",
            db_name=config.milvus_database_name,
        )
        exists = client.has_collection(config.milvus_collection_name)
        count = (
            int(client.get_collection_stats(
                config.milvus_collection_name
            ).get("row_count", 0))
            if exists else 0
        )
        checks.append(report(
            "Milvus 金融知识库",
            exists and count > 0,
            f"{config.milvus_database_name}.{config.milvus_collection_name} = {count} 条",
        ))
    except Exception as exc:
        checks.append(report("Milvus 金融知识库", False, str(exc)))

    try:
        from rag_qa.core.query_classifier import single_question_classifier

        label = single_question_classifier.predict_category("基金赎回费率是多少？")
        checks.append(report("BERT 分类器", label in {"通用知识", "专业咨询"}, label))
    except Exception as exc:
        checks.append(report("BERT 分类器", False, str(exc)))

    print("-" * 60)
    print(json.dumps({
        "passed": sum(checks),
        "total": len(checks),
        "ready": all(checks),
    }, ensure_ascii=False))
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
