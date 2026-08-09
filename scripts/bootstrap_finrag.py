"""初始化 FinRAG 的 FAQ 数据。

执行内容：
  1. 创建 MySQL qa_records 表和 question 唯一索引
  2. 将 data/finance_qa_dataset.csv 的问答幂等写入 MySQL
  3. 将问题、分词结果和问答缓存写入 Redis
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import get_mysql_client


CSV_PATH = PROJECT_ROOT / "data" / "finance_qa_dataset.csv"


def load_faq_records() -> tuple[list[tuple[str, str]], int]:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"FAQ 数据文件不存在: {CSV_PATH}")

    records_by_question: dict[str, str] = {}
    source_rows = 0
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        for line_number, row in enumerate(csv.DictReader(file), start=2):
            source_rows += 1
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                raise ValueError(f"FAQ 第 {line_number} 行问题或答案为空")
            # CSV 中存在少量同问题、不同表述答案；与 MySQL 唯一键语义一致，
            # 使用后出现的版本覆盖前一个版本。
            records_by_question[question] = answer

    if not records_by_question:
        raise ValueError("FAQ CSV 没有有效数据")
    return list(records_by_question.items()), source_rows


def main() -> None:
    records, source_rows = load_faq_records()
    mysql = get_mysql_client()
    mysql.create_table()
    mysql.ensure_question_unique_index()
    mysql.batch_upsert(records)

    rows = mysql.get_all_questions()
    if len(rows) != len(records):
        raise RuntimeError(
            f"MySQL FAQ 数量异常: CSV唯一问题={len(records)}, MySQL={len(rows)}"
        )

    redis_client = RedisClient()
    cache_result = redis_client.load_all_to_redis()

    print("=" * 60)
    print("FinRAG FAQ 初始化完成")
    print(
        f"CSV:   {source_rows} 行 / {len(records)} 个唯一问题 "
        f"/ {source_rows - len(records)} 个重复行"
    )
    print(f"MySQL: {len(rows)} 条")
    print(
        "Redis: "
        f"问题 {cache_result['questions']} / "
        f"分词 {cache_result['tokens']} / "
        f"问答 {cache_result['qa_pairs']}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
