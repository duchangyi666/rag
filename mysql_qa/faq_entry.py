"""
FAQ 模块入口

提供三条检索路线的测试和交互式控制台查询：

    路线① Redis 精确匹配 → 直接返回答案 (prob=1.0)
    路线② BM25 + softmax → prob≥0.85 → MySQL 查答案 → 回写 Redis
    路线③ prob<0.85 → 返回 None → 转交 RAG 模块

Usage:
    python -m mysql_qa.faq_entry          # 交互模式
    python -m mysql_qa.faq_entry --test   # 跑测试用例
"""

import sys
import time

from base.logger import setup_logger

logger = setup_logger('fin_rag.faq_entry')

# 分隔线
SEP = "─" * 60
DSEP = "=" * 60


def print_banner():
    """打印欢迎横幅。"""
    print()
    print(DSEP)
    print("  📊 FinRAG FAQ 智能问答模块")
    print(DSEP)
    print("  路线①  Redis 精确匹配   → prob=1.0 直接返回")
    print("  路线②  BM25 + softmax   → prob≥阈值 模糊命中")
    print("  路线③  低于阈值          → None → 转交 RAG")
    print(DSEP)


def run_route_tests():
    """跑三条路线的测试用例，覆盖所有分支。"""
    from mysql_qa.retriveal.bm25_search import BM25Searcher

    print()
    print(DSEP)
    print("  🧪 三条路线覆盖测试")
    print(DSEP)

    searcher = BM25Searcher()

    # 测试用例：(查询, 期望路线, 说明)
    test_cases = [
        (
            "基金到底是什么？和股票有什么区别？",
            "路线①",
            "问题原文一字不差，应被 Redis 精确命中",
        ),
        (
            "基金定投能赚钱吗",
            "路线②",
            "语义匹配「基金定投真的能赚钱吗？」，softmax 概率应≥0.85",
        ),
        (
            "怎么开户买股票",
            "路线②",
            "语义匹配「股票开户需要什么条件？」，取决于阈值",
        ),
        (
            "ROE是什么意思",
            "路线③",
            "数据集无 ROE 相关问题，top-1 概率<0.85，转 RAG",
        ),
        (
            "明天天气怎么样",
            "路线③",
            "完全无关查询，所有 BM25 得分为 0，均匀分布转 RAG",
        ),
        (
            "新手怎么买基金",
            "路线②",
            "模糊语义匹配，看是否能命中基金相关 FAQ",
        ),
        (
            "什么是ETF基金？怎么买卖？",
            "路线①",
            "数据集原文精确匹配（第15条）",
        ),
    ]

    for query, expected_route, desc in test_cases:
        print()
        print(f"  📝 查询: {query}")
        print(f"     预期路线: {expected_route} | 说明: {desc}")
        print(f"     {SEP}")

        start = time.time()
        answer, prob = searcher.search(query)
        elapsed = (time.time() - start) * 1000

        if answer is not None:
            actual_route = "路线①" if prob == 1.0 else "路线②"
        else:
            actual_route = "路线③"

        match_mark = "✅" if actual_route == expected_route else "⚠️"
        print(f"     {match_mark} 实际路线: {actual_route} | prob={prob:.6f} | 耗时={elapsed:.0f}ms")
        if answer:
            print(f"     答案: {answer[:100]}...")
        else:
            print(f"     结果: 未覆盖 → 转交 RAG 处理")

    searcher.close()
    print()
    print(DSEP)
    print("  测试完成")
    print(DSEP)


def run_interactive():
    """交互式控制台，用户输入问题进行查询。"""
    from mysql_qa.retriveal.bm25_search import BM25Searcher

    print_banner()
    print()
    print("  正在初始化 BM25 模型（首次加载从 Redis/MySQL 构建）...")
    print()

    searcher = BM25Searcher()

    # 首触发构建 + 显示语料信息
    _ = searcher.search("")  # 空查询触发 build 但不搜索
    if searcher.bm25 is not None:
        print(f"  ✅ 模型就绪 — 语料文档数: {len(searcher.tokenized_corpus)}")
        print(f"     阈值: {searcher.threshold}  温度: {searcher.temperature}")
        print(f"     k1: {searcher.k1}  b: {searcher.b}")
    print()
    print("  输入问题进行查询，输入 :q 退出，输入 :test 跑测试")
    print(DSEP)

    while True:
        try:
            query = input("\n  🔍 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if not query:
            continue

        if query == ':q':
            print("  再见！")
            break

        if query == ':test':
            searcher.close()
            run_route_tests()
            # 重新创建 searcher（run_route_tests 关了连接）
            searcher = BM25Searcher()
            continue

        if query == ':help':
            print("  命令:")
            print("    :q    退出")
            print("    :test 跑测试用例")
            print("    :help 显示帮助")
            print("  直接输入问题即可查询")
            continue

        # ── 执行查询 ──────────────────────────────────────────
        print(f"  {SEP}")
        start = time.time()
        answer, prob = searcher.search(query)
        elapsed = (time.time() - start) * 1000

        if answer is not None:
            route = "路线① Redis 精确命中" if prob == 1.0 else "路线② BM25 + softmax"
            print(f"  ✅ {route}")
            print(f"     prob={prob:.6f}  耗时={elapsed:.0f}ms")
            print(f"  {SEP}")
            print(f"  {answer}")
            print(f"  {SEP}")
        else:
            print(f"  ❌ 路线③ 低于阈值 → 转交 RAG")
            print(f"     prob={prob:.6f}  耗时={elapsed:.0f}ms")
            print(f"  {SEP}")

    searcher.close()


# ── 入口 ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if '--test' in sys.argv:
        run_route_tests()
    else:
        run_interactive()
