"""
基于已入库文档内容的精准测试问题

覆盖 5 种策略，每个问题都能在 Milvus 中找到对应文档内容。
"""
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')

from rag_qa.core.rag_system import get_rag_system, STRATEGY_LABELS

def test_query(rag, query, expected_strategy, explain=""):
    """测试单条查询的分类+检索效果"""
    t0 = time.perf_counter()
    result = rag.query(query, top_k=10, rerank_top_k=5, max_context_chunks=4)
    elapsed = time.perf_counter() - t0

    strategy = result['strategy']
    label = STRATEGY_LABELS.get(strategy, strategy)
    match = "OK" if strategy == expected_strategy else "MISMATCH"
    chunks = result['steps'].get('retrieved_count', 0)
    ans_len = len(result.get('answer', ''))

    print(f"  [{match}] 策略={strategy:20s} | 检索={chunks}块 | 回答={ans_len}字 | {elapsed:.1f}s")
    print(f"  问题: {query}")
    if explain:
        print(f"  依据: {explain}")
    if result.get('answer'):
        # 只打印前200字，过滤特殊字符
        preview = result['answer'][:250].replace('\n', ' ')
        for ch in ['#', '', '', '', '', '']:
            preview = preview.replace(ch, '')
        # 只保留可打印字符
        preview = ''.join(c for c in preview if c.isprintable() or c in '\t\n\r ')
        print(f"  回答: {preview}...")
    print()

    return result


def main():
    print("=" * 72)
    print("  FinRAG 精准测试 — 基于已入库文档")
    print("=" * 72)
    print()

    rag = get_rag_system()

    # ═══════════════════════════════════════════════════════════════
    # 1. direct_lookup — 直接事实查询
    # ═══════════════════════════════════════════════════════════════
    print("── 策略1: direct_lookup（直接事实查询）──\n")

    test_query(rag,
        "证券市场的参与者包括哪些？",
        "direct_lookup",
        "证券市场基础知识 第一章第二节：证券发行人、投资者、中介、自律组织、监管机构"
    )

    test_query(rag,
        "如何办理股票开户？开户流程是什么？",
        "direct_lookup",
        "证券市场基础知识 第二章第一节：开户具体流程三步走"
    )

    test_query(rag,
        "基金经理人是怎么运作的？",
        "direct_lookup",
        "证券市场基础知识 第一章第三节：基金由专家管理和运作"
    )

    test_query(rag,
        "什么是认购权证和认沽权证？",
        "direct_lookup",
        "证券市场基础知识 第一章第三节：权证分认购权证和认沽权证"
    )

    test_query(rag,
        "保险合同的构成部分有哪些？",
        "direct_lookup",
        "平安附加保险条款 1.1：保险单、投保单、批注、批单等"
    )

    # ═══════════════════════════════════════════════════════════════
    # 2. comparison — 对比分析
    # ═══════════════════════════════════════════════════════════════
    print("── 策略2: comparison（对比分析）──\n")

    test_query(rag,
        "一级市场和二级市场有什么区别？",
        "comparison",
        "证券市场基础知识 第一章第一节：按层次结构对比发行市场与交易市场"
    )

    test_query(rag,
        "系统性风险和非系统性风险有什么不同？",
        "comparison",
        "证券市场基础知识 第一章第四节：两类风险的详细对比"
    )

    test_query(rag,
        "封闭式基金和开放式基金有什么区别？",
        "comparison",
        "证券市场基础知识 第一章第三节末尾对比了两者"
    )

    test_query(rag,
        "股票和债券有什么不同？",
        "comparison",
        "证券市场基础知识 第一章第一节股票市场vs债券市场对比"
    )

    test_query(rag,
        "混合型基金和股票型基金有什么区别？",
        "comparison",
        "基金招募说明书中对比了不同类型基金的风险收益特征"
    )

    # ═══════════════════════════════════════════════════════════════
    # 3. condition_check — 条件判断
    # ═══════════════════════════════════════════════════════════════
    print("── 策略3: condition_check（条件判断）──\n")

    test_query(rag,
        "哪些人不可以开立股票账户？",
        "condition_check",
        "证券市场基础知识 第二章第一节：列出了不可开户的几类人"
    )

    test_query(rag,
        "购买理财产品需要满足什么条件？",
        "condition_check",
        "浦银理财产品说明书：仅向合格投资者发售，需要反洗钱审核"
    )

    test_query(rag,
        "基金合同在什么情况下会被终止？",
        "condition_check",
        "基金招募说明书：连续50工作日持有人不足200人或净值低于5000万"
    )

    # ═══════════════════════════════════════════════════════════════
    # 4. causal_analysis — 因果分析
    # ═══════════════════════════════════════════════════════════════
    print("── 策略4: causal_analysis（因果分析）──\n")

    test_query(rag,
        "为什么股票价格经常波动？",
        "causal_analysis",
        "证券市场基础知识：政治、社会、经济等多因素综合影响"
    )

    test_query(rag,
        "为什么市场利率提高时证券价格会下降？",
        "causal_analysis",
        "证券市场基础知识 第一章第四节：利率风险的详细解释"
    )

    test_query(rag,
        "为什么要进行投资者教育？",
        "causal_analysis",
        "证券市场基础知识 第六章：投资者教育的重要意义"
    )

    test_query(rag,
        "为什么理财产品的过往业绩不代表未来表现？",
        "causal_analysis",
        "理财产品说明书中有相关风险揭示"
    )

    # ═══════════════════════════════════════════════════════════════
    # 5. concept_explain — 概念详解
    # ═══════════════════════════════════════════════════════════════
    print("── 策略5: concept_explain（概念详解）──\n")

    test_query(rag,
        "介绍一下证券市场的主要风险有哪些？",
        "concept_explain",
        "证券市场基础知识 第一章第四节全面介绍系统性+非系统性风险"
    )

    test_query(rag,
        "什么是股指期货？它有什么特点？",
        "concept_explain",
        "证券市场基础知识 第一章第三节末尾介绍股指期货"
    )

    test_query(rag,
        "介绍一下QFII制度是什么？",
        "concept_explain",
        "证券市场基础知识 第一章第二节详细介绍QFII"
    )

    test_query(rag,
        "什么是ETF？它有哪些特点？",
        "concept_explain",
        "证券市场基础知识 第一章第三节详细介绍了ETF的三大特点"
    )

    test_query(rag,
        "介绍一下保险条款中的责任免除是什么意思？",
        "concept_explain",
        "保险条款第三章：责任免除和其他免责条款"
    )

    print("=" * 72)
    print("  测试完成")
    print("=" * 72)


if __name__ == '__main__':
    main()
