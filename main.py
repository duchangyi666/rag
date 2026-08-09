"""
FinRAG 统一入口 — FAQ + RAG 智能路由

三级级联策略：
    第一级  FAQ (BM25 + Redis/MySQL) → 命中直接返回，延迟 ~10ms
    第二级  BERT 意图分类           → 通用知识 / 专业咨询
    第三级A LLM 直接回答            → 通用金融概念，无需检索
    第三级B RAG 完整流程            → 专业问题，检索 + 生成

Usage:
    python main.py                          # 交互模式（多轮对话）
    python main.py "什么是ETF？"             # 单次查询
    python main.py --verbose "什么是ETF？"   # 详细输出
    python main.py -v                        # 交互 + 详细输出
"""

import sys
import time
import re
from typing import Optional, Dict

from base.logger import setup_logger

logger = setup_logger('fin_rag.main')

# ── 分隔线 ────────────────────────────────────────────────────────────────
SEP = "─" * 60
DSEP = "=" * 60

# ── 延迟加载的单例 ────────────────────────────────────────────────────────
_searcher = None
_rag = None
_classifier = None
_llm = None

CLARIFICATION_ANSWER = (
    "您好，请把问题描述完整一些，我才能准确判断。"
    "例如可以问：“基金赎回费怎么收？”或“股票开户需要什么条件？”"
)

FIXED_CHAT_PATTERNS = (
    (
        re.compile(r"^(你好|您好|嗨|hi|hello)[!！。,.，？?～~]*$", re.IGNORECASE),
        "您好！我是 FinSight AI 金融理财咨询助手，可以为您解答交易规则、"
        "产品费率、开户流程和常见理财问题。",
    ),
    (
        re.compile(
            r"^(你是谁|您是谁|你叫什么|你的名字|who are you)[!！。,.，？?]*$",
            re.IGNORECASE,
        ),
        "我是 FinSight AI 金融理财咨询助手，可以帮您查询金融知识库，"
        "也可以回答一般性的金融问题。",
    ),
    (
        re.compile(r"^(在吗|在不在|有人吗)[!！。,.，？?]*$", re.IGNORECASE),
        "我在，请告诉我您想咨询的金融或理财问题。",
    ),
    (
        re.compile(
            r"^(干嘛呢|你在干嘛|做什么|你能做什么)[!！。,.，？?]*$",
            re.IGNORECASE,
        ),
        "我正在待命，可以为您解答交易规则、产品费率、开户流程和理财知识。"
        "您想先了解哪一项？",
    ),
)


def _needs_clarification(query_str: str) -> bool:
    """拦截只有一个有效字符、无法表达完整意图的输入。"""
    compact = re.sub(r"[\W_]+", "", query_str, flags=re.UNICODE)
    return len(compact) <= 1


def _get_fixed_chat_reply(query_str: str) -> Optional[str]:
    """匹配常见问候和身份/状态询问，避免无意义地访问知识库。"""
    query_text = query_str.strip()
    for pattern, answer in FIXED_CHAT_PATTERNS:
        if pattern.fullmatch(query_text):
            return answer
    return None


def _get_searcher():
    """延迟加载 FAQ BM25 检索器。"""
    global _searcher
    if _searcher is None:
        from mysql_qa.retriveal.bm25_search import get_searcher
        _searcher = get_searcher()
    return _searcher


def _get_classifier():
    """延迟加载 BERT 意图分类器。"""
    global _classifier
    if _classifier is None:
        from rag_qa.core.query_classifier import single_question_classifier
        _classifier = single_question_classifier
    return _classifier


def _get_rag():
    """延迟加载 RAG 系统。"""
    global _rag
    if _rag is None:
        from rag_qa.core.rag_system import get_rag_system
        _rag = get_rag_system()
    return _rag


def _get_llm():
    """延迟加载 LLM 客户端（用于直接回答通用问题）。"""
    global _llm
    if _llm is None:
        from rag_qa.core.strategy_selector import get_strategy_selector
        _llm = get_strategy_selector()
    return _llm


# ═══════════════════════════════════════════════════════════════════════════
# 核心查询函数
# ═══════════════════════════════════════════════════════════════════════════

def query(user_query: str, history: str = "") -> Dict:
    """统一查询入口，三级级联路由。

    Args:
        user_query: 用户查询文本
        history:    多轮对话历史字符串

    Returns:
        {
            'answer':      str,     # 最终回答
            'source':      str,     # 'clarification' | 'faq' | 'llm_direct' | 'rag' | 'error'
            'confidence':  float | None,
            'intent':      str,     # '通用知识' | '专业咨询'
            'latency_ms':  float,
        }
    """
    t_start = time.time()
    query_str = user_query.strip()

    # ── 空查询保护 ──────────────────────────────────────────────────
    if not query_str:
        return {
            'answer': '请输入您的问题。',
            'source': 'error',
            'confidence': None,
            'intent': '未知',
            'latency_ms': (time.time() - t_start) * 1000,
        }

    if _needs_clarification(query_str):
        elapsed = (time.time() - t_start) * 1000
        logger.info("[路由] 输入信息不足，返回补充问题提示 | query=%r", query_str)
        return {
            'answer': CLARIFICATION_ANSWER,
            'source': 'clarification',
            'confidence': None,
            'intent': '待补充',
            'latency_ms': elapsed,
        }

    fixed_reply = _get_fixed_chat_reply(query_str)
    if fixed_reply:
        elapsed = (time.time() - t_start) * 1000
        logger.info("[路由] 闲聊固定话术命中 | query=%r", query_str)
        return {
            'answer': fixed_reply,
            'source': 'smalltalk',
            'confidence': None,
            'intent': '闲聊',
            'latency_ms': elapsed,
        }

    intent = '专业咨询'  # 默认值（降级时使用）

    # ════════════════════════════════════════════════════════════════
    # 第一级：FAQ (BM25 + Redis/MySQL)
    # ════════════════════════════════════════════════════════════════
    try:
        searcher = _get_searcher()
        answer, score = searcher.search(query_str)

        if answer is not None:
            elapsed = (time.time() - t_start) * 1000
            logger.info("[路由] FAQ 命中 | score=%.4f | latency=%.0fms", score, elapsed)
            return {
                'answer': answer,
                'source': 'faq',
                'confidence': score,
                'intent': '专业咨询',
                'latency_ms': elapsed,
            }

        logger.info("[路由] FAQ 未命中 (score=%.4f)，进入意图分类", score)
    except Exception as e:
        logger.warning("[降级] FAQ 异常，跳过: %s", e)

    # ════════════════════════════════════════════════════════════════
    # 第二级：BERT 意图分类
    # ════════════════════════════════════════════════════════════════
    try:
        classifier = _get_classifier()
        intent = classifier.predict_category(query_str)
        logger.info("[路由] BERT 意图分类: %s", intent)
    except Exception as e:
        logger.warning("[降级] BERT 分类异常，默认按'专业咨询'处理: %s", e)
        intent = '专业咨询'

    # ════════════════════════════════════════════════════════════════
    # 第三级：所有问题走 RAG 完整流程（检索 + 生成）
    # 绕过 BERT 意图分类结果，确保每次提问都检索知识库
    # ════════════════════════════════════════════════════════════════
    try:
        rag = _get_rag()
        result = rag.query(query_str, history=history)
        elapsed = (time.time() - t_start) * 1000
        rag_answer = result.get('answer', '')
        if rag_answer:
            logger.info(
                "[路由] RAG 回答 | strategy=%s | latency=%.0fms",
                result.get('strategy', '?'), elapsed,
            )
            return {
                'answer': rag_answer,
                'source': 'rag',
                'confidence': None,
                'intent': intent,
                'latency_ms': elapsed,
            }
        else:
            raise RuntimeError("RAG 返回空回答")
    except Exception as e:
        logger.error("[路由] RAG 失败: %s", e)

    # ════════════════════════════════════════════════════════════════
    # 最终兜底：LLM 再尝试
    # ════════════════════════════════════════════════════════════════
    try:
        llm = _get_llm()
        fallback_prompt = (
            f"用户问了一个金融相关问题，但我们的知识库暂时无法检索到相关资料。"
            f"请根据你的知识尽力回答，如果确实不知道请诚实说明。\n\n"
            f"用户问题: {query_str}"
        )
        answer = llm.call_with_retry(
            "你是一个诚实的金融助手。",
            fallback_prompt,
            temperature=0.3, max_tokens=1024,
        )
        elapsed = (time.time() - t_start) * 1000
        if answer:
            return {
                'answer': answer,
                'source': 'llm_direct',
                'confidence': None,
                'intent': intent,
                'latency_ms': elapsed,
            }
    except Exception:
        pass

    elapsed = (time.time() - t_start) * 1000
    return {
        'answer': (
            "抱歉，系统暂时无法处理您的问题。请稍后重试，或尝试：\n"
            "  1. 更换关键词重新提问\n"
            "  2. 简化问题的表述\n"
            "  3. 联系人工客服获取帮助"
        ),
        'source': 'error',
        'confidence': None,
        'intent': intent,
        'latency_ms': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 流式查询函数
# ═══════════════════════════════════════════════════════════════════════════

def query_stream(user_query: str, history: str = ""):
    """流式查询入口，与 query() 相同的三级级联，但 RAG/LLM 回答逐 token yield。

    Yields:
        {'type': 'status', 'step': str, 'value': str}
        {'type': 'token',  'content': str}
        {'type': 'done',   'source': str, 'intent': str, 'confidence': float|None,
                           'latency_ms': float}
    """
    t_start = time.time()
    query_str = user_query.strip()

    if not query_str:
        yield {'type': 'status', 'step': 'error', 'value': '空查询'}
        yield {
            'type': 'done', 'source': 'error', 'intent': '未知',
            'confidence': None, 'latency_ms': (time.time() - t_start) * 1000,
        }
        return

    if _needs_clarification(query_str):
        elapsed = (time.time() - t_start) * 1000
        logger.info("[路由·流式] 输入信息不足，返回补充问题提示 | query=%r", query_str)
        yield {'type': 'status', 'step': 'clarification', 'value': '请补充问题'}
        yield {'type': 'token', 'content': CLARIFICATION_ANSWER}
        yield {
            'type': 'done',
            'source': 'clarification',
            'intent': '待补充',
            'confidence': None,
            'latency_ms': elapsed,
        }
        return

    fixed_reply = _get_fixed_chat_reply(query_str)
    if fixed_reply:
        elapsed = (time.time() - t_start) * 1000
        logger.info("[路由·流式] 闲聊固定话术命中 | query=%r", query_str)
        yield {'type': 'status', 'step': 'smalltalk', 'value': '客服应答'}
        yield {'type': 'token', 'content': fixed_reply}
        yield {
            'type': 'done',
            'source': 'smalltalk',
            'intent': '闲聊',
            'confidence': None,
            'latency_ms': elapsed,
        }
        return

    intent = '专业咨询'

    # ════════════════════════════════════════════════════════════════
    # 第一级：FAQ
    # ════════════════════════════════════════════════════════════════
    try:
        searcher = _get_searcher()
        answer, score = searcher.search(query_str)

        if answer is not None:
            elapsed = (time.time() - t_start) * 1000
            logger.info("[路由·流式] FAQ 命中 | score=%.4f | latency=%.0fms", score, elapsed)
            yield {'type': 'status', 'step': 'faq', 'value': f'score={score:.4f}'}
            yield {'type': 'token', 'content': answer}
            yield {
                'type': 'done',
                'source': 'faq',
                'intent': '专业咨询',
                'confidence': score,
                'latency_ms': elapsed,
            }
            return

        logger.info("[路由·流式] FAQ 未命中 (score=%.4f)，进入意图分类", score)
    except Exception as e:
        logger.warning("[降级·流式] FAQ 异常: %s", e)

    # ════════════════════════════════════════════════════════════════
    # 第二级：BERT 意图分类
    # ════════════════════════════════════════════════════════════════
    try:
        classifier = _get_classifier()
        intent = classifier.predict_category(query_str)
        logger.info("[路由·流式] BERT 意图分类: %s", intent)
    except Exception as e:
        logger.warning("[降级·流式] BERT 异常: %s", e)
        intent = '专业咨询'

    yield {'type': 'status', 'step': 'intent', 'value': intent}

    # ════════════════════════════════════════════════════════════════
    # 第三级：所有问题走 RAG 流式流程（检索 + 生成）
    # 绕过 BERT 意图分类结果，确保每次提问都检索知识库
    # ════════════════════════════════════════════════════════════════
    try:
        rag = _get_rag()
        for event in rag.query_stream(query_str, history=history):
            if event['type'] == 'done':
                yield {
                    'type': 'done',
                    'source': 'rag',
                    'intent': event.get('intent', intent),
                    'confidence': None,
                    'latency_ms': (time.time() - t_start) * 1000,
                }
            else:
                yield event
        return
    except Exception as e:
        logger.error("[路由·流式] RAG 流式失败: %s", e)

    # ════════════════════════════════════════════════════════════════
    # 最终兜底
    # ════════════════════════════════════════════════════════════════
    try:
        llm = _get_llm()
        for token in llm.call_stream(
            "你是一个诚实的金融助手。",
            f"用户问题: {query_str}\n\n请根据你的知识尽力回答。",
            temperature=0.3, max_tokens=1024,
        ):
            yield {'type': 'token', 'content': token}
    except Exception:
        yield {'type': 'token', 'content': '抱歉，系统暂时无法处理您的问题，请稍后重试。'}

    yield {
        'type': 'done',
        'source': 'error',
        'intent': intent,
        'confidence': None,
        'latency_ms': (time.time() - t_start) * 1000,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 交互模式
# ═══════════════════════════════════════════════════════════════════════════

def _print_result(result: Dict, verbose: bool = False):
    """格式化打印查询结果。"""
    source_labels = {
        'clarification': '[提示] 请补充问题',
        'smalltalk': '[客服] 固定话术',
        'faq': '[FAQ] 知识库匹配',
        'llm_direct': '[LLM] 大模型直接回答',
        'rag': '[RAG] 检索增强生成',
        'error': '[ERROR] 系统错误',
    }
    source_label = source_labels.get(result['source'], result['source'])

    print(f"\n  {SEP}")
    print(f"  来源: {source_label}")
    if result.get('confidence') is not None:
        print(f"  置信度: {result['confidence']:.4f}")
    print(f"  延迟: {result['latency_ms']:.0f}ms")
    if verbose:
        print(f"  意图: {result.get('intent', '?')}")
    print(f"  {SEP}")
    print(f"  {result['answer']}")
    print(f"  {SEP}")


def _update_history(history_parts, user_input, answer):
    """更新对话历史列表（最近 5 轮）。"""
    answer_brief = answer[:200].replace('\n', ' ')
    history_parts.append(f"用户: {user_input}")
    history_parts.append(f"助手: {answer_brief}")
    if len(history_parts) > 10:
        history_parts[:] = history_parts[-10:]


def _run_streaming_query(query_str: str, history_str: str, verbose: bool = False):
    """执行流式查询并实时打印。"""
    print(f"\n  {SEP}")
    sys.stdout.write("  ")
    sys.stdout.flush()

    metadata = {}
    answer_parts = []
    first_token = True

    for event in query_stream(query_str, history=history_str):
        if event['type'] == 'status':
            if verbose:
                step = event.get('step', '?')
                value = event.get('value', '?')
                if first_token:
                    sys.stdout.write(f"\n  [{step}] {value}")
                else:
                    sys.stdout.write(f"\n  [{step}] {value}")
                sys.stdout.flush()

        elif event['type'] == 'token':
            if first_token:
                first_token = False
            sys.stdout.write(event['content'])
            sys.stdout.flush()
            answer_parts.append(event['content'])

        elif event['type'] == 'done':
            metadata = event

    full_answer = ''.join(answer_parts)
    # 更新对话历史
    # (history_parts is updated in the caller)

    print()
    print(f"  {SEP}")
    source_labels = {
        'clarification': '[提示] 请补充问题',
        'smalltalk': '[客服] 固定话术',
        'faq': '[FAQ] 知识库匹配',
        'llm_direct': '[LLM] 大模型直接回答',
        'rag': '[RAG] 检索增强生成',
        'error': '[ERROR] 系统错误',
    }
    source = source_labels.get(metadata.get('source', ''), metadata.get('source', '?'))
    print(f"  来源: {source}  |  延迟: {metadata.get('latency_ms', 0):.0f}ms")
    if verbose and metadata.get('intent'):
        print(f"  意图: {metadata.get('intent', '?')}")
    if metadata.get('confidence') is not None:
        print(f"  置信度: {metadata['confidence']:.4f}")
    print(f"  {SEP}")

    return full_answer


def init_components(show_progress: bool = True):
    """预加载所有组件，后续查询零等待。

    在项目启动时调用，将模型加载、连接建立等耗时操作提前完成。
    """
    components = []

    # 1. FAQ BM25 — Redis/MySQL 连接 + 语料加载
    try:
        if show_progress:
            print("  [1/4] 加载 FAQ (BM25)...", end=' ', flush=True)
        t0 = time.time()
        _get_searcher()
        elapsed = (time.time() - t0) * 1000
        components.append(('FAQ (BM25)', True, f'{elapsed:.0f}ms'))
        if show_progress:
            print(f"OK ({elapsed:.0f}ms)")
    except Exception as e:
        components.append(('FAQ (BM25)', False, str(e)))
        if show_progress:
            print(f"WARN: {e}")

    # 2. BERT 意图分类器 — 模型加载 (~3s)
    try:
        if show_progress:
            print("  [2/4] 加载 BERT 意图分类器...", end=' ', flush=True)
        t0 = time.time()
        _get_classifier()
        elapsed = (time.time() - t0) * 1000
        components.append(('BERT 分类器', True, f'{elapsed:.0f}ms'))
        if show_progress:
            print(f"OK ({elapsed:.0f}ms)")
    except Exception as e:
        components.append(('BERT 分类器', False, str(e)))
        if show_progress:
            print(f"WARN: {e}")

    # 3. RAG 系统 — LLM 客户端 + Milvus + BGE-M3 + Reranker (~10s)
    try:
        if show_progress:
            print("  [3/4] 加载 RAG 系统 (LLM + Milvus + 向量模型)...", end=' ', flush=True)
        t0 = time.time()
        _get_rag()
        elapsed = (time.time() - t0) * 1000
        components.append(('RAG 系统', True, f'{elapsed:.0f}ms'))
        if show_progress:
            print(f"OK ({elapsed:.0f}ms)")
    except Exception as e:
        components.append(('RAG 系统', False, str(e)))
        if show_progress:
            print(f"WARN: {e}")

    # 4. LLM 客户端 — API 验证
    try:
        if show_progress:
            print("  [4/4] 加载 LLM 客户端...", end=' ', flush=True)
        t0 = time.time()
        _get_llm()
        elapsed = (time.time() - t0) * 1000
        components.append(('LLM 客户端', True, f'{elapsed:.0f}ms'))
        if show_progress:
            print(f"OK ({elapsed:.0f}ms)")
    except Exception as e:
        components.append(('LLM 客户端', False, str(e)))
        if show_progress:
            print(f"WARN: {e}")

    return components


def run_interactive(verbose: bool = False, stream: bool = True):
    """交互式控制台，支持多轮对话和流式输出。"""
    print()
    print(DSEP)
    print("  FinRAG 金融智能问答系统")
    print(DSEP)
    print("  第一级  FAQ 知识库匹配         (~10ms)")
    print("  第二级  BERT 意图分类          (~50ms)")
    print("  第三级  LLM 直接回答 / RAG 检索增强")
    if stream:
        print("  输出模式: 流式 (token-by-token)")
    print(DSEP)
    print()
    print("  正在初始化所有组件...")
    print()

    init_components(show_progress=True)

    print()
    print("  输入问题进行查询，输入 :q 退出，输入 :help 查看帮助")
    print(DSEP)

    history_parts = []

    while True:
        try:
            user_input = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  再见！")
            break

        if not user_input:
            continue

        if user_input == ':q':
            print("  再见！")
            break

        if user_input == ':help':
            print()
            print("  命令:")
            print("    :q       退出系统")
            print("    :help    显示此帮助")
            print("    :clear   清除对话历史")
            print("    :verbose 切换详细模式")
            print()
            print("  直接输入您的问题即可获得回答，系统会自动判断：")
            print("    - FAQ 命中 → 直接返回知识库答案")
            print("    - 通用知识 → 大模型直接回答")
            print("    - 专业咨询 → RAG 检索增强回答")
            continue

        if user_input == ':clear':
            history_parts.clear()
            print("  [OK] 对话历史已清除")
            continue

        if user_input == ':verbose':
            verbose = not verbose
            status = "开启" if verbose else "关闭"
            print(f"  [OK] 详细模式已{status}")
            continue

        # ── 构建 history 字符串 ────────────────────────────────────
        history_str = '\n'.join(history_parts) if history_parts else ''

        # ── 执行查询 ───────────────────────────────────────────────
        if stream:
            full_answer = _run_streaming_query(user_input, history_str, verbose)
            _update_history(history_parts, user_input, full_answer)
        else:
            result = query(user_input, history=history_str)
            _print_result(result, verbose=verbose)
            _update_history(history_parts, user_input, result['answer'])


# ═══════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    use_stream = '--no-stream' not in sys.argv  # 默认启用流式

    # 提取 query（去掉脚本名和 flag 参数）
    args = [a for a in sys.argv[1:] if not a.startswith('-')]

    if args:
        # ── 命令行模式：静默预加载 + 单次查询 ────────────────────
        init_components(show_progress=verbose)
        query_text = ' '.join(args)
        if use_stream:
            if verbose:
                print(f"查询: {query_text}")
                print(SEP)
            _run_streaming_query(query_text, '', verbose)
        else:
            if verbose:
                print(f"查询: {query_text}")
                print(SEP)
            result = query(query_text)
            _print_result(result, verbose=verbose)
    else:
        # ── 交互模式 ──────────────────────────────────────────────
        run_interactive(verbose=verbose, stream=use_stream)
