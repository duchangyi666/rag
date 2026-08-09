"""
BM25 检索模块

职责：
    1. 首次加载时从 MySQL 读取问题数据，分词后缓存到 Redis
    2. 接收用户查询，先用 Redis 答案缓存命中（精确匹配），
       未命中则通过 BM25 检索最佳匹配问题，再从 MySQL 获取答案
    3. 当置信度（归一化得分）低于阈值时返回 None，交由上游 RAG 模块处理

数据流：
    MySQL ──(原始问题)──▶ jieba 分词 ──▶ Redis 缓存
                      │                    │
                      ▼                    ▼
                  用户查询 ──▶ 缓存命中? ──▶ 返回答案
                                │ 否
                                ▼
                         BM25 检索 ──▶ 得分 ≥ 阈值? ──▶ MySQL 查答案
                                            │ 否
                                            ▼
                                      返回 None → RAG

Usage:
    from mysql_qa.retriveal.bm25_search import BM25Searcher

    searcher = BM25Searcher()
    answer, score = searcher.search("什么是ROE")

    if answer:
        print(f"FAQ 命中 (score={score:.2f}): {answer}")
    else:
        print("FAQ 未覆盖，转交 RAG")
"""

import json
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from base.config import config
from base.logger import setup_logger
from mysql_qa.retriveal.text_utils import (
    has_sufficient_query_signal,
    tokenize_for_bm25,
)

# 模块级 logger
logger = setup_logger('fin_rag.bm25_search')

# ── Redis Key 常量（与 redis_client.py 保持同步） ─────────────────────
KEY_ANSWER_PREFIX = 'finrag_question:'     # String: 问题→答案（唯一带 TTL）
KEY_QUESTIONS_LIST = 'all_questions'      # List: 全量问题
KEY_TOKENIZER = 'tokenizer_questions'     # String: JSON 嵌套分词列表


class BM25Searcher:
    """
    BM25 FAQ 检索器。

    整合 Redis 缓存、BM25 检索、MySQL 回源三条路线：

    ┌─ 用户查询 ──────────────────────────────────────────┐
    │                                                      │
    │  路线① 原始 query → Redis 精确匹配                   │
    │     ├─ 命中 → 直接返回 (prob=1.0)                    │
    │     └─ 未命中 ↓                                      │
    │  路线② query 预处理(lower+分词) → BM25 + softmax     │
    │     ├─ prob ≥ 0.85                                   │
    │     │   → 取匹配到的原始问题                          │
    │     │   → Redis 查答案 → 有则返回                     │
    │     │   → 无则 MySQL 查 → 回写 Redis → 返回           │
    │     └─ prob < 0.85                                   │
    │        → 路线③ 返回 None → 转交 RAG                  │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        threshold: float = None,
        k1: float = None,
        b: float = None,
        softmax_temperature: float = None,
    ):
        """
        参数默认值从 config.ini [bm25] section 加载。
        也可通过构造函数参数覆盖配置值。

        Args:
            threshold:           softmax 概率阈值 [0, 1]
            k1:                  BM25 词频饱和参数
            b:                   BM25 长度归一化参数
            softmax_temperature: softmax 温度系数
        """
        self.threshold = threshold if threshold is not None else config.bm25_threshold
        self.k1 = k1 if k1 is not None else config.bm25_k1
        self.b = b if b is not None else config.bm25_b
        self.temperature = softmax_temperature if softmax_temperature is not None else config.bm25_softmax_temperature
        self.min_token_coverage = config.bm25_min_token_coverage

        # 语料库元数据（与 BM25 corpus 索引一一对应）
        self.questions: list[str] = []              # 原始问题文本
        self.qids: list[str] = []                   # 问题 ID
        self.tokenized_corpus: list[list[str]] = []  # 分词后的问题列表

        # BM25 模型（首次搜索时延迟构建）
        self.bm25: Optional[BM25Okapi] = None
        self._built = False

        # Redis / MySQL 客户端（延迟初始化）
        self._redis_client = None

    # ── 公共接口 ──────────────────────────────────────────────────────

    def search(self, query: str) -> tuple[Optional[str], float]:
        """
        检索最匹配的答案。

        Args:
            query: 用户原始查询（中文自然语言）

        Returns:
            (answer, prob):
                answer 不为 None 表示 FAQ 命中，prob∈[0,1] 为 softmax 后 top-1 的概率
                answer 为 None 表示 FAQ 未覆盖 → 转交 RAG
        """
        if not query or not query.strip():
            logger.warning("[BM25] 查询为空，跳过检索")
            return None, 0.0

        query = query.strip()
        logger.debug("[BM25] ═══════════════════════════════════════")
        logger.debug("[BM25] 用户查询: %s", query)

        # ── 路线①：原始 query 直接 Redis 精确匹配 ──────────────────
        logger.debug("[BM25] ① Redis 精确匹配 ...")
        answer = self._try_redis_cache(query)
        if answer is not None:
            logger.info("[BM25] ✅ 路线① Redis 精确命中 → prob=1.0 | answer len=%d", len(answer))
            return answer, 1.0
        logger.debug("[BM25] ① Redis 未命中，进入 BM25 语义检索")

        # ── ② 查询预处理：去除低信息词，再检查检索信号 ─────────────
        logger.debug("[BM25] ② 预处理 ...")
        query_lower = query.lower()
        query_tokens = tokenize_for_bm25(query_lower)
        if not has_sufficient_query_signal(query_tokens):
            logger.warning(
                "[BM25] ❌ 有效检索词不足，拒绝 FAQ 模糊匹配 | query:%s | tokens:%s",
                query[:50], query_tokens,
            )
            return None, 0.0
        logger.debug("[BM25] ② lower: '%s'", query_lower)
        logger.debug("[BM25] ② tokens: %s", query_tokens)

        # ── 构建/加载 BM25 模型 ────────────────────────────────────
        self._ensure_built()
        corpus_size = len(self.tokenized_corpus)
        if self.bm25 is None or corpus_size == 0:
            logger.warning("[BM25] ❌ BM25 模型为空（语料:%d），无法检索", corpus_size)
            return None, 0.0
        logger.debug(
            "[BM25] 模型就绪 | 语料:%d | threshold:%.2f | coverage:%.2f | "
            "T:%.2f | k1:%.2f | b:%.2f",
            corpus_size, self.threshold, self.min_token_coverage,
            self.temperature, self.k1, self.b,
        )

        # ── ③ BM25 与全部语料计算原始得分 ────────────────────────
        logger.debug("[BM25] ③ BM25 与全部 %d 条分词问题计算得分 ...", corpus_size)
        raw_scores: np.ndarray = self.bm25.get_scores(query_tokens)

        scored_docs = sorted(
            [(i, float(raw_scores[i]),
              self.questions[i] if i < len(self.questions) else '?')
             for i in range(len(raw_scores))],
            key=lambda x: x[1], reverse=True,
        )

        logger.debug("[BM25] ③ BM25 raw scores top-5:")
        for rank, (idx, raw, q) in enumerate(scored_docs[:5], 1):
            logger.debug("[BM25]    #%d idx=%d raw=%.6f | %s", rank, idx, raw, q[:60])

        # ── ④ softmax 概率分布 ────────────────────────────────────
        logger.debug("[BM25] ④ softmax (T=%.2f) ...", self.temperature)
        probs = self._softmax(raw_scores)

        best_idx = scored_docs[0][0]
        best_prob = float(probs[best_idx])
        best_question = scored_docs[0][2]
        raw_best = scored_docs[0][1]
        best_tokens = (
            set(self.tokenized_corpus[best_idx])
            if best_idx < len(self.tokenized_corpus)
            else set()
        )
        unique_query_tokens = set(query_tokens)
        matched_tokens = unique_query_tokens & best_tokens
        token_coverage = (
            len(matched_tokens) / len(unique_query_tokens)
            if unique_query_tokens
            else 0.0
        )

        prob_ranked = sorted(
            [(i, float(probs[i]),
              self.questions[i] if i < len(self.questions) else '?')
             for i in range(len(probs))],
            key=lambda x: x[1], reverse=True,
        )

        logger.debug("[BM25] ④ softmax 概率 top-5:")
        for rank, (idx, prob, q) in enumerate(prob_ranked[:5], 1):
            tag = " ◀ top-1" if idx == best_idx else ""
            logger.debug("[BM25]    #%d idx=%d prob=%.6f | %s%s", rank, idx, prob, q[:60], tag)

        # ── ⑤ 词覆盖率 + softmax 阈值联合判断 ─────────────────────
        logger.debug(
            "[BM25] ⑤ 联合判断 | coverage=%.3f vs %.2f | "
            "best_prob=%.6f vs %.2f | matched=%s",
            token_coverage, self.min_token_coverage,
            best_prob, self.threshold, sorted(matched_tokens),
        )

        if raw_best <= 0 or token_coverage < self.min_token_coverage:
            logger.warning(
                "[BM25] ❌ 有效词覆盖不足 → 转 RAG | coverage=%.3f < %.2f | "
                "raw=%.4f | query_tokens=%s | matched=%s | best_match:'%s'",
                token_coverage, self.min_token_coverage, raw_best,
                query_tokens, sorted(matched_tokens), best_question[:50],
            )
            return None, best_prob

        if best_prob < self.threshold:
            gap = prob_ranked[1][1] if len(prob_ranked) >= 2 else 0
            logger.warning(
                "[BM25] ❌ 路线③ 低于阈值 → 转 RAG | "
                "prob=%.6f < %.2f | best_match:'%s' | raw=%.4f | top1-top2_gap=%.6f",
                best_prob, self.threshold, best_question[:50], raw_best, best_prob - gap,
            )
            return None, best_prob

        # ── 路线②：BM25 命中，获取答案 ────────────────────────────
        logger.info("[BM25] ✅ 路线② BM25+softmax 命中 | prob=%.6f ≥ %.2f",
                    best_prob, self.threshold)
        logger.debug("[BM25] ② 最佳匹配问题: %s", best_question)

        # ⑥ 拿匹配到的原始问题查 Redis
        logger.debug("[BM25] ⑥ 以匹配问题查 Redis ...")
        answer = self._try_redis_cache(best_question)
        if answer is not None:
            logger.info("[BM25] ✅ Redis 缓存命中 → answer len=%d", len(answer))
            return answer, best_prob

        # ⑦ Redis 无缓存，降级 MySQL
        logger.debug("[BM25] ⑦ Redis 无缓存 → 降级 MySQL ...")
        answer = self._get_answer_from_mysql(best_question)
        if answer is not None:
            self._cache_answer(best_question, answer)
            logger.info("[BM25] ✅ MySQL 命中 → 已回写 Redis | answer len=%d", len(answer))
            return answer, best_prob

        logger.warning("[BM25] ❌ MySQL 也未查到 | question:%s", best_question[:50])
        return None, 0.0

    def batch_search(
        self,
        queries: list[str],
    ) -> list[tuple[Optional[str], float]]:
        """
        批量检索，单次模型构建、多次查询。

        Args:
            queries: 查询列表

        Returns:
            [(answer, score), ...] 与输入同序
        """
        self._ensure_built()
        return [self.search(q) for q in queries]

    # ── 内部：缓存交互 ────────────────────────────────────────────────

    def _try_redis_cache(self, query: str) -> Optional[str]:
        """
        尝试从 Redis 精确匹配缓存获取答案。

        Key: finrag:answer:{question}

        Returns:
            命中返回答案字符串，未命中返回 None
        """
        try:
            client = self._get_redis_client()
            if client is None:
                return None

            key = f'{KEY_ANSWER_PREFIX}{query}'
            # 使用 pipeline 将 GET + EXPIRE 合并为一次网络往返
            pipe = client.client.pipeline(transaction=False)
            pipe.get(key)
            pipe.expire(key, config.redis_ex)
            results = pipe.execute()
            answer = results[0] if results else None
            if answer:
                return answer
        except Exception as e:
            logger.warning("Redis 缓存查询失败 - error:%s", e)

        return None

    def _cache_answer(self, question: str, answer: str) -> None:
        """将问答对写入 Redis 缓存。"""
        try:
            client = self._get_redis_client()
            if client is None:
                return
            client.client.set(
                f'{KEY_ANSWER_PREFIX}{question}',
                answer,
                ex=config.redis_ex,
            )
        except Exception as e:
            logger.warning("Redis 缓存写入失败 - error:%s", e)

    def _get_answer_from_mysql(self, question: str) -> Optional[str]:
        """
        从 MySQL 精确查询答案。

        Returns:
            命中返回 answer 字符串，未命中返回 None
        """
        try:
            from mysql_qa.db.mysql_client import get_mysql_client

            mysql_client = get_mysql_client()
            row = mysql_client.get_answer_by_question(question)
            if row:
                return row.get('answer', '')
        except Exception as e:
            logger.error("MySQL 查询失败 - question:%s error:%s", question[:50], e)

        return None

    # ── 内部：模型构建 ────────────────────────────────────────────────

    def _ensure_built(self) -> None:
        """确保 BM25 模型已构建（首次调用时触发）。"""
        if self._built:
            return
        self._build_model()

    def _build_model(self) -> None:
        """
        构建 BM25 模型。

        优先从 Redis 读取已分词数据（由 RedisClient.load_all_to_redis 写入），
        不可用时降级为 MySQL + jieba 实时分词。
        """
        # ── 尝试从 Redis 加载 ────────────────────────────────────
        if self._load_from_redis():
            self._build_bm25()
            logger.info(
                "BM25 模型构建完成（数据源: Redis） - 文档数:%d",
                len(self.tokenized_corpus),
            )
            return

        # ── 降级：MySQL + jieba 实时分词 ─────────────────────────
        logger.info("Redis 无分词数据，从 MySQL 构建 BM25 模型")
        self._build_from_mysql()

        if self.tokenized_corpus:
            self._build_bm25()
            logger.info(
                "BM25 模型构建完成（数据源: MySQL） - 文档数:%d",
                len(self.tokenized_corpus),
            )
        else:
            logger.warning("MySQL 无数据，BM25 模型为空")

        self._built = True

    def _build_bm25(self) -> None:
        """用当前 tokenized_corpus 初始化 rank_bm25.BM25Okapi。"""
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )

    def _load_from_redis(self) -> bool:
        """
        从 Redis 加载已缓存的语料数据。

        Returns:
            True 表示成功加载到数据
        """
        try:
            client = self._get_redis_client()
            if client is None:
                return False

            # 读取分词嵌套列表：JSON 字符串 → [[tokens1], [tokens2], ...]
            token_json = client.client.get(KEY_TOKENIZER)
            if not token_json:
                logger.debug("Redis key %s 为空", KEY_TOKENIZER)
                return False

            tokenized: list[list[str]] = json.loads(token_json)
            if not tokenized:
                return False

            # 读取问题列表
            questions = client.client.lrange(KEY_QUESTIONS_LIST, 0, -1)

            self.tokenized_corpus = tokenized
            self.qids = [str(i) for i in range(len(tokenized))]
            self.questions = questions if questions else []
            logger.debug("从 Redis 加载 - 问题:%d 分词:%d", len(self.questions), len(tokenized))
            return True

        except Exception as e:
            logger.warning("从 Redis 加载语料失败 - error:%s", e)
            return False

    def _build_from_mysql(self) -> None:
        """从 MySQL 读取全量问题并 jieba 分词构建语料库。"""
        try:
            from mysql_qa.db.mysql_client import get_mysql_client

            mysql_client = get_mysql_client()
            records = mysql_client.get_all_questions()
            if not records:
                logger.warning("MySQL 查询结果为空")
                return

            self.questions = []
            self.qids = []
            self.tokenized_corpus = []

            for r in records:
                qid = r.get('id')
                question = (r.get('question') or '').strip()
                if not question or qid is None:
                    continue

                self.questions.append(question)
                self.qids.append(str(qid))

                tokens = tokenize_for_bm25(question)
                self.tokenized_corpus.append(tokens)

            logger.debug(
                "从 MySQL 构建语料 - 问题:%d",
                len(self.tokenized_corpus),
            )

        except Exception as e:
            logger.error("从 MySQL 构建语料失败 - error:%s", e)

    # ── 内部：softmax 概率计算 ──────────────────────────────────────────

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """
        对 BM25 原始得分做 softmax，转换为概率分布。

        P(d_i | q) = exp(score_i / T) / Σ_j exp(score_j / T)

        其中 T 为温度系数：
          - T < 1   → 分布更尖锐，top-1 概率更高（更严格）
          - T = 1   → 标准 softmax
          - T > 1   → 分布更平滑（更宽容）

        Args:
            scores: BM25 原始得分数组，shape=(N,)

        Returns:
            softmax 概率数组，shape=(N,) 且 Σ=1
        """
        # 减去最大值保证数值稳定性
        scores_shifted = scores - np.max(scores)
        exp_scores = np.exp(scores_shifted / self.temperature)
        probs = exp_scores / np.sum(exp_scores)

        logger.debug(
            "[BM25] softmax 完成 - max_raw:%.4f min_raw:%.4f sum_prob:%.6f",
            np.max(scores), np.min(scores), float(np.sum(probs)),
        )
        return probs

    # ── 内部：客户端获取 ──────────────────────────────────────────────

    def _get_redis_client(self):
        """懒加载 Redis 客户端，连接失败返回 None。"""
        if self._redis_client is not None:
            return self._redis_client

        try:
            from mysql_qa.cache.redis_client import RedisClient
            self._redis_client = RedisClient()
            return self._redis_client
        except Exception as e:
            logger.warning("Redis 不可用，仅使用 MySQL 检索 - error:%s", e)
            return None

    def close(self) -> None:
        """释放资源。"""
        if self._redis_client:
            try:
                self._redis_client.close()
            except Exception as e:
                logger.warning("关闭 Redis 连接失败 - error:%s", e)
            self._redis_client = None


# ── 模块级单例 ──────────────────────────────────────────────────────
_searcher_instance: Optional[BM25Searcher] = None


def get_searcher(
    threshold: float = None,
) -> BM25Searcher:
    """
    获取 BM25Searcher 单例。

    首次调用时构建模型，后续复用。
    参数默认值从 config.ini [bm25] section 加载。

    Args:
        threshold: softmax 概率阈值，None 则使用配置文件值

    Returns:
        BM25Searcher 实例
    """
    global _searcher_instance
    if _searcher_instance is None:
        _searcher_instance = BM25Searcher(threshold=threshold)
    return _searcher_instance


# ── 自测入口 ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("BM25 Searcher 自测")
    print("=" * 60)

    try:
        searcher = BM25Searcher()

        test_queries = [
            "什么是ROE",        # 精确匹配（Redis 缓存）
            "ROE是什么意思",    # BM25 模糊匹配
            "怎么开户",         # FAQ 未覆盖
        ]

        for q in test_queries:
            answer, score = searcher.search(q)
            print()
            print(f"查询: {q}")
            print(f"得分: {score:.4f}")
            print(f"结果: {answer if answer else '未覆盖 → 转 RAG'}")

        searcher.close()
        print("\n测试完成")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
