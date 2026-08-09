"""
Redis 缓存客户端

负责：
  1. Redis 连接初始化与管理
  2. 将 MySQL 全量问题列表和分词数据加载到 Redis（启动时一次性加载）
  3. 提供问题 → 答案的缓存查询能力（问答对由 FAQ 访问时按需写入，非启动全量加载）

Redis 数据结构：
  all_questions           → List   [问题1, 问题2, ...]         无 TTL
  tokenizer_questions     → String [[分词1], [分词2], ...]     无 TTL（JSON 嵌套列表）
  finrag_question:{问题}   → String 答案                       有 TTL（FAQ 命中时按需写入）

Usage:
    from mysql_qa.cache.redis_client import RedisClient
    client = RedisClient()
    client.load_all_to_redis()
    answer = client.get_answer_by_question("基金到底是什么？")
"""

import json
import redis

from base.config import config
from base.logger import setup_logger
from mysql_qa.retriveal.text_utils import tokenize_for_bm25

# 模块级 logger
logger = setup_logger('fin_rag.redis_client')

# ── Redis Key 命名常量 ────────────────────────────────────────────────
KEY_ALL_QUESTIONS = 'all_questions'                     # List: 所有问题文本
KEY_TOKENIZER = 'tokenizer_questions'                   # String: JSON 嵌套分词列表
KEY_QA_PREFIX = 'finrag_question:'                      # String: 问题→答案（唯一带 TTL）


class RedisClient:
    """Redis 缓存客户端，封装问答数据的缓存加载与查询。"""

    def __init__(self):
        """
        初始化 Redis 连接，参数从 config.ini [redis] section 加载。

        Raises:
            redis.RedisError: 连接失败时抛出（调用方可捕获后降级处理）。
        """
        try:
            self.client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                password=config.redis_password if config.redis_password else None,
                db=config.redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            self.client.ping()
            self.expire_seconds = config.redis_ex
            logger.info(
                "Redis 连接成功 - host:%s port:%d db:%d",
                config.redis_host, config.redis_port, config.redis_db,
            )
        except redis.RedisError as e:
            logger.error(
                "Redis 连接失败 - host:%s port:%d error:%s",
                config.redis_host, config.redis_port, e,
            )
            raise

    # ── 公共方法：一键全量加载 ────────────────────────────────────────

    def load_all_to_redis(self, table_name: str = 'qa_records') -> dict:
        """
        将 MySQL 全量问题列表和分词数据加载到 Redis（启动时调用）。

        注意：问答对（finrag_question:{问题} → 答案）不在此处全量加载，
        而是由 FAQ 访问时按需写入 Redis，避免启动时大量写入带 TTL 的数据。

          1. all_questions → 全量问题列表（无 TTL）
          2. tokenizer_questions → jieba 分词嵌套列表（无 TTL）

        Args:
            table_name: MySQL 表名

        Returns:
            {'questions': int, 'tokens': int}
        """
        from mysql_qa.db.mysql_client import get_mysql_client

        mysql_client = get_mysql_client()

        records = mysql_client.get_all_questions(table_name)
        if not records:
            logger.warning("MySQL 查询结果为空，跳过 Redis 加载")
            return {'questions': 0, 'tokens': 0}

        logger.info("开始全量加载 Redis，共 %d 条记录", len(records))

        q_count = self._load_questions_list(records)
        t_count = self._load_tokenizer(records)

        qa_count = self._load_qa_pairs(records)

        result = {'questions': q_count, 'tokens': t_count, 'qa_pairs': qa_count}
        logger.info(
            "Redis 全量加载完成 - 问题:%d 分词:%d 问答:%d",
            q_count, t_count, qa_count,
        )
        return result

    # ── 查询接口 ──────────────────────────────────────────────────────

    def get_answer_by_question(self, question: str) -> str | None:
        """
        通过问题从 Redis 缓存获取答案。
        命中后刷新 TTL，保持高频数据常驻。

        Args:
            question: 问题文本

        Returns:
            命中返回答案，未命中或异常返回 None
        """
        if not question:
            return None

        key = f'{KEY_QA_PREFIX}{question}'
        try:
            answer = self.client.get(key)
            if answer:
                self.client.expire(key, self.expire_seconds)
                logger.info("Redis 缓存命中 - question:%s", question[:50])
            else:
                logger.debug("Redis 缓存未命中 - question:%s", question[:50])
            return answer
        except redis.RedisError as e:
            logger.error("Redis 查询失败 - question:%s error:%s", question[:50], e)
            return None

    # ── 内部方法：数据加载 ────────────────────────────────────────────

    def _load_qa_pairs(self, records: list[dict]) -> int:
        """
        高频问答对写入 Redis String（唯一带 TTL 的数据）。

        Key:   finrag_question:{问题原文}
        Value: 答案
        TTL:   config.redis_ex 秒（调用时刷新）

        Returns:
            成功写入数量
        """
        count = 0
        pipe = self.client.pipeline(transaction=False)
        batch_size = 100

        for i, record in enumerate(records, 1):
            question = (record.get('question') or '').strip()
            answer = (record.get('answer') or '').strip()
            if not question:
                continue

            key = f'{KEY_QA_PREFIX}{question}'
            pipe.set(key, answer, ex=self.expire_seconds)
            count += 1

            if i % batch_size == 0:
                try:
                    pipe.execute()
                except redis.RedisError as e:
                    logger.error("批量写入问答对失败(第%d批) - error:%s", i // batch_size, e)
                pipe = self.client.pipeline(transaction=False)

        if count % batch_size != 0:
            try:
                pipe.execute()
            except redis.RedisError as e:
                logger.error("批量写入问答对失败(尾批) - error:%s", e)

        logger.info("高频问答对加载完成 - 共 %d 条（TTL=%ds）", count, self.expire_seconds)
        return count

    def _load_questions_list(self, records: list[dict]) -> int:
        """
        全量问题写入 Redis List。

        Key: all_questions（List）
        无 TTL，持久保存。

        Returns:
            成功写入数量
        """
        questions = [
            (r.get('question') or '').strip()
            for r in records
            if (r.get('question') or '').strip()
        ]
        if not questions:
            return 0

        try:
            self.client.delete(KEY_ALL_QUESTIONS)
            self.client.rpush(KEY_ALL_QUESTIONS, *questions)
            logger.info("全量问题列表加载完成 - 共 %d 条", len(questions))
        except redis.RedisError as e:
            logger.error("全量问题列表加载失败 - error:%s", e)
            return 0

        return len(questions)

    def _load_tokenizer(self, records: list[dict]) -> int:
        """
        jieba 分词后以嵌套列表 JSON 写入 Redis String。

        Key:   tokenizer_questions
        Value: [[词1, 词2, ...], [词1, 词2, ...], ...]
        无 TTL，持久保存。

        Returns:
            成功分词的问题数量
        """
        nested_tokens: list[list[str]] = []

        for record in records:
            question = (record.get('question') or '').strip()
            if not question:
                continue

            tokens = tokenize_for_bm25(question)
            nested_tokens.append(tokens)

        if not nested_tokens:
            return 0

        try:
            token_json = json.dumps(nested_tokens, ensure_ascii=False)
            self.client.delete(KEY_TOKENIZER)
            self.client.set(KEY_TOKENIZER, token_json)
            logger.info("分词嵌套列表加载完成 - 共 %d 个问题", len(nested_tokens))
        except redis.RedisError as e:
            logger.error("分词嵌套列表加载失败 - error:%s", e)
            return 0

        return len(nested_tokens)

    # ── 工具方法 ──────────────────────────────────────────────────────

    def close(self):
        """关闭 Redis 连接。"""
        if self.client:
            try:
                self.client.close()
                logger.info("Redis 连接已关闭")
            except redis.RedisError as e:
                logger.error("Redis 关闭连接失败 - error:%s", e)


# ── 模块级单例 ──────────────────────────────────────────────────────
_client_instance: RedisClient | None = None


def get_redis_client() -> RedisClient:
    """获取 Redis 客户端单例（懒初始化）。"""
    global _client_instance
    if _client_instance is None:
        _client_instance = RedisClient()
    return _client_instance


if __name__ == '__main__':
    try:
        from mysql_qa.db.mysql_client import get_mysql_client

        mysql_client = get_mysql_client()
        client = RedisClient()
        records = mysql_client.get_all_questions()

        if not records:
            print("MySQL 无数据")
            client.close()
            exit(0)

        # 1. 全量问题列表
        print("=" * 60)
        print("1. all_questions (List)")
        print("=" * 60)
        q_count = client._load_questions_list(records)
        print(f"写入: {q_count} 条")
        all_q = client.client.lrange(KEY_ALL_QUESTIONS, 0, 4)
        for i, q in enumerate(all_q, 1):
            print(f"  [{i}] {q[:60]}")

        # 2. 分词嵌套列表
        print()
        print("=" * 60)
        print("2. tokenizer_questions (JSON 嵌套列表)")
        print("=" * 60)
        t_count = client._load_tokenizer(records)
        print(f"写入: {t_count} 条")
        raw = client.client.get(KEY_TOKENIZER)
        nested = json.loads(raw)
        for i, tokens in enumerate(nested[:3]):
            print(f"  [{i}] {tokens}")

        client.close()
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
