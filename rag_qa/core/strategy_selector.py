"""
策略选择器

负责：
  1. 通过 OpenAI 兼容客户端加载智谱 GLM 大模型
  2. 为 RAG 管道中的各环节提供统一的 LLM 调用入口：
     - 意图识别（FINANCE / GENERAL）
     - Query 改写（12 种改写策略）
     - 最终回答生成

模型配置来自 config.ini [llm] section，API Key 优先从环境变量
ZHIPUAI_API_KEY 读取，fallback 到配置文件中的 api_key。

Usage:
    from rag_qa.core.strategy_selector import StrategySelector

    selector = StrategySelector()
    response = selector.call(system_prompt, user_prompt)
"""

import os
from typing import Optional, Dict, Any

from openai import OpenAI

from base.config import config
from base.logger import setup_logger

logger = setup_logger('fin_rag.strategy_selector')


class StrategySelector:
    """智谱 GLM 大模型客户端，封装 OpenAI 兼容接口。

    初始化时自动：
      1. 从 config.ini [llm] 读取模型名和 Base URL
      2. 从环境变量 ZHIPUAI_API_KEY 或配置文件读取 API Key
      3. 创建 OpenAI 兼容客户端

    Attributes:
        client:     OpenAI 客户端实例
        model:      模型名称（如 glm-4-flash）
        base_url:   API 端点地址
    """

    def __init__(self):
        """初始化 GLM 大模型客户端。

        通过智谱 OpenAI 兼容接口调用 GLM 模型。
        API Key 优先从环境变量 ZHIPUAI_API_KEY 读取，fallback 到配置文件。

        Raises:
            ValueError: API Key 未设置时抛出。
        """
        self.model = config.llm_model
        self.base_url = config.llm_base_url
        self.api_key = config.llm_api_key

        if not self.api_key:
            raise ValueError(
                "API Key 未设置，请通过以下任一方式设置：\n"
                "  1. export ZHIPUAI_API_KEY='your-key'\n"
                "  2. 在 config.ini [llm] 段中设置 api_key = xxx"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=1,
        )

        logger.info(
            "StrategySelector 初始化完成 - model:%s base_url:%s",
            self.model, self.base_url,
        )

    # ── LLM 调用接口 ────────────────────────────────────────────────

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs,
    ) -> Optional[str]:
        """调用 GLM 大模型。

        Args:
            system_prompt: 系统提示词
            user_prompt:   用户提示词
            temperature:   生成温度（默认 0.1，保证输出稳定）
            max_tokens:    最大输出 token 数
            **kwargs:      其他 OpenAI ChatCompletion 参数

        Returns:
            模型返回文本，失败返回 None
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            content = response.choices[0].message.content
            logger.debug(
                "LLM 调用成功 - model:%s tokens:%d",
                self.model,
                response.usage.total_tokens if response.usage else -1,
            )
            return content
        except Exception as e:
            logger.error("LLM 调用失败 - model:%s error:%s", self.model, e)
            return None

    def call_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs,
    ):
        """流式调用 GLM 大模型，逐 token yield。

        Args:
            system_prompt: 系统提示词
            user_prompt:   用户提示词
            temperature:   生成温度
            max_tokens:    最大输出 token 数
            **kwargs:      其他 OpenAI ChatCompletion 参数

        Yields:
            str: 每次 yield 一个文本 token 片段
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
        except Exception as e:
            logger.error("LLM 流式调用失败 - model:%s error:%s", self.model, e)
            raise RuntimeError(f"LLM 流式调用失败: {e}") from e

    def call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
        **kwargs,
    ) -> Optional[str]:
        """带重试的 LLM 调用。

        Args:
            system_prompt: 系统提示词
            user_prompt:   用户提示词
            max_retries:   最大重试次数
            **kwargs:      其他 OpenAI ChatCompletion 参数

        Returns:
            模型返回文本，全部重试失败返回 None
        """
        import time

        for attempt in range(1, max_retries + 1):
            result = self.call(system_prompt, user_prompt, **kwargs)
            if result is not None:
                return result
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning("LLM 调用重试 %d/%d，等待 %ds...", attempt, max_retries, wait)
                time.sleep(wait)

        logger.error("LLM 调用全部 %d 次重试失败", max_retries)
        return None


# ── 模块级单例 ──────────────────────────────────────────────────────
_selector_instance: Optional[StrategySelector] = None


def get_strategy_selector() -> StrategySelector:
    """获取 StrategySelector 单例（懒初始化）。"""
    global _selector_instance
    if _selector_instance is None:
        _selector_instance = StrategySelector()
    return _selector_instance


# ── 自测入口 ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("StrategySelector 自测")
    print("=" * 60)

    # 检查 API Key（环境变量优先，fallback 到配置文件）
    key = config.llm_api_key
    source = '环境变量' if os.environ.get('ZHIPUAI_API_KEY') else '配置文件'
    print(f"API Key: {'已设置 (' + key[:8] + '...)' if key else '未设置'} (来源: {source})")

    if not key:
        print("\n请先设置 API Key:")
        print("  export ZHIPUAI_API_KEY='your-api-key'")
        print("  或在 config.ini [llm] 段中设置 api_key = xxx")
        exit(1)

    # 初始化
    selector = StrategySelector()
    print(f"model: {selector.model}")
    print(f"base_url: {selector.base_url}")

    # 简单测试调用
    system_prompt = "你是一个助手，请用简洁的中文回答。"
    user_prompt = "什么是股票？请用一句话解释。"

    print(f"\n测试调用...")
    result = selector.call(system_prompt, user_prompt, max_tokens=200)

    if result:
        print(f"回答: {result}")
    else:
        print("调用失败")
