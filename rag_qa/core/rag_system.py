"""FinRAG 检索增强生成编排。

该模块连接当前仓库已经存在的三个组件：
  1. VectorStore：BGE-M3 + Milvus 混合检索 + CrossEncoder 精排
  2. rag_prompts：严格基于资料回答并附来源
  3. StrategySelector：智谱 GLM 普通/流式生成

对外提供 query() 和 query_stream()，返回结构与 main.py 约定一致。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from base.logger import setup_logger
from rag_qa.core.rag_prompts import format_prompt
from rag_qa.core.strategy_selector import get_strategy_selector
from rag_qa.core.vector_store import get_vector_store


logger = setup_logger("fin_rag.rag_system")
DEFAULT_TOP_K = 12
DEFAULT_RERANK_TOP_K = 5


class RAGSystem:
    """金融知识库检索与回答生成入口。"""

    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = get_strategy_selector()

    @staticmethod
    def _format_context(results: List[dict]) -> str:
        blocks: List[str] = []
        for index, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            title = (
                metadata.get("doc_title")
                or metadata.get("file_name")
                or f"资料 {index}"
            )
            section = metadata.get("headings", "")
            source = metadata.get("source", "")
            version = metadata.get("version", "")
            effective_date = metadata.get("effective_date", "")
            content = (
                metadata.get("parent_content")
                or result.get("content", "")
            )

            labels = [f"标题: {title}"]
            if section:
                labels.append(f"章节: {section}")
            if version:
                labels.append(f"版本: {version}")
            if effective_date:
                labels.append(f"生效日期: {effective_date}")
            if source:
                labels.append(f"来源: {source}")

            blocks.append(
                f"[来源: {title}]\n"
                + "\n".join(labels)
                + f"\n内容:\n{content}"
            )
        return "\n\n".join(blocks)

    def retrieve(self, user_query: str) -> List[dict]:
        """执行混合检索和精排。"""
        return self.vector_store.search(
            user_query,
            top_k=DEFAULT_TOP_K,
            rerank_top_k=DEFAULT_RERANK_TOP_K,
        )

    def query(self, user_query: str, history: str = "") -> Dict[str, Any]:
        """非流式 RAG 查询。"""
        results = self.retrieve(user_query)
        if not results:
            return {
                "answer": (
                    "根据目前可查阅的金融知识库，暂未找到与该问题直接相关的资料。"
                    "建议补充具体产品名称、市场或业务场景后重新提问。"
                ),
                "strategy": "hybrid_search_no_result",
                "sources": [],
            }

        prompt = format_prompt(
            "RAG最终回答生成",
            query=user_query,
            history=history,
            context=self._format_context(results),
        )
        answer = self.llm.call_with_retry(
            prompt["system"],
            prompt["user"],
            temperature=0.15,
            max_tokens=1200,
        )
        if not answer:
            raise RuntimeError("RAG 检索成功，但大模型未返回回答")

        return {
            "answer": answer,
            "strategy": "bge_m3_hybrid_rerank",
            "sources": [
                {
                    "title": item.get("metadata", {}).get("doc_title", ""),
                    "source": item.get("metadata", {}).get("source", ""),
                    "score": item.get("score"),
                }
                for item in results
            ],
        }

    def query_stream(
        self,
        user_query: str,
        history: str = "",
    ) -> Iterable[Dict[str, Any]]:
        """流式 RAG 查询，依次产生状态、文本片段和完成事件。"""
        yield {
            "type": "status",
            "step": "retrieval",
            "value": "正在执行稠密与稀疏混合检索",
        }
        results = self.retrieve(user_query)

        if not results:
            answer = (
                "根据目前可查阅的金融知识库，暂未找到与该问题直接相关的资料。"
                "建议补充具体产品名称、市场或业务场景后重新提问。"
            )
            yield {"type": "token", "content": answer}
            yield {
                "type": "done",
                "strategy": "hybrid_search_no_result",
                "result_count": 0,
            }
            return

        yield {
            "type": "status",
            "step": "rerank",
            "value": f"已筛选 {len(results)} 条高相关资料",
        }
        prompt = format_prompt(
            "RAG最终回答生成",
            query=user_query,
            history=history,
            context=self._format_context(results),
        )

        yield {
            "type": "status",
            "step": "generation",
            "value": "正在基于检索资料生成回答",
        }
        token_count = 0
        for token in self.llm.call_stream(
            prompt["system"],
            prompt["user"],
            temperature=0.15,
            max_tokens=1200,
        ):
            if token:
                token_count += 1
                yield {"type": "token", "content": token}

        if token_count == 0:
            raise RuntimeError("RAG 检索成功，但大模型流式响应为空")

        yield {
            "type": "done",
            "strategy": "bge_m3_hybrid_rerank",
            "result_count": len(results),
        }


_rag_instance: RAGSystem | None = None


def get_rag_system() -> RAGSystem:
    """获取 RAGSystem 单例。"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGSystem()
    return _rag_instance
