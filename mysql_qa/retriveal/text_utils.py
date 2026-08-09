"""FAQ 词法检索使用的文本预处理工具。"""

from __future__ import annotations

import re

import jieba


# 这些词本身不能表达金融检索意图。保留它们会让 BM25 因一个代词或语气词
# 命中整条 FAQ，并被后续 softmax 错误放大成很高的置信度。
FAQ_STOPWORDS = frozenset(
    {
        "你", "您", "我", "他", "她", "它", "我们", "你们", "他们",
        "的", "了", "着", "在", "是", "有", "和", "与", "或", "以及",
        "吗", "呢", "吧", "啊", "呀", "嘛",
        "请", "请问", "想", "要", "能", "可以", "觉得", "认为",
        "帮", "帮我", "告诉", "说", "一下", "一个", "这个", "那个",
        "什么", "怎么", "如何", "为什么", "是否", "有没有", "干嘛",
    }
)

_PUNCTUATION_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def tokenize_for_bm25(text: str) -> list[str]:
    """小写、jieba 分词，并移除标点与低信息词。"""
    tokens: list[str] = []
    for word in jieba.cut(str(text or "").lower()):
        token = word.strip()
        if not token or token in FAQ_STOPWORDS:
            continue
        if _PUNCTUATION_RE.fullmatch(token):
            continue
        tokens.append(token)
    return tokens


def has_sufficient_query_signal(tokens: list[str]) -> bool:
    """判断查询是否具备进行 FAQ 模糊匹配所需的最小信息量。

    精确匹配在此检查之前执行，因此这里可以对模糊匹配更严格：
    单个中文字符或单个拉丁字母都不足以稳定定位一条 FAQ；“股票”、
    “ETF”这类长度至少为 2 的业务词仍允许进入 BM25。
    """
    if not tokens:
        return False
    if len(tokens) == 1 and len(tokens[0]) < 2:
        return False
    return True
