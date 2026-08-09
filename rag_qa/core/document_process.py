"""
文档处理模块

负责：
  1. 加载多源文档：MinerU 输出目录（自动提取 .md）、直接 .md 文件
  2. 父-子双层分块：父块保留完整语义，子块用于精确检索
  3. 子块 metadata 携带父块完整信息，支持后续答案溯源验证

分块策略：
  - 父块（~1200 字符）：优先按 Markdown 标题（##）自然切分，
    过长 section 降级到中文递归分隔符切分
  - 子块（~300 字符）：在父块内部按自然段落 + 中文标点递归切分
  - 每个子块 metadata 包含：parent_id、parent_content、child_id、
    source、doc_title、headings

Usage:
    from rag_qa.core.document_process import DocumentProcessor

    processor = DocumentProcessor(parent_size=1200, child_size=300)
    chunks = processor.process([
        "data/mineru_output",
        "data/investment_knowledge",
    ])
    # chunks 可直接送入向量数据库
"""

import os
import re
import hashlib
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from base.config import config
from base.logger import setup_logger

logger = setup_logger('fin_rag.document_process')

# ── 默认分块参数（可通过构造函数覆盖）────────────────────────────────
DEFAULT_PARENT_SIZE = 1200   # 父块目标大小（字符数）
DEFAULT_CHILD_SIZE = 300     # 子块目标大小（字符数）
DEFAULT_OVERLAP = 50         # 块间重叠字符数

# ── Markdown 标题匹配 ─────────────────────────────────────────────────
# 匹配行首 ## / ### 标题（兼容 MinerU 正常格式和 investment_knowledge
# 中 `\##` 转义格式）
_HEADING_PATTERN = re.compile(r'^\\?#{1,3}\s+', re.MULTILINE)

# 提取第一个 # 或 ## 标题作为文档标题
_TITLE_PATTERN = re.compile(r'^\\?#{1,2}\s+(.+)', re.MULTILINE)

# ── 中文递归分隔符（优先级从高到低）──────────────────────────────────
_CHINESE_SEPARATORS = [
    "\n\n",              # 段落间空行（最高优先级）
    "\n",                # 换行
    "。|！|？",          # 中文句末标点
    r"\.\s|\!\s|\?\s",   # 英文句末标点
    "；|;\s",            # 分号
    "，|,\s",            # 逗号（最低优先级）
]


class DocumentProcessor:
    """文档加载与父-子双层分块处理器。

    参数全部通过构造函数传入，不硬编码分块大小：
        parent_size: 父块目标大小（字符数），默认 1200
        child_size:  子块目标大小（字符数），默认 300
        overlap:     块间重叠字符数，默认 50
    """

    def __init__(
        self,
        parent_size: int = DEFAULT_PARENT_SIZE,
        child_size: int = DEFAULT_CHILD_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ):
        if parent_size <= child_size:
            raise ValueError(
                f"parent_size({parent_size}) 必须大于 child_size({child_size})"
            )
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

        # 父块降级 splitter：当 section 超过 parent_size 时使用
        self._parent_fallback_splitter = RecursiveCharacterTextSplitter(
            separators=_CHINESE_SEPARATORS,
            keep_separator=True,
            is_separator_regex=True,
            chunk_size=self.parent_size,
            chunk_overlap=self.overlap,
        )

        # 子块 splitter
        self._child_splitter = RecursiveCharacterTextSplitter(
            separators=_CHINESE_SEPARATORS,
            keep_separator=True,
            is_separator_regex=True,
            chunk_size=self.child_size,
            chunk_overlap=self.overlap,
        )

    # ── 公共入口 ──────────────────────────────────────────────────────

    def process(self, sources: List[str]) -> List[Document]:
        """端到端处理：加载 → 父块切分 → 子块切分。

        Args:
            sources: 路径列表，每个元素可以是：
                     - MinerU 输出目录（自动提取 auto/*.md）
                     - 单个 .md 文件
                     - 含 .md 文件的普通目录

        Returns:
            子块 Document 列表，每个子块 metadata 携带父块完整信息。
        """
        documents = self.load_documents(sources)
        if not documents:
            logger.warning("未加载到任何文档，返回空列表")
            return []

        logger.info("共加载 %d 篇文档，开始父块切分...", len(documents))
        parent_docs = self._split_parents(documents)
        logger.info("父块切分完成，共 %d 个父块", len(parent_docs))

        child_docs = self._split_children(parent_docs)
        logger.info(
            "子块切分完成，共 %d 个子块（来源 %d 父块 %d 文档）",
            len(child_docs), len(parent_docs), len(documents),
        )
        return child_docs

    # ══════════════════════════════════════════════════════════════════
    # 文档加载
    # ══════════════════════════════════════════════════════════════════

    def load_documents(self, sources: List[str]) -> List[Document]:
        """自动检测路径类型并加载为 Document 列表。

        检测逻辑（按优先级）：
          1. 单个 .md 文件 → 直接加载
          2. 目录 → 优先查找 MinerU 结构（auto/*.md），
             找不到则递归扫描全部 .md 文件
        """
        documents: List[Document] = []

        for source in sources:
            if not os.path.exists(source):
                logger.warning("路径不存在，跳过: %s", source)
                continue

            if os.path.isfile(source):
                if source.endswith('.md'):
                    doc = self._load_markdown_file(source)
                    if doc:
                        documents.append(doc)
                else:
                    logger.debug("非 .md 文件，跳过: %s", source)
            elif os.path.isdir(source):
                docs = self._load_directory(source)
                documents.extend(docs)

        logger.info(
            "文档加载完成: %d 个源路径 → %d 篇文档",
            len(sources), len(documents),
        )
        return documents

    def _load_directory(self, dir_path: str) -> List[Document]:
        """加载目录中的 Markdown 文档，兼容 MinerU 输出和普通 .md 目录。

        MinerU 输出结构：
            some_pdf/
            └── auto/
                ├── some_pdf.md        ← 目标文件
                ├── images/            ← 跳过
                ├── *_middle.json      ← 跳过
                └── ...

        普通 .md 目录：
            knowledge/
            ├── a.md
            └── b.md
        """
        documents: List[Document] = []

        # 策略1：MinerU 输出目录（含 auto/*.md）
        auto_dir = os.path.join(dir_path, 'auto')
        if os.path.isdir(auto_dir):
            for name in os.listdir(auto_dir):
                if name.endswith('.md'):
                    filepath = os.path.join(auto_dir, name)
                    doc = self._load_markdown_file(filepath)
                    if doc:
                        documents.append(doc)
            if documents:
                logger.debug(
                    "MinerU 模式加载: %s → %d 篇", dir_path, len(documents)
                )
                return documents

        # 策略2：递归扫描全部 .md（排除 images 目录）
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in ('images',)]
            for name in files:
                if name.endswith('.md'):
                    filepath = os.path.join(root, name)
                    doc = self._load_markdown_file(filepath)
                    if doc:
                        documents.append(doc)

        logger.debug("目录扫描模式: %s → %d 篇", dir_path, len(documents))
        return documents

    def _load_markdown_file(self, file_path: str) -> Optional[Document]:
        """加载单个 .md 文件为 Document。

        自动提取文档标题（第一个 # / ## 标题行）写入 metadata。
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (OSError, UnicodeDecodeError) as e:
            logger.error("读取文件失败: %s - %s", file_path, e)
            return None

        if not content.strip():
            logger.debug("文件为空，跳过: %s", file_path)
            return None

        doc_title = self._extract_title(content, file_path)

        return Document(
            page_content=content,
            metadata={
                'source': os.path.abspath(file_path),
                'doc_title': doc_title,
                'file_name': os.path.basename(file_path),
            },
        )

    def _extract_title(self, content: str, file_path: str) -> str:
        """从 Markdown 内容中提取文档标题。

        优先取第一个 # / ## 标题行，回退取文件名（去掉扩展名）。
        """
        match = _TITLE_PATTERN.search(content)
        if match:
            return match.group(1).strip()
        basename = os.path.basename(file_path)
        return os.path.splitext(basename)[0]

    # ══════════════════════════════════════════════════════════════════
    # 父块切分
    # ══════════════════════════════════════════════════════════════════

    def _split_parents(self, documents: List[Document]) -> List[Document]:
        """将每篇文档切分为父块。

        策略（优先级）：
          1. 按 ## / ### 标题边界切分 → 语义完整的 section
          2. section 长度 > parent_size → 降级为递归分隔符切分
          3. 过短的 section → 合并到相邻 section（避免碎片）
        """
        parent_docs: List[Document] = []

        for doc in documents:
            sections = self._split_by_headings(doc.page_content)
            merged = self._merge_short_sections(sections)

            for heading, body in merged:
                section_text = (
                    f"{heading}\n{body}".strip() if heading else body.strip()
                )
                if not section_text:
                    continue

                parent_id = self._make_id(doc.metadata['source'], section_text[:80])

                if len(section_text) <= self.parent_size:
                    # 长度合适，直接作为父块
                    parent_docs.append(Document(
                        page_content=section_text,
                        metadata={
                            **doc.metadata,
                            'parent_id': parent_id,
                            'headings': heading,
                            'chunk_type': 'parent',
                        },
                    ))
                else:
                    # 过长：降级递归切分
                    sub_texts = self._parent_fallback_splitter.split_text(
                        section_text
                    )
                    for i, sub_text in enumerate(sub_texts):
                        sub_id = self._make_id(parent_id, str(i))
                        parent_docs.append(Document(
                            page_content=sub_text,
                            metadata={
                                **doc.metadata,
                                'parent_id': sub_id,
                                'headings': heading,
                                'chunk_type': 'parent',
                                'parent_part': f"{i + 1}/{len(sub_texts)}",
                            },
                        ))

        return parent_docs

    def _split_by_headings(self, text: str) -> List[Tuple[str, str]]:
        """按 ## / ### 标题将文本拆分为 (标题, 正文) 列表。

        第一个标题之前的内容（如文档大标题、前言）以空标题表示。
        """
        matches = list(_HEADING_PATTERN.finditer(text))

        if not matches:
            return [('', text)]

        sections: List[Tuple[str, str]] = []

        # 第一个标题之前的内容
        first = matches[0]
        if first.start() > 0:
            prefix = text[:first.start()].strip()
            if prefix:
                sections.append(('', prefix))

        # 按标题切分
        for i, match in enumerate(matches):
            # 提取完整标题行（从 ## 到行尾）
            line_end = text.find('\n', match.end())
            if line_end == -1:
                line_end = len(text)
            heading = text[match.start():line_end].strip()

            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((heading, body))

        return sections

    def _merge_short_sections(
        self, sections: List[Tuple[str, str]], min_chars: int = 200
    ) -> List[Tuple[str, str]]:
        """合并过短的 section，避免父块碎片化。

        从后向前遍历，将短 section 合并到前一个 section（前提是不超限）。
        """
        if len(sections) <= 1:
            return sections

        merged: List[Tuple[str, str]] = []
        for heading, body in sections:
            if (
                merged
                and len(body) < min_chars
                and len(merged[-1][1]) + len(body) < self.parent_size * 1.3
            ):
                prev_heading, prev_body = merged.pop()
                combined = f"{prev_body}\n{heading}\n{body}".strip()
                merged.append((prev_heading, combined))
            else:
                merged.append((heading, body))

        return merged

    # ══════════════════════════════════════════════════════════════════
    # 子块切分
    # ══════════════════════════════════════════════════════════════════

    def _split_children(self, parent_docs: List[Document]) -> List[Document]:
        """在每个父块内部切分子块，注入父块完整元数据。

        每个子块 metadata 包含：
          - child_id:       子块唯一 ID
          - parent_id:      父块 ID（用于关联查询）
          - parent_content: 父块完整文本（用于答案溯源验证）
          - source:         原始文件绝对路径
          - doc_title:      文档标题
          - file_name:      文件名
          - headings:       所属标题层级
          - chunk_type:     固定为 'child'
          - child_index:    子块在父块内的序号（从 0 开始）
        """
        child_docs: List[Document] = []

        for parent in parent_docs:
            parent_id = parent.metadata['parent_id']
            parent_content = parent.page_content

            # 父块内容较短：直接作为一个子块
            if len(parent_content) <= self.child_size:
                child_id = self._make_id(parent_id, 'single')
                child_docs.append(Document(
                    page_content=parent_content,
                    metadata={
                        'child_id': child_id,
                        'parent_id': parent_id,
                        'parent_content': parent_content,
                        'source': parent.metadata['source'],
                        'doc_title': parent.metadata['doc_title'],
                        'file_name': parent.metadata['file_name'],
                        'headings': parent.metadata.get('headings', ''),
                        'chunk_type': 'child',
                        'child_index': 0,
                    },
                ))
                continue

            # 在父块内递归切分子块
            sub_texts = self._child_splitter.split_text(parent_content)
            for i, sub_text in enumerate(sub_texts):
                child_id = self._make_id(parent_id, str(i))
                child_docs.append(Document(
                    page_content=sub_text,
                    metadata={
                        'child_id': child_id,
                        'parent_id': parent_id,
                        'parent_content': parent_content,
                        'source': parent.metadata['source'],
                        'doc_title': parent.metadata['doc_title'],
                        'file_name': parent.metadata['file_name'],
                        'headings': parent.metadata.get('headings', ''),
                        'chunk_type': 'child',
                        'child_index': i,
                    },
                ))

        return child_docs

    # ══════════════════════════════════════════════════════════════════
    # 工具方法
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _make_id(*parts: str) -> str:
        """基于内容生成稳定的短 ID（SHA-1 前 12 位）。

        相同内容 → 相同 ID，适合增量更新场景。
        """
        combined = '|'.join(parts)
        return hashlib.sha1(combined.encode('utf-8')).hexdigest()[:12]

    def get_parent_by_child_id(
        self, child_docs: List[Document], child_id: str
    ) -> Optional[str]:
        """根据子块 ID 查找其父块内容（用于答案验证时还原上下文）。"""
        for doc in child_docs:
            if doc.metadata.get('child_id') == child_id:
                return doc.metadata.get('parent_content')
        return None


# ── 便捷函数 ──────────────────────────────────────────────────────────


def process_documents(
    sources: List[str],
    parent_size: int = DEFAULT_PARENT_SIZE,
    child_size: int = DEFAULT_CHILD_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Document]:
    """一行式调用：加载 + 父-子分块。

    Args:
        sources:    源路径列表（MinerU 输出目录 / .md 文件 / 混合）
        parent_size: 父块目标大小，默认 1200
        child_size:  子块目标大小，默认 300
        overlap:     块间重叠字符数，默认 50

    Returns:
        子块 Document 列表，可直接送入向量数据库。
    """
    processor = DocumentProcessor(
        parent_size=parent_size,
        child_size=child_size,
        overlap=overlap,
    )
    return processor.process(sources)


# ── 自测入口 ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # 测试：investment_knowledge .md + MinerU 输出
    sources = [
        os.path.join(PROJECT_ROOT, 'data', 'investment_knowledge'),
        os.path.join(PROJECT_ROOT, 'data', 'mineru_output'),
    ]

    processor = DocumentProcessor(parent_size=1200, child_size=300, overlap=50)
    child_docs = processor.process(sources)

    print(f"\n{'=' * 70}")
    print(f"总计产生 {len(child_docs)} 个子块")
    print(f"{'=' * 70}")

    # 按文档分组统计
    from collections import Counter
    doc_counter = Counter(
        d.metadata.get('doc_title', '?') for d in child_docs
    )
    for title, count in doc_counter.most_common():
        print(f"  {title}: {count} 个子块")

    # 展示前 3 个子块详情
    print(f"\n{'=' * 70}")
    print("示例子块（前 3 个）：")
    print(f"{'=' * 70}")

    for i, doc in enumerate(child_docs[:3]):
        print(f"\n── 子块 #{i + 1} ──")
        print(f"  child_id:      {doc.metadata['child_id']}")
        print(f"  parent_id:     {doc.metadata['parent_id']}")
        print(f"  doc_title:     {doc.metadata['doc_title']}")
        print(f"  headings:      {doc.metadata['headings']}")
        print(f"  source:        {doc.metadata['source']}")
        print(f"  子块长度:      {len(doc.page_content)} 字符")
        print(f"  父块长度:      {len(doc.metadata['parent_content'])} 字符")
        print(f"  内容预览:      {doc.page_content}...")
        break
