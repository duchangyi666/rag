"""
向量存储模块

负责：
  1. Milvus 向量数据库连接管理（显式 schema，双向量字段 + 索引）
  2. BGE-M3 模型加载与推理（稠密 1024d + 稀疏 lexical weights）
  3. BGE Reranker 重排序模型加载与推理

Collection 字段设计：
  ┌─────────────────┬──────────────────────┬─────────────────────────┐
  │ 字段            │ 类型                  │ 说明                    │
  ├─────────────────┼──────────────────────┼─────────────────────────┤
  │ id              │ VARCHAR(32) PK        │ MD5(content) 主键       │
  │ dense_vector    │ FLOAT_VECTOR(1024)    │ BGE-M3 稠密向量         │
  │ sparse_vector   │ SPARSE_FLOAT_VECTOR   │ BGE-M3 稀疏词权重       │
  │ content         │ VARCHAR(65535)        │ 子块文本（动态字段）     │
  │ parent_id       │ VARCHAR(32)           │ 父块 ID                 │
  │ parent_content  │ VARCHAR(65535)        │ 父块完整文本             │
  │ doc_title       │ VARCHAR(512)          │ 文档标题                │
  │ headings        │ VARCHAR(512)          │ 标题层级                │
  │ source          │ VARCHAR(1024)         │ 原始文件路径             │
  │ file_name       │ VARCHAR(512)          │ 文件名                  │
  │ chunk_type      │ VARCHAR(32)           │ "child"                 │
  │ child_index     │ INT64                 │ 子块在父块内序号         │
  └─────────────────┴──────────────────────┴─────────────────────────┘

索引：
  - dense_vector  → HNSW（运行时读取现有索引的 IP/COSINE 配置）
  - sparse_vector → SPARSE_INVERTED_INDEX (IP)

Usage:
    from rag_qa.core.vector_store import VectorStore

    store = VectorStore()
    store.create_collection(drop_existing=True)
    store.insert(child_documents)                 # 写入（MD5 去重）
    results = store.search("什么是股票？", top_k=5)  # 检索 + 精排
"""

import os
import hashlib
from typing import List, Optional, Dict, Any

import numpy as np
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from langchain_core.documents import Document
from base.config import config
from base.logger import setup_logger
from sentence_transformers import CrossEncoder

logger = setup_logger('fin_rag.vector_store')

# ── Milvus 字段名常量 ─────────────────────────────────────────────────
FIELD_ID = 'id'
FIELD_DENSE = 'dense_vector'
FIELD_SPARSE = 'sparse_vector'

# ── 检索默认值 ────────────────────────────────────────────────────────
DEFAULT_ANN_TOP_K = 10      # ANN 粗排召回数
DEFAULT_RERANK_TOP_K = 5    # 精排返回数
INSERT_BATCH_SIZE = 100      # 写入批次大小


class VectorStore:
    """Milvus 向量存储 + BGE-M3 双向量推理 + CrossEncoder 精排。

    初始化时自动：
      1. 连接 Milvus（参数全部来自 config.ini）
      2. 加载 BGE-M3 模型（BGEM3EmbeddingFunction，稠密 + 稀疏输出）
      3. 加载 CrossEncoder 重排序模型（对父块内容精排）

    Attributes:
        client:    MilvusClient 实例
        embedding: BGEM3EmbeddingFunction（BGE-M3，稠密+稀疏）
        reranker:  CrossEncoder（BGE Reranker，cross-encoder 父块精排）
    """

    def __init__(self):
        """初始化 Milvus 连接和所有模型。

        Raises:
            Exception: Milvus 连接失败或任一模型加载失败时抛出。
        """
        # ── 1. 连接 Milvus ─────────────────────────────────────────
        self._init_milvus()

        # ── 2. 加载 BGE-M3 向量化模型 ──────────────────────────────
        self._init_embedding_model()

        # ── 3. 加载 BGE Reranker 重排序模型 ────────────────────────
        self._init_reranker_model()

        logger.info(
            "VectorStore 初始化完成 - Milvus:%s:%d/%s collection:%s dim:%d",
            config.milvus_host, config.milvus_port,
            config.milvus_database_name,
            config.milvus_collection_name,
            config.embedding_dim,
        )

    # ══════════════════════════════════════════════════════════════════
    # Milvus 连接
    # ══════════════════════════════════════════════════════════════════

    def _init_milvus(self) -> None:
        """初始化 Milvus 客户端连接。

        MilvusClient（pymilvus 2.5+）自动管理连接池，
        无需手动 connect / disconnect。
        """
        try:
            uri = f"http://{config.milvus_host}:{config.milvus_port}"
            self.client = MilvusClient(
                uri=uri,
                db_name=config.milvus_database_name,
            )
            collections = self.client.list_collections()
            self.collection_exists = (
                config.milvus_collection_name in collections
            )
            self.financial_schema = False
            self.dense_metric_type = 'COSINE'
            if self.collection_exists:
                description = self.client.describe_collection(
                    config.milvus_collection_name
                )
                field_names = {
                    field.get('name')
                    for field in description.get('fields', [])
                }
                self.financial_schema = 'chunk_text' in field_names
                for index_name in self.client.list_indexes(
                    config.milvus_collection_name
                ):
                    index_info = self.client.describe_index(
                        config.milvus_collection_name, index_name
                    )
                    if index_info.get('field_name') == FIELD_DENSE:
                        self.dense_metric_type = index_info.get(
                            'metric_type', 'COSINE'
                        )
                        break
            logger.info(
                "Milvus 连接成功 - %s db:%s collection:%s exists:%s schema:%s",
                uri, config.milvus_database_name,
                config.milvus_collection_name,
                self.collection_exists,
                'financial_v1' if self.financial_schema else 'finrag_dynamic',
            )
        except Exception as e:
            logger.error(
                "Milvus 连接失败 - host:%s port:%d db:%s error:%s",
                config.milvus_host, config.milvus_port,
                config.milvus_database_name, e,
            )
            raise

    # ══════════════════════════════════════════════════════════════════
    # 模型加载
    # ══════════════════════════════════════════════════════════════════

    def _init_embedding_model(self) -> None:
        """加载 BGE-M3 模型（milvus_model BGEM3EmbeddingFunction）。

        使用 milvus_model 官方封装，与 Milvus 原生集成：
          - dense:  List[List[float]]，1024 维稠密向量（语义级编码）
          - sparse: scipy csr_array，稀疏词权重（词汇级编码）

        混合检索时对两种向量分别检索后加权融合，兼顾语义和精确匹配。

        model_name 传入本地路径，不从 HuggingFace 远端下载。
        """
        model_path = config.embedding_model_path
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"BGE-M3 模型路径不存在: {model_path}")

        try:
            from milvus_model.hybrid import BGEM3EmbeddingFunction

            device = config.device
            self.embedding = BGEM3EmbeddingFunction(
                model_name=model_path,          # 本地路径，不去 HF 下载
                device=device,
                use_fp16=device == 'cuda',
                normalize_embeddings=True,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            logger.info(
                "BGE-M3 模型加载完成 (BGEM3EmbeddingFunction) - %s device:%s",
                model_path, device,
            )
        except Exception as e:
            logger.error("BGE-M3 模型加载失败 - path:%s error:%s", model_path, e)
            raise

    def _init_reranker_model(self) -> None:
        """加载 BGE Reranker 重排序模型（CrossEncoder）。

        混合检索粗排后，用 CrossEncoder 对 (query, parent_content) 逐对打分精排。
        使用父块内容（~1200 字符）而非子块（~300 字符），让 cross-encoder
        获得更完整的上下文做出更准确的 relevance 判断。
        """
        model_path = config.reranker_model_path
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Reranker 模型路径不存在: {model_path}")

        try:
            self.reranker = CrossEncoder(
                model_path,
                device=config.device,
                max_length=512,
            )
            logger.info(
                "BGE Reranker 模型加载完成 (CrossEncoder) - %s device:%s",
                model_path, config.device,
            )
        except Exception as e:
            self.reranker = None
            logger.warning(
                "Reranker 模型加载失败，降级使用 Milvus 混合分数 - path:%s error:%s",
                model_path, e,
            )

    # ══════════════════════════════════════════════════════════════════
    # 向量化
    # ══════════════════════════════════════════════════════════════════

    def encode(
        self,
        texts: List[str],
    ) -> Dict[str, Any]:
        """批量文本 → 稠密 + 稀疏向量。

        通过 BGEM3EmbeddingFunction 调用 BGE-M3，输出：
          - dense:  np.ndarray, shape=(n, 1024), float32, L2 归一化
          - sparse: scipy csr_array, shape=(n, vocab_size)

        sparse 为 scipy CSR 稀疏矩阵，可直接写入 Milvus SPARSE_FLOAT_VECTOR，
        也支持按行切片 [i] 获取单条稀疏向量。

        Args:
            texts: 文本列表

        Returns:
            {'dense': np.ndarray, 'sparse': csr_array}
        """
        if not texts:
            return {
                'dense': np.array([], dtype=np.float32),
                'sparse': None,
            }

        output = self.embedding(texts)

        dense = np.array(output['dense'], dtype=np.float32)
        sparse = output['sparse']  # csr_array, shape=(n, vocab_size)

        return {'dense': dense, 'sparse': sparse}

    def encode_query(self, query: str) -> Dict[str, Any]:
        """单条 query 向量化。

        Returns:
            {'dense': np.ndarray(1024,), 'sparse': Dict[int, float]}
        """
        result = self.encode([query])
        # 稀疏向量：csr_array → dict（Milvus AnnSearchRequest 兼容）
        sparse_csr = result['sparse']
        if hasattr(sparse_csr, 'tocoo') and sparse_csr.shape[0] == 1:
            coo = sparse_csr.tocoo()
            sparse_dict = {
                int(col): float(val)
                for col, val in zip(coo.col, coo.data)
            }
        else:
            sparse_dict = sparse_csr

        return {
            'dense': result['dense'][0],       # (1024,) ndarray
            'sparse': sparse_dict,              # Dict[int, float]
        }

    # ══════════════════════════════════════════════════════════════════
    # Collection 管理
    # ══════════════════════════════════════════════════════════════════

    def create_collection(self, drop_existing: bool = False) -> None:
        """创建 Milvus collection（显式 schema + 双向量索引）。

        字段：
          - id:            VARCHAR(32) 主键，MD5(content)
          - dense_vector:  FLOAT_VECTOR(1024)，COSINE 距离
          - sparse_vector: SPARSE_FLOAT_VECTOR，IP 距离
          - 其余标量字段通过 enable_dynamic_field 动态存储

        索引：
          - dense_vector  → HNSW
          - sparse_vector → SPARSE_INVERTED_INDEX

        Args:
            drop_existing: 是否先删除同名 collection 再重建。
        """
        collection_name = config.milvus_collection_name

        if drop_existing and self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            logger.info("已删除 collection: %s", collection_name)

        if self.client.has_collection(collection_name):
            logger.info("Collection 已存在，跳过创建: %s", collection_name)
            return

        # ── 1. 构建显式 schema ─────────────────────────────────────
        schema = self.client.create_schema(
            auto_id=False,                # 使用 MD5 主键，不自动生成
            enable_dynamic_field=True,    # 标量字段自动存储
        )

        # 主键：子块内容的 MD5 哈希
        schema.add_field(
            field_name=FIELD_ID,
            datatype=DataType.VARCHAR,
            max_length=32,
            is_primary=True,
        )

        # 稠密向量（1024 维，BGE-M3 dense output）
        schema.add_field(
            field_name=FIELD_DENSE,
            datatype=DataType.FLOAT_VECTOR,
            dim=config.embedding_dim,
        )

        # 稀疏向量（BGE-M3 lexical weights）
        schema.add_field(
            field_name=FIELD_SPARSE,
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        # ── 2. 构建索引参数 ────────────────────────────────────────
        index_params = self.client.prepare_index_params()

        # 稠密向量索引：HNSW，COSINE 距离
        index_params.add_index(
            field_name=FIELD_DENSE,
            index_type='HNSW',
            metric_type='COSINE',
            params={'M': 16, 'efConstruction': 200},
        )

        # 稀疏向量索引：SPARSE_INVERTED_INDEX，内积距离
        index_params.add_index(
            field_name=FIELD_SPARSE,
            index_type='SPARSE_INVERTED_INDEX',
            metric_type='IP',
        )

        # ── 3. 创建 collection ─────────────────────────────────────
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(
            "Collection 创建完成: %s (dense:HNSW/COSINE sparse:INVERTED/IP)",
            collection_name,
        )

    # ══════════════════════════════════════════════════════════════════
    # 数据写入
    # ══════════════════════════════════════════════════════════════════

    def insert(
        self,
        documents: List[Document],
        batch_size: int = INSERT_BATCH_SIZE,
    ) -> int:
        """子块文档 → 双向量化 → 批量 upsert 写入 Milvus。

        主键为 content 的 MD5，相同内容重复写入自动覆盖（upsert 语义），
        不会产生重复记录。

        Args:
            documents:  子块 Document 列表（来自 DocumentProcessor.process()）
            batch_size: 每批写入条数

        Returns:
            成功写入（含覆盖）的总条数
        """
        if not documents:
            logger.warning("文档列表为空，跳过写入")
            return 0

        collection_name = config.milvus_collection_name
        total = len(documents)
        upserted = 0

        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc.page_content for doc in batch]

            # 稠密 + 稀疏 双向量化
            vecs = self.encode(texts)

            # 构建写入数据
            data = []
            for i, doc in enumerate(batch):
                doc_id = self._content_md5(doc.page_content)
                # 稀疏向量：csr_array 切片 → dict 格式 (Milvus SPARSE_FLOAT_VECTOR 兼容)
                sparse_vec = vecs['sparse'][i]
                if hasattr(sparse_vec, 'tocoo'):
                    coo = sparse_vec.tocoo()
                    sparse_dict = {
                        int(col): float(val)
                        for col, val in zip(coo.col, coo.data)
                    }
                else:
                    sparse_dict = sparse_vec

                if self.financial_schema:
                    document_id = doc.metadata.get('document_id') or hashlib.sha256(
                        doc.metadata.get('source', '').encode('utf-8')
                    ).hexdigest()[:32]
                    row = {
                        FIELD_ID: doc_id,
                        FIELD_DENSE: vecs['dense'][i].tolist(),
                        FIELD_SPARSE: sparse_dict,
                        'chunk_text': doc.page_content[:8192],
                        'parent_id': doc.metadata.get('parent_id', ''),
                        'parent_text': doc.metadata.get('parent_content', '')[:16384],
                        'document_id': document_id,
                        'title': doc.metadata.get('doc_title', '')[:512],
                        'source_name': doc.metadata.get('file_name', '')[:512],
                        'source_url': doc.metadata.get('source_url', '')[:2048],
                        'business_scope': doc.metadata.get(
                            'business_scope', 'financial_general'
                        ),
                        'doc_type': doc.metadata.get('doc_type', 'markdown'),
                        'effective_date': doc.metadata.get('effective_date', ''),
                        'expire_date': doc.metadata.get('expire_date', ''),
                        'version': doc.metadata.get('version', ''),
                        'section': doc.metadata.get('headings', '')[:512],
                        'page_no': doc.metadata.get('page_no', 0),
                        'content_hash': doc_id,
                        'ingested_at': doc.metadata.get('ingested_at', ''),
                        'is_active': True,
                    }
                else:
                    row = {
                        FIELD_ID: doc_id,
                        FIELD_DENSE: vecs['dense'][i].tolist(),
                        FIELD_SPARSE: sparse_dict,
                        'content': doc.page_content,
                        'parent_id': doc.metadata.get('parent_id', ''),
                        'parent_content': doc.metadata.get('parent_content', ''),
                        'doc_title': doc.metadata.get('doc_title', ''),
                        'headings': doc.metadata.get('headings', ''),
                        'source': doc.metadata.get('source', ''),
                        'file_name': doc.metadata.get('file_name', ''),
                        'chunk_type': doc.metadata.get('chunk_type', 'child'),
                        'child_index': doc.metadata.get('child_index', 0),
                    }
                data.append(row)

            # 批量 upsert（主键已存在则覆盖，不存在则新增）
            try:
                result = self.client.upsert(
                    collection_name=collection_name,
                    data=data,
                )
                batch_count = result.get('upsert_count', len(batch))
                upserted += batch_count
                logger.debug(
                    "写入批次 %d/%d: %d 条",
                    i // batch_size + 1,
                    (total + batch_size - 1) // batch_size,
                    batch_count,
                )
            except Exception as e:
                logger.error("Milvus upsert 失败(批次%d): %s", i // batch_size, e)

        try:
            self.client.flush(collection_name=collection_name)
        except TypeError:
            # 兼容部分 pymilvus 版本的位置参数签名
            self.client.flush(collection_name)

        logger.info("Milvus upsert 完成并已 flush: %d/%d 条", upserted, total)
        return upserted

    # ══════════════════════════════════════════════════════════════════
    # 检索
    # ══════════════════════════════════════════════════════════════════

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_ANN_TOP_K,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        output_fields: Optional[List[str]] = None,
    ) -> List[dict]:
        """混合检索 + 父块去重 + CrossEncoder 精排。

        流程：
          1. Query → BGE-M3 双向量化（dense + sparse）
          2. AnnSearchRequest × 2：dense → 读取现有索引指标, sparse → IP
          3. WeightedRanker 加权融合 → 粗排 top_k
          4. 按 parent_id 去重（同一父块只保留加权分最高的子块）
          5. CrossEncoder 对 (query, parent_content) 逐对打分
          6. CrossEncoder 按相关性分数降序并应用最低相关性阈值

        Args:
            query:         用户查询
            top_k:         混合检索粗排召回数量
            rerank_top_k:  精排后返回数量（每个父块最多一条）
            dense_weight:  稠密向量权重（默认 0.7）
            sparse_weight: 稀疏向量权重（默认 0.3）
            output_fields: Milvus 返回字段，默认全部业务字段

        Returns:
            精排结果列表（按 score 降序，越大越相关），每条含 content / score / metadata
        """
        if output_fields is None:
            if self.financial_schema:
                output_fields = [
                    'chunk_text', 'parent_id', 'parent_text', 'title',
                    'section', 'source_name', 'source_url', 'business_scope',
                    'effective_date', 'expire_date', 'version', 'is_active',
                ]
            else:
                output_fields = [
                    'content', 'parent_id', 'parent_content',
                    'doc_title', 'headings', 'source', 'file_name', 'chunk_type',
                ]

        collection_name = config.milvus_collection_name

        # ── Step 1: Query 双向量化 ──────────────────────────────────
        query_vecs = self.encode_query(query)

        # ── Step 2: 稠密 + 稀疏 AnnSearchRequest ───────────────────
        dense_req = AnnSearchRequest(
            data=[query_vecs['dense'].tolist()],
            anns_field=FIELD_DENSE,
            param={'metric_type': self.dense_metric_type, 'ef': 200},
            limit=top_k,
        )

        sparse_req = AnnSearchRequest(
            data=[query_vecs['sparse']],
            anns_field=FIELD_SPARSE,
            param={'metric_type': 'IP'},
            limit=top_k,
        )

        # ── Step 3: WeightedRanker 加权融合 ────────────────────────
        ranker = WeightedRanker(dense_weight, sparse_weight)

        try:
            hybrid_results = self.client.hybrid_search(
                collection_name=collection_name,
                reqs=[dense_req, sparse_req],
                ranker=ranker,
                limit=top_k,
                output_fields=output_fields,
            )[0]
        except Exception as e:
            logger.error("混合检索失败: %s", e)
            return []

        if not hybrid_results:
            logger.info("混合检索无结果: %s", query[:50])
            return []

        logger.debug(
            "混合检索: dense(w:%.1f) + sparse(w:%.1f) → %d 条",
            dense_weight, sparse_weight, len(hybrid_results),
        )

        # ── Step 4: 按 parent_id 去重 ──────────────────────────────
        # 同一父块下可能有多个子块被召回，只保留加权分最高的那条
        # 避免后续 CrossEncoder 对同一父块内容重复打分浪费计算
        deduped = self._dedup_by_parent(hybrid_results)
        logger.debug(
            "父块去重: %d → %d 条", len(hybrid_results), len(deduped),
        )

        # ── Step 5: CrossEncoder 父块精排 ──────────────────────────
        # 对去重后的 (query, parent_content) 逐对打分
        if self.reranker is not None:
            pairs = []
            for item in deduped:
                entity = item.get('entity', {})
                parent_content = entity.get(
                    'parent_text',
                    entity.get('parent_content', entity.get(
                        'chunk_text', entity.get('content', '')
                    )),
                )
                pairs.append([query, parent_content])

            scores = self.reranker.predict(
                pairs,
                show_progress_bar=False,
            )

            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            if isinstance(scores, (int, float)):
                scores = [float(scores)]
        else:
            scores = [
                float(item.get('distance', item.get('score', 0.0)))
                for item in deduped
            ]

        # ── Step 6: 组装结果并按相关性降序排列 ──────────────────────
        scored_results = []
        for item, score in zip(deduped, scores):
            entity = item.get('entity', {})
            scored_results.append({
                'content': entity.get('chunk_text', entity.get('content', '')),
                'score': float(score),
                'metadata': {
                    'parent_id': entity.get('parent_id', ''),
                    'parent_content': entity.get(
                        'parent_text', entity.get('parent_content', '')
                    ),
                    'doc_title': entity.get('title', entity.get('doc_title', '')),
                    'headings': entity.get('section', entity.get('headings', '')),
                    'source': entity.get('source_url', entity.get('source', '')),
                    'file_name': entity.get(
                        'source_name', entity.get('file_name', '')
                    ),
                    'chunk_type': entity.get('chunk_type', ''),
                    'business_scope': entity.get('business_scope', ''),
                    'effective_date': entity.get('effective_date', ''),
                    'expire_date': entity.get('expire_date', ''),
                    'version': entity.get('version', ''),
                },
            })

        # CrossEncoder 与 Milvus 相似度均为分数越高越相关
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        relevant_results = [
            item for item in scored_results
            if item['score'] >= config.rerank_min_score
        ]
        top_results = relevant_results[:rerank_top_k]

        logger.info(
            "检索完成: '%s' → 混合:%d → 父块去重:%d → 相关性过滤:%d → 精排:%d "
            "| threshold:%.4f | top1_score:%.4f",
            query[:50], len(hybrid_results), len(deduped),
            len(relevant_results), len(top_results), config.rerank_min_score,
            top_results[0]['score'] if top_results else 0.0,
        )
        return top_results

    @staticmethod
    def _dedup_by_parent(hybrid_results: List[dict]) -> List[dict]:
        """按 parent_id 去重，每个父块只保留混合检索加权分最高的子块。

        hybrid_results 来自 Milvus hybrid_search，已按 WeightedRanker
        加权分降序排列，因此首次出现的 parent_id 即为该父块下最高分。
        """
        seen: set = set()
        deduped: List[dict] = []
        for item in hybrid_results:
            entity = item.get('entity', {})
            parent_id = entity.get('parent_id', '')
            if not parent_id or parent_id in seen:
                continue
            seen.add(parent_id)
            deduped.append(item)
        return deduped

    # ══════════════════════════════════════════════════════════════════
    # 工具方法
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _content_md5(content: str) -> str:
        """对子块文本内容取 MD5 哈希，作为主键。"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def get_collection_stats(self) -> dict:
        """获取 collection 统计信息。

        Returns:
            {'name': str, 'count': int}
        """
        collection_name = config.milvus_collection_name
        try:
            stats = self.client.get_collection_stats(collection_name)
            return {
                'name': collection_name,
                'count': stats.get('row_count', 0),
            }
        except Exception as e:
            logger.error("获取 collection 统计失败: %s", e)
            return {'name': collection_name, 'count': -1}

    def close(self) -> None:
        """释放 Milvus 连接。"""
        if self.client:
            try:
                self.client.close()
                logger.info("Milvus 连接已关闭")
            except Exception as e:
                logger.warning("关闭 Milvus 连接时出错: %s", e)


# ── 模块级单例 ──────────────────────────────────────────────────────
_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """获取 VectorStore 单例（懒初始化）。

    首次调用时完成 Milvus 连接 + BGE-M3 加载 + Reranker 加载，
    后续调用直接返回已初始化的实例。
    """
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore()
    return _store_instance


# ── 自测入口 ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    import time

    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    print("=" * 65)
    print("  FinRAG 端到端测试：文档加载 → 父-子分块 → 向量写入 → 检索")
    print("=" * 65)

    # ═══════════════════════════════════════════════════════════════
    # Step 1: 文档处理
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 1] 文档处理...")
    from rag_qa.core.document_process import DocumentProcessor

    mineru_dir = os.path.join(
        PROJECT_ROOT, 'data', 'mineru_output', '证券市场基础知识'
    )

    doc_processor = DocumentProcessor(parent_size=1200, child_size=300, overlap=50)
    t0 = time.perf_counter()
    child_docs = doc_processor.process([mineru_dir])
    t1 = time.perf_counter()

    print(f"  耗时: {t1 - t0:.1f}s")
    print(f"  子块总数: {len(child_docs)}")

    # 统计父块数和文档来源
    parents = set(d.metadata.get('parent_id', '') for d in child_docs)
    sources = set(d.metadata.get('source', '') for d in child_docs)
    print(f"  父块数: {len(parents)}")
    print(f"  文档数: {len(sources)}")
    for s in sources:
        title = next(
            (d.metadata.get('doc_title', '?') for d in child_docs
             if d.metadata.get('source') == s), '?'
        )
        count = sum(1 for d in child_docs if d.metadata.get('source') == s)
        print(f"    - {title}: {count} 子块")

    # ═══════════════════════════════════════════════════════════════
    # Step 2: 初始化向量库
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 2] 初始化向量库...")
    t0 = time.perf_counter()
    store = VectorStore()
    t1 = time.perf_counter()
    print(f"  模型加载耗时: {t1 - t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # Step 3: 创建 collection + 写入向量
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 3] 创建 collection 并写入向量...")
    store.create_collection()  # collection 已存在则跳过，upsert 自动去重

    t0 = time.perf_counter()
    inserted = store.insert(child_docs)
    t1 = time.perf_counter()
    print(f"  写入耗时: {t1 - t0:.1f}s")
    print(f"  写入条数: {inserted}")

    stats = store.get_collection_stats()
    print(f"  Collection 文档数: {stats['count']}")

    # ═══════════════════════════════════════════════════════════════
    # Step 4: 检索测试
    # ═══════════════════════════════════════════════════════════════
    print("\n[Step 4] 检索测试...")
    print("-" * 65)

    test_queries = [
        "什么是股票市场？",
        "证券市场的风险有哪些？",
        "如何进行价值投资？",
        "公司治理的主要内容是什么？",
    ]

    for query in test_queries:
        t0 = time.perf_counter()
        results = store.search(query, top_k=10, rerank_top_k=3)
        t1 = time.perf_counter()

        print(f"\n Query: {query}")
        print(f" 耗时: {(t1 - t0) * 1000:.0f}ms | 返回: {len(results)} 条")

        for rank, r in enumerate(results, 1):
            print(f"  [{rank}] score={r['score']:.4f} | "
                  f"doc={r['metadata']['doc_title'][:20]} | "
                  f"headings={r['metadata']['headings'][:30]}")
            # 展示内容片段
            snippet = r['content'][:100].replace('\n', ' ')
            print(f"      内容: {snippet}...")

    # ═══════════════════════════════════════════════════════════════
    # Step 5: 去重效果验证
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("  去重验证：同次查询中的 parent_id 是否唯一")
    print("-" * 65)
    query = "证券市场基础知识"
    results = store.search(query, top_k=20, rerank_top_k=10)
    parent_ids = [r['metadata']['parent_id'] for r in results]
    unique_parents = set(parent_ids)
    print(f"  返回 {len(results)} 条结果")
    print(f"  唯一 parent_id: {len(unique_parents)} 个")
    print(f"  去重: {'OK' if len(parent_ids) == len(unique_parents) else 'FAIL'}")

    store.close()
    print(f"\n{'=' * 65}")
    print("  端到端测试完成")
    print("=" * 65)
