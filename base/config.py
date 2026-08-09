"""
FinRAG 配置加载器

读取项目根目录的 config.ini，将参数加载为实例属性。

Usage:
    from base.config import config
    print(config.mysql_host)
    print(config.log_name)
"""

import configparser
import os


# 项目根目录（base/config.py 的父目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    """加载 config.ini，将各 section 参数设为实例属性。"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, 'config.ini')

        # 解析配置文件（interpolation=None 避免 %(xxx)s 被误解析为变量引用）
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(config_path, encoding='utf-8')

        # ── [mysql] 数据库配置 ────────────────────────────────────
        self.mysql_host = parser.get('mysql', 'host', fallback='localhost')
        self.mysql_port = parser.getint('mysql', 'port', fallback=3306)
        self.mysql_user = parser.get('mysql', 'user', fallback='root')
        self.mysql_password = parser.get('mysql', 'password', fallback='')
        self.mysql_database = parser.get('mysql', 'database', fallback='')
        self.mysql_charset = parser.get('mysql', 'charset', fallback='utf8mb4')

        # ── [redis] 缓存配置 ────────────────────────────────────
        self.redis_host = parser.get('redis', 'host', fallback='localhost')
        self.redis_port = parser.getint('redis', 'port', fallback=6379)
        self.redis_password = parser.get('redis', 'password', fallback='')
        self.redis_db = parser.getint('redis', 'db', fallback=0)
        self.redis_ex = parser.getint('redis', 'ex', fallback=86400)

        # ── [bm25] BM25 检索配置 ──────────────────────────────────
        self.bm25_threshold = parser.getfloat('bm25', 'threshold', fallback=0.85)
        self.bm25_k1 = parser.getfloat('bm25', 'k1', fallback=1.5)
        self.bm25_b = parser.getfloat('bm25', 'b', fallback=0.75)
        self.bm25_softmax_temperature = parser.getfloat('bm25', 'softmax_temperature', fallback=1.0)
        self.bm25_min_token_coverage = parser.getfloat(
            'bm25', 'min_token_coverage', fallback=0.6
        )

        # ── [milvus] 向量数据库配置 ───────────────────────────────
        self.milvus_host = parser.get('milvus', 'host', fallback='localhost')
        self.milvus_port = parser.getint('milvus', 'port', fallback=19530)
        self.milvus_database_name = parser.get('milvus', 'database_name', fallback='default')
        self.milvus_collection_name = parser.get('milvus', 'collection_name', fallback='finrag')

        # ── [models] 模型路径配置 ──────────────────────────────────
        models_dir_rel = parser.get('models', 'models_dir', fallback='rag_qa/models')
        embedding_rel = parser.get('models', 'embedding_model', fallback='rag_qa/models/bge-m3')
        reranker_rel = parser.get('models', 'reranker_model', fallback='rag_qa/models/bge-reranker-large')
        self.models_dir = os.path.abspath(os.path.join(PROJECT_ROOT, models_dir_rel))
        self.embedding_model_path = os.path.abspath(os.path.join(PROJECT_ROOT, embedding_rel))
        self.reranker_model_path = os.path.abspath(os.path.join(PROJECT_ROOT, reranker_rel))
        self.embedding_dim = parser.getint('models', 'embedding_dim', fallback=1024)

        # ── [retrieval] 检索质量控制 ────────────────────────────────
        self.rerank_min_score = parser.getfloat(
            'retrieval', 'rerank_min_score', fallback=0.005
        )

        # ── [bert] BERT 模型训练与推理配置 ───────────────────────────
        self.PRE_TRAINED_MODEL_NAME = parser.get('bert', 'pre_trained_model_name', fallback='bert-base-chinese')
        self.NUM_LABELS = parser.getint('bert', 'num_labels', fallback=2)
        self.MAX_LENGTH = parser.getint('bert', 'max_length', fallback=128)
        self.TEST_SIZE = parser.getfloat('bert', 'test_size', fallback=0.2)
        self.RANDOM_STATE = parser.getint('bert', 'random_state', fallback=42)
        self.NUM_TRAIN_EPOCHS = parser.getint('bert', 'num_train_epochs', fallback=3)
        self.PER_DEVICE_TRAIN_BATCH_SIZE = parser.getint('bert', 'per_device_train_batch_size', fallback=8)
        self.PER_DEVICE_EVAL_BATCH_SIZE = parser.getint('bert', 'per_device_eval_batch_size', fallback=8)
        self.WARMUP_STEPS = parser.getint('bert', 'warmup_steps', fallback=500)
        self.WEIGHT_DECAY = parser.getfloat('bert', 'weight_decay', fallback=0.01)
        self.LOGGING_STEPS = parser.getint('bert', 'logging_steps', fallback=10)
        self.EVALUATION_STRATEGY = parser.get('bert', 'evaluation_strategy', fallback='epoch')
        self.SAVE_STRATEGY = parser.get('bert', 'save_strategy', fallback='epoch')
        self.LOAD_BEST_MODEL_AT_END = parser.getboolean('bert', 'load_best_model_at_end', fallback=True)
        self.SAVE_TOTAL_LIMIT = parser.getint('bert', 'save_total_limit', fallback=1)
        self.METRIC_FOR_BEST_MODEL = parser.get('bert', 'metric_for_best_model', fallback='eval_loss')
        self.FP16 = parser.getboolean('bert', 'fp16', fallback=False)

        # 派生路径 — 预训练模型目录
        bert_model_rel = parser.get('bert', 'model_path', fallback='bert_query_classifier')
        logging_dir_rel = parser.get('bert', 'logging_dir', fallback='./bert_logs')
        output_dir_rel = parser.get('bert', 'output_dir', fallback='bert_results')

        self.PRE_TRAINED_MODEL_PATH = os.path.join(self.models_dir, self.PRE_TRAINED_MODEL_NAME)
        self.BERT_QUERY_CLASSIFIER = os.path.join(self.models_dir, bert_model_rel)
        self.BERT_TRAIN_DATA = os.path.join(PROJECT_ROOT, 'rag_qa', 'data', 'training_dataset_financial_5000.jsonl')
        self.OUTPUT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, output_dir_rel))
        self.LOGGING_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, logging_dir_rel))

        # 设备选择（延迟导入 torch，避免模块级强依赖）
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"

        # ── [llm] 大语言模型配置 ──────────────────────────────────
        self.llm_model = parser.get('llm', 'model', fallback='glm-4-flash')
        self.llm_base_url = parser.get('llm', 'llm_base_url', fallback='')
        self.llm_api_key = os.environ.get(
            'ZHIPUAI_API_KEY',
            parser.get('llm', 'api_key', fallback='')
        )

        # ── [log] 日志配置 ────────────────────────────────────────
        self.log_name = parser.get('log', 'name', fallback='fin_rag')
        self.log_console_level = parser.get('log', 'console_level', fallback='INFO')
        self.log_file_level = parser.get('log', 'file_level', fallback='DEBUG')
        self.log_enable_console = parser.getboolean('log', 'enable_console', fallback=True)
        self.log_format = parser.get(
            'log', 'log_format',
            fallback='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - '
                     '[%(threadName)s] - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        self.log_date_format = parser.get('log', 'date_format', fallback='%Y-%m-%d %H:%M:%S')
        self.log_max_bytes_mb = parser.getint('log', 'max_bytes_mb', fallback=10)
        self.log_backup_count = parser.getint('log', 'backup_count', fallback=7)
        self.log_encoding = parser.get('log', 'encoding', fallback='utf-8')

        # 日志路径（相对于项目根目录，解析为绝对路径）
        log_dir = parser.get('log', 'log_dir', fallback='logs')
        log_filename = parser.get('log', 'log_filename', fallback='app.log')
        self.log_dir_abs = os.path.abspath(os.path.join(PROJECT_ROOT, log_dir))
        self.log_filepath = os.path.join(self.log_dir_abs, log_filename)



# ── 模块级单例 ──────────────────────────────────────────────────────
config = Config()
