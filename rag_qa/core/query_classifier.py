import json
import os

import numpy as np
import torch
from base.config import config as conf
from base.logger import single_logger as logger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

"""
Query 分类模块。

基于 BERT 微调的二分类器，用于判断用户输入属于：
    1. 通用知识：金融概念、投资原理、市场机制等通用金融知识问答
    2. 专业咨询：具体银行/基金/理财产品咨询、费率查询、产品对比等需精准检索的咨询问题

分类结果决定检索策略：
    - 通用知识 → 向量语义检索（RAG）
    - 专业咨询 → 关键词检索（BM25）+ 知识库精确匹配
"""


class QueryClassifier:
    """
    BERT Query 分类器。

    职责：
        1. 加载本地 BERT 分词器。
        2. 加载已训练分类模型；如果不存在，则基于本地预训练 BERT 初始化新模型。
        3. 读取 JSONL 训练数据并完成训练。
        4. 对用户 query 进行类别预测。

    Args:
        model_path: 训练后分类模型保存目录，默认使用配置中的 conf.BERT_QUERY_CLASSIFIER。
    """

    def __init__(self, model_path=conf.BERT_QUERY_CLASSIFIER):
        self.pre_trained_model_path = conf.PRE_TRAINED_MODEL_PATH
        self.model_path = model_path
        self.model = None
        self.device = conf.device
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        self.id_to_label = {value: key for key, value in self.label_map.items()}

        logger.info(
            "初始化 QueryClassifier [pretrained=%s, model_path=%s, device=%s]",
            self.pre_trained_model_path,
            self.model_path,
            self.device
        )

        self.tokenizer = self._load_tokenizer()
        self.load_model()

    def _load_tokenizer(self):
        """从本地预训练模型目录加载 BERT 分词器。"""
        if not os.path.exists(self.pre_trained_model_path):
            logger.error("BERT 预训练模型目录不存在 [path=%s]", self.pre_trained_model_path)
            raise FileNotFoundError(f"BERT 预训练模型目录不存在: {self.pre_trained_model_path}")

        try:
            tokenizer = BertTokenizer.from_pretrained(
                self.pre_trained_model_path,
                local_files_only=True
            )
            logger.info("BERT 分词器加载成功 [path=%s]", self.pre_trained_model_path)
            return tokenizer
        except Exception as e:
            logger.exception("BERT 分词器加载失败 [path=%s]", self.pre_trained_model_path)
            raise RuntimeError(f"BERT 分词器加载失败: {e}") from e

    def load_model(self):
        """
        加载分类模型。

        优先加载已经训练好的分类模型；如果分类模型目录不存在，
        则使用本地 bert-base-chinese 初始化一个新的二分类模型。
        """
        try:
            if os.path.exists(self.model_path):
                self.model = BertForSequenceClassification.from_pretrained(
                    self.model_path,
                    local_files_only=True
                )
                logger.info("已加载训练好的 BERT 分类模型 [path=%s]", self.model_path)
            else:
                self.model = BertForSequenceClassification.from_pretrained(
                    self.pre_trained_model_path,
                    num_labels=conf.NUM_LABELS,
                    local_files_only=True
                )
                logger.warning(
                    "未找到训练好的分类模型，已基于本地预训练模型初始化新模型 [pretrained=%s, save_path=%s]",
                    self.pre_trained_model_path,
                    self.model_path
                )

            self.model.to(self.device)
            logger.info("BERT 分类模型已移动到设备 [device=%s]", self.device)
        except Exception as e:
            logger.exception(
                "BERT 分类模型加载失败 [model_path=%s, pretrained=%s]",
                self.model_path,
                self.pre_trained_model_path
            )
            raise RuntimeError(f"BERT 分类模型加载失败: {e}") from e

    def save_model(self):
        """保存训练后的分类模型和分词器。"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
            logger.info("模型和分词器保存成功 [path=%s]", self.model_path)
        except Exception as e:
            logger.exception("模型保存失败 [path=%s]", self.model_path)
            raise RuntimeError(f"模型保存失败: {e}") from e

    def preprocess_data(self, texts, labels):
        """
        将原始文本和中文标签转换为 BERT 训练输入。

        Args:
            texts: query 文本列表。
            labels: 中文标签列表，取值应为“通用知识”或“专业咨询”。

        Returns:
            Tuple[BatchEncoding, List[int]]: tokenizer 编码结果和数字标签列表。
        """
        try:
            unknown_labels = sorted(set(labels) - set(self.label_map.keys()))
            if unknown_labels:
                raise ValueError(f"存在未知标签: {unknown_labels}")

            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=conf.MAX_LENGTH,
                return_tensors="pt"
            )
            numeric_labels = [self.label_map[label] for label in labels]
            logger.debug("数据预处理完成 [samples=%d, max_length=%d]", len(texts), conf.MAX_LENGTH)
            return encodings, numeric_labels
        except Exception as e:
            logger.exception("数据预处理失败 [samples=%d]", len(texts) if texts else 0)
            raise RuntimeError(f"数据预处理失败: {e}") from e

    def create_dataset(self, encodings, labels):
        """创建 HuggingFace Trainer 可使用的 PyTorch Dataset。"""

        class QueryDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return QueryDataset(encodings, labels)

    def load_training_data(self, data_file):
        """
        按 JSONL 格式逐行加载训练数据。

        JSONL 格式要求每一行都是一个独立 JSON 对象，例如：
            {"query": "什么是货币市场基金？", "label": "通用知识"}
            {"query": "建行乾元系列理财产品的起购金额怎么查？", "label": "专业咨询"}

        Args:
            data_file: JSONL 训练数据文件路径。

        Returns:
            List[dict]: 包含 query 和 label 字段的样本列表。
        """
        if not os.path.exists(data_file):
            logger.error("训练数据文件不存在 [path=%s]", data_file)
            raise FileNotFoundError(f"训练数据文件不存在: {data_file}")

        data = []
        label_count = {label: 0 for label in self.label_map}

        try:
            with open(data_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.error("训练数据 JSON 解析失败 [line=%d, content=%s]", line_num, line)
                        raise ValueError(f"第 {line_num} 行不是合法 JSON: {line}") from e

                    if not isinstance(item, dict) or "query" not in item or "label" not in item:
                        raise ValueError(f"第 {line_num} 行数据格式错误，必须包含 query 和 label 字段: {item}")

                    if item["label"] not in self.label_map:
                        raise ValueError(f"第 {line_num} 行标签非法: {item['label']}")

                    data.append(item)
                    label_count[item["label"]] += 1

            if not data:
                raise ValueError(f"训练数据文件为空: {data_file}")

            logger.info(
                "训练数据加载完成 [path=%s, total=%d, label_count=%s]",
                data_file,
                len(data),
                label_count
            )
            return data
        except Exception as e:
            logger.exception("训练数据加载失败 [path=%s]", data_file)
            raise RuntimeError(f"训练数据加载失败: {e}") from e

    def train_model(self, data_file=conf.BERT_TRAIN_DATA):
        """训练 BERT 二分类模型。"""
        logger.info("开始训练 QueryClassifier [data_file=%s]", data_file)

        try:
            data = self.load_training_data(data_file)
            texts = [item["query"] for item in data]
            labels = [item["label"] for item in data]

            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts,
                labels,
                test_size=conf.TEST_SIZE,
                random_state=conf.RANDOM_STATE,
                stratify=labels
            )
            logger.info(
                "训练/验证数据划分完成 [train=%d, val=%d, test_size=%s, random_state=%s]",
                len(train_texts),
                len(val_texts),
                conf.TEST_SIZE,
                conf.RANDOM_STATE
            )

            # 保存原始字符串标签，供后续 evaluate_model 使用（preprocess_data 会将标签转为数值）
            val_labels_str = val_labels

            train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
            val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)
            train_dataset = self.create_dataset(train_encodings, train_labels)
            val_dataset = self.create_dataset(val_encodings, val_labels)

            training_args = TrainingArguments(
                output_dir=conf.OUTPUT_DIR,
                num_train_epochs=conf.NUM_TRAIN_EPOCHS,
                per_device_train_batch_size=conf.PER_DEVICE_TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=conf.PER_DEVICE_EVAL_BATCH_SIZE,
                warmup_steps=conf.WARMUP_STEPS,
                weight_decay=conf.WEIGHT_DECAY,
                logging_dir=conf.LOGGING_DIR,
                logging_steps=conf.LOGGING_STEPS,
                eval_strategy=conf.EVALUATION_STRATEGY,
                save_strategy=conf.SAVE_STRATEGY,
                load_best_model_at_end=conf.LOAD_BEST_MODEL_AT_END,
                save_total_limit=conf.SAVE_TOTAL_LIMIT,
                metric_for_best_model=conf.METRIC_FOR_BEST_MODEL,
                fp16=conf.FP16,
            )
            logger.info(
                "训练参数已创建 [epochs=%s, train_batch=%s, eval_batch=%s, output_dir=%s]",
                conf.NUM_TRAIN_EPOCHS,
                conf.PER_DEVICE_TRAIN_BATCH_SIZE,
                conf.PER_DEVICE_EVAL_BATCH_SIZE,
                conf.OUTPUT_DIR
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics
            )

            logger.info("开始执行 Trainer.train()")
            trainer.train()
            logger.info("模型训练完成，准备保存模型")

            self.save_model()
            self.evaluate_model(val_texts, val_labels_str)
            logger.info("QueryClassifier 训练流程完成")
        except Exception as e:
            logger.exception("QueryClassifier 训练失败 [data_file=%s]", data_file)
            raise RuntimeError(f"QueryClassifier 训练失败: {e}") from e

    def compute_metrics(self, eval_pred):
        """计算验证集评估指标。"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def evaluate_model(self, texts, labels):
        """在验证集上评估模型效果，并输出分类报告和混淆矩阵。"""
        try:
            encodings, numeric_labels = self.preprocess_data(texts, labels)
            dataset = self.create_dataset(encodings, numeric_labels)

            trainer = Trainer(model=self.model)
            predictions = trainer.predict(dataset)
            pred_labels = np.argmax(predictions.predictions, axis=-1)

            target_names = [self.id_to_label[i] for i in range(len(self.label_map))]
            report = classification_report(
                numeric_labels,
                pred_labels,
                target_names=target_names
            )
            matrix = confusion_matrix(numeric_labels, pred_labels)

            logger.info("分类报告:\n%s", report)
            logger.info("混淆矩阵:\n%s", matrix)
        except Exception as e:
            logger.exception("模型评估失败 [samples=%d]", len(texts) if texts else 0)
            raise RuntimeError(f"模型评估失败: {e}") from e

    def predict_category(self, query):
        """
        预测单条 query 的类别。

        Args:
            query: 用户输入问题。

        Returns:
            str: “专业咨询”或“通用知识”。
        """
        if self.model is None:
            logger.error("模型未加载，无法预测")
            return "通用知识"

        if not query or not str(query).strip():
            logger.warning("收到空 query，默认返回通用知识")
            return "通用知识"

        try:
            encoding = self.tokenizer(
                str(query),
                truncation=True,
                padding=True,
                max_length=conf.MAX_LENGTH,
                return_tensors="pt"
            )
            encoding = {key: value.to(self.device) for key, value in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**encoding)
                prediction = torch.argmax(outputs.logits, dim=1).item()

            label = self.id_to_label.get(prediction, "通用知识")
            logger.debug("Query 分类完成 [query=%s, label=%s, prediction=%s]", query, label, prediction)
            return label
        except Exception as e:
            logger.exception("Query 分类失败 [query=%s]", query)
            raise RuntimeError(f"Query 分类失败: {e}") from e


single_question_classifier = QueryClassifier()
if __name__ == "__main__":
    try:

        # 如果已存在训练后的分类模型，则直接预测；否则先训练再预测。
        if not os.path.exists(conf.BERT_QUERY_CLASSIFIER):
            single_question_classifier.train_model(data_file=conf.BERT_TRAIN_DATA)

        test_queries = [
            "什么是货币市场基金？",
            "建行乾元系列理财产品的起购金额和期限怎么查？",
            "如何理解年化收益率这个概念？",
            "招行日日鑫和工行添利宝的赎回时间有区别吗？",
            "为什么债券价格和利率呈反向关系？",
            "平安银行天天成长C的费率结构和投资范围是什么？",
            "如何评估一只基金的业绩表现？",
            "想对比农行安心快线和招行日日鑫，哪个更适合短期理财？",
            "什么是复利效应，它在投资中如何体现？",
            "中银理财稳富产品的风险等级和预期收益怎么样？",
        ]
        for query in test_queries:
            label = single_question_classifier.predict_category(query)
            print(f"{query}: {label}")
    except Exception as e:
        logger.exception("QueryClassifier 脚本执行失败")
        raise e