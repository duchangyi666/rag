# FinRAG — 金融智能问答系统

金融领域智能问答系统，采用 **FAQ + RAG 三级级联路由**策略，在保证回答质量的同时显著降低大模型调用成本。

## 系统架构

三级级联查询策略，逐级降级：

```
用户提问
   │
   ▼
① FAQ 快查 (BM25 + Redis/MySQL) ──命中──▶ 直接返回（延迟 ~10ms）
   │ 未命中
   ▼
② BERT 意图分类 ──▶ 通用金融概念 ──▶ ③A LLM 直接回答（无需检索）
   │ 专业问题
   ▼
③B RAG 完整流程（向量检索 + 重排序 + 生成）
```

## 功能特性

- **FAQ 快查**：基于 BM25 + Redis/MySQL 的常见问题秒级命中
- **BERT 意图分类**：区分通用金融概念与专业咨询问题（bert-base-chinese 微调）
- **RAG 检索增强**：BGE-M3 向量化 + BGE-Reranker 重排序 + Milvus 向量库
- **Web 界面**：FastAPI + WebSocket 流式输出（`static/` 前端）
- **多轮对话**：支持对话历史上下文
- **可观测**：结构化日志、路由策略可追踪

## 目录结构

```
├── main.py              # 命令行入口（交互/单次查询）
├── web_app.py           # Web 服务（FastAPI + WebSocket）
├── base/                # 基础组件（配置加载、日志）
├── mysql_qa/            # FAQ 查询链路（BM25 + Redis/MySQL）
├── rag_qa/              # RAG 链路（检索、重排序、生成、BERT 分类）
│   └── core/            # 路由、向量库、策略选择等核心逻辑
├── scripts/             # 运维脚本（环境检查、知识入库）
├── static/              # Web 前端（HTML/JS/CSS）
├── data/                # 知识文档与数据（不入库，本地维护）
└── config.ini.example   # 配置模板（复制为 config.ini 并填写凭据）
```

## 快速开始

### 1. 环境准备

```bash
conda create -n finrag python=3.10
conda activate finrag
pip install -r requirements.txt
```

### 2. 配置

```bash
cp config.ini.example config.ini
```

按需修改 `config.ini` 中的 MySQL、Redis、Milvus 连接信息和 LLM API Key。

### 3. 知识入库（可选）

```bash
python scripts/ingest_knowledge.py "data/你的知识文档目录"
```

### 4. 运行

```bash
# 命令行交互模式
python main.py

# 单次查询
python main.py "什么是ETF？"

# Web 服务（浏览器访问 http://127.0.0.1:8010）
python web_app.py
```

## Web 接口

| 接口 | 说明 |
|------|------|
| `GET /api/health` | 服务健康检查 |
| `POST /api/chat` | JSON 问答（WebSocket 降级方案） |
| `WS /ws/chat` | WebSocket 流式问答 |
| `GET /docs` | FastAPI 自动生成接口文档 |

## 技术栈

FastAPI · MySQL · Redis · Milvus · BM25 · BERT · BGE-M3 · BGE-Reranker · 智谱 GLM · WebSocket

## 说明

- **模型权重**（`rag_qa/models/`，约 5.9GB）与**知识数据**（`data/`）因体积较大不入库，需自行下载或按脚本构建。
- `config.ini` 含真实凭据，已被 `.gitignore` 排除，请勿提交。
