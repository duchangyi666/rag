# FinSight AI Web 客服

Web 页面复用 `main.py` 中现有的 FAQ、BERT、LLM 与 RAG 查询链路。

## 安装与启动

本机已经存在包含完整依赖的 `EduRAG` Conda 环境，建议固定使用它：

```powershell
conda activate EduRAG
python scripts/check_system.py
python scripts/bootstrap_finrag.py
python web_app.py
```

新增或更新知识文档后，可按路径增量写入 Milvus：

```powershell
python scripts/ingest_knowledge.py "data/mineru_output/证券市场基础知识"
```

浏览器访问 `http://127.0.0.1:8010`。

如果需要在新环境安装完整依赖：

```powershell
pip install -r requirements.txt
```

仅预览页面和流式交互（不连接 MySQL、Redis、Milvus 与大模型）：

```powershell
$env:FINRAG_DEMO = "1"
python web_app.py
```

## 接口

- `GET /api/health`：服务状态，前端通过 `fetch` 调用。
- `POST /api/chat`：普通 JSON 问答，也是 WebSocket 不可用时的降级接口。
- `WS /ws/chat`：发送 `{"message":"问题","history":[]}`，接收 `status`、`token`、`done` 事件。
- `GET /docs`：FastAPI 自动生成的接口文档。
