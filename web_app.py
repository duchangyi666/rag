"""FinRAG Web 服务入口。

提供：
  - GET  /api/health  服务状态
  - POST /api/chat    普通 HTTP 问答
  - WS   /ws/chat     大模型流式问答
  - GET  /            金融咨询客服页面

默认复用 main.py 中的真实 query / query_stream。设置环境变量
FINRAG_DEMO=1 时使用轻量演示回答，便于只启动前端进行联调。
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from main import init_components, query, query_stream


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEMO_MODE = os.getenv("FINRAG_DEMO", "").lower() in {"1", "true", "yes", "on"}
PRELOAD_COMPONENTS = os.getenv("FINRAG_PRELOAD", "1").lower() in {
    "1", "true", "yes", "on"
}


@asynccontextmanager
async def lifespan(application: FastAPI):
    """生产模式启动时预加载模型，避免首个用户承担初始化等待。"""
    application.state.components = []
    if not DEMO_MODE and PRELOAD_COMPONENTS:
        application.state.components = await asyncio.to_thread(
            init_components, False
        )
    yield

app = FastAPI(
    title="FinRAG 金融理财咨询 API",
    description="FAQ、意图识别与 RAG 三级级联金融问答服务",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("FINRAG_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: List[HistoryItem] = Field(default_factory=list, max_length=10)


class ChatResponse(BaseModel):
    answer: str
    source: str
    confidence: float | None = None
    intent: str
    latency_ms: float


def _history_to_text(items: List[HistoryItem]) -> str:
    labels = {"user": "用户", "assistant": "助手"}
    return "\n".join(f"{labels[item.role]}: {item.content}" for item in items[-10:])


def _demo_result(message: str) -> Dict[str, Any]:
    started = time.perf_counter()
    answer = (
        f"关于“{message}”，建议先从资金用途、投资期限和风险承受能力三个维度评估。\n\n"
        "如果这笔资金在一年内可能使用，应优先关注流动性与本金波动；如果投资期限较长，"
        "可以在充分理解风险的前提下考虑分散配置。购买任何产品前，请重点核对产品说明书中的"
        "风险等级、费用、赎回规则和历史最大回撤。\n\n"
        "以上内容仅作金融知识参考，不构成具体投资建议。"
    )
    return {
        "answer": answer,
        "source": "demo",
        "confidence": None,
        "intent": "专业咨询",
        "latency_ms": (time.perf_counter() - started) * 1000,
    }


def _demo_stream(message: str):
    result = _demo_result(message)
    yield {"type": "status", "step": "faq", "value": "正在匹配金融知识库"}
    time.sleep(0.25)
    yield {"type": "status", "step": "intent", "value": "专业咨询"}
    time.sleep(0.25)
    yield {"type": "status", "step": "retrieval", "value": "已召回 6 条相关资料"}
    time.sleep(0.25)
    yield {"type": "status", "step": "rerank", "value": "相关性重排完成"}
    time.sleep(0.2)
    for char in result["answer"]:
        yield {"type": "token", "content": char}
        time.sleep(0.012)
    yield {
        "type": "done",
        "source": result["source"],
        "intent": result["intent"],
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"] + 950,
    }


def _iter_stream(message: str, history: str):
    if DEMO_MODE:
        return _demo_stream(message)
    return query_stream(message, history=history)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    component_rows = [
        {"name": name, "ready": ready, "detail": detail}
        for name, ready, detail in getattr(app.state, "components", [])
    ]
    all_ready = all(row["ready"] for row in component_rows)
    return {
        "status": "ok" if all_ready else "degraded",
        "service": "FinRAG",
        "mode": "demo" if DEMO_MODE else "production",
        "streaming": True,
        "components": component_rows,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> Dict[str, Any]:
    history = _history_to_text(payload.history)
    if DEMO_MODE:
        return _demo_result(payload.message)
    return await asyncio.to_thread(query, payload.message, history)


@app.websocket("/ws/chat")
async def chat_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json(
        {"type": "ready", "message": "FinRAG streaming channel connected"}
    )

    try:
        while True:
            raw_payload = await websocket.receive_json()
            try:
                payload = ChatRequest.model_validate(raw_payload)
            except Exception as exc:
                await websocket.send_json(
                    {"type": "error", "message": f"请求格式错误：{exc}"}
                )
                continue

            queue: asyncio.Queue[Dict[str, Any] | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()
            history = _history_to_text(payload.history)

            def produce() -> None:
                try:
                    for event in _iter_stream(payload.message, history):
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                except Exception as exc:
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"type": "error", "message": f"服务处理失败：{exc}"},
                    )
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            producer = asyncio.create_task(asyncio.to_thread(produce))

            while True:
                event = await queue.get()
                if event is None:
                    break
                await websocket.send_json(event)

            await producer
    except (WebSocketDisconnect, RuntimeError):
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web_app:app",
        host=os.getenv("FINRAG_HOST", "0.0.0.0"),
        port=int(os.getenv("FINRAG_PORT", "8010")),
        reload=os.getenv("FINRAG_RELOAD", "").lower() in {"1", "true"},
    )
