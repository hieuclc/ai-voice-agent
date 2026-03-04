"""
OpenAI-compatible REST API server wrapping the LangGraph RAG agent.

Endpoints:
  POST /v1/chat/completions   (streaming + non-streaming)
  GET  /v1/models
  GET  /health

Run:
  uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dotenv import load_dotenv
from typing import AsyncIterator, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from pydantic import BaseModel
import asyncio

from agent import preload_bge_model
from agent_routing import create_router_agent

# ---------------------------------------------------------------------------
# Extra tools  ← add your independently-implemented tools here
# ---------------------------------------------------------------------------
# from my_tools.web import tavily_search
# EXTRA_TOOLS = [tavily_search]
EXTRA_TOOLS: list = []

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
LLM_MODEL       = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# TTS Normalizer — import sau khi load_dotenv để OPENAI_API_KEY sẵn sàng
# ---------------------------------------------------------------------------
from tts_normalizer import TTSNormalizerAgent

_tts = TTSNormalizerAgent(
    model=LLM_MODEL,
    use_llm_fallback=True,
    openai_api_key=OPENAI_API_KEY,
    openai_base_url=OPENAI_BASE_URL,
)

# ---------------------------------------------------------------------------
# OpenAI request / response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = LLM_MODEL
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[list[str] | str] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None


# ---------------------------------------------------------------------------
# LangChain message conversion
# ---------------------------------------------------------------------------

def to_lc_messages(messages: list[ChatMessage]) -> list:
    out = []
    for m in messages:
        if m.role == "system":
            out.append(SystemMessage(content=m.content))
        elif m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            out.append(AIMessage(content=m.content))
    return out


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _chunk_payload(
    completion_id: str,
    model: str,
    delta: dict,
    finish_reason: Optional[str] = None,
) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------

async def stream_agent_response(
    graph,
    lc_messages: list,
    model: str,
    completion_id: str,
) -> AsyncIterator[str]:
    """
    Stream agent response với thinking sentences tiếng Việt.

    Cơ chế TTS normalization trong streaming:
      - Các token thực từ LLM được GOM vào buffer (không stream ngay).
      - Khi agent kết thúc, buffer được normalize bằng TTSNormalizerAgent.
      - Sau đó mới stream từng ký tự/từ của text đã normalize ra client.
      - Thinking sentences vẫn stream real-time như cũ (không cần normalize).
    """
    from agent import (
        THINKING_INTERVAL_SECONDS,
        pick_thinking_start_sentence,
        pick_thinking_ongoing_sentence,
    )

    yield _sse(_chunk_payload(completion_id, model, {"role": "assistant", "content": ""}))

    initial_state = {
        "messages": lc_messages,
        "hop_count": 0,
        "thinking_streamed": False,
    }

    output_queue: asyncio.Queue = asyncio.Queue()
    tool_started_event = asyncio.Event()
    llm_started_event  = asyncio.Event()

    # Buffer gom toàn bộ token thực từ LLM trước khi normalize
    token_buffer: list[str] = []

    async def agent_consumer():
        try:
            async for event in graph.astream_events(initial_state, version="v2"):
                kind = event.get("event", "")

                if kind == "on_tool_start":
                    if not tool_started_event.is_set():
                        tool_started_event.set()
                        await output_queue.put(("thinking", pick_thinking_start_sentence()))

                elif kind == "on_chat_model_stream":
                    chunk: AIMessageChunk = event["data"]["chunk"]
                    if chunk.tool_call_chunks:
                        continue
                    token = chunk.content
                    if not token:
                        continue
                    if not llm_started_event.is_set():
                        llm_started_event.set()
                    # Gom token vào buffer, KHÔNG put vào queue ngay
                    token_buffer.append(token)

        except Exception as exc:
            logger.error("agent_consumer error: %s", exc)
        finally:
            await output_queue.put(("done", None))

    async def thinking_producer():
        await tool_started_event.wait()
        while True:
            try:
                await asyncio.wait_for(
                    asyncio.shield(llm_started_event.wait()),
                    timeout=THINKING_INTERVAL_SECONDS,
                )
                break
            except asyncio.TimeoutError:
                if not llm_started_event.is_set():
                    await output_queue.put(("thinking", pick_thinking_ongoing_sentence()))
                else:
                    break

    agent_task = asyncio.create_task(agent_consumer())
    think_task = asyncio.create_task(thinking_producer())

    # Drain thinking sentences trong khi agent chạy
    try:
        while True:
            kind, value = await output_queue.get()
            if kind == "done":
                break
            elif kind == "thinking":
                yield _sse(_chunk_payload(completion_id, model, {"content": f"\n\n{value}\n\n"}))
            # kind == "token" không còn xuất hiện ở đây nữa (đã gom vào buffer)
    finally:
        think_task.cancel()
        if not agent_task.done():
            agent_task.cancel()

    # --- Normalize toàn bộ response rồi stream từng từ ---
    if token_buffer:
        raw_text = "".join(token_buffer)
        try:
            normalized = await _tts.anormalize(raw_text)
        except Exception as exc:
            logger.warning("TTS normalize lỗi, dùng text gốc: %s", exc)
            normalized = raw_text

        # Stream từng từ (split by space) để client vẫn thấy hiệu ứng streaming
        words = normalized.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else word + " "
            yield _sse(_chunk_payload(completion_id, model, {"content": chunk}))

    yield _sse(_chunk_payload(completion_id, model, {}, finish_reason="stop"))
    yield _sse_done()


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------

async def get_full_response(graph, lc_messages: list, model: str) -> str:
    result = await graph.ainvoke(
        {"messages": lc_messages, "hop_count": 0, "thinking_streamed": False}
    )
    raw = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            raw = msg.content
            break

    # Normalize tại đây — đảm bảo cả non-streaming path đều được xử lý
    try:
        return await _tts.anormalize(raw)
    except Exception as exc:
        logger.warning("TTS normalize lỗi, dùng text gốc: %s", exc)
        return raw


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm all models on startup so first request is fast."""
    logger.info("Startup: loading BGE-M3 embedding model…")
    await preload_bge_model()
    logger.info("Startup: all services ready.")
    yield
    logger.info("Shutdown: bye.")


app = FastAPI(title="RAG Agent – OpenAI Compatible API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the agent once at startup
graph = create_router_agent(
    extra_tools=EXTRA_TOOLS,
    model=LLM_MODEL,
    openai_api_key=OPENAI_API_KEY,
    openai_base_url=OPENAI_BASE_URL,
)
logger.info("Agent ready. LLM: %s", LLM_MODEL)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": LLM_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rag-agent",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    lc_messages   = to_lc_messages(request.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    model         = request.model

    if request.stream:
        return StreamingResponse(
            stream_agent_response(graph, lc_messages, model, completion_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    content = await get_full_response(graph, lc_messages, model)
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )