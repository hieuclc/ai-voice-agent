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
from openai import AsyncOpenAI
from pydantic import BaseModel
import asyncio

from agent import preload_bge_model
from agent_routing import create_router_agent, pre_route, get_tts_agents

EXTRA_TOOLS: list = []

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
load_dotenv(override=True)

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
LLM_MODEL       = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# Domains mà cần tool call → cần emit thinking sentences
_TOOL_DOMAINS = {"law", "admission", "tour"}

# AsyncOpenAI client dùng để stream trực tiếp cho normal_talk
_openai_client: Optional[AsyncOpenAI] = None


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
    return _openai_client


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

def _strip_pipecat_persona(messages: list[ChatMessage]) -> list[ChatMessage]:
    """
    Strip persona system prompt (index 0) cho tool domains (law/admission/tour).
    Backend có system prompt riêng cho từng domain → tránh conflict.

    KHÔNG strip cho normal_talk — persona prompt của Pipecat đã đủ tốt
    (tiếng Việt, không markdown, giọng tự nhiên) và bỏ đi sẽ mất context.
    """
    if messages and messages[0].role == "system":
        return messages[1:]
    return messages


def to_lc_messages(messages: list[ChatMessage], strip_persona: bool = True) -> list:
    """
    Convert ChatMessage list sang LangChain messages.

    strip_persona=True  : dùng cho tool domains, bỏ system prompt đầu tiên.
    strip_persona=False : dùng cho normal_talk, giữ nguyên toàn bộ messages.
    """
    source = _strip_pipecat_persona(messages) if strip_persona else messages
    out = []
    for m in source:
        if m.role == "system":
            out.append(SystemMessage(content=m.content))
        elif m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            out.append(AIMessage(content=m.content))
    return out


def to_openai_messages(messages: list[ChatMessage]) -> list[dict]:
    """
    Convert ChatMessage list sang OpenAI-format dicts, giữ nguyên toàn bộ
    (kể cả system prompt đầu tiên) — dùng cho normal_talk direct stream.
    """
    return [{"role": m.role, "content": m.content} for m in messages]


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
    raw_messages: list[ChatMessage],
    model: str,
    completion_id: str,
) -> AsyncIterator[str]:
    """
    Stream agent response.

    normal_talk:
      - Giữ nguyên persona system prompt từ Pipecat (không strip).
      - Stream trực tiếp qua AsyncOpenAI.create(), bypass hoàn toàn graph
        và TTSNormalizerAgent.
      - Token đầu tiên về ~200ms thay vì ~1.6s.

    Tool domains (law/admission/tour):
      - Strip persona prompt, inject domain system prompt riêng.
      - Emit thinking sentence ngay lập tức.
      - ainvoke graph song song, sau đó stream qua TTSNormalizerAgent.
    """
    from agent import (
        THINKING_INTERVAL_SECONDS,
        pick_thinking_start_sentence,
        pick_thinking_ongoing_sentence,
    )

    yield _sse(_chunk_payload(completion_id, model, {"role": "assistant", "content": ""}))

    # ------------------------------------------------------------------
    # Bước 1: Pre-route — xác định domain nhanh
    # ------------------------------------------------------------------
    # Pre-route dùng lc_messages (bỏ system[0]) để không bị nhiễu bởi persona
    lc_messages_for_route = to_lc_messages(raw_messages, strip_persona=True)
    domain = await pre_route(lc_messages_for_route)
    logger.info("stream_agent_response: pre_route → %s", domain)

    needs_thinking = domain in _TOOL_DOMAINS

    # ------------------------------------------------------------------
    # Bước 2: normal_talk — direct OpenAI stream, giữ nguyên persona prompt
    # ------------------------------------------------------------------
    if not needs_thinking:
        openai_messages = to_openai_messages(raw_messages)
        client = _get_openai_client()

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=openai_messages,
                stream=True,
                temperature=0,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield _sse(_chunk_payload(completion_id, model, {"content": delta.content}))
        except Exception as exc:
            logger.error("normal_talk direct stream error: %s", exc)

        yield _sse(_chunk_payload(completion_id, model, {}, finish_reason="stop"))
        yield _sse_done()
        return

    # ------------------------------------------------------------------
    # Bước 3: Tool domains — strip persona, dùng domain system prompt
    # ------------------------------------------------------------------
    lc_messages = to_lc_messages(raw_messages, strip_persona=True)

    initial_state = {
        "messages":          lc_messages,
        "hop_count":         0,
        "thinking_streamed": False,
        "domain":            domain,
        "skip_tts":          True,
    }

    output_queue: asyncio.Queue = asyncio.Queue()
    agent_done_event = asyncio.Event()
    raw_agent_text: list[str] = []

    async def agent_runner():
        try:
            result = await graph.ainvoke(initial_state)
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None) and msg.content.strip():
                    raw_agent_text.append(msg.content)
                    break
        except Exception as exc:
            logger.error("agent_runner error: %s", exc)
        finally:
            agent_done_event.set()
            await output_queue.put(("done", None))

    async def thinking_producer():
        await output_queue.put(("thinking", pick_thinking_start_sentence()))
        while True:
            try:
                await asyncio.wait_for(
                    asyncio.shield(agent_done_event.wait()),
                    timeout=THINKING_INTERVAL_SECONDS,
                )
                break
            except asyncio.TimeoutError:
                if not agent_done_event.is_set():
                    await output_queue.put(("thinking", pick_thinking_ongoing_sentence()))
                else:
                    break

    agent_task = asyncio.create_task(agent_runner())
    think_task = asyncio.create_task(thinking_producer())

    try:
        while True:
            kind, value = await output_queue.get()
            if kind == "done":
                break
            elif kind == "thinking":
                logger.info("Emitting thinking sentence: %r", value)
                yield _sse(_chunk_payload(completion_id, model, {"content": value}))
                yield _sse(_chunk_payload(completion_id, model, {"content": " "}))
    finally:
        think_task.cancel()
        if not agent_task.done():
            agent_task.cancel()

    # ------------------------------------------------------------------
    # Bước 4: Stream TTS normalize cho tool domains
    # ------------------------------------------------------------------
    raw_text = raw_agent_text[0] if raw_agent_text else ""
    if raw_text:
        tts_agents = get_tts_agents()
        tts = tts_agents.get(domain) or tts_agents.get("normal_talk")
        if tts is not None:
            async for token_chunk in tts.astream_normalize(raw_text):
                if token_chunk:
                    yield _sse(_chunk_payload(completion_id, model, {"content": token_chunk}))
        else:
            words = raw_text.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == len(words) - 1 else word + " "
                yield _sse(_chunk_payload(completion_id, model, {"content": chunk}))

    yield _sse(_chunk_payload(completion_id, model, {}, finish_reason="stop"))
    yield _sse_done()


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------

async def get_full_response(graph, lc_messages: list) -> str:
    domain = await pre_route(lc_messages)
    result = await graph.ainvoke(
        {
            "messages":          lc_messages,
            "hop_count":         0,
            "thinking_streamed": False,
            "domain":            domain,
        }
    )
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return msg.content
    return ""


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
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
        "data": [{"id": LLM_MODEL, "object": "model", "created": int(time.time()), "owned_by": "rag-agent"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    model         = request.model

    if request.stream:
        return StreamingResponse(
            # Truyền raw_messages để normal_talk có thể giữ persona prompt
            stream_agent_response(graph, request.messages, model, completion_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection":    "keep-alive",
            },
        )

    lc_messages = to_lc_messages(request.messages, strip_persona=True)
    content = await get_full_response(graph, lc_messages)
    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )