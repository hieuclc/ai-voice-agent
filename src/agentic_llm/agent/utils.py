"""
utils.py — Shared singletons, constants, and RAG utility functions.

Exports:
  Constants   : RETRIEVAL_K, TOP_K, MAX_HOPS, MAX_REWRITE,
                THINKING_INTERVAL_SECONDS, THINKING_RESPONSE_DELAY_SECONDS
  Prompt      : OUTPUT_FORMAT
  Startup     : preload_models()
  RAG         : encode_query(), hybrid_retrieve(), rerank()
  Qdrant      : get_async_qdrant()
  Thinking    : pick_thinking_start_sentence(), pick_thinking_ongoing_sentence()
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Literal, Annotated, Optional

import numpy as np
import torch
from dotenv import load_dotenv

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels

from reranker import Reranker

from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from typing_extensions import TypedDict

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QDRANT_HOST     = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.environ.get("QDRANT_PORT", "6333"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL_NAME", "AITeamVN/Vietnamese_Embedding_v2")
DEVICE          = os.environ.get("DEVICE", "cpu")

RETRIEVAL_K = 10
TOP_K       = 5
MAX_HOPS    = 6
MAX_REWRITE = 3

THINKING_INTERVAL_SECONDS: float      = 10.0
THINKING_RESPONSE_DELAY_SECONDS: float = 0.5

# ---------------------------------------------------------------------------
# Shared output-format prompt snippet
# ---------------------------------------------------------------------------

OUTPUT_FORMAT = """\
QUY TẮC ĐỊNH DẠNG ĐẦU RA — BẮT BUỘC:
- Luôn trả lời bằng tiếng Việt.
- Chỉ trả lời đúng phạm vi câu hỏi. Không liệt kê thông tin thừa không được hỏi.
  Ví dụ: hỏi "điểm chuẩn ngành khoa học máy tính" -> chỉ nói về điểm chuẩn, không nói về thông tin khác
- Không thêm câu tổng kết, lời chúc, hay lời mời hỏi thêm sau khi đã trả lời xong.
- Không giải thích lại nội dung vừa nói bằng câu khác.
- Dừng lại ngay sau khi đã trả lời đầy đủ câu hỏi.
- Tuyệt đối không dùng markdown, bullet, số thứ tự, emoji, header, dấu gạch đầu dòng.
- Số thập phân phải giữ nguyên dấu phẩy như trong nguồn gốc: viết "27,58" không phải "27.58".
- Không tự tính toán số liệu phái sinh (delta, tổng, hiệu) rồi đưa vào answer.
  Nếu context không có sẵn kết quả tính toán → chỉ trích số gốc, không tự tính.
  Ngoại lệ: phép tính đơn giản đúng 100% (ví dụ: 420 + 420 + 420 = 1.260) thì được phép.
- Không giải thích cách suy nghĩ, không nói "Theo dữ liệu tôi tìm được...".\
"""

# ---------------------------------------------------------------------------
# Thinking sentences
# ---------------------------------------------------------------------------

_THINKING_SENTENCES_START: list[str] = [
    "Tôi đang thực hiện tìm kiếm thông tin, vui lòng chờ trong giây lát.",
    "Tôi sẽ tìm kiếm dữ liệu ngay bây giờ, vui lòng chờ đợi.",
    "Để trả lời chính xác, tôi cần tra cứu dữ liệu, xin vui lòng chờ.",
]

_THINKING_SENTENCES_ONGOING: list[str] = [
    "Quá trình tìm kiếm vẫn đang tiếp tục, vui lòng chờ thêm.",
    "Hệ thống đang truy xuất dữ liệu liên quan, xin vui lòng đợi.",
    "Đang phân tích các nguồn tài liệu, vui lòng kiên nhẫn chờ đợi.",
    "Tìm kiếm vẫn đang được thực hiện, kết quả sẽ có trong chốc lát.",
    "Hệ thống vẫn đang xử lý yêu cầu, vui lòng chờ thêm một chút.",
]


def pick_thinking_start_sentence() -> str:
    return random.choice(_THINKING_SENTENCES_START)


def pick_thinking_ongoing_sentence() -> str:
    return random.choice(_THINKING_SENTENCES_ONGOING)


# ---------------------------------------------------------------------------
# build_sub_agent — generic agent factory (dùng bởi tất cả domain agents)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages:          Annotated[list[BaseMessage], add_messages]
    hop_count:         int
    thinking_streamed: bool

def build_sub_agent(
    llm_with_tools,
    tools: list[BaseTool],
    system_prompt: str,
):
    """
    Factory tạo một LangGraph sub-agent từ LLM đã bind tools + danh sách tools + system prompt.

    Args:
        llm_with_tools : LLM đã gọi .bind_tools(tools).
        tools          : Danh sách BaseTool tương ứng.
        system_prompt  : System prompt cho agent này.

    Returns:
        Compiled LangGraph (AgentState).
    """
    tool_node = ToolNode(tools)

    async def call_model(state: AgentState) -> dict:
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        try:
            response = await llm_with_tools.ainvoke(messages)
        except Exception as exc:
            import traceback
            logger.error(
                "LLM call failed [%s]: %s\n%s",
                type(exc).__name__, exc, traceback.format_exc(),
            )
            raise
        return {"messages": [response], "hop_count": state["hop_count"]}

    async def run_tools(state: AgentState) -> dict:
        result = await tool_node.ainvoke(state)
        return {**result, "hop_count": state["hop_count"] + 1, "thinking_streamed": False}

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and state["hop_count"] < MAX_HOPS:
            return "tools"
        return "__end__"

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", run_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "__end__": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


# ---------------------------------------------------------------------------
# BGE-M3 singleton
# ---------------------------------------------------------------------------

_bge_model: Optional[BGEM3FlagModel] = None
_bge_init_lock = asyncio.Lock()


def _load_bge_sync() -> BGEM3FlagModel:
    logger.info("Loading BGE-M3 model: %s on %s", EMBEDDING_MODEL, DEVICE)
    m = BGEM3FlagModel(
        EMBEDDING_MODEL,
        use_fp16=(DEVICE == "cuda"),
        device=DEVICE,
    )
    m.model.eval()
    m.encode(["warmup"], batch_size=1, max_length=32)
    logger.info("BGE-M3 ready.")
    return m


async def _ensure_bge() -> BGEM3FlagModel:
    global _bge_model
    if _bge_model is not None:
        return _bge_model
    async with _bge_init_lock:
        if _bge_model is not None:
            return _bge_model
        loop = asyncio.get_running_loop()
        _bge_model = await loop.run_in_executor(None, _load_bge_sync)
    return _bge_model


# ---------------------------------------------------------------------------
# AsyncQdrantClient singleton
# ---------------------------------------------------------------------------

_async_qdrant: Optional[AsyncQdrantClient] = None


def get_async_qdrant() -> AsyncQdrantClient:
    global _async_qdrant
    if _async_qdrant is None:
        _async_qdrant = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info("AsyncQdrantClient connected: %s:%s", QDRANT_HOST, QDRANT_PORT)
    return _async_qdrant


# ---------------------------------------------------------------------------
# Reranker singleton
# ---------------------------------------------------------------------------

_reranker: Optional[Reranker] = None


# ---------------------------------------------------------------------------
# Startup: load BGE-M3 + Reranker
# ---------------------------------------------------------------------------

async def preload_models() -> None:
    """Preload BGE-M3 embedding model và Reranker tại startup."""
    global _reranker
    await _ensure_bge()
    loop = asyncio.get_running_loop()
    r = Reranker()
    await loop.run_in_executor(None, r.startup)
    _reranker = r
    logger.info("All models loaded: BGE-M3 + Reranker.")


# ---------------------------------------------------------------------------
# Core encoding
# ---------------------------------------------------------------------------

def _encode_sync(texts: list[str]) -> tuple[np.ndarray, list[dict]]:
    m = _bge_model
    assert m is not None, "BGE model not loaded"

    tok      = m.tokenizer
    specials = {tok.cls_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id}

    with torch.no_grad():
        res = m.encode(
            texts,
            batch_size=min(16, len(texts)),
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    dense = np.array(res["dense_vecs"], dtype=np.float32)

    sparse_list = []
    for lw in res["lexical_weights"]:
        d: dict[int, float] = {}
        for k, w in lw.items():
            tid = int(k)
            fw  = float(w)
            if tid not in specials and fw > 0:
                d[tid] = max(d.get(tid, 0.0), fw)
        sparse_list.append(d)

    return dense, sparse_list


async def encode_query(query: str) -> tuple[list[float], dict[int, float]]:
    """Encode một query thành dense vector và sparse dict."""
    await _ensure_bge()
    loop = asyncio.get_running_loop()
    dense_mat, sparse_list = await loop.run_in_executor(None, _encode_sync, [query])
    return dense_mat[0].tolist(), sparse_list[0]


# ---------------------------------------------------------------------------
# Core hybrid retrieval (generic — dùng cho law và admission)
# ---------------------------------------------------------------------------

async def hybrid_retrieve(collection: str, query: str, k: int = RETRIEVAL_K) -> list[dict]:
    """
    Hybrid (dense + sparse RRF) retrieval từ một Qdrant collection.
    """
    client = get_async_qdrant()
    dense_vec, sparse_dict = await encode_query(query)

    sparse_indices = sorted(sparse_dict.keys())
    sparse_values  = [sparse_dict[i] for i in sparse_indices]

    results = await client.query_points(
        collection_name=collection,
        prefetch=[
            qmodels.Prefetch(query=dense_vec, using="dense", limit=k),
            qmodels.Prefetch(
                query=qmodels.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=k,
            ),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=k,
        with_payload=True,
    )

    docs = []
    for pt in results.points:
        payload = pt.payload or {}
        text    = payload.get("page_content", "")
        meta    = payload.get("metadata", {})
        if isinstance(meta, dict):
            docs.append({"text": text, **meta})
        else:
            docs.append({"text": text})
    return docs


# ---------------------------------------------------------------------------
# Core rerank (generic)
# ---------------------------------------------------------------------------

async def rerank(query: str, docs: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Rerank docs bằng cross-encoder. Fallback về top-k nếu reranker lỗi."""
    if not docs:
        return []
    if _reranker is None:
        logger.warning("Reranker not initialized, returning top-%d without reranking.", top_k)
        return docs[:top_k]
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _reranker.rerank, query, docs, top_k)
    except Exception as exc:
        logger.warning("Reranker error (%s), falling back to top-%d", exc, top_k)
        return docs[:top_k]