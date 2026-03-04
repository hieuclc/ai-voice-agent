"""
agent.py — LangGraph RAG Agent với Qdrant hybrid retrieval, cross-encoder reranker,
           và agentic workflow (route → retrieve → grade → rewrite → generate).

RAG tools:
  - search_law       : Qdrant hybrid search (dense + sparse) + reranker
  - search_admission : Qdrant hybrid search (dense + sparse) + reranker
  - search_tours     : Qdrant hybrid search (summary only)
  - get_tour_detail  : Qdrant exact-match tour detail

Agentic workflow cho search_law:
  retrieve → grade_documents → [done | rewrite → retrieve (loop ≤ MAX_REWRITE)]

Kiến trúc async:
  - BGE-M3 model được load SYNCHRONOUSLY tại startup (preload_bge_model)
  - Encode query chạy trong executor (CPU-bound) → không block event loop
  - Qdrant query dùng AsyncQdrantClient → hoàn toàn async, không dùng run_in_executor
  - Không dùng QdrantVectorStore (vì nó gọi embed bên trong thread không có event loop)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from typing import Annotated, List, Literal, Optional

import numpy as np
import torch
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from typing_extensions import TypedDict

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    NamedSparseVector,
    NamedVector,
    SparseVector,
    SearchRequest,
    FusionQuery,
    Prefetch,
    Query,
)
from qdrant_client import models as qmodels

from reranker import LawReranker

logger = logging.getLogger(__name__)
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_HOST          = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT          = int(os.environ.get("QDRANT_PORT", "6333"))
EMBEDDING_MODEL      = os.environ.get("EMBEDDING_MODEL_NAME", "AITeamVN/Vietnamese_Embedding_v2")
DEVICE               = os.environ.get("DEVICE", "cpu")

LAW_COLLECTION       = "law"
ADMISSION_COLLECTION = "admission"
TOUR_COLLECTION      = "tours"

RETRIEVAL_K = 10   # candidates trước rerank
TOP_K       = 5    # kết quả cuối sau rerank
MAX_HOPS    = 6    # max tool calls cho agent chính
MAX_REWRITE = 3    # max query rewrite trong agentic loop

# ---------------------------------------------------------------------------
# Thinking sentences
# ---------------------------------------------------------------------------

THINKING_INTERVAL_SECONDS: float = 10.0
THINKING_RESPONSE_DELAY_SECONDS: float = 0.5

THINKING_SENTENCES_START: list[str] = [
    "Tôi đang thực hiện tìm kiếm thông tin, vui lòng chờ trong giây lát.",
    "Tôi sẽ tìm kiếm dữ liệu ngay bây giờ, vui lòng chờ đợi.",
    "Để trả lời chính xác, tôi cần tra cứu dữ liệu, xin vui lòng chờ.",
]

THINKING_SENTENCES_ONGOING: list[str] = [
    "Quá trình tìm kiếm vẫn đang tiếp tục, vui lòng chờ thêm.",
    "Hệ thống đang truy xuất dữ liệu liên quan, xin vui lòng đợi.",
    "Đang phân tích các nguồn tài liệu, vui lòng kiên nhẫn chờ đợi.",
    "Tìm kiếm vẫn đang được thực hiện, kết quả sẽ có trong chốc lát.",
    "Hệ thống vẫn đang xử lý yêu cầu, vui lòng chờ thêm một chút.",
]


def pick_thinking_start_sentence() -> str:
    return random.choice(THINKING_SENTENCES_START)

def pick_thinking_ongoing_sentence() -> str:
    return random.choice(THINKING_SENTENCES_ONGOING)


# ---------------------------------------------------------------------------
# BGE-M3 singleton — load SYNCHRONOUSLY, encode trong executor
# ---------------------------------------------------------------------------

_bge_model: Optional[BGEM3FlagModel] = None
_bge_init_lock = asyncio.Lock()


def _load_bge_sync() -> BGEM3FlagModel:
    """Load và warmup BGE model. Chạy synchronously trong thread executor."""
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
    """Đảm bảo model đã load. Gọi từ async context an toàn."""
    global _bge_model
    if _bge_model is not None:
        return _bge_model
    async with _bge_init_lock:
        if _bge_model is not None:
            return _bge_model
        loop = asyncio.get_running_loop()
        _bge_model = await loop.run_in_executor(None, _load_bge_sync)
    return _bge_model


async def preload_bge_model() -> None:
    """Preload tại startup để request đầu tiên không bị chậm."""
    await _ensure_bge()


def _encode_sync(texts: list[str]) -> tuple[np.ndarray, list[dict]]:
    """
    Encode texts → (dense_matrix, sparse_list).
    Chạy synchronously — dùng trong run_in_executor.
    """
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
            tid = int(k); fw = float(w)
            if tid not in specials and fw > 0:
                d[tid] = max(d.get(tid, 0.0), fw)
        sparse_list.append(d)

    return dense, sparse_list


async def _encode_query(query: str) -> tuple[list[float], dict[int, float]]:
    """
    Encode một query string → (dense_vec, sparse_dict).
    Chạy encode trong executor để không block event loop.
    """
    await _ensure_bge()
    loop = asyncio.get_running_loop()
    dense_mat, sparse_list = await loop.run_in_executor(None, _encode_sync, [query])
    return dense_mat[0].tolist(), sparse_list[0]


# ---------------------------------------------------------------------------
# AsyncQdrantClient singleton
# ---------------------------------------------------------------------------

_async_qdrant: Optional[AsyncQdrantClient] = None


def _get_async_qdrant() -> AsyncQdrantClient:
    global _async_qdrant
    if _async_qdrant is None:
        _async_qdrant = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info("AsyncQdrantClient connected: %s:%s", QDRANT_HOST, QDRANT_PORT)
    return _async_qdrant


# ---------------------------------------------------------------------------
# Reranker singleton
# ---------------------------------------------------------------------------

_reranker: Optional[LawReranker] = None
_reranker_lock = asyncio.Lock()


async def _get_reranker() -> LawReranker:
    global _reranker
    if _reranker is not None:
        return _reranker
    async with _reranker_lock:
        if _reranker is not None:
            return _reranker
        loop = asyncio.get_running_loop()
        r = LawReranker()
        await loop.run_in_executor(None, r.startup)
        _reranker = r
    return _reranker


# ---------------------------------------------------------------------------
# Core hybrid retrieval
# ---------------------------------------------------------------------------

async def _hybrid_retrieve(collection: str, query: str, k: int = RETRIEVAL_K) -> list[dict]:
    """
    Hybrid search (dense + sparse → Qdrant RRF fusion) trực tiếp qua AsyncQdrantClient.
    Trả về list[dict] với key "text" + toàn bộ metadata.
    """
    client = _get_async_qdrant()
    dense_vec, sparse_dict = await _encode_query(query)

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


async def _rerank(query: str, docs: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Apply cross-encoder reranker async. Fallback nếu lỗi."""
    if not docs:
        return []
    try:
        reranker = await _get_reranker()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, reranker.rerank, query, docs, top_k)
    except Exception as exc:
        logger.warning("Reranker error (%s), falling back to top-%d", exc, top_k)
        return docs[:top_k]


# ---------------------------------------------------------------------------
# Agentic law retrieval: retrieve → grade → rewrite (loop ≤ MAX_REWRITE)
# ---------------------------------------------------------------------------

async def _grade_documents(query: str, docs: list[dict], llm: ChatOpenAI) -> list[dict]:
    """Grade relevance của từng doc. Song song hoá để giảm latency."""
    if not docs:
        return []

    prompt_tpl = (
        "Bạn là bộ lọc tài liệu pháp lý.\n"
        "Query: {query}\n\n"
        "Tài liệu:\n{doc}\n\n"
        "Tài liệu này có liên quan đến query không? Trả lời DUY NHẤT 'yes' hoặc 'no':"
    )

    async def _grade_one(doc: dict) -> bool:
        try:
            resp = await llm.ainvoke([
                HumanMessage(content=prompt_tpl.format(query=query, doc=doc["text"][:1500]))
            ])
            return resp.content.strip().lower().startswith("yes")
        except Exception:
            return True  # fail-open

    flags = await asyncio.gather(*[_grade_one(d) for d in docs])
    return [d for d, keep in zip(docs, flags) if keep]


async def _rewrite_query(original: str, current: str, llm: ChatOpenAI) -> str:
    """Rewrite query để cải thiện retrieval."""
    prompt = (
        "Bạn là công cụ rewrite query cho hệ thống tìm kiếm pháp luật Việt Nam.\n"
        "Query hiện tại không tìm được đủ tài liệu liên quan.\n\n"
        f"Query gốc: {original}\n"
        f"Query hiện tại: {current}\n\n"
        "Viết lại query (5–12 từ, thuật ngữ pháp lý chính xác hơn).\n"
        "Chỉ trả về query mới, không giải thích:"
    )
    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        new_q = resp.content.strip()
        logger.info("[Rewrite] %r → %r", current, new_q)
        return new_q
    except Exception:
        return current


def _format_law_chunk(doc: dict) -> str:
    text        = doc.get("text", "")
    source      = doc.get("source", doc.get("source_file", ""))
    article     = doc.get("article", "")
    clause      = doc.get("clause", "")
    clause_full = doc.get("clause_full", "")

    header = f"Văn bản: {source}"
    if article: header += f" | Điều: {article}"
    if clause:  header += f" | Khoản: {clause}"

    parts = [header, "", text]
    if clause_full and clause_full != text:
        parts += ["", f"Toàn văn khoản:\n{clause_full}"]
    return "\n".join(parts)


async def _agentic_law_retrieve(
    query: str,
    grader_llm: ChatOpenAI,
) -> tuple[str, list[dict]]:
    """
    Retrieve → grade → rewrite loop cho pháp luật.
    Trả về (context_text, source_docs).
    """
    MIN_RELEVANT  = 2
    current_query = query
    rewrite_count = 0

    while True:
        raw    = await _hybrid_retrieve(LAW_COLLECTION, current_query, RETRIEVAL_K)
        ranked = await _rerank(current_query, raw, TOP_K)

        logger.info(
            "[AgenticLaw] rewrite=%d retrieved=%d ranked=%d query=%r",
            rewrite_count, len(raw), len(ranked), current_query,
        )

        if not ranked:
            if rewrite_count >= MAX_REWRITE:
                return "", []
            rewrite_count += 1
            current_query = await _rewrite_query(query, current_query, grader_llm)
            continue

        relevant = await _grade_documents(query, ranked, grader_llm)
        logger.info("[AgenticLaw] relevant=%d/%d", len(relevant), len(ranked))

        if len(relevant) >= MIN_RELEVANT or rewrite_count >= MAX_REWRITE:
            final   = relevant if relevant else ranked
            context = "\n\n---\n\n".join(_format_law_chunk(d) for d in final)
            return context, final

        rewrite_count += 1
        current_query = await _rewrite_query(query, current_query, grader_llm)


# ---------------------------------------------------------------------------
# Tool: search_law
# ---------------------------------------------------------------------------

class _SearchInput(BaseModel):
    query: str

_grader_llm: Optional[ChatOpenAI] = None


async def _run_search_law(query: str) -> str:
    if _grader_llm is None:
        raw    = await _hybrid_retrieve(LAW_COLLECTION, query, RETRIEVAL_K)
        ranked = await _rerank(query, raw, TOP_K)
        if not ranked:
            return "Không tìm thấy thông tin pháp luật phù hợp."
        return "\n\n-------------------\n\n".join(_format_law_chunk(d) for d in ranked)

    context, sources = await _agentic_law_retrieve(query, _grader_llm)

    if not context:
        return "Không tìm thấy thông tin pháp luật phù hợp."

    logger.info(
        "===== LAW RESULT =====\n%d sources\n%s\n======================",
        len(sources),
        "\n".join(f"  [{i+1}] {d.get('source','?')} | {d.get('article','')}"
                  for i, d in enumerate(sources)),
    )
    return context


search_law = StructuredTool.from_function(
    coroutine=_run_search_law,
    name="search_law",
    description=(
        "Tìm kiếm văn bản pháp luật, điều khoản, quy định, mức phạt, xử phạt hành chính. "
        "Dùng khi câu hỏi liên quan đến luật, nghị định, thông tư, xử phạt, v.v. "
        "Tham số: query (câu hỏi hoặc từ khóa pháp luật)."
    ),
    args_schema=_SearchInput,
)


# ---------------------------------------------------------------------------
# Tool: search_admission
# ---------------------------------------------------------------------------

_ADMISSION_QUERY_REWRITER_SYSTEM_PROMPT = """\
Bạn là công cụ chuyển câu hỏi tuyển sinh tự nhiên thành danh sách keyword query tối ưu \
cho hệ thống tìm kiếm vector (Qdrant hybrid search).

NHIỆM VỤ: Tách câu hỏi thành 1-3 keyword query ngắn gọn, tập trung vào thực thể chính.

QUY TẮC:
1. Mỗi query: 3–8 từ, chứa tên ngành/mã ngành/năm/loại thông tin cần tìm.
2. LUÔN có một query chứa TÊN NGÀNH CHÍNH XÁC như trong câu hỏi gốc.
3. Tách riêng: một query cho điểm chuẩn, một query cho học phí nếu cần cả hai.
4. Bỏ hết ngữ cảnh cá nhân ("tôi đạt", "con tôi", "năm nay mình").
5. Bỏ phần reasoning/điều kiện ("nếu đạt X thì", "có nên đăng ký không").
6. GIỮ nguyên năm (2024, 2025, 2026) nếu có trong câu hỏi.
7. KHÔNG thêm từ khóa không có trong câu hỏi gốc.

Trả về JSON: {"queries": ["query1", "query2", ...]}
Chỉ trả về JSON, không có gì khác.\
"""


async def _rewrite_query_for_admission(natural_query: str) -> list[str]:
    """Tách câu hỏi tuyển sinh phức tạp thành danh sách keyword query tối ưu."""
    if _rewriter_llm is None:
        return [natural_query]
    try:
        response = await _rewriter_llm.ainvoke([
            SystemMessage(content=_ADMISSION_QUERY_REWRITER_SYSTEM_PROMPT),
            HumanMessage(content=natural_query),
        ])
        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        queries = parsed.get("queries") or [natural_query]
        logger.info("ADMISSION QUERY REWRITE: %r → %s", natural_query, queries)
        return queries
    except Exception as exc:
        logger.warning("Admission query rewriter error (%s), fallback to original.", exc)
        return [natural_query]


async def _run_search_admission(query: str) -> str:
    sub_queries = await _rewrite_query_for_admission(query)

    seen_texts: set = set()
    merged: list[dict] = []

    async def _fetch_one(sq: str) -> list[dict]:
        try:
            return await _hybrid_retrieve(ADMISSION_COLLECTION, sq, RETRIEVAL_K)
        except Exception as exc:
            logger.error("admission retrieve error for sub-query %r: %s", sq, exc)
            return []

    all_batches = await asyncio.gather(*[_fetch_one(sq) for sq in sub_queries])
    for batch in all_batches:
        for doc in batch:
            key = doc.get("text", "")[:120]
            if key not in seen_texts:
                seen_texts.add(key)
                merged.append(doc)

    try:
        ranked = await _rerank(query, merged, TOP_K)
    except Exception as exc:
        logger.error("admission rerank error: %s", exc)
        return f"Retrieval error: {exc}"

    if not ranked:
        return "Không tìm thấy thông tin tuyển sinh phù hợp."

    logger.info(
        "===== ADMISSION: query=%r → %d/%d results =====\n%s",
        query, len(ranked), len(merged),
        "\n".join(
            f"  [{i+1}] score={d.get('_rerank_score', 0):.4f} | {d.get('text','')[:80].replace(chr(10),' ')}"
            for i, d in enumerate(ranked)
        ),
    )

    parts = []
    for i, item in enumerate(ranked, 1):
        text = item.get("text", "")
        meta = {k: v for k, v in item.items() if k not in ("text", "_rerank_score")}
        parts.append(
            f"[Kết quả {i}]\n"
            f"Metadata: {json.dumps(meta, ensure_ascii=False)}\n\n"
            f"Nội dung:\n{text}"
        )

    body = "\n\n-------------------\n\n".join(parts)

    return body


search_admission = StructuredTool.from_function(
    coroutine=_run_search_admission,
    name="search_admission",
    description=(
        "Tìm kiếm thông tin tuyển sinh, điểm chuẩn, chỉ tiêu, học phí, ngành học. "
        "Tham số: query (câu hỏi hoặc từ khóa tuyển sinh)."
    ),
    args_schema=_SearchInput,
)

_rewriter_llm: Optional[ChatOpenAI] = None


# ---------------------------------------------------------------------------
# Tour Query Rewriter
# ---------------------------------------------------------------------------

_TOUR_QUERY_REWRITER_SYSTEM_PROMPT = """\
Bạn là công cụ phân tích câu hỏi về tour du lịch để tạo ra các sub-query tìm kiếm hiệu quả.

NHIỆM VỤ:
Phân tích câu hỏi gốc và trả về JSON với cấu trúc sau:
{
  "sub_queries": ["query1", "query2", ...],
  "clause_types": ["policies", "departures", "services", "summary", "day_1", ...]
}

QUY TẮC QUAN TRỌNG NHẤT — TÊN TOUR:
Tên tour thường gồm nhiều tỉnh/thành phố nối nhau, ví dụ:
  "Đồng Tháp, Sa Đéc, Cần Thơ", "Hà Nội, Hạ Long, Ninh Bình", "Đà Nẵng, Hội An, Huế"
ĐÂY LÀ MỘT TOUR DUY NHẤT, KHÔNG PHẢI NHIỀU TOUR KHÁC NHAU.
TUYỆT ĐỐI không tách tên tour thành từng tỉnh riêng lẻ khi tạo sub_queries.
Luôn giữ nguyên tên tour đầy đủ trong mọi sub_query.
Ngoài ra, tên tour cũng có thể là mã số. Ví dụ "Tour số 1" hay "tour có mã số 2", thì viết lại query là "Tour số 1" hoặc "Tour số 2".

QUY TẮC PHÂN TÍCH:
1. Câu hỏi chỉ hỏi về thông tin tour (lịch trình, địa điểm):
   → sub_queries: [tên tour đầy đủ], clause_types: null

2. Câu hỏi về hủy tour, phí hủy, hoàn tiền:
   → clause_types: ["policies"]
   → sub_queries: [tên tour đầy đủ nếu có, "chính sách hủy tour hoàn tiền phí", "điều kiện hủy tour trước ngày khởi hành"]
   Lưu ý: câu hỏi "trước X ngày" cần tìm TOÀN BỘ bảng phí hủy để suy luận khoảng.

3. Câu hỏi về thanh toán, đặt cọc, deadline thanh toán:
   → clause_types: ["policies"]
   → sub_queries: [tên tour đầy đủ nếu có, "chính sách thanh toán đặt cọc", "thời hạn hoàn tất thanh toán"]

4. Câu hỏi về lịch khởi hành, ngày đi, còn chỗ:
   → clause_types: ["departures"]
   → sub_queries: [tên tour đầy đủ nếu có]

5. Câu hỏi về dịch vụ tổng quát (bao gồm gì, không bao gồm gì, phương tiện, hướng dẫn viên):
   → clause_types: ["services"]

6. Câu hỏi về lịch trình ngày cụ thể:
   → clause_types: ["day_1", "day_2", ...]

7. Câu hỏi về một địa điểm/hoạt động cụ thể có bao gồm trong giá tour không,
   hoặc chi phí của một dịch vụ/vé tham quan cụ thể:
   → clause_types: ["services", "day_1", "day_2", "day_3", "day_4", "day_5"]
   → sub_queries: [tên tour đầy đủ, tên địa điểm/hoạt động đó, "dịch vụ bao gồm không bao gồm"]

CHỈ trả về JSON hợp lệ, không có preamble hay giải thích.\
"""


async def _rewrite_query_for_tour(natural_query: str) -> dict:
    """
    Phân tích câu hỏi tour → {sub_queries: [...], clause_types: [...]}.
    Fallback về query gốc nếu LLM chưa init hoặc lỗi.
    """
    if _rewriter_llm is None:
        return {"sub_queries": [natural_query], "clause_types": None}
    try:
        response = await _rewriter_llm.ainvoke([
            SystemMessage(content=_TOUR_QUERY_REWRITER_SYSTEM_PROMPT),
            HumanMessage(content=natural_query),
        ])
        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        sub_queries  = parsed.get("sub_queries") or [natural_query]
        clause_types = parsed.get("clause_types") or None
        logger.info(
            "TOUR QUERY REWRITE: %r → sub_queries=%s clause_types=%s",
            natural_query, sub_queries, clause_types,
        )
        return {"sub_queries": sub_queries, "clause_types": clause_types}
    except Exception as exc:
        logger.warning("Tour query rewriter error (%s), fallback to original.", exc)
        return {"sub_queries": [natural_query], "clause_types": None}


# ---------------------------------------------------------------------------
# Tour tools
# ---------------------------------------------------------------------------

class _TourSearchInput(BaseModel):
    query: str
    tour_type: Optional[str] = None

class _TourInfoInput(BaseModel):
    query: str
    tour_type: Optional[str] = None

class _TourDetailInput(BaseModel):
    tour_id: str


async def _hybrid_search_tours(
    query: str,
    clause_filter: Optional[list[str]],
    tour_type: Optional[str],
    prefetch_limit: int,
    final_limit: int,
    pinned_tour_id: Optional[str] = None,
) -> list[dict]:
    """Hàm hybrid search dùng chung cho cả 2 tour tools."""
    dense_vec, sparse_dict = await _encode_query(query)
    client = _get_async_qdrant()

    sparse_indices = sorted(sparse_dict.keys())
    sparse_values  = [sparse_dict[i] for i in sparse_indices]

    must_conditions = []
    if clause_filter:
        must_conditions.append(
            qmodels.FieldCondition(
                key="metadata.clause",
                match=qmodels.MatchAny(any=clause_filter),
            )
        )
    if tour_type:
        must_conditions.append(
            qmodels.FieldCondition(
                key="metadata.chapter",
                match=qmodels.MatchValue(value=tour_type),
            )
        )
    if pinned_tour_id:
        must_conditions.append(
            qmodels.FieldCondition(
                key="metadata.point",
                match=qmodels.MatchValue(value=pinned_tour_id),
            )
        )

    q_filter = qmodels.Filter(must=must_conditions) if must_conditions else None

    results = await client.query_points(
        collection_name=TOUR_COLLECTION,
        prefetch=[
            qmodels.Prefetch(query=dense_vec, using="dense", limit=prefetch_limit, filter=q_filter),
            qmodels.Prefetch(
                query=qmodels.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=prefetch_limit,
                filter=q_filter,
            ),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=final_limit,
        with_payload=True,
    )

    docs = []
    for pt in results.points:
        payload = pt.payload or {}
        meta    = payload.get("metadata", {})
        text    = payload.get("page_content", "")
        docs.append({"text": text, **meta})
    return docs


async def _run_search_tours(query: str, tour_type: Optional[str] = None) -> str:
    """Tìm kiếm và liệt kê các tour phù hợp. Chỉ search summary chunks."""
    # Tăng prefetch và final_limit để không bỏ sót tour khi query reasoning cần tổng hợp
    try:
        candidates = await _hybrid_search_tours(
            query=query,
            clause_filter=["summary"],
            tour_type=tour_type,
            prefetch_limit=20,
            final_limit=12,
        )
    except Exception as exc:
        logger.error("search_tours error: %s", exc)
        return f"Lỗi tìm kiếm tour: {exc}"

    if not candidates:
        return "Không tìm thấy tour phù hợp."

    ranked = await _rerank(query, candidates, top_k=8)

    lines = []
    for item in ranked:
        name    = item.get("section", "")
        tour_id = item.get("point", item.get("article", ""))
        try:
            cf    = json.loads(item.get("clause_full", "{}"))
            dur   = f"{cf.get('d','?')} ngày {cf.get('n','?')} đêm"
            p0, p1 = cf.get("p0", 0), cf.get("p1", 0)
            price = f"{p0:,}–{p1:,} VND".replace(",", ".")
            dep   = cf.get("dc", "")
        except Exception:
            dur = price = dep = ""
        lines.append(f"{name} — mã số {tour_id} — {dur} — {price} — khởi hành từ {dep}")

    logger.info("===== SEARCH_TOURS: query=%r → %d results =====", query, len(lines))
    return "\n".join(lines)


async def _run_search_tour_info(query: str, tour_type: Optional[str] = None) -> str:
    """Tìm kiếm thông tin cụ thể bên trong nội dung tour."""
    rewrite_result = await _rewrite_query_for_tour(query)
    sub_queries    = rewrite_result["sub_queries"]
    clause_filter  = rewrite_result["clause_types"]

    # Extract tour_id nếu query đề cập "Tour số X", "tour mã X" rõ ràng
    import re as _re
    _tour_id_match = _re.search(r"(?:tour\s*(?:số|mã|số mã)?\s*)(\d+)", query, _re.IGNORECASE)
    pinned_tour_id: Optional[str] = _tour_id_match.group(1) if _tour_id_match else None

    async def _fetch_one(sq: str) -> list[dict]:
        try:
            return await _hybrid_search_tours(
                query=sq,
                clause_filter=clause_filter,
                tour_type=tour_type,
                prefetch_limit=20,
                final_limit=10,
                pinned_tour_id=pinned_tour_id,
            )
        except Exception as exc:
            logger.error("search_tour_info sub-query error (%r): %s", sq, exc)
            return []

    all_results_nested = await asyncio.gather(*[_fetch_one(sq) for sq in sub_queries])

    seen: set = set()
    merged: list[dict] = []
    for batch in all_results_nested:
        for doc in batch:
            key = (doc.get("point", doc.get("article", "")), doc.get("clause", ""), doc.get("text", "")[:80])
            if key not in seen:
                seen.add(key)
                merged.append(doc)

    if not merged:
        return "Không tìm thấy thông tin tour phù hợp."

    ranked = await _rerank(query, merged, top_k=8)

    # Ưu tiên kết quả đúng tour nếu đã pin
    if pinned_tour_id:
        pinned = [r for r in ranked if str(r.get("point", r.get("article", ""))) == pinned_tour_id]
        others = [r for r in ranked if str(r.get("point", r.get("article", ""))) != pinned_tour_id]
        ranked = (pinned + others)[:8]

    parts = []
    for item in ranked:
        tour_name = item.get("section", "")
        tour_id   = item.get("point", item.get("article", ""))
        clause    = item.get("clause", "")
        text      = item.get("text", "")
        parts.append(f"[Tour: {tour_name} | mã {tour_id} | {clause}]\n{text}")

    logger.info(
        "===== SEARCH_TOUR_INFO: query=%r sub_queries=%s clause_filter=%s pinned=%s → %d merged → %d ranked =====",
        query, sub_queries, clause_filter, pinned_tour_id, len(merged), len(ranked),
    )
    return "\n\n---\n\n".join(parts)
async def _run_get_tour_detail(tour_id: str) -> str:
    """Lấy toàn bộ thông tin của một tour theo mã tour."""
    client = _get_async_qdrant()
    try:
        results, _ = await client.scroll(
            collection_name=TOUR_COLLECTION,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.point",
                        match=qmodels.MatchValue(value=tour_id),
                    )
                ]
            ),
            limit=50,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        logger.error("get_tour_detail error: %s", exc)
        return f"Lỗi truy vấn chi tiết tour: {exc}"

    if not results:
        return f"Không tìm thấy tour với mã số {tour_id}."

    summary_meta: Optional[dict] = None
    departures_text: str         = ""
    services_text: str           = ""
    policies_text: str           = ""
    day_chunks: list             = []

    for pt in results:
        payload = pt.payload or {}
        meta    = payload.get("metadata", {})
        text    = payload.get("page_content", "")
        clause  = meta.get("clause", "")

        if clause == "summary":
            summary_meta = meta
        elif clause == "departures":
            departures_text = text
        elif clause == "services":
            services_text = text
        elif clause == "policies":
            policies_text = text
        elif clause.startswith("day_"):
            day_num = int(clause.replace("day_", "") or 0)
            day_chunks.append((day_num, text))

    if not summary_meta:
        return f"Không tìm thấy thông tin tổng quan của tour {tour_id}."

    day_chunks.sort(key=lambda x: x[0])

    lines = []
    name  = summary_meta.get("section", "")
    try:
        cf        = json.loads(summary_meta.get("clause_full", "{}"))
        dur       = f"{cf.get('d','?')} ngày {cf.get('n','?')} đêm"
        p0, p1    = cf.get("p0", 0), cf.get("p1", 0)
        price     = f"{p0:,}–{p1:,} VND".replace(",", ".")
        dep       = cf.get("dc", "")
        transport = cf.get("tr", "")
        dests     = ", ".join(cf.get("ds", []))
    except Exception:
        dur = price = dep = transport = dests = ""

    lines.append(f"Tên tour: {name} (mã số {tour_id})")
    if dur:       lines.append(f"Thời gian: {dur}")
    if price:     lines.append(f"Giá từ: {price}")
    if dep:       lines.append(f"Khởi hành từ: {dep}")
    if transport: lines.append(f"Phương tiện: {transport}")
    if dests:     lines.append(f"Điểm đến: {dests}")
    if departures_text: lines.append(f"\n{departures_text}")
    if services_text:   lines.append(f"\n{services_text}")
    if policies_text:   lines.append(f"\n{policies_text}")
    if day_chunks:
        lines.append("\nLịch trình chi tiết:")
        for _, text in day_chunks:
            lines.append(f"\n{text}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: count_tour_meals
# ---------------------------------------------------------------------------

class _CountMealsInput(BaseModel):
    tour_ids: list[str]


async def _run_count_tour_meals(tour_ids: list[str]) -> str:
    """Đếm bữa ăn từ metadata.clause_full — exact-match, không dùng embedding."""
    client = _get_async_qdrant()
    try:
        results, _ = await client.scroll(
            collection_name=TOUR_COLLECTION,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.point",
                        match=qmodels.MatchAny(any=tour_ids),
                    ),
                ]
            ),
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        logger.error("count_tour_meals error: %s", exc)
        return f"Lỗi truy vấn: {exc}"

    from collections import defaultdict as _dd
    tour_days: dict = _dd(list)
    tour_names: dict = {}

    for pt in results:
        payload = pt.payload or {}
        meta    = payload.get("metadata", {})
        clause  = meta.get("clause", "")
        tid     = meta.get("point", meta.get("article", ""))
        section = meta.get("section", "")

        if not clause.startswith("day_"):
            continue
        if tid not in tour_names and section:
            tour_names[tid] = section

        try:
            day_num = int(clause.replace("day_", ""))
        except ValueError:
            continue

        try:
            cf = json.loads(meta.get("clause_full", "{}"))
            meals     = cf.get("meals", [])
            day_label = cf.get("day_label", clause)
        except Exception:
            meals     = []
            day_label = clause

        day_text = payload.get("page_content", "")
        tour_days[tid].append((day_num, day_label, meals, day_text))

    if not tour_days:
        return "Không tìm thấy dữ liệu bữa ăn cho các tour yêu cầu."

    lines = []
    for tid in sorted(tour_days.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        name = tour_names.get(tid, f"Tour {tid}")
        days = sorted(tour_days[tid], key=lambda x: x[0])

        total_sang  = sum(m.count("sáng")  for _, _, m, _ in days)
        total_trua  = sum(m.count("trưa")  for _, _, m, _ in days)
        total_toi   = sum(m.count("tối")   for _, _, m, _ in days)
        total_chinh = total_sang + total_trua + total_toi

        lines.append(f"[Tour {tid}: {name}]")
        for _, day_label, meals, day_text in days:
            meal_str = ", ".join(meals) if meals else "không có bữa ăn"
            lines.append(f"  {day_label}: {meal_str}")
            if day_text:
                lines.append(f"    {day_text}")
        lines.append(f"  TỔNG: sáng={total_sang} | trưa={total_trua} | tối={total_toi}")
        lines.append(f"  TỔNG BỮA CHÍNH (sáng + trưa + tối): {total_chinh}")
        lines.append("")

    logger.info("===== COUNT_TOUR_MEALS: tour_ids=%s =====", tour_ids)
    return "\n".join(lines)


count_tour_meals = StructuredTool.from_function(
    coroutine=_run_count_tour_meals,
    name="count_tour_meals",
    description=(
        "Đếm số lượng bữa ăn (sáng, trưa, chiều, tối, bữa chính) của một hoặc nhiều tour. "
        "Đọc trực tiếp từ metadata — kết quả chính xác 100%, không qua reranker. "
        "Dùng cho mọi câu hỏi: 'tour X có bao nhiêu bữa', 'so sánh số bữa ăn tour 1 và tour 4', "
        "'tour nào có nhiều bữa chính hơn'. "
        "Tham số: tour_ids — danh sách mã tour, ví dụ ['1'] hoặc ['1','4']."
    ),
    args_schema=_CountMealsInput,
)

search_tours = StructuredTool.from_function(
    coroutine=_run_search_tours,
    name="search_tours",
    description=(
        "Tìm kiếm và liệt kê danh sách tour du lịch phù hợp với yêu cầu của khách. "
        "Trả về tên tour, mã số, thời gian, giá và điểm khởi hành. "
        "Dùng khi khách hỏi 'có tour nào đi X không', 'cho tôi xem danh sách tour'. "
        "KHÔNG dùng khi đã biết mã số tour — dùng scan_all_tours hoặc get_tour_detail thay thế. "
        "KHÔNG dùng với query rỗng. "
        "Tham số: query (từ khóa địa điểm/loại tour, KHÔNG được rỗng), "
        "tour_type (tuỳ chọn: 'tour miền bắc'/'tour miền nam'/'tour miền trung')."
    ),
    args_schema=_TourSearchInput,
)

search_tour_info = StructuredTool.from_function(
    coroutine=_run_search_tour_info,
    name="search_tour_info",
    description=(
        "Tìm kiếm thông tin chi tiết bên trong nội dung tour: "
        "địa điểm tham quan, hoạt động theo ngày, dịch vụ bao gồm/không bao gồm, "
        "chính sách hủy tour, chính sách trẻ em, lịch khởi hành và giá theo ngày. "
        "Dùng khi khách hỏi thông tin cụ thể như 'Chùa Tam Chúc có diện tích bao nhiêu', "
        "'tour có bao gồm vé máy bay không', 'chính sách hủy tour như thế nào', "
        "'ngày nào còn chỗ'. "
        "Khi so sánh 2 tour: gọi 2 lần với cùng query nhưng tour_type KHÁC NHAU. "
        "Tham số: query (nội dung cần tìm), "
        "tour_type (BẮT BUỘC khi đã biết loại tour: "
        "'tour miền bắc' / 'tour miền nam' / 'tour miền trung' / 'tour miền tây')."
    ),
    args_schema=_TourInfoInput,
)

get_tour_detail = StructuredTool.from_function(
    coroutine=_run_get_tour_detail,
    name="get_tour_detail",
    description=(
        "Lấy toàn bộ thông tin của một tour cụ thể theo mã tour: "
        "lịch trình từng ngày, lịch khởi hành và giá theo hạng khách sạn, "
        "dịch vụ bao gồm/không bao gồm, chính sách hủy tour, trẻ em, thanh toán. "
        "Dùng sau khi khách đã chọn được mã tour từ search_tours."
    ),
    args_schema=_TourDetailInput,
)


class _TourScanInput(BaseModel):
    clause_types: list[str]
    tour_ids: Optional[list[str]] = None


async def _run_scan_all_tours(
    clause_types: list[str],
    tour_ids: Optional[list[str]] = None,
) -> str:
    """Scroll toàn bộ collection để lấy chunks theo clause_types."""
    client = _get_async_qdrant()
    try:
        must: list = [
            qmodels.FieldCondition(
                key="metadata.clause",
                match=qmodels.MatchAny(any=clause_types),
            )
        ]
        if tour_ids:
            must.append(
                qmodels.FieldCondition(
                    key="metadata.point",
                    match=qmodels.MatchAny(any=tour_ids),
                )
            )
        results, _ = await client.scroll(
            collection_name=TOUR_COLLECTION,
            scroll_filter=qmodels.Filter(must=must),
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        import traceback
        logger.error("scan_all_tours error: %s\n%s", exc, traceback.format_exc())
        return f"Lỗi scan tour: {exc}"

    if not results:
        return "Không tìm thấy dữ liệu tour."

    from collections import defaultdict as _dd
    grouped: dict = _dd(list)
    for pt in results:
        payload = pt.payload or {}
        meta    = payload.get("metadata", {})
        text    = payload.get("page_content", "")
        tid     = meta.get("point", meta.get("article", "?"))
        clause  = meta.get("clause", "")
        clause_full = meta.get("clause_full", "")
        grouped[tid].append((clause, text, clause_full))

    clause_order = {"summary": 0, "departures": 1, "services": 2, "policies": 3}
    def _clause_key(x):
        c = x[0]  # x is (clause, text, clause_full)
        if c in clause_order:
            return (clause_order[c], c)
        if c.startswith("day_"):
            n = int(c.replace("day_", "") or 0)
            return (10 + n, c)
        return (99, c)

    parts = []
    for tid, items in sorted(grouped.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
        items.sort(key=_clause_key)
        tour_texts = []
        for clause, text, clause_full in items:
            if clause == "summary" and clause_full:
                # Bổ sung transport, departure city rõ ràng vào summary để LLM dễ trích xuất
                try:
                    import json as _j
                    cf = _j.loads(clause_full)
                    tr = cf.get("tr", "")
                    dc = cf.get("dc", "")
                    extra_lines = []
                    if tr:
                        extra_lines.append(f"Phương tiện di chuyển: {tr}")
                    if dc:
                        extra_lines.append(f"Điểm khởi hành: {dc}")
                    if extra_lines:
                        text = text + "\n" + "\n".join(extra_lines)
                except Exception:
                    pass
            tour_texts.append(text)
        parts.append(f"[Tour mã {tid}]\n" + "\n\n".join(tour_texts))

    logger.info(
        "===== SCAN_ALL_TOURS: clause_types=%s tour_ids=%s → %d tours =====",
        clause_types, tour_ids, len(grouped),
    )
    return "\n\n" + ("\n\n" + "="*60 + "\n\n").join(parts)


scan_all_tours = StructuredTool.from_function(
    coroutine=_run_scan_all_tours,
    name="scan_all_tours",
    description=(
        "Lấy dữ liệu từ NHIỀU hoặc TẤT CẢ tour cùng lúc để tổng hợp, liệt kê, so sánh. "
        "Dùng khi câu hỏi liên quan đến nhiều tour: "
        "'tour nào dùng máy bay', 'liệt kê tour có Tràng An', "
        "'điểm chung giữa các tour', 'tổng hợp giấy tờ cần mang', "
        "'so sánh chính sách trẻ em giữa tour miền Bắc và miền Tây'. "
        "Tham số: "
        "clause_types (BẮT BUỘC, danh sách loại chunk: "
        "'summary' cho tổng quan/phương tiện/điểm đến, "
        "'policies' cho chính sách trẻ em/hủy tour/ghi chú, "
        "'services' cho dịch vụ bao gồm/quà tặng, "
        "'departures' cho lịch khởi hành/tiêu chuẩn khách sạn, "
        "'day_1'..'day_5' cho lịch trình ngày cụ thể); "
        "tour_ids (tuỳ chọn, ví dụ ['1','3'] để chỉ lấy tour cụ thể, "
        "bỏ trống để lấy tất cả)."
    ),
    args_schema=_TourScanInput,
)

# ---------------------------------------------------------------------------
# Master tool list
# ---------------------------------------------------------------------------

RAG_TOOLS: list[BaseTool] = [
    search_law,
    search_admission,
    search_tours,
    search_tour_info,
    scan_all_tours,
    get_tour_detail,
    count_tour_meals,
]


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages:          Annotated[list[BaseMessage], add_messages]
    hop_count:         int
    thinking_streamed: bool


# ---------------------------------------------------------------------------
# System prompt — định hướng chung, không chứa domain rules
# (domain rules nằm trong agent_routing.py)
# ---------------------------------------------------------------------------

def _build_base_system_prompt() -> str:
    current_date = datetime.now().strftime("%d/%m/%Y")
    return f"""Bạn là trợ lý giọng nói tiếng Việt. Ngày hiện tại: {current_date}.

QUY TẮC ĐỊNH DẠNG ĐẦU RA — BẮT BUỘC:
- Luôn trả lời bằng tiếng Việt.
- Văn bản thuần, tự nhiên như đang nói chuyện trực tiếp.
- Tuyệt đối không dùng markdown, bullet, số thứ tự, emoji, header, dấu gạch đầu dòng.
- Không dùng chữ số — mọi con số phải viết bằng chữ (ví dụ: "hai mươi bảy phẩy năm").
- Không giải thích cách suy nghĩ hay quá trình xử lý.
- Mỗi câu phải có ít nhất bốn từ.
- Viết tắt (THPT, UET, VND...) phải viết IN HOA.
- Chỉ trả lời dựa trên dữ liệu truy xuất được; không ước tính hay suy diễn số liệu cụ thể."""


# ---------------------------------------------------------------------------
# Shared sub-agent builder (dùng bởi agent_routing.py)
# ---------------------------------------------------------------------------

def build_sub_agent(llm_with_tools, tools: list[BaseTool], system_prompt: str):
    """
    Xây dựng một compiled LangGraph sub-agent.
    Dùng chung cho agent đơn lẻ và từng sub-agent trong router.
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
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def create_agent(
    extra_tools: Optional[list[BaseTool]] = None,
    model: str = "gpt-4o-mini",
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    temperature: float = 0.0,
):
    global _grader_llm, _rewriter_llm

    tools = RAG_TOOLS + list(extra_tools or [])

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=openai_base_url,
        streaming=True,
    )

    _non_stream_llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=openai_base_url,
        streaming=False,
    )
    _grader_llm   = _non_stream_llm
    _rewriter_llm = _non_stream_llm

    llm_with_tools = llm.bind_tools(tools)
    system_prompt  = _build_system_prompt(tools)

    return _build_graph(llm_with_tools, tools, system_prompt)

# ---------------------------------------------------------------------------
# Thinking stream helper (dùng bởi server.py)
# ---------------------------------------------------------------------------

async def stream_thinking_while_running(agent_task):
    yield pick_thinking_start_sentence()
    task_done = False
    while not task_done:
        try:
            await asyncio.wait_for(asyncio.shield(agent_task), timeout=THINKING_INTERVAL_SECONDS)
            task_done = True
        except asyncio.TimeoutError:
            if not agent_task.done():
                yield pick_thinking_ongoing_sentence()
            else:
                task_done = True
    await asyncio.sleep(THINKING_RESPONSE_DELAY_SECONDS)