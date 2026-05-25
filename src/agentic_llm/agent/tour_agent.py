"""
tour_agent.py — Domain agent cho tư vấn du lịch.

Exports:
  build_tour_agent(streaming_llm, rewriter_llm, date) → compiled LangGraph
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from qdrant_client import models as qmodels

from utils import (
    OUTPUT_FORMAT,
    RETRIEVAL_K,
    TOP_K,
    encode_query,
    get_async_qdrant,
    rerank,
)

logger = logging.getLogger(__name__)

TOUR_COLLECTION = "tours"

# ---------------------------------------------------------------------------
# Tour-specific hybrid search (cần filter đặc thù: clause, chapter, point)
# ---------------------------------------------------------------------------

async def _hybrid_search_tours(
    query: str,
    clause_filter: Optional[list[str]],
    tour_type: Optional[str],
    prefetch_limit: int,
    final_limit: int,
    pinned_tour_id: Optional[str] = None,
) -> list[dict]:
    dense_vec, sparse_dict = await encode_query(query)
    client = get_async_qdrant()

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
            qmodels.Prefetch(
                query=dense_vec,
                using="dense",
                limit=prefetch_limit,
                filter=q_filter,
            ),
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


# ---------------------------------------------------------------------------
# Tour query rewriter
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

3. Câu hỏi về thanh toán, đặt cọc, deadline thanh toán:
   → clause_types: ["policies"]
   → sub_queries: [tên tour đầy đủ nếu có, "chính sách thanh toán đặt cọc", "thời hạn hoàn tất thanh toán"]

4. Câu hỏi về lịch khởi hành, ngày đi, còn chỗ:
   → clause_types: ["departures"]
   → sub_queries: [tên tour đầy đủ nếu có]

4b. Câu hỏi về giá cụ thể theo hạng khách sạn, giá tối đa, giá cao nhất, giá thấp nhất của một tour:
   → clause_types: ["departures"]
   → sub_queries: [tên tour đầy đủ, "giá khách sạn", "lịch khởi hành giá"]

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

_rewriter_llm_ref: Optional[ChatOpenAI] = None


async def _rewrite_query_for_tour(natural_query: str) -> dict:
    if _rewriter_llm_ref is None:
        return {"sub_queries": [natural_query], "clause_types": None}
    try:
        response = await _rewriter_llm_ref.ainvoke([
            SystemMessage(content=_TOUR_QUERY_REWRITER_SYSTEM_PROMPT),
            HumanMessage(content=natural_query),
        ])
        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        parsed       = json.loads(raw)
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
# Tool input schemas
# ---------------------------------------------------------------------------

class _TourSearchInput(BaseModel):
    query: str
    tour_type: Optional[str] = None

class _TourInfoInput(BaseModel):
    query: str
    tour_type: Optional[str] = None

class _TourDetailInput(BaseModel):
    tour_id: str

class _TourScanInput(BaseModel):
    clause_types: list[str]
    tour_ids: Optional[list[str]] = None

class _CountMealsInput(BaseModel):
    tour_ids: list[str]


# ---------------------------------------------------------------------------
# Tool: search_tours
# ---------------------------------------------------------------------------

async def _run_search_tours(query: str, tour_type: Optional[str] = None) -> str:
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

    ranked = await rerank(query, candidates, top_k=8)

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


# ---------------------------------------------------------------------------
# Tool: search_tour_info
# ---------------------------------------------------------------------------

async def _run_search_tour_info(query: str, tour_type: Optional[str] = None) -> str:
    rewrite_result = await _rewrite_query_for_tour(query)
    sub_queries    = rewrite_result["sub_queries"]
    clause_filter  = rewrite_result["clause_types"]

    _tour_id_match = re.search(r"(?:tour\s*(?:số|mã|số mã)?\s*)(\d+)", query, re.IGNORECASE)
    pinned_tour_id: Optional[str] = _tour_id_match.group(1) if _tour_id_match else None

    _prefetch = 20 if pinned_tour_id else 30
    _final    = 10 if pinned_tour_id else 15

    async def _fetch_one(sq: str) -> list[dict]:
        try:
            return await _hybrid_search_tours(
                query=sq,
                clause_filter=clause_filter,
                tour_type=tour_type,
                prefetch_limit=_prefetch,
                final_limit=_final,
                pinned_tour_id=pinned_tour_id,
            )
        except Exception as exc:
            logger.error("search_tour_info sub-query error (%r): %s", sq, exc)
            return []

    all_results_nested = await asyncio.gather(*[_fetch_one(sq) for sq in sub_queries])

    seen: set        = set()
    merged: list[dict] = []
    for batch in all_results_nested:
        for doc in batch:
            key = (
                doc.get("point", doc.get("article", "")),
                doc.get("clause", ""),
                doc.get("text", "")[:80],
            )
            if key not in seen:
                seen.add(key)
                merged.append(doc)

    if not merged:
        return "Không tìm thấy thông tin tour phù hợp."

    ranked = await rerank(query, merged, top_k=8)

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


# ---------------------------------------------------------------------------
# Tool: get_tour_detail
# ---------------------------------------------------------------------------

async def _run_get_tour_detail(tour_id: str) -> str:
    client = get_async_qdrant()
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
    if dur:             lines.append(f"Thời gian: {dur}")
    if price:           lines.append(f"Giá từ: {price}")
    if dep:             lines.append(f"Khởi hành từ: {dep}")
    if transport:       lines.append(f"Phương tiện: {transport}")
    if dests:           lines.append(f"Điểm đến: {dests}")
    if departures_text: lines.append(f"\n{departures_text}")
    if services_text:   lines.append(f"\n{services_text}")
    if policies_text:   lines.append(f"\n{policies_text}")
    if day_chunks:
        lines.append("\nLịch trình chi tiết:")
        for _, text in day_chunks:
            lines.append(f"\n{text}")

    return "\n".join(lines)


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


# ---------------------------------------------------------------------------
# Tool: scan_all_tours
# ---------------------------------------------------------------------------

async def _run_scan_all_tours(
    clause_types: list[str],
    tour_ids: Optional[list[str]] = None,
) -> str:
    client = get_async_qdrant()
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

    grouped: dict = defaultdict(list)
    for pt in results:
        payload     = pt.payload or {}
        meta        = payload.get("metadata", {})
        text        = payload.get("page_content", "")
        tid         = meta.get("point", meta.get("article", "?"))
        clause      = meta.get("clause", "")
        clause_full = meta.get("clause_full", "")
        grouped[tid].append((clause, text, clause_full))

    clause_order = {"summary": 0, "departures": 1, "services": 2, "policies": 3}

    def _clause_key(x):
        c = x[0]
        if c in clause_order:
            return (clause_order[c], c)
        if c.startswith("day_"):
            n = int(c.replace("day_", "") or 0)
            return (10 + n, c)
        return (99, c)

    parts = []
    for tid, items in sorted(
        grouped.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999
    ):
        items.sort(key=_clause_key)
        tour_texts = []
        for clause, text, clause_full in items:
            if clause == "summary" and clause_full:
                try:
                    cf = json.loads(clause_full)
                    tr = cf.get("tr", "")
                    dc = cf.get("dc", "")
                    extra_lines = []
                    if tr: extra_lines.append(f"Phương tiện di chuyển: {tr}")
                    if dc: extra_lines.append(f"Điểm khởi hành: {dc}")
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
    return "\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(parts)


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
# Tool: count_tour_meals
# ---------------------------------------------------------------------------

async def _run_count_tour_meals(tour_ids: list[str]) -> str:
    client = get_async_qdrant()
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

    tour_days: dict  = defaultdict(list)
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
            cf        = json.loads(meta.get("clause_full", "{}"))
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

        total_sang  = sum(m.count("sáng") for _, _, m, _ in days)
        total_trua  = sum(m.count("trưa") for _, _, m, _ in days)
        total_toi   = sum(m.count("tối")  for _, _, m, _ in days)
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


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def tour_system_prompt(tool_lines: str, date: str) -> str:
    return f"""Bạn là chuyên gia tư vấn du lịch. Ngày hiện tại: {date}.

{OUTPUT_FORMAT}

CÔNG CỤ:
{tool_lines}

CHỌN ĐÚNG CÔNG CỤ:
search_tours      → CHƯA biết mã số, cần tìm tour theo từ khóa/địa điểm. KHÔNG dùng khi đã có mã số. KHÔNG gọi với query rỗng.
search_tour_info  → Tìm thông tin cụ thể trong một tour: địa điểm, dịch vụ, chính sách, lịch khởi hành.
get_tour_detail   → Đã có mã số, muốn xem toàn bộ chi tiết 1 tour.
scan_all_tours    → Đã có mã số hoặc cần tổng hợp/so sánh nhiều tour. Truyền tour_ids=["2","3"] khi biết mã.
count_tour_meals  → Đếm số bữa ăn của tour. Dùng cho mọi câu hỏi về số lượng bữa.

QUY TẮC KHI ĐÃ BIẾT MÃ SỐ TOUR:
Khi câu hỏi đã chỉ rõ "Tour 2", "Tour số 3", "tour mã 4"... TUYỆT ĐỐI không gọi search_tours.
  - Hỏi thông tin TỔNG QUAN 1 tour (lịch trình đầy đủ, giá, điểm đến, chính sách) → get_tour_detail(tour_id="2")
  - Hỏi thông tin CỤ THỂ trong 1 tour (nghỉ đêm ngày 1, chính sách hủy, chi phí vé tham quan) → search_tour_info(query="...", tour_type=...)
  - So sánh/tổng hợp nhiều tour → scan_all_tours(tour_ids=["2","3"], clause_types=[...])

QUY TẮC BỮA ĂN:
Câu hỏi đếm hoặc so sánh số lượng bữa ăn → luôn dùng count_tour_meals với tour_ids cụ thể.
Câu hỏi về loại bữa cụ thể ("buffet sáng", "ăn tối du thuyền") → dùng search_tour_info để lấy raw itinerary.
Không tự đếm từ text mô tả.

QUY TẮC DỊCH VỤ BAO GỒM / KHÔNG BAO GỒM:
Câu hỏi về một địa điểm hoặc hoạt động cụ thể có bao gồm trong giá tour không,
hoặc hỏi chi phí của một vé tham quan cụ thể:
Luôn kiểm tra cả services (included lẫn excluded) — thông tin thường nằm trong excluded.
Nếu search_tour_info chưa trả về services chunk → gọi thêm lần nữa với query tập trung vào dịch vụ.

QUY TẮC CHÍNH SÁCH:
Câu hỏi về một điều kiện cụ thể trong chính sách (một độ tuổi, một mức phí):
Chỉ trả lời đúng phần được hỏi. Không liệt kê toàn bộ chính sách.
Khi so sánh: nêu rõ điểm khác nhau, chỉ kết luận "tương tự" khi nội dung hoàn toàn trùng khớp.

QUY TẮC SỐ NGÀY HỦY TOUR:
Khi câu hỏi đề cập "trước X ngày": lấy toàn bộ bảng chính sách, xác định khoảng X rơi vào giữa hai mốc nào, áp dụng đúng mức phí của khoảng đó.

SAU KHI search_tours TRẢ VỀ KẾT QUẢ:
Đọc nguyên văn từng dòng. Không bỏ cụm "mã số". Cuối cùng hỏi khách muốn biết thêm tour nào.
"mã số" là ID thực tế trong hệ thống — KHÔNG phải số thứ tự trong danh sách.
Ví dụ: "[mã 3] TOUR HẠ LONG" → gọi là "tour mã ba", không phải "tour thứ nhất" hay "tour số một"."""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

_TOUR_TOOLS = [
    search_tours,
    search_tour_info,
    scan_all_tours,
    get_tour_detail,
    count_tour_meals,
]


def build_tour_agent(
    streaming_llm: ChatOpenAI,
    rewriter_llm: ChatOpenAI,
    date: str,
    extra_tools: Optional[list] = None,
):
    """
    Tạo compiled LangGraph cho Tour domain.

    Args:
        streaming_llm : LLM dùng để sinh câu trả lời (streaming=True).
        rewriter_llm  : LLM dùng để rewrite query (streaming=False).
        date          : Ngày hiện tại dạng "dd/mm/yyyy".
        extra_tools   : Danh sách tool bổ sung (tuỳ chọn).

    Returns:
        Compiled LangGraph sub-agent.
    """
    from utils import build_sub_agent

    global _rewriter_llm_ref
    _rewriter_llm_ref = rewriter_llm

    tools = list(extra_tools or []) + _TOUR_TOOLS
    tool_lines = "\n".join(
        f"- {t.name}: {t.description.splitlines()[0]}" for t in tools
    )
    system_prompt = tour_system_prompt(tool_lines, date)

    return build_sub_agent(
        llm_with_tools=streaming_llm.bind_tools(tools),
        tools=tools,
        system_prompt=system_prompt,
    )