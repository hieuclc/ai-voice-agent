"""
admission_agent.py — Domain agent cho tuyển sinh UET.

Exports:
  build_admission_agent(streaming_llm, rewriter_llm, date) → compiled LangGraph
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from utils import (
    OUTPUT_FORMAT,
    RETRIEVAL_K,
    TOP_K,
    hybrid_retrieve,
    rerank,
)

logger = logging.getLogger(__name__)

ADMISSION_COLLECTION = "admission"

# ---------------------------------------------------------------------------
# Query rewriter
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

# rewriter_llm được inject khi build_admission_agent() được gọi
_rewriter_llm_ref: Optional[ChatOpenAI] = None


async def _rewrite_query_for_admission(natural_query: str) -> list[str]:
    if _rewriter_llm_ref is None:
        return [natural_query]
    try:
        response = await _rewriter_llm_ref.ainvoke([
            SystemMessage(content=_ADMISSION_QUERY_REWRITER_SYSTEM_PROMPT),
            HumanMessage(content=natural_query),
        ])
        raw    = response.content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        queries = parsed.get("queries") or [natural_query]
        logger.info("ADMISSION QUERY REWRITE: %r → %s", natural_query, queries)
        return queries
    except Exception as exc:
        logger.warning("Admission query rewriter error (%s), fallback to original.", exc)
        return [natural_query]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class _SearchInput(BaseModel):
    query: str


async def _run_search_admission(query: str) -> str:
    sub_queries = await _rewrite_query_for_admission(query)

    seen_texts: set   = set()
    merged: list[dict] = []

    async def _fetch_one(sq: str) -> list[dict]:
        try:
            return await hybrid_retrieve(ADMISSION_COLLECTION, sq, RETRIEVAL_K)
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
        ranked = await rerank(query, merged, TOP_K)
    except Exception as exc:
        logger.error("admission rerank error: %s", exc)
        return f"Retrieval error: {exc}"

    if not ranked:
        return "Không tìm thấy thông tin tuyển sinh phù hợp."

    logger.info(
        "===== ADMISSION: query=%r → %d/%d results =====\n%s",
        query, len(ranked), len(merged),
        "\n".join(
            f"  [{i+1}] score={d.get('_rerank_score', 0):.4f} | "
            f"{d.get('text','')[:80].replace(chr(10),' ')}"
            for i, d in enumerate(ranked)
        ),
    )

    parts = []
    for i, item in enumerate(ranked, 1):
        text = item.get("text", "")
        parts.append(f"[Kết quả {i}]\nNội dung:\n{text}")

    return "\n\n-------------------\n\n".join(parts)


search_admission = StructuredTool.from_function(
    coroutine=_run_search_admission,
    name="search_admission",
    description=(
        "Tìm kiếm thông tin tuyển sinh, điểm chuẩn, chỉ tiêu, học phí, ngành học. "
        "Tham số: query (câu hỏi hoặc từ khóa tuyển sinh)."
    ),
    args_schema=_SearchInput,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def admission_system_prompt(tool_lines: str, date: str) -> str:
    return f"""Bạn là chuyên gia tư vấn tuyển sinh Đại học Công nghệ — Đại học Quốc gia Hà Nội (UET). Ngày hiện tại: {date}.

{OUTPUT_FORMAT}

CÔNG CỤ:
{tool_lines}

MAPPING MÃ XÉT TUYỂN CÁC NGÀNH VÀ TÊN NGÀNH:
CN1 - Công nghệ thông tin
CN2 - Kỹ thuật máy tính
CN3 - Vật lý kỹ thuật
CN4 - Cơ kỹ thuật
CN5 - Công nghệ kỹ thuật xây dựng
CN6 - Công nghệ kỹ thuật cơ - điện tử
CN7 - Công nghệ hàng không vũ trụ
CN8 - Khoa học máy tính
CN9 - Công nghệ kỹ thuật điện tử - viễn thông
CN10 - Công nghệ nông nghiệp
CN11 - Kỹ thuật điều khiển và tự động hoá
CN12 - Trí tuệ nhân tạo
CN13 - Kỹ thuật năng lượng
CN14 - Hệ thống thông tin
CN15 - Mạng máy tính và truyền thông dữ liệu
CN17 - Kỹ thuật Robot
CN18 - Thiết kế công nghiệp và đồ họa

QUY TẮC TRA CỨU:
Mọi thông tin liên quan đến UET — bao gồm nhưng không giới hạn ở: điểm chuẩn,
học phí, chỉ tiêu, ngành học, mã ngành, địa chỉ, thông tin liên hệ, lịch thi,
chương trình đào tạo, quy định tuyển sinh — đều phải tra cứu qua search_admission.
Tuyệt đối không tự trả lời từ kiến thức có sẵn khi chưa gọi search_admission.
Luôn dùng search_admission trước khi trả lời. Không ước tính hay suy diễn số liệu.
Khi thực hiện tìm kiếm thông tin, nếu tìm tên ngành không ra kết quả thì hãy tìm theo mã ngành tương ứng.

QUY TẮC PHÂN TÍCH SAU KHI CÓ DỮ LIỆU:
Nếu câu hỏi yêu cầu đánh giá hoặc tư vấn (xu hướng điểm, nên đăng ký không):
  Lấy điểm chuẩn các năm trong data, so sánh với điểm thí sinh.
  Nhận xét xu hướng tăng giảm. Đưa ra đánh giá rõ: an toàn, cạnh tranh hoặc rủi ro cao.
  Nói rõ dự đoán tương lai là xu hướng, không phải con số chính xác.

QUY TẮC SO SÁNH ĐIỂM — BẮT BUỘC TUÂN THỦ:
Điều kiện đỗ: điểm thí sinh ≥ điểm chuẩn.
Điều kiện TRƯỢT: điểm thí sinh < điểm chuẩn. Dù chỉ chênh 0.01 điểm cũng là TRƯỢT.
Ví dụ bắt buộc ghi nhớ:
  - Thí sinh 27.0, điểm chuẩn 27.58 → 27.0 < 27.58 → TRƯỢT. Không nên nộp.
  - Thí sinh 27.6, điểm chuẩn 27.58 → 27.6 > 27.58 → ĐỖ. Có thể nộp nhưng sát.
  - Thí sinh 28.0, điểm chuẩn 27.58 → 28.0 > 27.58 → AN TOÀN.
Khi so sánh số thập phân: 27 = 27.00, không phải 27.99. Phải tính đủ phần thập phân.
TUYỆT ĐỐI không kết luận "có thể nộp" khi điểm thí sinh thấp hơn điểm chuẩn.

Nếu kết quả có mã ngành chưa có tên đầy đủ → gọi thêm search_admission để lấy tên.
Không liệt kê ngành "phù hợp" khi học phí thực tế vượt ngân sách người hỏi.
Không xác nhận thông tin người hỏi đề cập nếu dữ liệu không chứa thông tin đó."""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_admission_agent(
    streaming_llm: ChatOpenAI,
    rewriter_llm: ChatOpenAI,
    date: str,
    extra_tools: Optional[list] = None,
):
    """
    Tạo compiled LangGraph cho Admission domain.

    Args:
        streaming_llm : LLM dùng để sinh câu trả lời (streaming=True).
        rewriter_llm  : LLM dùng để rewrite query (streaming=False).
        date          : Ngày hiện tại dạng "dd/mm/yyyy".
        extra_tools   : Danh sách tool bổ sung (tuỳ chọn).

    Returns:
        Compiled LangGraph sub-agent.
    """
    from agent_routing import build_sub_agent

    global _rewriter_llm_ref
    _rewriter_llm_ref = rewriter_llm

    tools = list(extra_tools or []) + [search_admission]
    tool_lines = "\n".join(
        f"- {t.name}: {t.description.splitlines()[0]}" for t in tools
    )
    system_prompt = admission_system_prompt(tool_lines, date)

    return build_sub_agent(
        llm_with_tools=streaming_llm.bind_tools(tools),
        tools=tools,
        system_prompt=system_prompt,
    )