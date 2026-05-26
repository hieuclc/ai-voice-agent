"""
law_agent.py — Domain agent cho pháp luật Việt Nam.

Exports:
  build_law_agent(streaming_llm, grader_llm, date) → compiled LangGraph
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from utils import (
    OUTPUT_FORMAT,
    MAX_REWRITE,
    RETRIEVAL_K,
    TOP_K,
    hybrid_retrieve,
    rerank,
)

logger = logging.getLogger(__name__)

LAW_COLLECTION = "law"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_law_chunk(doc: dict) -> str:
    text        = doc.get("text", "")
    source      = doc.get("source", doc.get("source_file", ""))
    article     = doc.get("article", "")
    clause      = doc.get("clause", "")
    clause_full = doc.get("clause_full", "")

    header = f"Văn bản: {source}"
    if article:
        header += f" | Điều: {article}"
    if clause:
        header += f" | Khoản: {clause}"

    parts = [header, "", text]
    if clause_full and clause_full != text:
        parts += ["", f"Toàn văn khoản:\n{clause_full}"]
    return "\n".join(parts)


async def _grade_documents(
    query: str,
    docs: list[dict],
    llm: ChatOpenAI,
) -> list[dict]:
    if not docs:
        return []

    prompt_tpl = (
        "Bạn là bộ lọc tài liệu pháp lý.\n"
        "Query: {query}\n\n"
        "Tài liệu:\n{doc}\n\n"
        "Tài liệu này có liên quan đến query không? Trả lời DUY NHẤT 'yes' hoặc 'no':"
    )

    import asyncio

    async def _grade_one(doc: dict) -> bool:
        try:
            resp = await llm.ainvoke([
                HumanMessage(content=prompt_tpl.format(query=query, doc=doc["text"][:1500]))
            ])
            return resp.content.strip().lower().startswith("yes")
        except Exception:
            return True

    flags = await asyncio.gather(*[_grade_one(d) for d in docs])
    return [d for d, keep in zip(docs, flags) if keep]


async def _rewrite_query(
    original: str,
    current: str,
    llm: ChatOpenAI,
) -> str:
    prompt = (
        "Bạn là công cụ rewrite query cho hệ thống tìm kiếm pháp luật Việt Nam.\n"
        "Query hiện tại không tìm được đủ tài liệu liên quan.\n\n"
        f"Query gốc: {original}\n"
        f"Query hiện tại: {current}\n\n"
        "Viết lại query (5-12 từ, thuật ngữ pháp lý chính xác hơn).\n"
        "Chỉ trả về query mới, không giải thích:"
    )
    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        new_q = resp.content.strip()
        logger.info("[LawRewrite] %r → %r", current, new_q)
        return new_q
    except Exception:
        return current


async def _agentic_law_retrieve(
    query: str,
    grader_llm: ChatOpenAI,
) -> tuple[str, list[dict]]:
    MIN_RELEVANT  = 2
    current_query = query
    rewrite_count = 0

    while True:
        raw    = await hybrid_retrieve(LAW_COLLECTION, current_query, RETRIEVAL_K)
        ranked = await rerank(current_query, raw, TOP_K)

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
# Tool
# ---------------------------------------------------------------------------

class _SearchInput(BaseModel):
    query: str


# grader_llm được inject khi build_law_agent() được gọi
_grader_llm_ref: Optional[ChatOpenAI] = None


async def _run_search_law(query: str) -> str:
    if _grader_llm_ref is None:
        # Fallback: không có grader → retrieve + rerank thẳng
        raw    = await hybrid_retrieve(LAW_COLLECTION, query, RETRIEVAL_K)
        ranked = await rerank(query, raw, TOP_K)
        if not ranked:
            return "Không tìm thấy thông tin pháp luật phù hợp."
        return "\n\n-------------------\n\n".join(_format_law_chunk(d) for d in ranked)

    context, sources = await _agentic_law_retrieve(query, _grader_llm_ref)

    if not context:
        return "Không tìm thấy thông tin pháp luật phù hợp."

    logger.info(
        "===== LAW RESULT =====\n%d sources\n%s\n======================",
        len(sources),
        "\n".join(
            f"  [{i+1}] {d.get('source','?')} | {d.get('article','')}"
            for i, d in enumerate(sources)
        ),
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
# System prompt
# ---------------------------------------------------------------------------

def law_system_prompt(tool_lines: str, date: str) -> str:
    return f"""Bạn là chuyên gia tư vấn pháp luật Việt Nam. Ngày hiện tại: {date}.

{OUTPUT_FORMAT}

CÔNG CỤ:
{tool_lines}

QUY TẮC TRA CỨU PHÁP LUẬT:
Với mọi câu hỏi về luật, nghị định, mức phạt, điều khoản — bắt buộc tra cứu trước khi trả lời. Sử dụng công cụ search_law.
Nếu tra lại vẫn không có → trả lời: "Không tìm thấy thông tin phù hợp trong dữ liệu hiện có."

QUY TẮC QUERY KHI GỌI search_law — BẮT BUỘC:
Query phải là từ khóa NGẮN (3-6 từ), KHÔNG dùng câu hỏi đầy đủ.
Chỉ giữ lại hành vi vi phạm + đối tượng/phương tiện. Bỏ hết từ thừa.

Ví dụ ĐÚNG:
  "vượt đèn đỏ ô tô"
  "vượt đèn đỏ xe máy"
  "nồng độ cồn xe máy"
  "không đội mũ bảo hiểm"
  "chạy quá tốc độ ô tô"

QUY TẮC TRÍCH DẪN:
Mọi thông tin pháp luật phải có nguồn ngay sau nội dung, dạng:
"theo Luật Giao thông đường bộ, Điều ba mươi bảy, Khoản một"
Không bịa số điều khoản khi không có trong dữ liệu."""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_law_agent(
    streaming_llm: ChatOpenAI,
    grader_llm: ChatOpenAI,
    date: str,
    extra_tools: Optional[list] = None,
):
    """
    Tạo compiled LangGraph cho Law domain.

    Args:
        streaming_llm : LLM dùng để sinh câu trả lời (streaming=True).
        grader_llm    : LLM dùng để grade và rewrite query (streaming=False).
        date          : Ngày hiện tại dạng "dd/mm/yyyy".
        extra_tools   : Danh sách tool bổ sung (tuỳ chọn).

    Returns:
        Compiled LangGraph sub-agent.
    """
    from utils import build_sub_agent

    global _grader_llm_ref
    _grader_llm_ref = grader_llm

    tools = list(extra_tools or []) + [search_law]
    tool_lines = "\n".join(
        f"- {t.name}: {t.description.splitlines()[0]}" for t in tools
    )
    system_prompt = law_system_prompt(tool_lines, date)

    return build_sub_agent(
        llm_with_tools=streaming_llm.bind_tools(tools),
        tools=tools,
        system_prompt=system_prompt,
    )