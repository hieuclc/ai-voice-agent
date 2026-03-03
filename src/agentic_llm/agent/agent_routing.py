"""
agent_routing.py — Multi-agent routing layer.

Kiến trúc:
  RouterAgent  → phân loại query → route tới 1 trong 3 sub-agent:
    LawAgent       : pháp luật, nghị định, xử phạt
    AdmissionAgent : tư vấn tuyển sinh UET
    TourAgent      : tư vấn du lịch

Mỗi sub-agent có:
  - System prompt riêng, tập trung đúng domain
  - Tool set riêng (chỉ các tool liên quan)
  - Cùng output format: văn bản hội thoại tiếng Việt, không ký tự đặc biệt

Public API (tương thích với create_agent cũ):
  create_router_agent(...)  → trả về compiled LangGraph runnable
  preload_bge_model()       → re-export từ agent.py
  stream_thinking_while_running(...) → re-export từ agent.py
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from typing_extensions import TypedDict

# Re-export tiện ích từ agent gốc
from agent import (
    preload_bge_model,
    stream_thinking_while_running,
    # Tools
    search_law,
    search_lightrag,
    search_admission,
    search_tours,
    search_tour_info,
    scan_all_tours,
    get_tour_detail,
    count_tour_meals,
    # LLM globals setter
    _grader_llm,
    _rewriter_llm,
)
import agent as _agent_module

logger = logging.getLogger(__name__)

MAX_HOPS = 6

# ---------------------------------------------------------------------------
# Output format chung — áp dụng cho tất cả sub-agent
# ---------------------------------------------------------------------------

_OUTPUT_FORMAT = """\
QUY TẮC ĐỊNH DẠNG ĐẦU RA — BẮT BUỘC:
- Luôn trả lời bằng tiếng Việt.
- Văn bản thuần, tự nhiên như đang nói chuyện trực tiếp.
- Tuyệt đối không dùng markdown, bullet, số thứ tự, emoji, header, dấu gạch đầu dòng.
- Không dùng chữ số — mọi con số phải viết bằng chữ (ví dụ: "hai mươi bảy phẩy năm").
- Không giải thích cách suy nghĩ, không nói "Theo dữ liệu tôi tìm được...".
- Mỗi câu phải có ít nhất bốn từ.
- Viết tắt (THPT, UET, VND...) phải viết IN HOA.\
"""

# ---------------------------------------------------------------------------
# System prompt từng sub-agent
# ---------------------------------------------------------------------------

def _law_system_prompt(tool_lines: str, date: str) -> str:
    return f"""Bạn là chuyên gia tư vấn pháp luật Việt Nam. Ngày hiện tại: {date}.

{_OUTPUT_FORMAT}

CÔNG CỤ:
{tool_lines}

QUY TẮC TRA CỨU PHÁP LUẬT:
Với mọi câu hỏi về luật, nghị định, mức phạt, điều khoản — bắt buộc tra cứu trước khi trả lời.
Dùng search_law trước. Nếu không đủ hoặc cần quan hệ nhiều văn bản → dùng search_lightrag.
Khi gọi search_lightrag: truyền nguyên văn câu hỏi gốc của người dùng.
Nếu tra lại vẫn không có → trả lời: "Không tìm thấy thông tin phù hợp trong dữ liệu hiện có."

QUY TẮC TRÍCH DẪN:
Mọi thông tin pháp luật phải có nguồn ngay sau nội dung, dạng:
"theo Luật Giao thông đường bộ, Điều ba mươi bảy, Khoản một"
Không bịa số điều khoản khi không có trong dữ liệu."""


def _admission_system_prompt(tool_lines: str, date: str) -> str:
    return f"""Bạn là chuyên gia tư vấn tuyển sinh Đại học Công nghệ — Đại học Quốc gia Hà Nội (UET). Ngày hiện tại: {date}.

{_OUTPUT_FORMAT}

CÔNG CỤ:
{tool_lines}

QUY TẮC TRA CỨU:
Câu hỏi về điểm chuẩn, học phí, chỉ tiêu, ngành học, mã ngành — không bao giờ là kiến thức phổ thông.
Luôn dùng search_admission trước khi trả lời. Không ước tính hay suy diễn số liệu.

QUY TẮC PHÂN TÍCH SAU KHI CÓ DỮ LIỆU:
Nếu câu hỏi yêu cầu đánh giá hoặc tư vấn (xu hướng điểm, nên đăng ký không):
  Lấy điểm chuẩn các năm trong data, so sánh với điểm thí sinh.
  Nhận xét xu hướng tăng giảm. Đưa ra đánh giá rõ: an toàn, cạnh tranh hoặc rủi ro cao.
  Nói rõ dự đoán tương lai là xu hướng, không phải con số chính xác.

Nếu kết quả có mã ngành chưa có tên đầy đủ → gọi thêm search_admission để lấy tên.
Không liệt kê ngành "phù hợp" khi học phí thực tế vượt ngân sách người hỏi.
Không xác nhận thông tin người hỏi đề cập nếu dữ liệu không chứa thông tin đó."""


def _tour_system_prompt(tool_lines: str, date: str) -> str:
    return f"""Bạn là chuyên gia tư vấn du lịch. Ngày hiện tại: {date}.

{_OUTPUT_FORMAT}

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
  - Hỏi thông tin 1 tour → get_tour_detail(tour_id="2")
  - So sánh/tổng hợp nhiều tour → scan_all_tours(tour_ids=["2","3"], clause_types=[...])
  - Ví dụ: "so sánh thời gian Tour 2 và Tour 3" → scan_all_tours(tour_ids=["2","3"], clause_types=["summary"])
  - Ví dụ: "Tour 2 đi những đâu" → get_tour_detail(tour_id="2") hoặc scan_all_tours(tour_ids=["2"], clause_types=["summary"])

QUY TẮC BỮA ĂN:
Câu hỏi đếm hoặc so sánh bữa ăn → luôn dùng count_tour_meals với tour_ids cụ thể.
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
# AgentState
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages:          Annotated[list[BaseMessage], add_messages]
    hop_count:         int
    thinking_streamed: bool


# ---------------------------------------------------------------------------
# Hàm dựng graph cho một sub-agent
# ---------------------------------------------------------------------------

def _build_sub_agent(llm_with_tools, tools: list[BaseTool], system_prompt: str):
    tool_node = ToolNode(tools)

    async def call_model(state: AgentState) -> dict:
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response], "hop_count": state["hop_count"]}

    async def run_tools(state: AgentState) -> dict:
        result = await tool_node.ainvoke(state)
        return {**result, "hop_count": state["hop_count"] + 1, "thinking_streamed": False}

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and state["hop_count"] < MAX_HOPS:
            return "tools"
        return "__end__"

    g = StateGraph(AgentState)
    g.add_node("agent", call_model)
    g.add_node("tools", run_tools)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    g.add_edge("tools", "agent")
    return g.compile()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_ROUTER_PROMPT = """\
Bạn là bộ định tuyến câu hỏi. Phân loại câu hỏi sau vào đúng một trong ba domain:

- "law"       : câu hỏi về luật, nghị định, thông tư, xử phạt, quy định pháp luật Việt Nam
- "admission" : câu hỏi về tuyển sinh, điểm chuẩn, học phí, ngành học, mã ngành (UET)
- "tour"      : câu hỏi về tour du lịch, lịch trình, chính sách tour, bữa ăn, giá tour

Chỉ trả về đúng một trong ba từ: law, admission, hoặc tour. Không giải thích.\
"""


class _RouterOutput(BaseModel):
    domain: Literal["law", "admission", "tour"]


async def _route(query: str, llm: ChatOpenAI) -> str:
    """Gọi LLM để phân loại query. Fallback về 'law' nếu lỗi."""
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=_ROUTER_PROMPT),
            HumanMessage(content=query),
        ])
        domain = resp.content.strip().lower()
        if domain in ("law", "admission", "tour"):
            logger.info("ROUTER: %r → %s", query[:60], domain)
            return domain
        logger.warning("ROUTER unexpected output %r, defaulting to law", domain)
        return "law"
    except Exception as exc:
        logger.error("ROUTER error: %s", exc)
        return "law"


# ---------------------------------------------------------------------------
# Router graph — gọi đúng sub-agent dựa vào domain
# ---------------------------------------------------------------------------

class RouterState(TypedDict):
    messages:          Annotated[list[BaseMessage], add_messages]
    hop_count:         int
    thinking_streamed: bool
    domain:            str   # "law" | "admission" | "tour" | ""


def _build_router_graph(
    router_llm: ChatOpenAI,
    law_graph,
    admission_graph,
    tour_graph,
):
    async def route_node(state: RouterState) -> dict:
        # Lấy câu hỏi mới nhất của user
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        query = user_msgs[-1].content if user_msgs else ""
        domain = await _route(query, router_llm)
        return {"domain": domain}

    async def run_law(state: RouterState) -> dict:
        sub_state = {"messages": state["messages"], "hop_count": 0, "thinking_streamed": False}
        result = await law_graph.ainvoke(sub_state)
        return {"messages": result["messages"], "hop_count": state["hop_count"]}

    async def run_admission(state: RouterState) -> dict:
        sub_state = {"messages": state["messages"], "hop_count": 0, "thinking_streamed": False}
        result = await admission_graph.ainvoke(sub_state)
        return {"messages": result["messages"], "hop_count": state["hop_count"]}

    async def run_tour(state: RouterState) -> dict:
        sub_state = {"messages": state["messages"], "hop_count": 0, "thinking_streamed": False}
        result = await tour_graph.ainvoke(sub_state)
        return {"messages": result["messages"], "hop_count": state["hop_count"]}

    def dispatch(state: RouterState) -> str:
        return state["domain"]

    g = StateGraph(RouterState)
    g.add_node("router",    route_node)
    g.add_node("law",       run_law)
    g.add_node("admission", run_admission)
    g.add_node("tour",      run_tour)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        dispatch,
        {"law": "law", "admission": "admission", "tour": "tour"},
    )
    g.add_edge("law",       END)
    g.add_edge("admission", END)
    g.add_edge("tour",      END)

    return g.compile()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_router_agent(
    extra_tools: Optional[list[BaseTool]] = None,
    model: str = "gpt-4o-mini",
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    temperature: float = 0.0,
):
    """
    Thay thế create_agent(). Trả về router graph tương thích LangGraph.
    """
    import os
    api_key  = openai_api_key  or os.environ.get("OPENAI_API_KEY")
    base_url = openai_base_url

    def _make_llm(streaming: bool) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            temperature=temperature if streaming else 0,
            api_key=api_key,
            base_url=base_url,
            streaming=streaming,
        )

    streaming_llm    = _make_llm(streaming=True)
    non_streaming_llm = _make_llm(streaming=False)
    router_llm       = _make_llm(streaming=False)

    # Cập nhật globals trong agent.py để rewriter/grader dùng đúng LLM
    _agent_module._grader_llm   = non_streaming_llm
    _agent_module._rewriter_llm = non_streaming_llm

    date = datetime.now().strftime("%d/%m/%Y")

    # ── Law sub-agent ────────────────────────────────────────────────────
    law_tools = list(extra_tools or []) + [search_law, search_lightrag]
    law_tool_lines = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in law_tools)
    law_graph = _build_sub_agent(
        streaming_llm.bind_tools(law_tools),
        law_tools,
        _law_system_prompt(law_tool_lines, date),
    )

    # ── Admission sub-agent ──────────────────────────────────────────────
    admission_tools = list(extra_tools or []) + [search_admission]
    adm_tool_lines = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in admission_tools)
    admission_graph = _build_sub_agent(
        streaming_llm.bind_tools(admission_tools),
        admission_tools,
        _admission_system_prompt(adm_tool_lines, date),
    )

    # ── Tour sub-agent ───────────────────────────────────────────────────
    tour_tools = list(extra_tools or []) + [
        search_tours,
        search_tour_info,
        scan_all_tours,
        get_tour_detail,
        count_tour_meals,
    ]
    tour_tool_lines = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in tour_tools)
    tour_graph = _build_sub_agent(
        streaming_llm.bind_tools(tour_tools),
        tour_tools,
        _tour_system_prompt(tour_tool_lines, date),
    )

    return _build_router_graph(router_llm, law_graph, admission_graph, tour_graph)