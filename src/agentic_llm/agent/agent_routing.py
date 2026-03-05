"""
agent_routing.py — Multi-agent routing layer.

Domains:
  law         — pháp luật Việt Nam
  admission   — tuyển sinh UET
  tour        — tư vấn du lịch
  normal_talk — xã giao, chào hỏi, câu hỏi ngoài 3 domain trên
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
from pydantic import BaseModel
from typing_extensions import TypedDict

from agent import (
    preload_bge_model,
    stream_thinking_while_running,
    build_sub_agent,
    AgentState,
    search_law,
    search_admission,
    search_tours,
    search_tour_info,
    scan_all_tours,
    get_tour_detail,
    count_tour_meals,
)
import agent as _agent_module
from tts_normalizer import TTSNormalizerAgent

logger = logging.getLogger(__name__)

# TTS singleton — khởi tạo lại trong create_router_agent() khi có đủ config
_tts: Optional[TTSNormalizerAgent] = None

# Router LLM singleton — dùng bởi pre_route() trong server.py
_router_llm_instance: Optional[ChatOpenAI] = None

MAX_HOPS = 6

_OUTPUT_FORMAT = """\
QUY TẮC ĐỊNH DẠNG ĐẦU RA — BẮT BUỘC:
- Luôn trả lời bằng tiếng Việt.
- Chỉ trả lời đúng phạm vi câu hỏi. Không liệt kê thông tin thừa không được hỏi.
  Ví dụ: hỏi "khởi hành từ đâu" → chỉ trả lời tên địa điểm, không thêm tên tour, mã số, giá.
  Ví dụ: hỏi "chính sách hủy ngay sau đăng ký" → chỉ cần mức phí đó, không liệt kê toàn bộ bảng.
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

def _law_system_prompt(tool_lines: str, date: str) -> str:
    return f"""Bạn là chuyên gia tư vấn pháp luật Việt Nam. Ngày hiện tại: {date}.

{_OUTPUT_FORMAT}

CÔNG CỤ:
{tool_lines}

QUY TẮC TRA CỨU PHÁP LUẬT:
Với mọi câu hỏi về luật, nghị định, mức phạt, điều khoản — bắt buộc tra cứu trước khi trả lời. Sử dụng công cụ search_law.
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


def _normal_talk_system_prompt(date: str) -> str:
    return f"""Bạn là trợ lý giọng nói tiếng Việt thân thiện. Ngày hiện tại: {date}.

QUY TẮC:
- Luôn trả lời bằng tiếng Việt.
- Ngắn gọn, tự nhiên như đang trò chuyện trực tiếp.
- Tuyệt đối không dùng markdown, bullet, số thứ tự, emoji, header, dấu gạch đầu dòng.
- Không giải thích cách suy nghĩ.
- Mỗi câu phải có ít nhất bốn từ.
- Nếu được hỏi về chủ đề nằm ngoài khả năng (pháp luật chuyên sâu, tuyển sinh UET,
  tour du lịch cụ thể), hãy nói rõ bạn có thể hỗ trợ những chủ đề đó và mời người dùng hỏi."""


# ---------------------------------------------------------------------------
# Router prompt — 4 domains
# ---------------------------------------------------------------------------

_ROUTER_PROMPT = """\
Bạn là bộ định tuyến câu hỏi. Phân loại câu hỏi sau vào đúng một trong bốn domain:

- "law"         : câu hỏi về luật, nghị định, thông tư, xử phạt, quy định pháp luật Việt Nam
- "admission"   : câu hỏi về tuyển sinh, điểm chuẩn, học phí, ngành học, mã ngành (UET)
- "tour"        : câu hỏi về tour du lịch, lịch trình, chính sách tour, bữa ăn, giá tour
- "normal_talk" : lời chào hỏi, cảm ơn, hỏi thăm, tán gẫu, câu hỏi xã giao,
                  hoặc bất kỳ nội dung nào không thuộc ba domain trên

Chỉ trả về đúng một trong bốn từ: law, admission, tour, hoặc normal_talk. Không giải thích.\
"""

_VALID_DOMAINS = {"law", "admission", "tour", "normal_talk"}


async def _route(query: str, llm: ChatOpenAI) -> str:
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=_ROUTER_PROMPT),
            HumanMessage(content=query),
        ])
        domain = resp.content.strip().lower()
        if domain in _VALID_DOMAINS:
            logger.info("ROUTER: %r → %s", query[:60], domain)
            return domain
        logger.warning("ROUTER unexpected output %r, defaulting to normal_talk", domain)
        return "normal_talk"
    except Exception as exc:
        logger.error("ROUTER error: %s", exc)
        return "normal_talk"


# ---------------------------------------------------------------------------
# Public helper — gọi từ server.py để pre-route trước khi ainvoke graph
# ---------------------------------------------------------------------------

async def pre_route(messages: list) -> str:
    """
    Phân loại nhanh domain từ danh sách messages.
    - Nếu không có HumanMessage nào → normal_talk (ví dụ: kickstart system prompt)
    - Ngược lại → gọi router LLM với nội dung HumanMessage cuối cùng

    Server.py gọi hàm này TRƯỚC khi ainvoke graph, sau đó truyền domain vào
    initial_state["domain"] để route_node bỏ qua LLM call thứ hai.
    """
    if _router_llm_instance is None:
        logger.warning("pre_route called before create_router_agent — defaulting to normal_talk")
        return "normal_talk"

    user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    if not user_msgs:
        logger.info("ROUTER: no HumanMessage found → normal_talk")
        return "normal_talk"

    return await _route(user_msgs[-1].content, _router_llm_instance)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class RouterState(TypedDict):
    messages:          Annotated[list[BaseMessage], add_messages]
    hop_count:         int
    thinking_streamed: bool
    domain:            str


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_router_graph(
    router_llm: ChatOpenAI,
    law_graph,
    admission_graph,
    tour_graph,
    normal_talk_llm: ChatOpenAI,
    normal_talk_system_prompt: str,
):
    # ------------------------------------------------------------------
    # route_node: nếu domain đã được pre-seeded từ server.py thì bỏ qua
    #             LLM call (tiết kiệm 1 round-trip); nếu không có
    #             HumanMessage thì default về normal_talk
    # ------------------------------------------------------------------
    async def route_node(state: RouterState) -> dict:
        # Domain đã được server.py điền trước → dùng luôn
        if state.get("domain") and state["domain"] in _VALID_DOMAINS:
            logger.info("ROUTER: domain pre-seeded as %s, skipping LLM", state["domain"])
            return {}

        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not user_msgs:
            logger.info("ROUTER: no HumanMessage → normal_talk")
            return {"domain": "normal_talk"}

        domain = await _route(user_msgs[-1].content, router_llm)
        return {"domain": domain}

    # ------------------------------------------------------------------
    # Sub-agent runners
    # ------------------------------------------------------------------
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

    async def run_normal_talk(state: RouterState) -> dict:
        """
        Agent xã giao — không dùng tool, trả lời trực tiếp từ LLM.
        Không cần emit thinking sentences.
        """
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=normal_talk_system_prompt)] + messages
        response = await normal_talk_llm.ainvoke(messages)
        return {"messages": [response], "hop_count": state["hop_count"]}

    # ------------------------------------------------------------------
    # TTS normalization node (chạy sau mọi sub-agent)
    # ------------------------------------------------------------------
    async def tts_node(state: RouterState) -> dict:
        """
        Node TTS duy nhất trong toàn bộ pipeline.
        Tìm AIMessage cuối, normalize, thay thế in-place.
        """
        messages = list(state["messages"])
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
                if _tts is not None:
                    try:
                        normalized = await _tts.anormalize(msg.content)
                    except Exception as exc:
                        logger.warning("TTS normalize error, keeping original: %s", exc)
                        normalized = msg.content
                else:
                    normalized = msg.content
                new_msg = AIMessage(
                    content=normalized,
                    id=msg.id,
                    name=getattr(msg, "name", None),
                    additional_kwargs=msg.additional_kwargs,
                )
                messages = messages[:i] + [new_msg] + messages[i + 1:]
                break
        return {"messages": messages}

    # ------------------------------------------------------------------
    # Conditional dispatch
    # ------------------------------------------------------------------
    def dispatch(state: RouterState) -> str:
        return state.get("domain", "normal_talk")

    # ------------------------------------------------------------------
    # Assemble graph
    # ------------------------------------------------------------------
    g = StateGraph(RouterState)
    g.add_node("router",      route_node)
    g.add_node("law",         run_law)
    g.add_node("admission",   run_admission)
    g.add_node("tour",        run_tour)
    g.add_node("normal_talk", run_normal_talk)
    g.add_node("tts",         tts_node)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        dispatch,
        {
            "law":         "law",
            "admission":   "admission",
            "tour":        "tour",
            "normal_talk": "normal_talk",
        },
    )

    # Tất cả sub-agent đều qua TTS trước khi kết thúc
    g.add_edge("law",         "tts")
    g.add_edge("admission",   "tts")
    g.add_edge("tour",        "tts")
    g.add_edge("normal_talk", "tts")
    g.add_edge("tts",         END)

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
    import os
    api_key  = openai_api_key or os.environ.get("OPENAI_API_KEY")
    base_url = openai_base_url

    # TTS singleton
    global _tts
    _tts = TTSNormalizerAgent(
        model=model,
        openai_api_key=api_key,
        openai_base_url=base_url,
    )

    def _make_llm(streaming: bool) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            temperature=temperature if streaming else 0,
            api_key=api_key,
            base_url=base_url,
            streaming=streaming,
        )

    streaming_llm     = _make_llm(streaming=True)
    non_streaming_llm = _make_llm(streaming=False)
    router_llm        = _make_llm(streaming=False)

    # Expose router LLM cho pre_route()
    global _router_llm_instance
    _router_llm_instance = router_llm

    _agent_module._grader_llm   = non_streaming_llm
    _agent_module._rewriter_llm = non_streaming_llm

    date = datetime.now().strftime("%d/%m/%Y")

    # --- Law sub-agent ---
    law_tools = list(extra_tools or []) + [search_law]
    law_tool_lines = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in law_tools)
    law_graph = build_sub_agent(
        streaming_llm.bind_tools(law_tools),
        law_tools,
        _law_system_prompt(law_tool_lines, date),
    )

    # --- Admission sub-agent ---
    admission_tools = list(extra_tools or []) + [search_admission]
    adm_tool_lines = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in admission_tools)
    admission_graph = build_sub_agent(
        streaming_llm.bind_tools(admission_tools),
        admission_tools,
        _admission_system_prompt(adm_tool_lines, date),
    )

    # --- Tour sub-agent ---
    tour_tools = list(extra_tools or []) + [
        search_tours, search_tour_info, scan_all_tours, get_tour_detail, count_tour_meals,
    ]
    tour_tool_lines = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in tour_tools)
    tour_graph = build_sub_agent(
        streaming_llm.bind_tools(tour_tools),
        tour_tools,
        _tour_system_prompt(tour_tool_lines, date),
    )

    # --- Normal talk LLM (no tools) ---
    normal_talk_llm = _make_llm(streaming=False)
    nt_system_prompt = _normal_talk_system_prompt(date)

    return _build_router_graph(
        router_llm,
        law_graph,
        admission_graph,
        tour_graph,
        normal_talk_llm,
        nt_system_prompt,
    )