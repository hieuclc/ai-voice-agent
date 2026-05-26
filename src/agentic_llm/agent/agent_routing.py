"""
agent_routing.py — Router graph ghép các domain sub-agents.

Kiến trúc:
  RouterState → route_node → [law | admission | normal_talk] → tts_node → END

Exports:
  build_sub_agent(llm_with_tools, tools, system_prompt) → compiled LangGraph
  create_router_agent(...)                               → compiled RouterGraph
  pre_route(messages)                                    → domain str
  get_tts_agents()                                       → dict[str, TTSNormalizerAgent]
  preload_models (re-export từ utils)
  pick_thinking_start_sentence (re-export từ utils)
  pick_thinking_ongoing_sentence (re-export từ utils)
  THINKING_INTERVAL_SECONDS (re-export từ utils)
  _normal_talk_system_prompt
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from tts_normalizer import TTSNormalizerAgent, create_tts_normalizers
from utils import (
    THINKING_INTERVAL_SECONDS,
    THINKING_RESPONSE_DELAY_SECONDS,
    pick_thinking_ongoing_sentence,
    pick_thinking_start_sentence,
    preload_models,
)

logger = logging.getLogger(__name__)

# Re-export để server.py import từ một chỗ
__all__ = [
    "create_router_agent",
    "pre_route",
    "get_tts_agents",
    "preload_models",
    "pick_thinking_start_sentence",
    "pick_thinking_ongoing_sentence",
    "THINKING_INTERVAL_SECONDS",
    "THINKING_RESPONSE_DELAY_SECONDS",
    "_normal_talk_system_prompt",
]

# ---------------------------------------------------------------------------
# Graph states
# ---------------------------------------------------------------------------

class RouterState(TypedDict):
    messages:          Annotated[list[BaseMessage], add_messages]
    hop_count:         int
    thinking_streamed: bool
    domain:            str


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_ROUTER_PROMPT = """\
Bạn là bộ định tuyến câu hỏi. Phân loại câu hỏi sau vào đúng một trong bốn domain:

- "law"         : câu hỏi về luật, nghị định, thông tư, xử phạt, quy định pháp luật Việt Nam
- "admission"   : câu hỏi về tuyển sinh, điểm chuẩn, học phí, ngành học, mã ngành (UET)
- "normal_talk" : lời chào hỏi, cảm ơn, hỏi thăm, tán gẫu, câu hỏi xã giao,
                  hoặc bất kỳ nội dung nào không thuộc ba domain trên

Chỉ trả về đúng một trong bốn từ: law, admission, hoặc normal_talk. Không giải thích.\
"""

_VALID_DOMAINS = {"law", "admission", "normal_talk"}

_router_llm_instance: Optional[ChatOpenAI] = None

# TTS per-domain singletons
_tts_agents: dict[str, TTSNormalizerAgent] = {}


def get_tts_agents() -> dict[str, TTSNormalizerAgent]:
    return _tts_agents


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


async def pre_route(messages: list) -> str:
    """Pre-route trước khi vào graph, tránh overhead cho normal_talk."""
    if _router_llm_instance is None:
        logger.warning("pre_route called before create_router_agent — defaulting to normal_talk")
        return "normal_talk"

    user_msgs = [m for m in messages if isinstance(m, HumanMessage)]

    if not user_msgs:
        logger.info("ROUTER: no HumanMessage → normal_talk (kickstart)")
        return "normal_talk"

    last_content = user_msgs[-1].content.strip()

    if not last_content or len(last_content) < 3:
        logger.info(
            "ROUTER: empty/short HumanMessage %r → normal_talk (kickstart trigger)",
            last_content,
        )
        return "normal_talk"

    return await _route(last_content, _router_llm_instance)


# ---------------------------------------------------------------------------
# Normal talk system prompt
# ---------------------------------------------------------------------------

def _normal_talk_system_prompt(date: str) -> str:
    return f"""Bạn là trợ lý giọng nói tiếng Việt thân thiện. Ngày hiện tại: {date}.

QUY TẮC:
- Luôn trả lời bằng tiếng Việt.
- Câu đầu tiên của phản hồi phải có dưới 10 từ.
- Ngắn gọn, tự nhiên như đang trò chuyện trực tiếp.
- Tuyệt đối không dùng markdown, bullet, số thứ tự, emoji, header, dấu gạch đầu dòng.
- Không giải thích cách suy nghĩ.
- Mỗi câu phải có ít nhất bốn từ.
- Nếu được hỏi về chủ đề nằm ngoài khả năng (pháp luật chuyên sâu, tuyển sinh UET), hãy nói rõ bạn có thể hỗ trợ những chủ đề đó và mời người dùng hỏi."""


# ---------------------------------------------------------------------------
# Router graph builder
# ---------------------------------------------------------------------------

def _build_router_graph(
    router_llm: ChatOpenAI,
    law_graph,
    admission_graph,
    normal_talk_llm: ChatOpenAI,
    normal_talk_system_prompt: str,
):
    async def route_node(state: RouterState) -> dict:
        if state.get("domain") and state["domain"] in _VALID_DOMAINS:
            logger.info("ROUTER: domain pre-seeded as %s, skipping LLM", state["domain"])
            return {}
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not user_msgs:
            return {"domain": "normal_talk"}
        domain = await _route(user_msgs[-1].content, router_llm)
        return {"domain": domain}

    async def run_law(state: RouterState) -> dict:
        sub_state = {"messages": state["messages"], "hop_count": 0, "thinking_streamed": False}
        result    = await law_graph.ainvoke(sub_state)
        return {"messages": result["messages"], "hop_count": state["hop_count"]}

    async def run_admission(state: RouterState) -> dict:
        sub_state = {"messages": state["messages"], "hop_count": 0, "thinking_streamed": False}
        result    = await admission_graph.ainvoke(sub_state)
        return {"messages": result["messages"], "hop_count": state["hop_count"]}

    async def run_normal_talk(state: RouterState) -> dict:
        raw = list(state["messages"])
        system_msgs  = [m for m in raw if isinstance(m, SystemMessage)]
        non_sys_msgs = [m for m in raw if not isinstance(m, SystemMessage)]

        if not non_sys_msgs and system_msgs:
            greeting_instruction = HumanMessage(content=system_msgs[0].content)
            messages = [SystemMessage(content=normal_talk_system_prompt), greeting_instruction]
        else:
            messages = [SystemMessage(content=normal_talk_system_prompt)] + non_sys_msgs

        response = await normal_talk_llm.ainvoke(messages)
        return {"messages": [response], "hop_count": state["hop_count"]}

    def dispatch(state: RouterState) -> str:
        return state.get("domain", "normal_talk")

    g = StateGraph(RouterState)
    g.add_node("router",      route_node)
    g.add_node("law",         run_law)
    g.add_node("admission",   run_admission)
    g.add_node("normal_talk", run_normal_talk)
    
    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        dispatch,
        {
            "law":         "law",
            "admission":   "admission",
            "normal_talk": "normal_talk",
        },
    )

    g.add_edge("law",         END)
    g.add_edge("admission",   END)
    g.add_edge("normal_talk", END)

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
    Khởi tạo toàn bộ router graph gồm 3 domain sub-agents + normal_talk + TTS.

    Args:
        extra_tools     : Tool bổ sung inject vào tất cả domain agents.
        model           : OpenAI model name.
        openai_api_key  : API key (fallback về env OPENAI_API_KEY).
        openai_base_url : Base URL tuỳ chọn.
        temperature     : Temperature cho streaming LLM.

    Returns:
        Compiled RouterGraph.
    """
    from law_agent import build_law_agent
    from admission_agent import build_admission_agent

    api_key  = openai_api_key or os.environ.get("OPENAI_API_KEY")
    base_url = openai_base_url

    global _tts_agents
    _tts_agents = create_tts_normalizers(
        model=model,
        openai_api_key=api_key,
        openai_base_url=base_url,
    )
    logger.info("TTS normalizers initialized for domains: %s", list(_tts_agents.keys()))

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

    global _router_llm_instance
    _router_llm_instance = router_llm

    date = datetime.now().strftime("%d/%m/%Y")

    law_graph       = build_law_agent(streaming_llm, non_streaming_llm, date, extra_tools)
    admission_graph = build_admission_agent(streaming_llm, non_streaming_llm, date, extra_tools)

    normal_talk_llm  = _make_llm(streaming=False)
    nt_system_prompt = _normal_talk_system_prompt(date)

    return _build_router_graph(
        router_llm,
        law_graph,
        admission_graph,
        normal_talk_llm,
        nt_system_prompt,
    )