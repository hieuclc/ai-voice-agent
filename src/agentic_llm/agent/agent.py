"""
LangGraph RAG Agent with multi-hop reasoning.

RAG tools:
  - search_law      : ChromaDB vector search (law domain)
  - search_admission: ChromaDB vector search (admission domain)
  - search_tours    : ChromaDB vector search (tour domain)
  - get_tour_detail : ChromaDB exact-match tour detail
  - search_lightrag : LightRAG hybrid (graph + vector) search  ← NEW
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dotenv import load_dotenv
from typing import Annotated, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from typing_extensions import TypedDict
from datetime import datetime

logger = logging.getLogger(__name__)
load_dotenv(override=True)

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# ChromaDB (LAW domain only)
# ---------------------------------------------------------------------------

CHROMA_LAW_PATH = os.environ.get("CHROMA_PATH")
LAW_COLLECTION_NAME = "law"
ADMISSION_COLLECTION_NAME = "admission"
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
DEVICE = os.environ.get("DEVICE")
logger.info("Initializing LAW ChromaDB at path: %s", CHROMA_LAW_PATH)

# Persistent client
_client = chromadb.PersistentClient(path=CHROMA_LAW_PATH)

# Embedding function (must match the one used when indexing!)
_law_embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME,
    device=DEVICE,
)

# Get collection
_law_collection = _client.get_or_create_collection(
    name=LAW_COLLECTION_NAME,
    embedding_function=_law_embedding_fn,
)

_admission_collection = _client.get_or_create_collection(
    name=ADMISSION_COLLECTION_NAME,
    embedding_function=_law_embedding_fn,
)

try:
    count = _law_collection.count()
    logger.info(
        "LAW collection '%s' loaded. Document count = %d",
        LAW_COLLECTION_NAME,
        count,
    )
    count = _admission_collection.count()
    logger.info(
        "Admission collection '%s' loaded. Document count = %d",
        ADMISSION_COLLECTION_NAME,
        count,
    )
except Exception as e:
    logger.exception("Failed to load LAW collection: %s", e)


# ---------------------------------------------------------------------------
# LightRAG singleton — initialized once at startup
# ---------------------------------------------------------------------------

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag.utils import setup_logger as lightrag_setup_logger
from embedding_service import VIETNAMESE_EMBEDDING_FUNC, get_embedding_service

lightrag_setup_logger("lightrag", level="WARNING")

LIGHTRAG_WORKING_DIR = os.environ.get("LIGHTRAG_WORKING_DIR", "./lightrag_data")

# ---------------------------------------------------------------------------
# LightRAG instance
# ---------------------------------------------------------------------------

_lightrag_instance: Optional[LightRAG] = None
_lightrag_lock = asyncio.Lock()


async def get_lightrag() -> LightRAG:
    """Return (and lazily initialize) the shared LightRAG instance."""
    global _lightrag_instance
    if _lightrag_instance is not None:
        return _lightrag_instance

    async with _lightrag_lock:
        # Double-checked locking
        if _lightrag_instance is not None:
            return _lightrag_instance

        os.makedirs(LIGHTRAG_WORKING_DIR, exist_ok=True)

        # Pre-warm the embedding model before LightRAG uses it
        await get_embedding_service()

        rag = LightRAG(
            working_dir=LIGHTRAG_WORKING_DIR,
            llm_model_func=gpt_4o_mini_complete,
            embedding_func=VIETNAMESE_EMBEDDING_FUNC,
        )
        await rag.initialize_storages()
        _lightrag_instance = rag
        logger.info("LightRAG initialized. working_dir=%s", LIGHTRAG_WORKING_DIR)

    return _lightrag_instance


# ---------------------------------------------------------------------------
# Thinking / waiting sentences
# ---------------------------------------------------------------------------

# Khoảng thời gian (giây) giữa các câu thông báo trong lúc tìm kiếm
THINKING_INTERVAL_SECONDS: float = 10.0

# Độ trễ (giây) trước khi trả LLM response — tạo cảm giác tự nhiên sau khi yield câu chờ
THINKING_RESPONSE_DELAY_SECONDS: float = 0.5

# 3 câu mở đầu — yield ngay khi tool bắt đầu lần đầu tiên trong mỗi lifecycle
THINKING_SENTENCES_START: list[str] = [
    "Tôi đang thực hiện tìm kiếm thông tin, vui lòng chờ trong giây lát.",
    "Tôi sẽ tìm kiếm dữ liệu ngay bây giờ, vui lòng chờ đợi.",
    "Để trả lời chính xác, tôi cần tra cứu dữ liệu, xin vui lòng chờ.",
]

# 5 câu ongoing — yield mỗi THINKING_INTERVAL_SECONDS nếu agent vẫn đang xử lý
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


def pick_thinking_sentence() -> str:
    """Backward-compatible: picks from the combined pool."""
    return random.choice(THINKING_SENTENCES_START + THINKING_SENTENCES_ONGOING)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    hop_count: int           # number of tool-execution rounds completed
    thinking_streamed: bool  # bookkeeping used by the streaming server


# ---------------------------------------------------------------------------
# Shared input schema for all RAG tools
# ---------------------------------------------------------------------------

class _SearchInput(BaseModel):
    query: str


# ---------------------------------------------------------------------------
# RAG Tools — ChromaDB (unchanged)
# ---------------------------------------------------------------------------

def law_rag_query(question: str, k: int = 3):
    result = _law_collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas"],
    )

    answers = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    for meta, text in zip(metadatas, documents):
        answers.append({
            "text": text,
            "article": meta.get("article", ""),
            "clause": meta.get("clause", ""),
            "article_full": meta.get("article_full", ""),
            "clause_full": meta.get("clause_full", ""),
            "source": meta.get("source", meta.get("source_file", ""))
        })

    return answers


def admission_rag_query(question: str, k: int = 3):
    result = _admission_collection.query(
        query_texts=[question],
        n_results=k,
    )

    answers = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    for meta, text in zip(metadatas, documents):
        answers.append({
            "text": text,
            "metadata": meta
        })

    return answers


async def _run_search_law(query: str) -> str:
    loop = asyncio.get_event_loop()
    try:
        results: list[dict] = await loop.run_in_executor(
            None, lambda: law_rag_query(query, k=5)
        )
    except Exception as exc:
        logger.error("law_rag_query error: %s", exc)
        return f"Retrieval error: {exc}"

    if not results:
        return "Không tìm thấy thông tin pháp luật phù hợp."

    parts: list[str] = []

    for i, item in enumerate(results, 1):
        text = item.get("text", "")
        article = item.get("article", "")
        clause = item.get("clause", "")
        source = item.get("source", "")
        clause_full = item.get("clause_full", "")

        parts.append(
            f"[Kết quả {i}]\n"
            f"Văn bản: {source}\n"
            f"Điều: {article}\n"
            f"Khoản: {clause}\n\n"
            f"Nội dung:\n{text}\n\n"
            f"Toàn văn khoản:\n{clause_full}"
        )

    logger.info("===== LAW RETRIEVAL RESULT =====")
    for i, item in enumerate(results, 1):
        logger.info(
            "Rank %s | Source: %s | Article: %s | Clause: %s",
            i,
            item.get("source"),
            item.get("article"),
            item.get("clause"),
        )
    logger.info("================================")
    return "\n\n-------------------\n\n".join(parts)


async def _run_search_admission(query: str) -> str:
    loop = asyncio.get_event_loop()
    logger.info(query)
    try:
        results: list[dict] = await loop.run_in_executor(
            None,
            lambda: admission_rag_query(query, k=5)
        )
    except Exception as exc:
        logger.error("admission_rag_query error: %s", exc)
        return f"Retrieval error: {exc}"

    if not results:
        return "Không tìm thấy thông tin tuyển sinh phù hợp."

    parts: list[str] = []
    for i, item in enumerate(results, 1):
        text = item.get("text", "")
        meta = item.get("metadata", {})
        parts.append(
            f"[Kết quả {i}]\n"
            f"Metadata: {json.dumps(meta, ensure_ascii=False)}\n\n"
            f"Nội dung:\n{text}"
        )

    return "\n\n-------------------\n\n".join(parts)


# ---------------------------------------------------------------------------
# Query Rewriter — dedicated LLM for converting natural language → retrieval keywords
# ---------------------------------------------------------------------------

_REWRITER_SYSTEM_PROMPT = """\
Bạn là công cụ chuyên biệt: chuyển câu hỏi tự nhiên thành keyword query \
tối ưu cho hệ thống tìm kiếm đồ thị tri thức pháp luật (LightRAG).

NHIỆM VỤ: Tách bỏ ngữ cảnh cá nhân, giữ nguyên nội dung pháp lý + loại thông tin cần tìm.

QUY TẮC:
1. Chỉ trả về keyword query, KHÔNG giải thích, KHÔNG câu hoàn chỉnh.
2. Query ngắn gọn: 5–15 từ.
3. GIỮ NGUYÊN mọi thực thể pháp lý trong câu hỏi gốc:
   - Tên đối tượng cụ thể: loại phương tiện, loại hàng hóa, loại người, tên tổ chức...
   - Hành vi vi phạm hoặc chủ đề cần tra cứu
   - Số tiền, số ngày, mức độ nếu có
   Ví dụ: "ô tô", "xe máy", "doanh nghiệp", "người lao động", "hợp đồng lao động" đều là thực thể cần giữ.
4. Xác định loại thông tin cần tìm từ ý định câu hỏi, thêm từ khóa tương ứng:
   - Hỏi về phạt / xử phạt / hình phạt / bị gì / hậu quả → thêm "mức phạt tiền xử phạt hành chính"
   - Hỏi về thủ tục / cách làm / quy trình → thêm "thủ tục"
   - Hỏi về điều kiện / tiêu chuẩn / yêu cầu → thêm "điều kiện"
   - Hỏi về quyền / nghĩa vụ / được làm gì → thêm "quyền nghĩa vụ"
   - Hỏi về hợp đồng / tranh chấp dân sự → thêm "dân sự"
   - Hỏi về tội phạm / truy cứu hình sự → thêm "hình sự"
5. KHÔNG giữ lại: "tôi", "của tôi", "hôm nay", "lỡ", "nghiêm túc", câu chuyện cá nhân.
6. KHÔNG thêm từ khóa không có trong câu hỏi gốc ngoài các từ khóa loại thông tin ở bước 4.

VÍ DỤ (minh họa nguyên tắc, KHÔNG phải danh sách cần học thuộc):
Input:  "tôi lỡ đi xe ô tô vượt đèn đỏ, sẽ bị xử phạt như nào"
Output: ô tô vượt đèn đỏ mức phạt tiền xử phạt hành chính

Input:  "công ty tôi chậm đóng bảo hiểm xã hội cho nhân viên thì bị gì"
Output: doanh nghiệp chậm đóng bảo hiểm xã hội mức phạt tiền xử phạt hành chính

Input:  "hợp đồng lao động không có thời hạn thì chấm dứt như thế nào"
Output: chấm dứt hợp đồng lao động không xác định thời hạn điều kiện thủ tục

Input:  "muốn mở cửa hàng kinh doanh thực phẩm cần giấy tờ gì"
Output: điều kiện kinh doanh thực phẩm giấy phép thủ tục

Input:  "người nước ngoài mua nhà tại Việt Nam được không"
Output: người nước ngoài mua nhà bất động sản Việt Nam điều kiện quyền

Chỉ trả về keyword query, không có gì khác.\
"""

_rewriter_llm: Optional[ChatOpenAI] = None


async def _rewrite_query_for_lightrag(natural_query: str) -> str:
    """Call the rewriter LLM to convert natural language → retrieval keywords."""
    if _rewriter_llm is None:
        logger.warning("Rewriter LLM not initialized, using original query.")
        return natural_query
    response = await _rewriter_llm.ainvoke([
        SystemMessage(content=_REWRITER_SYSTEM_PROMPT),
        HumanMessage(content=natural_query),
    ])
    rewritten = response.content.strip()
    logger.info(
        "===== QUERY REWRITE =====\nOriginal : %s\nRewritten: %s\n=========================",
        natural_query, rewritten,
    )
    return rewritten


# ---------------------------------------------------------------------------
# RAG Tool — LightRAG hybrid search  ← NEW
# ---------------------------------------------------------------------------



async def _run_search_lightrag(query: str) -> str:
    """
    Query LightRAG with local mode (graph neighborhood + vector search).

    Why local instead of hybrid:
    - hybrid = local + global + naive → 3 internal LLM calls, slow (~8-15s)
    - local = 1 internal LLM call, typically sufficient for specific legal Q&A
    - top_k reduced from default 60 → 20 to shrink context and speed up synthesis

    Query is sent as-is (no appended instructions) to preserve embedding accuracy.
    Citation formatting is handled by the outer agent system prompt.
    """
    # Always rewrite through dedicated LLM before hitting LightRAG.
    # This ensures query embeds into the correct subgraph regardless of how
    # the agent phrased the tool call.
    effective_query = await _rewrite_query_for_lightrag(query)

    logger.info("===== LIGHTRAG INPUT QUERY =====\n%s\n================================", effective_query)

    try:
        rag = await get_lightrag()
        answer: str = await rag.aquery(
            effective_query,
            param=QueryParam(mode="mix", top_k=20, user_prompt = "Hãy tìm thông tin chi tiết về truy vấn. Lưu ý thông tin phải chính xác tuyệt đối và bám sát truy vấn, không được trả về câu trả lời sai lệch"),
        )
        if not answer or not answer.strip():
            return "Không tìm thấy thông tin phù hợp trong kho dữ liệu."
        logger.info("===== LIGHTRAG RESULT =====\n%s\n=====", answer)
        return answer
    except Exception as exc:
        logger.error("LightRAG query error: %s", exc)
        return f"Lỗi truy xuất LightRAG: {exc}"


search_lightrag = StructuredTool.from_function(
    coroutine=_run_search_lightrag,
    name="search_lightrag",
    description=(
        "Tìm kiếm thông tin chuyên sâu bằng đồ thị tri thức kết hợp tìm kiếm vector (hybrid). "
        "Dùng khi cần tra cứu văn bản pháp luật, điều khoản, quy định, mức phạt, xử phạt hành chính. "
        "QUAN TRỌNG: tham số query PHẢI là nguyên văn câu hỏi gốc của người dùng, "
        "KHÔNG được tóm tắt, paraphrase hay rút gọn — hệ thống sẽ tự tối ưu hóa bên trong."
    ),
    args_schema=_SearchInput,
)


# ---------------------------------------------------------------------------
# Tour domain tools (kept from original agent.py)
# ---------------------------------------------------------------------------

TOUR_COLLECTION_NAME = "tours"

_tour_collection = _client.get_or_create_collection(
    name=TOUR_COLLECTION_NAME,
    embedding_function=_law_embedding_fn,
)


class _TourSearchInput(BaseModel):
    query: str
    tour_type: Optional[str] = None


class _TourDetailInput(BaseModel):
    tour_id: str


def _tour_detail_prose(summary_meta: dict, day_chunks: list) -> str:
    lines = []
    name = summary_meta.get("name", "")
    tour_id = summary_meta.get("point", "")
    duration = summary_meta.get("duration", "")
    price = summary_meta.get("price", "")
    departure = summary_meta.get("departure", "")
    transport = summary_meta.get("transport", "")
    highlights = summary_meta.get("highlights", "")
    includes = summary_meta.get("includes", "")
    excludes = summary_meta.get("excludes", "")
    cancellation = summary_meta.get("cancellation", "")
    children_policy = summary_meta.get("children_policy", "")
    payment = summary_meta.get("payment", "")

    lines.append(f"Tên tour: {name} (mã số {tour_id})")
    if duration:
        lines.append(f"Thời gian: {duration}")
    if price:
        lines.append(f"Giá: {price}")
    if departure:
        lines.append(f"Khởi hành từ: {departure}")
    if transport:
        lines.append(f"Phương tiện: {transport}")
    if highlights:
        lines.append(f"Điểm nổi bật: {highlights}")
    if includes:
        lines.append(f"Bao gồm: {includes}")
    if excludes:
        lines.append(f"Không bao gồm: {excludes}")

    if day_chunks:
        lines.append("\nLịch trình:")
        for doc, meta in day_chunks:
            day_label = meta.get("day_label", meta.get("clause", ""))
            lines.append(f"  {day_label}: {doc}")

    if cancellation:
        lines.append(f"\nChính sách hủy: {cancellation}")
    if children_policy:
        lines.append(f"Chính sách trẻ em: {children_policy}")
    if payment:
        lines.append(f"Thanh toán: {payment}")

    return "\n".join(lines)


async def _run_search_tours(query: str, tour_type: Optional[str] = None) -> str:
    loop = asyncio.get_event_loop()
    where_filter = {"clause": {"$eq": "summary"}}
    if tour_type:
        where_filter = {
            "$and": [
                {"clause": {"$eq": "summary"}},
                {"tour_type": {"$eq": tour_type}},
            ]
        }

    try:
        results = await loop.run_in_executor(
            None,
            lambda: _tour_collection.query(
                query_texts=[query],
                n_results=5,
                where=where_filter,
                include=["documents", "metadatas"],
            ),
        )
    except Exception as exc:
        logger.error("tour search error: %s", exc)
        return f"Lỗi tìm kiếm tour: {exc}"

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not metadatas:
        return "Không tìm thấy tour phù hợp."

    lines = []
    for meta in metadatas:
        name = meta.get("name", "")
        tour_id = meta.get("point", "")
        duration = meta.get("duration", "")
        price = meta.get("price", "")
        departure = meta.get("departure", "")
        lines.append(f"{name} — mã số {tour_id} — {duration} — {price} — khởi hành từ {departure}")

    return "\n".join(lines)


async def _run_get_tour_detail(tour_id: str) -> str:
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None,
            lambda: _tour_collection.get(
                where={"point": {"$eq": tour_id}},
                include=["documents", "metadatas"],
            ),
        )
    except Exception as exc:
        logger.error("tour detail error: %s", exc)
        return f"Lỗi truy vấn chi tiết tour: {exc}"

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    if not metadatas:
        return f"Không tìm thấy tour với mã số {tour_id}."

    summary_meta: dict | None = None
    day_chunks: list = []

    for doc, meta in zip(documents, metadatas):
        clause = meta.get("clause", "")
        if clause == "summary":
            summary_meta = meta
        elif clause.startswith("day_"):
            day_chunks.append((doc, meta))

    if not summary_meta:
        return f"Không tìm thấy thông tin tổng quan của tour {tour_id}."

    day_chunks.sort(key=lambda x: int(x[1].get("clause", "day_0").replace("day_", "") or 0))

    return _tour_detail_prose(summary_meta, day_chunks)


# ---------------------------------------------------------------------------
# StructuredTool wrappers
# ---------------------------------------------------------------------------

search_law = StructuredTool.from_function(
    coroutine=_run_search_law,
    name="search_law",
    description=(
        "Tìm kiếm văn bản pháp luật, điều khoản, quy định trong ChromaDB. "
        "Dùng khi câu hỏi liên quan đến luật, nghị định, thông tư, xử phạt, v.v. "
        "Tham số: query (câu hỏi hoặc từ khóa pháp luật)."
    ),
    args_schema=_SearchInput,
)

search_admission = StructuredTool.from_function(
    coroutine=_run_search_admission,
    name="search_admission",
    description=(
        "Tìm kiếm thông tin tuyển sinh, điểm chuẩn, chỉ tiêu trong ChromaDB. "
        "Tham số: query (câu hỏi hoặc từ khóa tuyển sinh)."
    ),
    args_schema=_SearchInput,
)

search_tours = StructuredTool.from_function(
    coroutine=_run_search_tours,
    name="search_tours",
    description=(
        "Tìm kiếm danh sách tour du lịch theo khu vực hoặc từ khóa. "
        "Dùng khi khách hỏi có tour nào, tour miền nào, điểm đến cụ thể, "
        "tour mấy ngày, giá khoảng bao nhiêu. "
        "Tham số: query (từ khóa), tour_type (tuỳ chọn: tour miền bắc/nam/trung)."
    ),
    args_schema=_TourSearchInput,
)

get_tour_detail = StructuredTool.from_function(
    coroutine=_run_get_tour_detail,
    name="get_tour_detail",
    description=(
        "Lấy thông tin chi tiết của một tour theo mã tour (tour_id). "
        "Dùng khi khách hỏi về lịch trình, giá theo ngày khởi hành, "
        "dịch vụ, chính sách hủy, trẻ em, thanh toán của một tour cụ thể. "
        "Sau khi gọi thành công, ghi nhớ đây là tour đang tư vấn cho đến khi khách đổi tour."
    ),
    args_schema=_TourDetailInput,
)

# ---------------------------------------------------------------------------
# Master tool list
# ---------------------------------------------------------------------------

RAG_TOOLS: list[BaseTool] = [
    search_law,
    search_admission,
    search_tours,
    get_tour_detail,
    search_lightrag,   # ← LightRAG hybrid search
]


# ---------------------------------------------------------------------------
# System prompt  (auto-lists all registered tools)
# ---------------------------------------------------------------------------

def _build_system_prompt(tools: list[BaseTool]) -> str:
    tool_lines = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in tools)
    current_date = datetime.now().strftime("%d/%m/%Y")
    return f'''Bạn là một trợ lý giọng nói tiếng Việt chuyên tư vấn và nghiên cứu pháp lý, dữ liệu chuyên ngành.

Ngày hiện tại là {current_date} theo định dạng ngày tháng năm.

=====================
VAI TRÒ VÀ PHONG CÁCH
=====================

- Luôn trả lời bằng tiếng Việt.
- Câu trả lời phải là văn bản thuần, tự nhiên như đang nói trực tiếp.
- Không dùng markdown, bullet list, emoji hoặc ký tự đặc biệt.
- Không dùng chữ số, mọi con số phải viết bằng chữ.
- Chỉ trả về nội dung trả lời, không giải thích thêm về cách bạn suy nghĩ.
- Mỗi câu phải có ít nhất bốn từ.
- Các cụm từ viết tắt phải viết IN HOA.


=====================
QUY TẮC SỬ DỤNG CÔNG CỤ
=====================

Các công cụ hiện có:
{tool_lines}

1. Nếu câu hỏi yêu cầu dữ liệu chuyên ngành lưu trong hệ thống
   như luật, điểm chuẩn, chỉ tiêu, học phí, điều khoản pháp lý, dữ liệu cấu trúc,
   bạn BẮT BUỘC phải sử dụng công cụ phù hợp TRƯỚC KHI trả lời.

2. Với câu hỏi pháp luật, mức phạt, xử phạt hành chính, điều khoản cụ thể,
   quan hệ giữa các văn bản pháp luật, hãy dùng search_lightrag.
   Khi gọi search_lightrag, tham số query BẮT BUỘC phải là NGUYÊN VĂN lời người dùng vừa nói,
   KHÔNG được diễn đạt lại, tóm tắt hay rút gọn dù chỉ một từ.

3. Khi đã sử dụng công cụ, bạn PHẢI trả lời NGHIÊM NGẶT dựa trên dữ liệu truy xuất được.
   TUYỆT ĐỐI không được bổ sung, ước đoán hoặc suy diễn bất kỳ con số nào ngoài dữ liệu.

4. Nếu kết quả truy xuất chưa chứa thông tin trực tiếp trả lời câu hỏi,
   bạn phải thử truy xuất lại trước khi kết luận không tìm thấy.

   Các cách truy xuất lại bao gồm:
   - Viết lại câu truy vấn cụ thể hơn.
   - Mở rộng hoặc diễn đạt lại từ khóa.
   - Tăng phạm vi truy xuất (ví dụ: lấy nhiều kết quả hơn).
   - Chia câu hỏi thành nhiều phần và truy xuất từng phần.
   - Gọi lại công cụ với truy vấn đầy đủ và rõ ràng hơn.
   - Nếu search_law không cho kết quả tốt, thử search_lightrag và ngược lại.

   Bạn được phép gọi công cụ nhiều lần cho đến khi:
   - Tìm được thông tin phù hợp, hoặc
   - Đã thử truy xuất lại mà vẫn không có kết quả liên quan.

5. Chỉ khi đã thử truy xuất lại mà vẫn không có thông tin phù hợp,
   bạn mới được phép trả lời:
   "Không tìm thấy thông tin phù hợp trong dữ liệu hiện có."

6. Câu hỏi về tuyển sinh (học phí, điểm chuẩn, chỉ tiêu, ngành học, mã ngành)
   KHÔNG BAO GIỜ là "kiến thức phổ thông" — LUÔN LUÔN phải dùng search_admission.
   TUYỆT ĐỐI không được trả lời câu hỏi tuyển sinh bằng hiểu biết chung hay ước tính.

7. Chỉ những câu hỏi hoàn toàn không liên quan đến số liệu, dữ liệu cụ thể
   (ví dụ: giải thích khái niệm chung chung) mới được trả lời bằng hiểu biết chung.

8. Không được đoán, ước tính, hoặc suy diễn bất kỳ số liệu nào.

9. Không được suy luận dựa trên năm tương tự hoặc đối tượng tương tự.

10. Với dữ liệu số, phải giữ nguyên nội dung theo kết quả truy xuất, không được làm tròn
    hoặc diễn giải lại con số.

=====================
QUY TẮC TRÍCH DẪN
=====================

Mọi thông tin pháp luật PHẢI có trích dẫn nguồn rõ ràng ngay sau nội dung đó.

Định dạng trích dẫn bắt buộc:
- Văn bản luật: theo <Tên luật/nghị định/thông tư>, <Điều X>, <Khoản Y> (nếu có)
- Ví dụ: theo Luật Giao thông đường bộ, Điều ba mươi bảy, Khoản một
- Ví dụ: theo Nghị định một trăm linh tám năm hai nghìn không trăm mười ba, Điều năm

Quy tắc:
- Trích dẫn phải đặt ngay sau luận điểm, không gom tất cả về cuối.
- Nếu nhiều điều khoản hỗ trợ một luận điểm, trích dẫn tất cả cách nhau bằng dấu chấm phẩy.
- Nếu dữ liệu truy xuất không cung cấp số điều khoản cụ thể, ghi rõ tên văn bản mà không bịa số điều.
- Tuyệt đối không bịa hoặc suy đoán số điều, khoản khi không có trong dữ liệu.

=====================
QUY TẮC TƯ VẤN TUYỂN SINH — BẮT BUỘC MULTI-HOP
=====================

Mọi câu hỏi tuyển sinh PHẢI thực hiện đầy đủ các bước sau:

BƯỚC 1 — Dùng search_admission để truy xuất dữ liệu liên quan (học phí, điểm chuẩn, v.v.)

BƯỚC 2 — Phân tích kết quả dựa trên dữ liệu thực tế:
  - So sánh chính xác các con số trong dữ liệu với điều kiện người hỏi đưa ra.
  - Nếu câu hỏi về học phí: so sánh học phí thực tế với ngân sách người hỏi.
  - Nếu KHÔNG có ngành nào đáp ứng điều kiện → phải trả lời trung thực là không có ngành nào phù hợp,
    đồng thời thông báo mức học phí thực tế thấp nhất của trường để người hỏi tham khảo.

BƯỚC 3 — Nếu kết quả chứa MÃ NGÀNH mà CHƯA có TÊN NGÀNH đầy đủ,
  BẮT BUỘC phải gọi thêm search_admission để lấy tên đầy đủ từng mã ngành
  (ví dụ: query "ngành CN1 tên là gì", "CN17 tên ngành", "CN19 học phí").
  Thực hiện cho TẤT CẢ các mã ngành cần hiển thị trong câu trả lời.

BƯỚC 4 — Chỉ sau khi đã có đủ TÊN NGÀNH và SỐ LIỆU CHÍNH XÁC, mới tổng hợp và trả lời.

QUY TẮC QUAN TRỌNG:
- TUYỆT ĐỐI không được bịa, ước tính hoặc suy diễn bất kỳ số học phí, điểm chuẩn nào.
- TUYỆT ĐỐI không được trả lời với mã ngành thuần túy (CN1, CN15...) mà không kèm tên ngành.
- TUYỆT ĐỐI không được liệt kê ngành "phù hợp" khi học phí thực tế VƯỢT QUÁ ngân sách.
- Luôn viết đầy đủ: "Ngành <Tên ngành đầy đủ> (mã <CN...>) có học phí <số tiền> mỗi năm".
- Ví dụ đúng khi không có ngành phù hợp:
  "Với mức ngân sách ba mươi lăm triệu đồng mỗi năm, hiện tại không có ngành nào tại trường
   có học phí trong mức này. Mức học phí thấp nhất của trường là ba mươi tám triệu đồng mỗi năm,
   áp dụng cho ngành Công nghệ vật liệu và Vi điện tử (mã CN mười chín).""

=====================
QUY TẮC TƯ VẤN TOUR
=====================

Khi công cụ search_tours trả về kết quả:

- Đọc nguyên văn từng dòng cho khách nghe.
- Không được bỏ cụm "mã số ...".
- Không được chỉnh sửa nội dung từng dòng.
- Không dùng bullet, markdown hay emoji.

Sau khi liệt kê xong, nói:
"Bạn muốn biết thêm về tour nào, vui lòng cho biết mã số tour hoặc tên tour."

Khi khách nói mã số:
- Gọi công cụ get_tour_detail.

Khi khách nói tên tour:
- Gọi search_tours để xác định rồi gọi get_tour_detail.

Sau khi đã tư vấn chi tiết một tour:
- Các câu hỏi tiếp theo về tour đó không cần gọi lại công cụ,
  trừ khi người dùng yêu cầu thông tin mới ngoài dữ liệu đã có.'''


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

MAX_HOPS = 6


def _build_graph(llm_with_tools, tools: list[BaseTool], system_prompt: str):
    tool_node = ToolNode(tools)

    async def call_model(state: AgentState) -> dict:
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response], "hop_count": state["hop_count"]}

    async def run_tools(state: AgentState) -> dict:
        result = await tool_node.ainvoke(state)
        return {
            **result,
            "hop_count": state["hop_count"] + 1,
            "thinking_streamed": False,
        }

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
        "agent", should_continue, {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_agent(
    extra_tools: Optional[list[BaseTool]] = None,
    model: str = "gpt-4o-mini",
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    temperature: float = 0.0,
):
    """
    Create and return the compiled LangGraph agent.

    Parameters
    ----------
    extra_tools : list[BaseTool] | None
        Additional tools to inject alongside the RAG tools defined in this file.
    model : str
        OpenAI-compatible model name.
    openai_api_key : str | None
        Falls back to OPENAI_API_KEY env var.
    openai_base_url : str | None
        Custom LLM backend URL.
    temperature : float
        LLM sampling temperature.
    """
    tools: list[BaseTool] = RAG_TOOLS + list(extra_tools or [])

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=openai_base_url,
        streaming=True,
    )

    # Rewriter LLM: same model, temperature=0 for deterministic keyword output
    global _rewriter_llm
    _rewriter_llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=openai_base_url,
        streaming=False,  # rewriter output is consumed whole, not streamed
    )

    llm_with_tools = llm.bind_tools(tools)
    system_prompt = _build_system_prompt(tools)

    return _build_graph(llm_with_tools, tools, system_prompt)


# ---------------------------------------------------------------------------
# Thinking stream — yields interaction sentences while agent is working
# ---------------------------------------------------------------------------

async def stream_thinking_while_running(agent_task):
    """
    Async generator: yield câu chờ tiếng Việt trong khi agent đang chạy,
    sau đó đợi thêm THINKING_RESPONSE_DELAY_SECONDS trước khi kết thúc
    để tạo khoảng nghỉ tự nhiên trước khi LLM response được gửi đi.

    Yield behaviour:
    - Yield ngay lập tức câu đầu tiên khi tool bắt đầu chạy (không delay).
    - Cứ mỗi THINKING_INTERVAL_SECONDS yield thêm một câu nếu task chưa xong.
    - Khi task xong, đợi THINKING_RESPONSE_DELAY_SECONDS rồi mới kết thúc generator.

    Usage in streaming server::

        task = asyncio.create_task(agent.ainvoke(initial_state))
        async for sentence in stream_thinking_while_running(task):
            # send sentence to client ngay khi nhận được
            ...
        # generator kết thúc -> đã qua delay, gửi LLM response luôn
        result = await task
    """
    # Yield ngay — agent vừa bắt đầu dùng tool
    yield pick_thinking_start_sentence()

    task_done = False
    while not task_done:
        try:
            await asyncio.wait_for(
                asyncio.shield(agent_task),
                timeout=THINKING_INTERVAL_SECONDS,
            )
            # Task hoàn thành trong khoảng chờ
            task_done = True
        except asyncio.TimeoutError:
            # Vẫn còn đang chạy — yield câu tiếp theo
            if not agent_task.done():
                yield pick_thinking_ongoing_sentence()
            else:
                task_done = True

    # Khoảng nghỉ tự nhiên trước khi trả response —
    # giúp tránh cảm giác response xuất hiện ngay lập tức sau câu chờ cuối
    await asyncio.sleep(THINKING_RESPONSE_DELAY_SECONDS)