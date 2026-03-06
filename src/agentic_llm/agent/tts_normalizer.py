"""
tts_normalizer.py — LLM-based TTS Text Normalization Agent for Vietnamese.

Mỗi domain có system prompt riêng, gọn và tập trung vào đặc thù của domain đó.
Class TTSNormalizerAgent nhận domain param để chọn đúng prompt khi normalize.

Domains: law | admission | tour | normal_talk
Fallback: nếu domain không xác định → dùng prompt chung (minimal)

Usage:
    agent = TTSNormalizerAgent(domain="law")
    clean = await agent.anormalize("Phạt 18.000.000 đồng theo NĐ-CP.")

    agent = TTSNormalizerAgent(domain="tour")
    clean = await agent.anormalize("TOUR PHÚ QUỐC 3N2Đ | GRAND WORLD - VINWONDERS")
"""

from __future__ import annotations
import re
import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)


# ============================================================
# Shared: regex post-processor (chạy sau LLM, đánh vần ký tự
# viết tắt còn sót lại dạng CHỮ+SỐ hoặc ALL-CAPS)
# ============================================================

LETTER_MAP = {
    "A": "a", "B": "bê", "C": "xê", "D": "đê", "E": "e",
    "F": "ép", "G": "gờ", "H": "hát", "I": "i", "J": "di",
    "K": "ca", "L": "lờ", "M": "mờ", "N": "nờ", "O": "o",
    "P": "pê", "Q": "quy", "R": "rờ", "S": "ét", "T": "tê",
    "U": "u", "V": "vê", "W": "vê kép", "X": "ích",
    "Y": "i", "Z": "dét",
}

DIGIT_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba",
    "4": "bốn", "5": "năm", "6": "sáu",
    "7": "bảy", "8": "tám", "9": "chín",
}


# Tập ký tự chữ thường tiếng Việt có dấu — dùng cho boundary lookbehind/lookahead
_VI_LOWER = "a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"

# Match token bắt đầu bằng CHỮ HOA, tiếp theo là CHỮ HOA hoặc SỐ (≥0 ký tự).
# Boundary Unicode-aware: không match nếu đứng liền chữ thường/hoa/số khác.
_ABBR_RE = re.compile(
    rf"(?<![{_VI_LOWER}A-Z0-9\-])"   # không đứng sau chữ/số/dấu gạch ngang
    r"[A-Z][A-Z0-9]*"                 # token: chữ HOA đầu + CHỮ HOA/SỐ tiếp theo
    rf"(?![{_VI_LOWER}ĐA-Z0-9\-])",  # không đứng trước chữ/số/Đ/dấu gạch ngang
    re.UNICODE,
)


def _is_code_token(token: str) -> bool:
    """Token có cả chữ lẫn số xen kẽ → mã kỹ thuật (A1, B2, CN12, QH2026...) → giữ nguyên."""
    return bool(re.search(r"[A-Z]", token)) and bool(re.search(r"[0-9]", token))


def _spell_token(token: str) -> str:
    """Đánh vần từng chữ cái của token ALL-CAPS thuần (GPLX, SAT, HCM, IELTS...)."""
    return " ".join(LETTER_MAP.get(ch, ch) for ch in token)


def _regex_post_process(text: str) -> str:
    """
    Đánh vần các token ALL-CAPS còn sót sau LLM, giữ nguyên mã CHỮ+SỐ.

    Cải thiện so với pattern cũ:
    - Boundary dùng lookahead/lookbehind Unicode → hoạt động đúng sau dấu câu (, . ;)
      Ví dụ: 'A1,'  → A1 giữ nguyên  |  '.IELTS,' → IELTS đánh vần
    - Token CHỮ+SỐ xen kẽ (A1, B2, CN12, QH2026) → giữ nguyên
    - Token chỉ chữ hoa thuần (GPLX, SAT, IELTS, HCM) → đánh vần
    - Đơn chữ cái đứng độc lập (A, B, C) → đánh vần
    """
    def _replacer(m: re.Match) -> str:
        token = m.group(0)
        return token if _is_code_token(token) else _spell_token(token)

    return _ABBR_RE.sub(_replacer, text)


# ============================================================
# Shared rules (nhúng vào mọi prompt)
# ============================================================

_SHARED_RULES = """\
QUY TẮC CHUNG (BẮT BUỘC):

- Chỉ trả về văn bản đã chuẩn hóa.
- Không giải thích.
- Không thêm hoặc bớt nội dung.
- Luôn trả lời bằng tiếng Việt.
- Không sử dụng markdown hay ký hiệu đặc biệt.

==================================================
XỬ LÝ VIẾT TẮT — ƯU TIÊN CAO NHẤT
==================================================

- Với mọi từ viết tắt ALL-CAPS:
  → Tra bảng viết tắt của domain TRƯỚC.
  → Nếu có trong bảng: dùng đúng giá trị trong bảng.
  → Nếu KHÔNG có trong bảng: giữ nguyên.
- TUYỆT ĐỐI không tự đánh vần khi đã có trong bảng.
- Mã dạng CHỮ+SỐ (CN12, A1, QH2026...) → giữ nguyên.

==================================================
XỬ LÝ DẤU CÂU
==================================================

- Mỗi câu phải kết thúc bằng dấu chấm.
- Không dùng dấu chấm than hoặc dấu chấm hỏi.
- Mỗi câu chỉ được chứa một ý chính.
- Mỗi câu tối đa một dấu phẩy.
- Không dùng dấu phẩy để nối hai ý lớn.

- Khi liệt kê nhiều thông tin cùng loại:
  → BẮT BUỘC dùng liên từ tự nhiên như:
    "cụ thể là", "và", "lần lượt là".
  → Không liệt kê khô chỉ bằng dấu phẩy.
  → Ưu tiên:
      + Một câu giới thiệu nhóm thông tin.
      + Giữa các nhóm thông tin có thể liệt kê như: "Thứ nhất là", "Thứ hai là", "Thứ ba là", ... Kết thúc có thể bằng "Cuối cùng là"

- Dấu hai chấm sau nhãn → thay bằng dấu phẩy.
- Dấu hai chấm dạng "X: Y" → thay bằng "là".
- Dấu | → thay bằng dấu phẩy.
- Dấu " - " giữa các mục → thay bằng dấu phẩy.
- Bullet list → chuyển thành câu hoàn chỉnh.
- Xuống dòng giữa câu → chuyển thành dấu chấm.

==================================================
QUY TẮC NGẮT CÂU — RẤT QUAN TRỌNG
==================================================

BẮT BUỘC tách câu khi:

- Xuất hiện mốc năm khác.
- Xuất hiện chủ đề mới.
- Xuất hiện phép so sánh giữa hai đối tượng.
- Xuất hiện liệt kê nhiều hơn hai giá trị.
- Câu dài và chứa hai mệnh đề độc lập.
- Câu dài quá 20 từ

ĐÚNG:
"Năm hai nghìn không trăm hai mươi bốn, điểm chuẩn là hai mươi bảy.
Năm hai nghìn không trăm hai mươi lăm, điểm chuẩn là hai mươi tám."

SAI:
"Năm hai nghìn không trăm hai mươi bốn, điểm chuẩn là hai mươi bảy, năm hai nghìn không trăm hai mươi lăm, điểm chuẩn là hai mươi tám."

==================================================
XỬ LÝ TÊN RIÊNG
==================================================

- Tên viết HOA TOÀN BỘ nhưng không phải viết tắt → chuyển sang Title Case.
- Không thay đổi tên riêng đã đúng chuẩn.

==================================================
YÊU CẦU CUỐI CÙNG
==================================================

- Văn bản phải tự nhiên khi đọc thành tiếng.
- Câu ngắn, rõ ràng.
- Không nối quá nhiều thông tin trong một câu.
- Nếu có nhiều số liệu trong cùng một nhóm, phải dùng cấu trúc tự nhiên thay vì đọc dạng bảng.
"""

# ============================================================
# Domain prompts
# ============================================================

def _build_law_prompt() -> str:
    return f"""\
Bạn là chuyên gia chuẩn hóa văn bản pháp luật tiếng Việt cho hệ thống TTS.
Nhiệm vụ: chuyển văn bản pháp luật thành dạng đọc được — không còn viết tắt pháp lý, số, ký tự đặc biệt.

{_SHARED_RULES}

QUY TẮC SỐ PHÁP LÝ:
- Số nguyên kiểu VN (dấu chấm ngăn nghìn): 18.000.000 → mười tám triệu; 35.000 → ba mươi lăm nghìn.
- Số thập phân (dấu phẩy): 28,19 → hai mươi tám phẩy mười chín.
- 15 → mười lăm; 25 → hai mươi lăm; 21 → hai mươi mốt. Nhóm thiếu trăm → "không trăm".
- Mã văn bản (số/năm/LOẠI-CQUAN): 168/2024/NĐ-CP → số một trăm sáu mươi tám năm hai nghìn không trăm hai mươi bốn nghị định chính phủ.
- Điều khoản: Điều 6 Khoản 9 Điểm b → Điều sáu Khoản chín Điểm b.

BẢNG VIẾT TẮT PHÁP LÝ (CHỈ CÁC TỪ NÀY MỚI ĐƯỢC EXPAND):
  NĐ-CP → nghị định chính phủ | NĐ → nghị định | QĐ-TTg → quyết định thủ tướng | QĐ → quyết định
  TT-BCA → thông tư bộ công an | TT-BGTVT → thông tư bộ giao thông vận tải | TT → thông tư
  VBHN-VPQH → văn bản hợp nhất văn phòng quốc hội | VBHN → văn bản hợp nhất
  BLHS → bộ luật hình sự | BLDS → bộ luật dân sự | BLTTDS → bộ luật tố tụng dân sự
  BGTVT → bộ giao thông vận tải | BCA → bộ công an | BTC → bộ tài chính | BYT → bộ y tế
  HDND → hội đồng nhân dân | UBND → ủy ban nhân dân | TAND → tòa án nhân dân
  CP → chính phủ | TTg → thủ tướng
  CSGT → cảnh sát giao thông | ATGT → an toàn giao thông | GPLX → giấy phép lái xe
  PCCC → phòng cháy chữa cháy | BVMT → bảo vệ môi trường
  VND → đồng | km/h → ki lô mét trên giờ | % → phần trăm | m2 → mét vuông
  TP.HCM → thành phố hồ chí minh | TP. → thành phố | TP → thành phố | VN → việt nam

VÍ DỤ:
IN:  Theo Nghị định 168/2024/NĐ-CP, phạt từ 18.000.000 đồng đến 20.000.000 đồng.
OUT: Theo Nghị định số một trăm sáu mươi tám năm hai nghìn không trăm hai mươi bốn nghị định chính phủ, phạt từ mười tám triệu đồng đến hai mươi triệu đồng.

IN:  UBND TP.HCM xử phạt vi phạm ATGT từ 800.000 đến 1.000.000 đồng.
OUT: Ủy ban nhân dân thành phố hồ chí minh xử phạt vi phạm an toàn giao thông từ tám trăm nghìn đến một triệu đồng.

IN:  **Lưu ý:** Các hạng A1, A và B1 cần mang GPLX.
OUT: Lưu ý, Các hạng A1, A và B1 cần mang giấy phép lái xe.

IN:  Điều 6 Khoản 9 Điểm b — phạt từ 18.000.000 đến 20.000.000 đồng.
OUT: Điều sáu Khoản chín Điểm b, phạt từ mười tám triệu đến hai mươi triệu đồng.\
"""


def _build_admission_prompt() -> str:
    return f"""\Bạn là chuyên gia chuẩn hóa văn bản tuyển sinh đại học tiếng Việt cho hệ thống TTS.
Nhiệm vụ: chuyển văn bản tuyển sinh thành dạng đọc được — không còn số, viết tắt, ký tự đặc biệt.

{_SHARED_RULES}

QUY TẮC SỐ TUYỂN SINH:
- Số nguyên: 44.000.000 → bốn mươi bốn triệu; 420 → bốn trăm hai mươi; 80 → tám mươi.
- Số thập phân điểm chuẩn (dấu phẩy): 28,19 → hai mươi tám phẩy mười chín.
- Năm: 2025 → hai nghìn không trăm hai mươi lăm.
- 15 → mười lăm; 25 → hai mươi lăm; 21 → hai mươi mốt.
- Mã ngành (CN1, CN12...): đọc là "xê nờ" + số. CN1 → xê nờ một, CN12 → xê nờ mười hai, CN18 → xê nờ mười tám.
- Số điện thoại → đọc từng chữ số theo nhóm: 024 37 547 865 → không hai bốn, ba bảy năm bốn, bảy tám sáu lăm.

BẢNG VIẾT TẮT TUYỂN SINH — TRA BẢNG NÀY TRƯỚC KHI XỬ LÝ BẤT KỲ TỪ VIẾT TẮT NÀO:
  UET → trường đại học công nghệ | VNU → đại học quốc gia hà nội | ĐHQGHN → đại học quốc gia hà nội
  ĐGNL → đánh giá năng lực | THPT → trung học phổ thông
  IELTS → ai eo | TOEIC → tô ách | A-Level → ây le vồ | ACT → ây xi ti | SAT → ét ây ti
  CNTT → công nghệ thông tin | KHMT → khoa học máy tính | KTMT → Kỹ thuật máy tính | ĐTVT → Điện tử viễn thông
  AI → ây ai | IT → ai ti | IoT → ai ô ti
  VND → đồng | % → phần trăm | VND/năm → đồng mỗi năm | VND/tháng → đồng mỗi tháng

VÍ DỤ:
IN:  Điểm chuẩn ngành CN12 năm 2025 là 28,19. Học phí 44.000.000 VND/năm.
OUT: Điểm chuẩn ngành xê nờ mười hai năm hai nghìn không trăm hai mươi lăm là hai mươi tám phẩy mười chín. Học phí bốn mươi bốn triệu đồng mỗi năm.

IN:  Chỉ tiêu CN1: 420, CN12: 80, CN18: 60.
OUT: Chỉ tiêu xê nờ một là bốn trăm hai mươi, xê nờ mười hai là tám mươi, xê nờ mười tám là sáu mươi.

IN:  Điểm chuẩn: IELTS là 6.5, SAT là 1200, ĐGNL là 900.
OUT: Điểm chuẩn, ai eo là sáu phẩy năm, ét ây ti là một nghìn hai trăm, đánh giá năng lực là chín trăm.

IN:  Sinh viên tốt nghiệp có thể làm về IT, AI và IoT.
OUT: Sinh viên tốt nghiệp có thể làm về ai ti, ây ai và ai ô ti."""
def _build_tour_prompt() -> str:
    return f"""\
Bạn là chuyên gia chuẩn hóa văn bản du lịch tiếng Việt cho hệ thống TTS.
Nhiệm vụ: chuyển văn bản tour du lịch thành dạng đọc được — không còn số, viết tắt, ký tự đặc biệt.

{_SHARED_RULES}

QUY TẮC SỐ TOUR:
- Giá tour: 5.990.000 → năm triệu chín trăm chín mươi nghìn.
- Thời lượng: 5N4D / 5N4Đ → năm ngày bốn đêm; 3N2D → ba ngày hai đêm.
- Số nguyên thông thường: 2 → hai; 15 → mười lăm; 25 → hai mươi lăm.
- % → phần trăm; VND → đồng.
- Mã tour dạng CHỮ+SỐ → GIỮ NGUYÊN.

QUY TẮC ĐỊA DANH / THƯƠNG HIỆU:
- Tên viết HOA TOÀN BỘ không phải viết tắt → Title Case.
  Ví dụ: GRAND WORLD → Grand World; VINWONDERS → Vinwonders; CÁP TREO HÒN THƠM → Cáp Treo Hòn Thơm.
- Dấu " - " giữa các địa danh → dấu phẩy.
- buffet → búp phê.

BẢNG VIẾT TẮT TOUR:
  VND → đồng | % → phần trăm | TP.HCM → thành phố hồ chí minh | TP → thành phố | VN → việt nam

VÍ DỤ:
IN:  TOUR PHÚ QUỐC MÙA XUÂN 3N2Đ | GRAND WORLD - VINWONDERS - SAFARI - CÁP TREO HÒN THƠM
OUT: Tour Phú Quốc Mùa Xuân ba ngày hai đêm, Grand World, Vinwonders, Safari, Cáp Treo Hòn Thơm.

IN:  Tour Miền Bắc 5N4Đ | Tam Chúc – Ninh Bình – Hạ Long
OUT: Tour Miền Bắc năm ngày bốn đêm, Tam Chúc, Ninh Bình, Hạ Long.

IN:  Giá tour 5.990.000 VND, bao gồm buffet sáng và vé tham quan.
OUT: Giá tour năm triệu chín trăm chín mươi nghìn đồng, bao gồm búp phê sáng và vé tham quan.\
"""


def _build_normal_talk_prompt() -> str:
    return f"""\
Bạn là chuyên gia chuẩn hóa văn bản hội thoại tiếng Việt cho hệ thống TTS.
Nhiệm vụ: chuyển văn bản hội thoại thành dạng đọc được tự nhiên — không còn ký tự đặc biệt hay markdown.

{_SHARED_RULES}

QUY TẮC HỘI THOẠI:
- Số thông thường đọc thành chữ: 3 → ba; 10 → mười; 2025 → hai nghìn không trăm hai mươi lăm.
- Giữ giọng văn tự nhiên, thân thiện. Không thêm/bớt nội dung.
- Tên riêng viết HOA TOÀN BỘ → Title Case.\
"""


# Map domain → build function
_PROMPT_BUILDERS: dict[str, object] = {
    "law":         _build_law_prompt,
    "admission":   _build_admission_prompt,
    "tour":        _build_tour_prompt,
    "normal_talk": _build_normal_talk_prompt,
}


def build_tts_prompt(domain: str) -> str:
    """Trả về system prompt TTS cho domain tương ứng. Fallback về normal_talk."""
    builder = _PROMPT_BUILDERS.get(domain, _build_normal_talk_prompt)
    return builder()


# ============================================================
# TTSNormalizerAgent — nhận domain param
# ============================================================

class TTSNormalizerAgent:
    """
    LLM-based TTS normalizer với system prompt per-domain.

    Args:
        domain: "law" | "admission" | "tour" | "normal_talk"
                Nếu None hoặc không xác định → dùng normal_talk prompt.
    """

    def __init__(
        self,
        domain: str = "normal_talk",
        model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ):
        self._domain          = domain
        self._model           = model
        self._openai_api_key  = openai_api_key
        self._openai_base_url = openai_base_url
        self._system_prompt   = build_tts_prompt(domain)
        self._llm             = None  # lazy init
        logger.debug("TTSNormalizerAgent init: domain=%s model=%s", domain, model)

    @property
    def domain(self) -> str:
        return self._domain

    def _get_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self._model,
                temperature=0,
                api_key=self._openai_api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self._openai_base_url,
            )
        return self._llm

    def normalize(self, text: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        if not text.strip():
            return text
        try:
            resp = self._get_llm().invoke([
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=text),
            ])
            return _regex_post_process(resp.content.strip())
        except Exception as exc:
            logger.warning("TTS [%s] normalize error, returning original: %s", self._domain, exc)
            return text

    async def anormalize(self, text: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        if not text.strip():
            return text
        try:
            resp = await self._get_llm().ainvoke([
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=text),
            ])
            return _regex_post_process(resp.content.strip())
        except Exception as exc:
            logger.warning("TTS [%s] anormalize error, returning original: %s", self._domain, exc)
            return text

    async def astream_normalize(self, text: str):
        """
        Stream token từ LLM TTS normalizer — yield chunk ngay khi LLM trả về,
        không chờ full response. Giảm latency đáng kể so với anormalize().

        Chiến lược buffer:
        - Gom token vào buffer cho đến khi xuất hiện word-boundary (dấu cách, dấu câu).
        - Yield ngay khi có boundary → client nhận chữ hoàn chỉnh, không bị vỡ giữa từ.
        - _regex_post_process được bỏ qua trong stream (LLM thường không emit ALL-CAPS
          sau normalize); nếu cần có thể áp dụng trên full text sau khi collect.

        Yields:
            str: chunk văn bản đã normalize, sẵn sàng gửi tới client.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        if not text.strip():
            return

        buffer = ""
        # Ranh giới để flush buffer: khoảng trắng hoặc dấu câu
        _BOUNDARIES = {" ", "\n", ".", ",", ";", ":", "!", "?"}

        try:
            async for chunk in self._get_llm().astream([
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=text),
            ]):
                token: str = getattr(chunk, "content", "") or ""
                if not token:
                    continue

                buffer += token

                # Tìm boundary cuối cùng trong buffer để flush phần trước nó
                last_boundary = -1
                for idx in range(len(buffer) - 1, -1, -1):
                    if buffer[idx] in _BOUNDARIES:
                        last_boundary = idx
                        break

                if last_boundary >= 0:
                    to_yield = buffer[: last_boundary + 1]
                    buffer   = buffer[last_boundary + 1 :]
                    yield to_yield

            # Flush phần còn lại (đuôi câu không kết thúc bằng dấu câu)
            if buffer:
                yield buffer

        except Exception as exc:
            logger.warning(
                "TTS [%s] astream_normalize error, falling back word-split: %s",
                self._domain, exc,
            )
            # Fallback: stream từng từ của text gốc (không normalize)
            words = text.split(" ")
            for i, word in enumerate(words):
                yield word if i == len(words) - 1 else word + " "


# ============================================================
# Factory helpers
# ============================================================

def create_tts_normalizers(
    model: str = "gpt-4o-mini",
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
) -> dict[str, TTSNormalizerAgent]:
    """
    Tạo dict {domain: TTSNormalizerAgent} cho tất cả domains.
    Dùng bởi create_router_agent() trong agent_routing.py.
    """
    return {
        domain: TTSNormalizerAgent(
            domain=domain,
            model=model,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
        )
        for domain in _PROMPT_BUILDERS
    }


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import asyncio

    TEST_CASES: list[tuple[str, str]] = [
        ("law",  "Theo Nghị định 168/2024/NĐ-CP, phạt từ 18.000.000 đồng đến 20.000.000 đồng."),
        ("law",  "UBND TP.HCM xử phạt vi phạm ATGT từ 800.000 đến 1.000.000 đồng."),
        ("law",  "**Lưu ý:** Các hạng A1, A và B1 cần mang GPLX."),
        ("admission", "Điểm chuẩn ngành CN12 năm 2025 là 28,19. Học phí 44.000.000 VND/năm."),
        ("admission", "Chỉ tiêu CN1: 420, CN12: 80, CN18: 60."),
        ("admission", "Sinh viên tốt nghiệp có thể làm về IT, AI và IoT."),
        ("tour", "TOUR PHÚ QUỐC MÙA XUÂN 3N2Đ | GRAND WORLD - VINWONDERS - SAFARI"),
        ("tour", "Tour Miền Bắc 5N4Đ | Tam Chúc – Ninh Bình – Hạ Long"),
        ("tour", "Giá tour 5.990.000 VND, bao gồm buffet sáng."),
        ("normal_talk", "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"),
    ]

    async def run():
        agents = create_tts_normalizers()
        for domain, text in TEST_CASES:
            result = await agents[domain].anormalize(text)
            print(f"[{domain:12s}] IN : {text}")
            print(f"             OUT: {result}\n")

    asyncio.run(run())