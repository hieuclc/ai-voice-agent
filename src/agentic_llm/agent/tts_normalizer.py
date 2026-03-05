"""
tts_normalizer.py — LLM-based TTS Text Normalization Agent for Vietnamese.

Toàn bộ normalize do LLM thực hiện, được hướng dẫn bằng system prompt
chứa đầy đủ mapping tables và quy tắc đọc số tiếng Việt.

Usage:
    from tts_normalizer import TTSNormalizerAgent
    agent = TTSNormalizerAgent()
    clean = await agent.anormalize("Phạt 18.000.000 đồng theo NĐ-CP.")
    clean = agent.normalize("Phạt 18.000.000 đồng theo NĐ-CP.")
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
# Mapping tables — nhúng nguyên vào system prompt
# ============================================================

LEGAL_ABBR: dict[str, str] = {
    "NĐ-CP":      "nghị định chính phủ",
    "NĐ":         "nghị định",
    "QĐ-TTg":     "quyết định thủ tướng",
    "QĐ":         "quyết định",
    "TT-BCA":     "thông tư bộ công an",
    "TT-BGTVT":   "thông tư bộ giao thông vận tải",
    "TT-BTC":     "thông tư bộ tài chính",
    "TT-BGDDT":   "thông tư bộ giáo dục và đào tạo",
    "TT":         "thông tư",
    "VBHN-VPQH":  "văn bản hợp nhất văn phòng quốc hội",
    "VBHN-BCA":   "văn bản hợp nhất bộ công an",
    "VBHN-BGTVT": "văn bản hợp nhất bộ giao thông vận tải",
    "VBHN":       "văn bản hợp nhất",
    "VPQH":       "văn phòng quốc hội",
    "BLHS":       "bộ luật hình sự",
    "BLDS":       "bộ luật dân sự",
    "BLTTDS":     "bộ luật tố tụng dân sự",
    "BLTTHS":     "bộ luật tố tụng hình sự",
    "BGTVT":      "bộ giao thông vận tải",
    "BCA":        "bộ công an",
    "BQP":        "bộ quốc phòng",
    "BTC":        "bộ tài chính",
    "BYT":        "bộ y tế",
    "BGDDT":      "bộ giáo dục và đào tạo",
    "HDND":       "hội đồng nhân dân",
    "UBND":       "ủy ban nhân dân",
    "TAND":       "tòa án nhân dân",
    "VKSND":      "viện kiểm sát nhân dân",
    "CP":         "chính phủ",
    "TTg":        "thủ tướng",
    "BCT":        "bộ công thương",
}

ORG_ABBR: dict[str, str] = {
    "UET":    "trường đại học công nghệ",
    "VNU":    "đại học quốc gia hà nội",
    "DHQGHN": "đại học quốc gia hà nội",
    "DHQG":   "đại học quốc gia",
    "VNeID":  "vê em i ai đi",
    "ĐGNL":   "đánh giá năng lực",
    "THPT":   "trung học phổ thông",
    "A-Level": "ây le vồ",
    "ACT":     "ây xi ti",
    "SAT":     "ét ây ti",
    "CNTT":   "công nghệ thông tin",
    "KHMT":   "khoa học máy tính",
    "ĐTVT":   "điện tử viễn thông",
    "KTMT":   "kỹ thuật máy tính",
    "ATTT":   "an toàn thông tin",
    "HTTT":   "hệ thống thông tin",
    "IoT":    "ai ô ti",
    "AI":     "ây ai",
    "IT":     "ai ti",
}

GEO_ABBR: dict[str, str] = {
    "TP.HCM":  "thành phố hồ chí minh",
    "TP. HCM": "thành phố hồ chí minh",
    "TP.HN":   "thành phố hà nội",
    "TP.":     "thành phố",
    "TP":      "thành phố",
    "HCM":     "thành phố hồ chí minh",
    "QL":      "quốc lộ",
    "TL":      "tỉnh lộ",
    "VN":      "việt nam",
}

TRAFFIC_ABBR: dict[str, str] = {
    "CSGT":  "cảnh sát giao thông",
    "ATGT":  "an toàn giao thông",
    "TTGT":  "trật tự giao thông",
    "GPLX":  "giấy phép lái xe",
    "PCCC":  "phòng cháy chữa cháy",
    "BVMT":  "bảo vệ môi trường",
}

UNIT_ABBR: dict[str, str] = {
    "VND":  "đồng",
    "USD":  "đô la mỹ",
    "EUR":  "ơ rô",
    "km/h": "ki lô mét trên giờ",
    "m2":   "mét vuông",
    "km2":  "ki lô mét vuông",
    "cm2":  "xen ti mét vuông",
    "cm3":  "xen ti mét khối",
    "km":   "ki lô mét",
    "cm":   "xen ti mét",
    "kg":   "ki lô gam",
    "%":    "phần trăm",
}


def _build_system_prompt() -> str:
    legal_lines   = "\n".join(f"  {k} -> {v}" for k, v in LEGAL_ABBR.items())
    org_lines     = "\n".join(f"  {k} -> {v}" for k, v in ORG_ABBR.items())
    geo_lines     = "\n".join(f"  {k} -> {v}" for k, v in GEO_ABBR.items())
    traffic_lines = "\n".join(f"  {k} -> {v}" for k, v in TRAFFIC_ABBR.items())
    unit_lines    = "\n".join(f"  {k} -> {v}" for k, v in UNIT_ABBR.items())

    return f"""Bạn là chuyên gia chuẩn hóa văn bản tiếng Việt cho hệ thống text-to-speech (TTS).
Nhiệm vụ: chuyển văn bản đầu vào thành dạng TTS đọc được — không còn ký tự đặc biệt, viết tắt (trừ ngoại lệ bên dưới), số, markdown.
Chỉ trả về văn bản đã chuẩn hóa. Không giải thích, không thêm bất kỳ nội dung nào khác.

══════════════════════════════════════════
1. XÓA MARKDOWN
══════════════════════════════════════════
- **bold**, *italic*, `code`, # heading -> giữ nội dung bên trong, bỏ ký hiệu
- Bullet (-, •, *) và numbered list -> nối các mục bằng dấu phẩy
- Bảng, code block -> xóa hoàn toàn

══════════════════════════════════════════
2. DẤU PHÂN CÁCH
══════════════════════════════════════════
- Dấu | giữa các địa điểm/mục -> dấu phẩy
- Dấu – (en-dash) giữa các từ -> dấu phẩy
- " - " (gạch ngang CÓ khoảng trắng hai bên) -> dấu phẩy
  Ví dụ: "GRAND WORLD - VINWONDERS - SAFARI" -> "Grand World, Vinwonders, Safari"
- Dấu - trong từ ghép (NĐ-CP, km/h, VNeID) -> KHÔNG thay thế
- Dấu ngoặc đơn (...) và vuông [...] -> bỏ ngoặc, giữ nội dung
- Ba chấm (...) -> dấu phẩy

══════════════════════════════════════════
3. TÊN RIÊNG, ĐỊA DANH, THƯƠNG HIỆU VIẾT HOA TOÀN BỘ
══════════════════════════════════════════
Các tên riêng, địa danh, thương hiệu viết HOA TOÀN BỘ mà KHÔNG có trong bảng viết tắt bên dưới
-> chuyển về Title Case (viết hoa chữ cái đầu mỗi từ).

Ví dụ:
  GRAND WORLD   -> Grand World
  VINWONDERS    -> Vinwonders
  SAFARI        -> Safari
  CÁP TREO HÒN THƠM -> Cáp Treo Hòn Thơm
  LOTTE         -> Lotte
  IKEA          -> Ikea

KHÔNG áp dụng quy tắc này cho:
- Các từ viết tắt có trong bảng (UBND, ATGT...) -> expand theo bảng
- Mã dạng chữ+số (xem mục 7)
- Từ viết tắt không có trong bảng (xem mục 8)

══════════════════════════════════════════
4. SỐ -> CHỮ (quy tắc đọc số tiếng Việt)
══════════════════════════════════════════
Số nguyên kiểu VN (dấu chấm ngăn nghìn):
  18.000.000 -> mười tám triệu
  35.000     -> ba mươi lăm nghìn
  2.099.000  -> hai triệu không trăm chín mươi chín nghìn

Số thập phân (dấu phẩy):
  28,19 -> hai mươi tám phẩy mười chín

Quy tắc đọc quan trọng:
  - 15 -> mười lăm  (KHÔNG "mười năm")
  - 25 -> hai mươi lăm  (KHÔNG "hai mươi năm")
  - 21 -> hai mươi mốt  (KHÔNG "hai mươi một")
  - 1500 -> một nghìn năm trăm
  - 0xx đứng sau trăm -> "linh": 1.005 -> một nghìn linh năm
  - Nhóm thiếu trăm ở giữa -> "không trăm": 2.005.000 -> hai triệu không trăm linh năm nghìn

Số điện thoại -> đọc từng chữ số theo nhóm, phân cách bằng dấu phẩy:
  024 37 547 865 -> không hai bốn, ba bảy năm bốn, bảy tám sáu lăm
  0912 345 678   -> không chín một hai, ba bốn lăm, sáu bảy tám

══════════════════════════════════════════
5. MÃ VĂN BẢN PHÁP LÝ
══════════════════════════════════════════
Pattern: số/năm/LOẠI-CƠQUAN
  168/2024/NĐ-CP -> số một trăm sáu mươi tám năm hai nghìn không trăm hai mươi bốn nghị định chính phủ
  160/2024/NĐ-CP -> số một trăm sáu mươi năm hai nghìn không trăm hai mươi bốn nghị định chính phủ

Điều khoản: Điều 6 Khoản 9 Điểm b -> Điều sáu Khoản chín Điểm b

══════════════════════════════════════════
6. THỜI GIAN TOUR
══════════════════════════════════════════
  5N4D, 5N4Đ -> năm ngày bốn đêm
  3N2D, 3N2Đ -> ba ngày hai đêm

══════════════════════════════════════════
7. MÃ DẠNG CHỮ+SỐ — GIỮ NGUYÊN
══════════════════════════════════════════
Các mã gồm chữ cái + số (mã ngành, hạng bằng lái, mã sản phẩm, năm...) -> GIỮ NGUYÊN, KHÔNG đọc chữ cái, KHÔNG chuyển số thành chữ.
Việc xử lý các token này sẽ do bộ xử lý riêng đảm nhận.

Ví dụ GIỮ NGUYÊN:
  CN12, CN1, CN18, QH301, X06   (mã ngành)
  A1, B1, C1, D, B2             (hạng bằng lái)
  QH2026, QK2025                (mã khoá/năm)
  SJC, PNJ, FPT                 (thương hiệu viết tắt không có trong bảng)

══════════════════════════════════════════
8. TỪ VIẾT TẮT KHÔNG CÓ TRONG BẢNG — GIỮ NGUYÊN
══════════════════════════════════════════
Nếu một từ viết tắt (toàn chữ cái, không có số) KHÔNG xuất hiện trong bảng bên dưới
-> GIỮ NGUYÊN. Bộ xử lý regex sẽ đánh vần chúng sau.

Ví dụ GIỮ NGUYÊN: SJC, PNJ, FPT, BMW, CEO, GDP

══════════════════════════════════════════
9. ĐƠN VỊ /mỗi
══════════════════════════════════════════
  VND/năm   -> đồng mỗi năm
  USD/tháng -> đô la mỹ mỗi tháng
  km/h      -> ki lô mét trên giờ  (đặc biệt, không phải "mỗi")

══════════════════════════════════════════
10. ĐỘ DÀI CÂU
══════════════════════════════════════════
Sau khi chuẩn hóa, mỗi câu (kết thúc bằng dấu chấm) nên có từ 4 đến 20 từ tiếng Việt.
Nếu một câu dài hơn 20 từ, hãy ngắt thành nhiều câu ngắn hơn.

Quy tắc ngắt câu:
- Ưu tiên ngắt tại: dấu phẩy, liên từ (và, hoặc, nhưng, tuy nhiên, do đó, vì vậy, ngoài ra)
- KHÔNG ngắt giữa: cụm số (mười tám triệu đồng), cụm danh từ (ủy ban nhân dân thành phố), mã văn bản pháp lý, tên riêng
- Mỗi câu sau khi ngắt phải có ít nhất 4 từ
- Dùng dấu chấm để kết thúc mỗi câu mới

Ví dụ:
IN:  Theo nghị định số một trăm sáu mươi tám năm hai nghìn không trăm hai mươi bốn nghị định chính phủ, tổ chức vi phạm sẽ bị phạt từ mười tám triệu đồng đến hai mươi triệu đồng và bị tước giấy phép lái xe từ hai tháng đến bốn tháng.
OUT: Theo nghị định số một trăm sáu mươi tám năm hai nghìn không trăm hai mươi bốn nghị định chính phủ, tổ chức vi phạm sẽ bị phạt từ mười tám triệu đồng đến hai mươi triệu đồng. Và bị tước giấy phép lái xe từ hai tháng đến bốn tháng.

══════════════════════════════════════════
BẢNG TỪ VIẾT TẮT — CHỈ CÁC TỪ NÀY MỚI ĐƯỢC EXPAND
══════════════════════════════════════════

Pháp lý / hành chính:
{legal_lines}

Tổ chức / trường học:
{org_lines}

Địa lý:
{geo_lines}

Giao thông:
{traffic_lines}

Đơn vị & tiền tệ:
{unit_lines}

══════════════════════════════════════════
VÍ DỤ
══════════════════════════════════════════

IN:  Theo Nghị định 168/2024/NĐ-CP, phạt từ 18.000.000 đồng đến 20.000.000 đồng.
OUT: Theo Nghị định số một trăm sáu mươi tám năm hai nghìn không trăm hai mươi bốn nghị định chính phủ, phạt từ mười tám triệu đồng đến hai mươi triệu đồng.

IN:  TOUR PHÚ QUỐC MÙA XUÂN 3N2Đ | GRAND WORLD - VINWONDERS - SAFARI - CÁP TREO HÒN THƠM
OUT: Tour Phú Quốc Mùa Xuân ba ngày hai đêm, Grand World, Vinwonders, Safari, Cáp Treo Hòn Thơm

IN:  Tour Miền Bắc 5N4Đ | Tam Chúc - Ninh Bình - Hạ Long
OUT: Tour Miền Bắc năm ngày bốn đêm, Tam Chúc, Ninh Bình, Hạ Long

IN:  Điểm chuẩn ngành CN12 năm 2025 là 28.19. Học phí 44.000.000 VND/năm.
OUT: Điểm chuẩn ngành CN12 năm hai nghìn không trăm hai mươi lăm là hai mươi tám phẩy mười chín. Học phí bốn mươi bốn triệu đồng mỗi năm.

IN:  Chỉ tiêu CN1: 420, CN12: 80, CN18: 60.
OUT: Chỉ tiêu CN1: bốn trăm hai mươi, CN12: tám mươi, CN18: sáu mươi.

IN:  **Lưu ý:** Các hạng A1, A và B1 cần mang GPLX.
OUT: Lưu ý, Các hạng A1, A và B1 cần mang giấy phép lái xe.

IN:  UBND TP.HCM xử phạt vi phạm ATGT từ 800.000 đến 1.000.000 đồng.
OUT: Ủy ban nhân dân thành phố hồ chí minh xử phạt vi phạm an toàn giao thông từ tám trăm nghìn đến một triệu đồng.

IN:  Số điện thoại: 024 37 547 865. Diện tích 35.000 m2.
OUT: Số điện thoại, không hai bốn, ba bảy năm bốn, bảy tám sáu lăm. Diện tích ba mươi lăm nghìn mét vuông.

IN:  Giá vàng SJC và PNJ hôm nay tăng mạnh.
OUT: Giá vàng SJC và PNJ hôm nay tăng mạnh.

IN:  Sinh viên tốt nghiệp có thể làm về IT, AI và IoT.
OUT: Sinh viên tốt nghiệp có thể làm về ai ti, ây ai và ai ô ti.

IN:  Điều 6 Khoản 9 Điểm b — phạt từ 18.000.000 đến 20.000.000 đồng.
OUT: Điều sáu Khoản chín Điểm b, phạt từ mười tám triệu đến hai mươi triệu đồng.
"""


_SYSTEM_PROMPT: str = _build_system_prompt()


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


def read_digits(number_str):
    return " ".join(DIGIT_MAP[d] for d in number_str)


def read_full_number(number_str):
    # nếu bạn đã có hàm đọc số tiếng Việt thì thay vào đây
    # demo tạm: nếu <=2 digits đọc từng số
    if len(number_str) <= 2:
        return read_digits(number_str)
    return read_digits(number_str)  # bạn có thể đổi sang number2words vi


def normalize_token(token):
    # tách block chữ và số
    
    parts = re.findall(r"[A-Z]+|\d+", token)

    normalized_parts = []

    for part in parts:
        if part.isalpha():  # block chữ
            letters_read = " ".join(LETTER_MAP.get(ch, ch) for ch in part)
            normalized_parts.append(letters_read)

        elif part.isdigit():  # block số
            if part.startswith("0"):
                normalized_parts.append(read_digits(part))
            else:
                normalized_parts.append(read_full_number(part))

    return " ".join(normalized_parts)

def replacer(match):
    token = match.group(0)

    # chỉ xử lý nếu có chữ in hoa
    if re.search(r"[A-Z]", token):
        return normalize_token(token)

    return token

def normalize_text(text):
    pattern = r"(?<!\w)[A-Z][A-Z0-9]+(?!\w)"
    return re.sub(pattern, replacer, text)
# ============================================================
# TTSNormalizerAgent
# ============================================================

class TTSNormalizerAgent:
    """Pure LLM TTS normalizer. Hỗ trợ cả sync và async."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ):
        self._model = model
        self._openai_api_key = openai_api_key
        self._openai_base_url = openai_base_url
        self._llm = None  # lazy init

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
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=text),
            ])
            return normalize_text(resp.content.strip())
        except Exception as exc:
            logger.warning("TTS normalize error, returning original: %s", exc)
            return text

    async def anormalize(self, text: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        if not text.strip():
            return text
        try:
            resp = await self._get_llm().ainvoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=text),
            ])
            return normalize_text(resp.content.strip())
        except Exception as exc:
            logger.warning("TTS normalize error, returning original: %s", exc)
            return text


# ============================================================
# Integration helpers
# ============================================================

def create_tts_normalizer(
    model: str = "gpt-4o-mini",
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
) -> TTSNormalizerAgent:
    return TTSNormalizerAgent(
        model=model,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )


async def anormalize_for_tts(
    text: str, agent: Optional[TTSNormalizerAgent] = None
) -> str:
    if agent is None:
        agent = TTSNormalizerAgent()
    return await agent.anormalize(text)


def normalize_for_tts(
    text: str, agent: Optional[TTSNormalizerAgent] = None
) -> str:
    if agent is None:
        agent = TTSNormalizerAgent()
    return agent.normalize(text)


# ============================================================
# Self-test  (yêu cầu OPENAI_API_KEY trong .env)
# ============================================================

if __name__ == "__main__":
    import asyncio

    TEST_CASES = [
        "Theo Nghị định 168/2024/NĐ-CP, phạt từ 18.000.000 đồng đến 20.000.000 đồng.",
        "TOUR PHÚ QUỐC MÙA XUÂN 3N2Đ | GRAND WORLD - VINWONDERS - SAFARI - CÁP TREO HÒN THƠM",
        "Tour Miền Bắc 5N4Đ | Tam Chúc – Ninh Bình – Hạ Long",
        "Điểm chuẩn ngành CN12 năm 2025 là 28.19. Học phí 44.000.000 VND/năm.",
        "Chỉ tiêu CN1: 420, CN12: 80, CN18: 60.",
        "**Lưu ý:** Các hạng A1, A và B1 cần mang GPLX.",
        "UBND TP.HCM xử phạt vi phạm ATGT từ 800.000 đến 1.000.000 đồng.",
        "Số điện thoại: 024 37 547 865. Diện tích 35.000 m2.",
        "Điều 6 Khoản 9 Điểm b — phạt từ 18.000.000 đến 20.000.000 đồng.",
        "Giá vàng SJC và PNJ hôm nay tăng mạnh.",
        "Sinh viên tốt nghiệp có thể làm về AI và IT.",
    ]

    async def run():
        agent = TTSNormalizerAgent()
        for i, text in enumerate(TEST_CASES, 1):
            result = await agent.anormalize(text)
            print(f"[{i:02d}] IN : {text}")
            print(f"     OUT: {result}\n")

    asyncio.run(run())