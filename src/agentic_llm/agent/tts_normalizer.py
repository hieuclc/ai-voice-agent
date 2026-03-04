"""
tts_normalizer.py — TTS Text Normalization Agent for Vietnamese LLM output.

Pipeline (rule-based fast path + LLM fallback):
  1. strip_markdown          – xóa **bold**, *italic*, `code`, # heading, bullets
  2. expand_legal_codes      – 168/2024/NĐ-CP → "số một sáu tám năm 2024 nghị định chính phủ"
  3. expand_abbreviations    – NĐ-CP, UET, ĐHQGHN, UBND, TP.HCM, …
  4. expand_tour_duration    – 5N4Đ → "năm ngày bốn đêm"
  5. normalize_units         – km/h, m2, cm3, %, VNĐ, VND
  6. normalize_phone         – 024 37 547 865 → "không hai bốn, ba bảy, năm bốn bảy, tám sáu lăm"
  7. normalize_numbers       – 18.000.000 → "mười tám triệu" | 28.19 → "hai mươi tám phẩy mười chín"
  8. expand_legal_refs       – Điều 6 Khoản 9 Điểm b → đọc số thành chữ
  9. clean_special_chars     – loại bỏ ký tự đặc biệt còn lại
 10. final_cleanup           – chuẩn hóa khoảng trắng, dấu câu

LLM fallback: phát hiện từ viết tắt chưa biết → gọi LLM để giải nghĩa → cache.

Usage:
    from tts_normalizer import TTSNormalizerAgent
    agent = TTSNormalizerAgent()
    clean = agent.normalize("Phạt tiền từ 18.000.000 đồng theo NĐ-CP năm 2024.")
    # → "Phạt tiền từ mười tám triệu đồng theo nghị định chính phủ năm hai nghìn không trăm hai mươi tư."
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)


# ============================================================
# 1. CONSTANTS — Từ điển viết tắt
# ============================================================

# --- Từ viết tắt pháp lý / hành chính ---
LEGAL_ABBR: dict[str, str] = {
    "NĐ-CP":        "nghị định chính phủ",
    "NĐ":           "nghị định",
    "QĐ-TTg":       "quyết định thủ tướng",
    "QĐ":           "quyết định",
    "TT-BCA":       "thông tư bộ công an",
    "TT-BGTVT":     "thông tư bộ giao thông vận tải",
    "TT-BTC":       "thông tư bộ tài chính",
    "TT-BGDĐT":     "thông tư bộ giáo dục và đào tạo",
    "TT":           "thông tư",
    "VBHN-VPQH":    "văn bản hợp nhất văn phòng quốc hội",
    "VBHN-BCA":     "văn bản hợp nhất bộ công an",
    "VBHN-BGTVT":   "văn bản hợp nhất bộ giao thông vận tải",
    "VBHN":         "văn bản hợp nhất",
    "VPQH":         "văn phòng quốc hội",
    "BLHS":         "bộ luật hình sự",
    "BLDS":         "bộ luật dân sự",
    "BLTTDS":       "bộ luật tố tụng dân sự",
    "BLTTHS":       "bộ luật tố tụng hình sự",
    "BGTVT":        "bộ giao thông vận tải",
    "BCA":          "bộ công an",
    "BQP":          "bộ quốc phòng",
    "BTC":          "bộ tài chính",
    "BYT":          "bộ y tế",
    "BGDĐT":        "bộ giáo dục và đào tạo",
    "HĐND":         "hội đồng nhân dân",
    "UBND":         "ủy ban nhân dân",
    "TAND":         "tòa án nhân dân",
    "VKSND":        "viện kiểm sát nhân dân",
    "CP":           "chính phủ",
    "TTg":          "thủ tướng",
    "BCT":          "bộ công thương",
}

# --- Từ viết tắt tổ chức / trường học ---
ORG_ABBR: dict[str, str] = {
    # Đọc từng chữ cái (spell-out)
    "UET":      "u ê tê",
    "VNU":      "vê en u",
    "ĐHQGHN":   "đại học quốc gia hà nội",
    "ĐHQG":     "đại học quốc gia",
    "VNeID":    "vê ne i đi",
    "VNPT":     "vê en pê tê",
    "VIETTEL":  "viê ten",
    "AI":       "a i",
    "IT":       "i tê",
    "ICT":      "i xê tê",
    "IoT":      "i ô tê",
    "CNTT":     "công nghệ thông tin",
    "KHMT":     "khoa học máy tính",
    "ĐTVT":     "điện tử viễn thông",
    "KTMT":     "kỹ thuật máy tính",
    "ATTT":     "an toàn thông tin",
    "HTTT":     "hệ thống thông tin",
    "KHTN":     "khoa học tự nhiên",
    "ĐHBK":     "đại học bách khoa",
    "ĐHSP":     "đại học sư phạm",
    "ĐHKT":     "đại học kinh tế",
    "ĐHQY":     "đại học y",
    "GPA":      "gê pê a",
    "CV":       "xê vê",
    "ID":       "ai đi",
    "PIN":      "pim",
    "OTP":      "ô tê pê",
}

# --- Từ viết tắt địa lý ---
GEO_ABBR: dict[str, str] = {
    "TP.HCM":       "thành phố hồ chí minh",
    "TP. HCM":      "thành phố hồ chí minh",
    "HCM":          "hồ chí minh",
    "TP.HN":        "thành phố hà nội",
    "TP.":          "thành phố",
    "TP":           "thành phố",
    "Q.":           "quận",
    "H.":           "huyện",
    "P.":           "phường",
    "X.":           "xã",
    "TT.":          "thị trấn",
    "TX.":          "thị xã",
    "TL":           "tỉnh lộ",
    "QL":           "quốc lộ",
    "VN":           "việt nam",
}

# --- Từ viết tắt đơn vị & tiền tệ ---
UNIT_ABBR: dict[str, str] = {
    # Tiền tệ
    "VNĐ":      "đồng",
    "VND":      "đồng",
    "đ":        "đồng",
    "USD":      "đô la mỹ",
    "EUR":      "ơ rô",
    # Đơn vị đo
    "km/h":     "ki lô mét trên giờ",
    "m/s":      "mét trên giây",
    "km²":      "ki lô mét vuông",
    "km2":      "ki lô mét vuông",
    "m²":       "mét vuông",
    "m2":       "mét vuông",
    "cm²":      "xen ti mét vuông",
    "cm2":      "xen ti mét vuông",
    "cm³":      "xen ti mét khối",
    "cm3":      "xen ti mét khối",
    "mm":       "mi li mét",
    "cm":       "xen ti mét",
    "km":       "ki lô mét",
    "kg":       "ki lô gam",
    "mg":       "mi li gam",
    "%":        "phần trăm",
    "°C":       "độ xê",
    "°":        "độ",
    "kWh":      "ki lô oát giờ",
    "kW":       "ki lô oát",
    "W":        "oát",
    "GHz":      "ghi ga héc",
    "MHz":      "mê ga héc",
    "GB":       "gì bai",
    "MB":       "mê ga bai",
    "KB":       "ki lô bai",
    "TB":       "tê ra bai",
}

# --- Từ viết tắt giao thông & phương tiện ---
TRAFFIC_ABBR: dict[str, str] = {
    "CSGT":     "cảnh sát giao thông",
    "ATGT":     "an toàn giao thông",
    "TTGT":     "trật tự giao thông",
    "GPLX":     "giấy phép lái xe",
    "GPKD":     "giấy phép kinh doanh",
    "ĐKLX":     "đăng ký lái xe",
    "BVMT":     "bảo vệ môi trường",
    "PCCC":     "phòng cháy chữa cháy",
    "ĐKKT":     "đăng ký kiểm tra",
}

# --- Giấy phép lái xe ---
DRIVING_LICENSE_ABBR: dict[str, str] = {
    "hạng A1":  "hạng a một",
    "hạng A2":  "hạng a hai",
    "hạng A":   "hạng a",
    "hạng B1":  "hạng bê một",
    "hạng B2":  "hạng bê hai",
    "hạng B":   "hạng bê",
    "hạng C1":  "hạng xê một",
    "hạng C2":  "hạng xê hai",
    "hạng C":   "hạng xê",
    "hạng D1":  "hạng đê một",
    "hạng D2":  "hạng đê hai",
    "hạng D":   "hạng đê",
    "hạng E":   "hạng ê",
    "hạng F":   "hạng ép",
    "hạng FB":  "hạng ép bê",
    "hạng FC":  "hạng ép xê",
}

# Gộp tất cả vào một dict, ưu tiên theo độ dài (dài trước)
ALL_ABBR: dict[str, str] = {
    **LEGAL_ABBR,
    **ORG_ABBR,
    **GEO_ABBR,
    **TRAFFIC_ABBR,
    **DRIVING_LICENSE_ABBR,
}

# UNIT_ABBR xử lý riêng trong bước normalize_units

# ============================================================
# 2. Vietnamese Number-to-Words
# ============================================================

_UNITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_SCALES = ["", "nghìn", "triệu", "tỷ"]


def _three_digit_to_words(n: int, is_leading: bool = True) -> str:
    """Chuyển số 0-999 thành chữ tiếng Việt."""
    if n == 0:
        return ""
    hundreds = n // 100
    tens = (n % 100) // 10
    ones = n % 10
    parts = []

    if hundreds > 0:
        parts.append(f"{_UNITS[hundreds]} trăm")
    elif not is_leading and (tens > 0 or ones > 0):
        # Không có hàng trăm nhưng là nhóm giữa/cuối → "không trăm"
        parts.append("không trăm")

    if tens == 0:
        if ones > 0:
            # 0x → "linh x" nếu đứng sau trăm
            prefix = "linh " if (hundreds > 0 or not is_leading) else ""
            parts.append(f"{prefix}{_UNITS[ones]}")
    elif tens == 1:
        parts.append("mười")
        if ones > 0:
            # 10 → mười, 15 → mười lăm (không "mười năm")
            word = "lăm" if ones == 5 else _UNITS[ones]
            parts.append(word)
    else:
        parts.append(f"{_UNITS[tens]} mươi")
        if ones == 1:
            parts.append("mốt")
        elif ones == 5:
            parts.append("lăm")
        elif ones > 0:
            parts.append(_UNITS[ones])

    return " ".join(parts)


def number_to_words_vi(n: int) -> str:
    """
    Chuyển số nguyên dương thành chữ tiếng Việt.
    Ví dụ:
        18_000_000 → "mười tám triệu"
        35_000     → "ba mươi lăm nghìn"
        200_000    → "hai trăm nghìn"
        1_234_567  → "một triệu hai trăm ba mươi tư nghìn năm trăm sáu mươi bảy"
    """
    if n == 0:
        return "không"
    if n < 0:
        return "âm " + number_to_words_vi(-n)

    # Tách thành nhóm 3 chữ số từ phải sang trái
    groups: list[int] = []
    while n > 0:
        groups.append(n % 1000)
        n //= 1000

    # Giới hạn tới tỷ (10^12) → dùng tỷ lặp cho số lớn hơn
    parts: list[str] = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        is_leading = (i == len(groups) - 1)
        word = _three_digit_to_words(g, is_leading)
        scale = _SCALES[i] if i < len(_SCALES) else f"10^{i*3}"
        if scale:
            word = f"{word} {scale}"
        parts.append(word.strip())

    return " ".join(parts)


def decimal_to_words_vi(text: str) -> str:
    """
    Chuyển số thập phân (dấu chấm) thành chữ.
    28.19 → "hai mươi tám phẩy mười chín"
    """
    parts = text.split(".")
    if len(parts) != 2:
        return text
    int_part = int(parts[0]) if parts[0] else 0
    frac_str = parts[1]

    int_words = number_to_words_vi(int_part)
    # Đọc phần thập phân như số nguyên
    frac_words = number_to_words_vi(int(frac_str)) if frac_str.isdigit() else frac_str
    return f"{int_words} phẩy {frac_words}"


def ordinal_digits_vi(digits: str) -> str:
    """
    Đọc từng chữ số riêng lẻ (dùng cho số điện thoại, mã số).
    "024" → "không hai bốn"
    """
    return " ".join(_UNITS[int(d)] if d.isdigit() else d for d in digits)


# ============================================================
# 3. Core Normalizer
# ============================================================

class TTSNormalizer:
    """Rule-based normalizer — nhanh, không cần LLM."""

    # Regex detect số điện thoại VN (10 chữ số, có thể có khoảng trắng/dấu chấm)
    _PHONE_RE = re.compile(
        r"(?<!\d)"                          # không đứng sau chữ số
        r"(0[3-9]\d)\s*[-.]?\s*"            # đầu số di động
        r"(\d{3,4})\s*[-.]?\s*(\d{3,4})"   # phần còn lại
        r"(?!\d)"
    )
    # Regex số cố định Hà Nội: 024 XXXX XXXX
    _PHONE_LANDLINE_RE = re.compile(
        r"(?<!\d)(02[0-9])\s+(\d{2}\s?\d{3})\s+(\d{3})(?!\d)"
    )

    # Regex số nguyên có dấu chấm ngăn cách nghìn (VN style: 18.000.000)
    _VN_INT_RE = re.compile(r"(?<!\d)(\d{1,3})(?:\.\d{3})+(?!\d)")

    # Regex số thập phân tiếng Anh: 28.19 (2+ chữ số + dấu chấm + 2+ chữ số)
    _DECIMAL_RE = re.compile(r"(?<!\d)(\d+)\.(\d{2,})(?!\d)")

    # Regex số nguyên đơn thuần
    _INT_RE = re.compile(r"(?<!\w)(\d+)(?!\w)")

    # Regex mã văn bản pháp lý: 168/2024/NĐ-CP, 160/2024/NĐ-CP
    _LEGAL_CODE_RE = re.compile(
        r"(\d+)/(\d{4})/([A-ZĐÂÊÔĂÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ-]+)"
    )

    # Regex tour duration: 5N4Đ, 3N2Đ
    _TOUR_DUR_RE = re.compile(r"(\d+)[Nn](\d+)[ĐđDd]")

    # Regex hạng GPLX: hạng A1, B2, v.v. — ưu tiên trước expand_abbr chung
    _LICENSE_RE = re.compile(r"\bhạng\s+([ABCDEF][12]?)\b", re.IGNORECASE)

    # Regex điều khoản: "Điều 6", "Khoản 9", "Điểm b"
    _LEGAL_REF_RE = re.compile(
        r"\b(Điều|điều|Khoản|khoản|Điểm|điểm|Mục|mục|Chương|chương)\s+(\d+|[a-zđ])\b"
    )

    # Regex phát hiện từ viết tắt chưa biết: 2+ chữ hoa liên tiếp
    _UNKNOWN_ABBR_RE = re.compile(
        r"\b([A-ZĐÂÊÔĂÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ]{2,})\b"
    )

    def normalize(self, text: str) -> str:
        text = self._strip_markdown(text)
        text = self._expand_legal_codes(text)
        text = self._expand_tour_duration(text)
        text = self._expand_driving_license(text)
        text = self._expand_abbreviations(text)
        text = self._normalize_units(text)
        text = self._normalize_phone(text)
        text = self._normalize_vn_integers(text)
        text = self._normalize_decimals(text)
        text = self._normalize_plain_integers(text)
        text = self._normalize_legal_refs(text)
        text = self._clean_special_chars(text)
        text = self._final_cleanup(text)
        return text

    def detect_unknown_abbreviations(self, text: str) -> list[str]:
        """Trả về danh sách từ viết tắt chưa có trong từ điển."""
        known = set(ALL_ABBR.keys()) | set(UNIT_ABBR.keys())
        found = self._UNKNOWN_ABBR_RE.findall(text)
        return [w for w in found if w not in known]

    # ----------------------------------------------------------
    # Step 1: Strip Markdown
    # ----------------------------------------------------------
    def _strip_markdown(self, text: str) -> str:
        # Headers: # Tiêu đề
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Bold: **text** or __text__
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"__(.+?)__", r"\1", text)
        # Italic: *text* or _text_
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"_(.+?)_", r"\1", text)
        # Code inline: `code`
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Code block: ```...```
        text = re.sub(r"```[\s\S]+?```", "", text)
        # Bullet points: - item, • item, * item (đầu dòng)
        text = re.sub(r"^\s*[-•*]\s+", "", text, flags=re.MULTILINE)
        # Numbered list: 1. item
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        # Links: [text](url) → text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Tables: dòng có |
        text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[-|: ]+$", "", text, flags=re.MULTILINE)
        return text

    # ----------------------------------------------------------
    # Step 2: Expand Legal Codes (phải làm trước expand_abbr)
    # ----------------------------------------------------------
    def _expand_legal_codes(self, text: str) -> str:
        """
        168/2024/NĐ-CP → "số một trăm sáu mươi tám năm hai nghìn không trăm hai mươi tư nghị định chính phủ"
        """
        def replace_code(m: re.Match) -> str:
            num_str = m.group(1)
            year_str = m.group(2)
            suffix = m.group(3)

            num_words = number_to_words_vi(int(num_str))
            year_words = number_to_words_vi(int(year_str))

            # Giải nghĩa suffix (NĐ-CP, TT-BGTVT, ...)
            suffix_clean = suffix.strip("-")
            suffix_words = self._lookup_abbr(suffix_clean)

            return f"số {num_words} năm {year_words} {suffix_words}"

        return self._LEGAL_CODE_RE.sub(replace_code, text)

    # ----------------------------------------------------------
    # Step 3: Tour Duration
    # ----------------------------------------------------------
    def _expand_tour_duration(self, text: str) -> str:
        """5N4Đ → 'năm ngày bốn đêm'"""
        def replace_dur(m: re.Match) -> str:
            days = number_to_words_vi(int(m.group(1)))
            nights = number_to_words_vi(int(m.group(2)))
            return f"{days} ngày {nights} đêm"
        return self._TOUR_DUR_RE.sub(replace_dur, text)

    # ----------------------------------------------------------
    # Step 4: Driving License Grades
    # ----------------------------------------------------------
    def _expand_driving_license(self, text: str) -> str:
        """
        hạng A1 → 'hạng a một', hạng B → 'hạng bê'
        Cũng expand danh sách: hạng A1, A và B1 → hạng a một, a và bê một
        """
        _GRADE_MAP = {
            "A1": "a một", "A2": "a hai", "A": "a",
            "B1": "bê một", "B2": "bê hai", "B": "bê",
            "C1": "xê một", "C2": "xê hai", "C": "xê",
            "D1": "đê một", "D2": "đê hai", "D": "đê",
            "E": "ê", "F": "ép", "FB": "ép bê", "FC": "ép xê",
        }

        def replace_license(m: re.Match) -> str:
            grade = m.group(1).upper()
            return f"hạng {_GRADE_MAP.get(grade, grade.lower())}"

        # Bước 1: "hạng X" → "hạng x..."
        text = self._LICENSE_RE.sub(replace_license, text)

        # Bước 2: Expand các grade đơn lẻ trong danh sách sau "hạng"
        # Ví dụ: "hạng a một, A và B1" → "hạng a một, a và bê một"
        _GRADE_IN_LIST_RE = re.compile(r"([,]\s*|và\s+|hoặc\s+)([ABCDEF][12]?)(?!\w)")

        def replace_in_list(m: re.Match) -> str:
            sep = m.group(1)
            g = m.group(2).upper()
            return sep + _GRADE_MAP.get(g, g.lower())

        # Áp dụng nhiều lần cho list dài
        for _ in range(5):
            new = _GRADE_IN_LIST_RE.sub(replace_in_list, text)
            if new == text:
                break
            text = new

        return text

    # ----------------------------------------------------------
    # Step 5: Expand Abbreviations
    # ----------------------------------------------------------
    def _expand_abbreviations(self, text: str) -> str:
        # Sắp xếp theo độ dài giảm dần để match greedy
        sorted_abbr = sorted(ALL_ABBR.items(), key=lambda x: len(x[0]), reverse=True)
        for abbr, expansion in sorted_abbr:
            # Escape đặc biệt, word boundary
            pattern = r"(?<![A-Za-zÀ-ỹ\-/])" + re.escape(abbr) + r"(?![A-Za-zÀ-ỹ\-/])"
            text = re.sub(pattern, expansion, text)
        return text

    # ----------------------------------------------------------
    # Step 6: Normalize Units
    # ----------------------------------------------------------
    def _normalize_units(self, text: str) -> str:
        # Sắp xếp dài trước để tránh match ngắn sớm (km/h trước km)
        sorted_units = sorted(UNIT_ABBR.items(), key=lambda x: len(x[0]), reverse=True)
        for unit, expansion in sorted_units:
            # Cho phép khoảng trắng tuỳ chọn giữa số và đơn vị
            pattern = r"(?<=\d)\s*" + re.escape(unit) + r"(?!\w)"
            text = re.sub(pattern, f" {expansion}", text)
            # Và khi đứng độc lập
            pattern2 = r"(?<!\w)" + re.escape(unit) + r"(?!\w)"
            text = re.sub(pattern2, f" {expansion} ", text)
        return text

    # ----------------------------------------------------------
    # Step 7: Phone Numbers
    # ----------------------------------------------------------
    def _normalize_phone(self, text: str) -> str:
        def replace_phone(m: re.Match) -> str:
            full = re.sub(r"[\s.\-]", "", m.group(0))
            # Đọc theo nhóm 3-4 chữ số
            groups = []
            i = 0
            # Đầu số: 3 chữ số
            groups.append(ordinal_digits_vi(full[:3]))
            rest = full[3:]
            # Chia phần còn lại thành nhóm 4 và 3
            while rest:
                chunk = rest[:4]
                groups.append(ordinal_digits_vi(chunk))
                rest = rest[4:]
            return ", ".join(groups)

        # Số di động: 0xxx xxx xxxx
        text = self._PHONE_RE.sub(replace_phone, text)
        # Số bàn: 024 xx xxx xxx
        text = self._PHONE_LANDLINE_RE.sub(replace_phone, text)
        return text

    # ----------------------------------------------------------
    # Step 8: VN Integer (dấu chấm ngăn nghìn)
    # ----------------------------------------------------------
    def _normalize_vn_integers(self, text: str) -> str:
        def replace_vn_int(m: re.Match) -> str:
            raw = m.group(0).replace(".", "")
            return number_to_words_vi(int(raw))
        return self._VN_INT_RE.sub(replace_vn_int, text)

    # ----------------------------------------------------------
    # Step 9: Decimal Numbers
    # ----------------------------------------------------------
    def _normalize_decimals(self, text: str) -> str:
        def replace_decimal(m: re.Match) -> str:
            return decimal_to_words_vi(m.group(0))
        return self._DECIMAL_RE.sub(replace_decimal, text)

    # ----------------------------------------------------------
    # Step 10: Plain Integers
    # ----------------------------------------------------------
    def _normalize_plain_integers(self, text: str) -> str:
        def replace_int(m: re.Match) -> str:
            return number_to_words_vi(int(m.group(1)))
        return self._INT_RE.sub(replace_int, text)

    # ----------------------------------------------------------
    # Step 11: Legal References (Điều X, Khoản Y)
    # ----------------------------------------------------------
    def _normalize_legal_refs(self, text: str) -> str:
        def replace_ref(m: re.Match) -> str:
            keyword = m.group(1)
            ref = m.group(2)
            if ref.isdigit():
                ref_word = number_to_words_vi(int(ref))
            else:
                ref_word = ref  # chữ cái a, b, c, đ → giữ nguyên
            return f"{keyword} {ref_word}"
        return self._LEGAL_REF_RE.sub(replace_ref, text)

    # ----------------------------------------------------------
    # Step 12: Clean Special Characters
    # ----------------------------------------------------------

    # Đơn vị "mỗi": /năm, /tháng, /ngày, /người, /vé, /xe, /lần
    _PER_UNIT_RE = re.compile(
        r"/\s*(năm|tháng|ngày|người|vé|xe|lần|giờ|phút|giây|tuần|km|m|lít|kg)\b",
        re.IGNORECASE,
    )

    def _clean_special_chars(self, text: str) -> str:
        # "X/năm" → "X mỗi năm" (trước khi xử lý / chung)
        text = self._PER_UNIT_RE.sub(lambda m: f" mỗi {m.group(1)}", text)
        # Dấu ngoặc đơn: (nội dung) → ", nội dung,"
        text = re.sub(r"\(([^)]+)\)", r", \1,", text)
        # Dấu ngoặc vuông: [nội dung] → nội dung
        text = re.sub(r"\[([^\]]+)\]", r"\1", text)
        # Dấu gạch ngang kép (em dash, en dash)
        text = re.sub(r"[–—]", ", ", text)
        # Ba chấm
        text = re.sub(r"\.{2,}", ", ", text)
        # Dấu /  trong ngữ cảnh "a/b" → "a hoặc b"
        text = re.sub(r"(?<=\w)/(?=\w)", " hoặc ", text)
        # Ký tự đặc biệt còn lại: giữ dấu câu tiêu chuẩn
        text = re.sub(r"[^\w\sÀ-ỹàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ,.:!?]", " ", text)
        return text

    # ----------------------------------------------------------
    # Step 13: Final Cleanup
    # ----------------------------------------------------------
    def _final_cleanup(self, text: str) -> str:
        # Gộp khoảng trắng nhiều
        text = re.sub(r" {2,}", " ", text)
        # Xóa khoảng trắng trước dấu câu
        text = re.sub(r"\s+([,.:!?])", r"\1", text)
        # Khoảng trắng sau dấu câu
        text = re.sub(r"([,.:!?])(?!\s)", r"\1 ", text)
        # Xóa dòng trống nhiều
        text = re.sub(r"\n{2,}", "\n", text)
        # Xóa khoảng trắng đầu/cuối mỗi dòng
        lines = [line.strip() for line in text.splitlines()]
        # Bỏ dòng rỗng
        lines = [l for l in lines if l]
        return " ".join(lines).strip()

    # ----------------------------------------------------------
    # Helper
    # ----------------------------------------------------------
    def _lookup_abbr(self, key: str) -> str:
        """Tra cứu từ điển, trả về key nếu không tìm thấy."""
        for d in [ALL_ABBR, UNIT_ABBR]:
            if key in d:
                return d[key]
            # Thử ghép components: NĐ-CP → NĐ + CP
            parts = re.split(r"[-/]", key)
            expansions = []
            for p in parts:
                expansions.append(d.get(p, p.lower()))
            return " ".join(expansions)
        return key.lower()


# ============================================================
# 4. LLM Fallback Agent  (OpenAI)
# ============================================================

_ABBR_SYSTEM = (
    "Bạn là chuyên gia chuẩn hóa văn bản tiếng Việt cho TTS. "
    "Trả về JSON object duy nhất: key = từ viết tắt, value = dạng đầy đủ tiếng Việt thường. "
    "Tên riêng quốc tế hoặc không rõ → đánh vần từng chữ cái (UNESCO → 'u nê ét xê ô'). "
    "Không giải thích, không markdown, chỉ JSON thuần."
)


class TTSNormalizerAgent:
    """
    Agent chuẩn hóa TTS = rule-based normalizer + OpenAI LLM fallback.

    LLM fallback chỉ gọi khi phát hiện từ viết tắt chưa có trong từ điển.
    Kết quả được cache để tránh gọi API nhiều lần.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        use_llm_fallback: bool = True,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ):
        self._rule = TTSNormalizer()
        self._use_llm = use_llm_fallback
        self._cache: dict[str, str] = {}
        self._model = model
        self._openai_api_key = openai_api_key
        self._openai_base_url = openai_base_url
        self._llm = None  # lazy init khi thực sự cần

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

    # ── Public API ──────────────────────────────────────────

    def normalize(self, text: str) -> str:
        """Đồng bộ: rule-based → LLM fallback nếu có từ lạ."""
        if self._use_llm:
            unknowns = [w for w in self._rule.detect_unknown_abbreviations(text)
                        if w not in self._cache]
            if unknowns:
                expansions = self._llm_expand_sync(unknowns)
                text = self._apply_expansions(text, expansions)
        return self._rule.normalize(text)

    async def anormalize(self, text: str) -> str:
        """Bất đồng bộ (dùng trong async LangGraph pipeline)."""
        if self._use_llm:
            unknowns = [w for w in self._rule.detect_unknown_abbreviations(text)
                        if w not in self._cache]
            if unknowns:
                expansions = await self._llm_expand_async(unknowns)
                text = self._apply_expansions(text, expansions)
        return self._rule.normalize(text)

    # ── LLM Fallback (OpenAI via langchain_openai) ──────────

    def _llm_expand_sync(self, unknowns: list[str]) -> dict[str, str]:
        from langchain_core.messages import SystemMessage, HumanMessage as HMsg
        llm = self._get_llm()
        resp = llm.invoke([
            SystemMessage(content=_ABBR_SYSTEM),
            HMsg(content=json.dumps(unknowns, ensure_ascii=False)),
        ])
        return self._parse_and_cache(resp.content, unknowns)

    async def _llm_expand_async(self, unknowns: list[str]) -> dict[str, str]:
        from langchain_core.messages import SystemMessage, HumanMessage as HMsg
        llm = self._get_llm()
        resp = await llm.ainvoke([
            SystemMessage(content=_ABBR_SYSTEM),
            HMsg(content=json.dumps(unknowns, ensure_ascii=False)),
        ])
        return self._parse_and_cache(resp.content, unknowns)

    def _parse_and_cache(self, raw: str, unknowns: list[str]) -> dict[str, str]:
        try:
            clean = re.sub(r"```json|```", "", raw).strip()
            parsed: dict[str, str] = json.loads(clean)
            self._cache.update(parsed)
            return parsed
        except (json.JSONDecodeError, TypeError):
            logger.warning("TTS LLM fallback JSON không hợp lệ: %s", raw[:200])
            # Fallback: đánh vần từng chữ cái cho tất cả unknowns
            return {w: " ".join(c.lower() for c in w) for w in unknowns}

    def _apply_expansions(self, text: str, expansions: dict[str, str]) -> str:
        for abbr, expansion in expansions.items():
            text = re.sub(r"\b" + re.escape(abbr) + r"\b", expansion, text)
        return text


# ============================================================
# 6. Integration Helper — dùng trong agent_routing.py
# ============================================================

def create_tts_normalizer(use_llm_fallback: bool = True) -> TTSNormalizerAgent:
    """Factory function để khởi tạo normalizer agent."""
    return TTSNormalizerAgent(use_llm_fallback=use_llm_fallback)


def normalize_for_tts(text: str, agent: Optional[TTSNormalizerAgent] = None) -> str:
    """
    Convenience function — dùng nhanh không cần khởi tạo agent.
    Rule-based only (không LLM).
    """
    if agent is not None:
        return agent.normalize(text)
    return TTSNormalizer().normalize(text)


async def anormalize_for_tts(text: str, agent: Optional[TTSNormalizerAgent] = None) -> str:
    """Async version của normalize_for_tts."""
    if agent is not None:
        return await agent.anormalize(text)
    return TTSNormalizer().normalize(text)
