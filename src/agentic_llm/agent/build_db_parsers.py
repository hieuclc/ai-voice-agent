"""
build_db_parsers.py — Parsers và data models, extracted từ build_db.py gốc.
Imported bởi build_db.py mới (Qdrant version).
"""

import re
import uuid
from dataclasses import dataclass
from typing import Optional, List, Dict
from docx import Document


# =========================================================
# Data Model
# =========================================================

@dataclass
class LegalChunk:
    id: str
    text: str
    doc_type: str
    source_file: str

    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    point: Optional[str] = None

    article_full: str = ""
    clause_full: str = ""

    def to_chromadb(self):
        meta = {
            "doc_type": self.doc_type,
            "source_file": self.source_file,
            "source": ".".join(self.source_file.split("/")[-1].split('.')[:-1]),   # ⭐ NEW FIELD
            "chapter": self.chapter or "",
            "section": self.section or "",
            "article": self.article or "",
            "clause": self.clause or "",
            "point": self.point or "",
            "article_full": self.article_full,
            "clause_full": self.clause_full,
        }
        return self.id, self.text, meta


# =========================================================
# Helpers
# =========================================================

def load_docx_paragraphs(path: str) -> List[str]:
    doc = Document(path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


def load_docx_text(path: str) -> str:
    return "\n".join(load_docx_paragraphs(path))


def make_id(prefix: str = "") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# =========================================================
# Regex
# =========================================================

RE_ARTICLE = re.compile(r"^(Điều\s+\d+[a-z]?\.\s*.*)$")
RE_CHAPTER = re.compile(r"^(Chương\s+[IVXLCDM\d]+)$", re.IGNORECASE)
RE_SECTION = re.compile(r"^(Mục\s+\d+[a-z]?)$", re.IGNORECASE)
RE_CLAUSE = re.compile(r"^(\d+)\.\s+(.+)")
RE_POINT = re.compile(r"^([a-zđ])\)\s+(.+)")

# =========================================================
# BaseDocumentParser (domain-agnostic)
# =========================================================

class BaseDocumentParser:
    """
    Base parser chung cho mọi loại tài liệu.
    Trả về list[LegalChunk] để thống nhất index pipeline.
    """

    def __init__(self, doc_type: str, source_file: str = ""):
        self.doc_type = doc_type
        self.source_file = source_file
        self.chunks: List[LegalChunk] = []

    def _new_chunk(self, text: str, **metadata):
        self.chunks.append(
            LegalChunk(
                id=make_id(self.doc_type),
                text=text,
                doc_type=self.doc_type,
                source_file=self.source_file,
                **metadata
            )
        )

    def parse(self, *args, **kwargs):
        raise NotImplementedError

# =========================================================
# Base Class
# =========================================================

class BaseLegalParser(BaseDocumentParser):
    """
    Base cho văn bản pháp luật.
    """
    def __init__(self, doc_type: str, source_file: str = ""):
        super().__init__(doc_type, source_file)

    def _new_chunk(self, **kwargs) -> None:
        self.chunks.append(
            LegalChunk(
                id=make_id(self.doc_type),
                doc_type=self.doc_type,
                source_file=self.source_file,
                **kwargs
            )
        )

    def parse(self, *args, **kwargs) -> List[LegalChunk]:
        raise NotImplementedError


# =========================================================
# LawParser
# =========================================================

class LawParser(BaseLegalParser):
    """
    Luật:
    Chương → [Mục] → Điều → Khoản → Điểm

    Granularity:
    - Không khoản → chunk toàn Điều
    - Có khoản → chunk mỗi Khoản
    - Có điểm → chunk mỗi Điểm
    """

    def __init__(self, source_file: str = ""):
        super().__init__(doc_type="luat", source_file=source_file)

    def parse(self, paragraphs: List[str]) -> List[LegalChunk]:
        self.chunks = []

        chapter = None
        section = None
        article_title = None
        article_lines = []

        def flush():
            if article_title and article_lines:
                self._process_article(
                    article_title,
                    article_lines,
                    chapter,
                    section
                )

        for para in paragraphs:
            if RE_CHAPTER.match(para):
                flush()
                chapter = para
                section = None
                article_title = None
                article_lines = []
                continue

            if RE_SECTION.match(para):
                flush()
                section = para
                article_title = None
                article_lines = []
                continue

            if RE_ARTICLE.match(para):
                flush()
                article_title = para
                article_lines = []
                continue

            if article_title:
                article_lines.append(para)

        flush()
        return self.chunks

    def _process_article(self, title, lines, chapter, section):
        article_full = title + "\n" + "\n".join(lines)
        clauses = self._split_clauses(lines)

        if not clauses:
            self._new_chunk(
                text=article_full,
                chapter=chapter,
                section=section,
                article=title,
                article_full=article_full,
            )
            return

        for num, clause_lines in clauses.items():
            clause_title = f"Khoản {num}"
            clause_full = "\n".join(clause_lines)
            points = self._split_points(clause_lines)

            if not points:
                self._new_chunk(
                    text=f"{title}\n{clause_full}",
                    chapter=chapter,
                    section=section,
                    article=title,
                    clause=clause_title,
                    article_full=article_full,
                    clause_full=clause_full,
                )
            else:
                for letter, point_text in points.items():
                    self._new_chunk(
                        text=f"{title} – {clause_title}\n{point_text}",
                        chapter=chapter,
                        section=section,
                        article=title,
                        clause=clause_title,
                        point=f"Điểm {letter}",
                        article_full=article_full,
                        clause_full=clause_full,
                    )

    @staticmethod
    def _split_clauses(lines: List[str]) -> Dict[str, List[str]]:
        clauses = {}
        current = None
        for line in lines:
            m = RE_CLAUSE.match(line)
            if m:
                current = m.group(1)
                clauses[current] = [line]
            elif current:
                clauses[current].append(line)
        return clauses

    @staticmethod
    def _split_points(lines: List[str]) -> Dict[str, str]:
        points = {}
        for line in lines:
            m = RE_POINT.match(line)
            if m:
                points[m.group(1)] = line
        return points


# =========================================================
# CircularParser
# =========================================================

class CircularParser(BaseLegalParser):
    """
    Thông tư / Nghị định:
    Điều → Khoản → Điểm

    Granularity:
    - Ưu tiên chunk ở cấp Điểm (behavior)
    - Nếu không có điểm → chunk Khoản
    """

    def __init__(self, source_file: str = ""):
        super().__init__(doc_type="thong_tu", source_file=source_file)

    def parse(self, raw_text: str) -> List[LegalChunk]:
        self.chunks = []

        articles = re.findall(
            r"(Điều\s+\d+\.?.*?)(?=Điều\s+\d+\.|$)",
            raw_text,
            flags=re.S,
        )

        for art in articles:
            art_title = art.split("\n")[0].strip()
            article_full = art.strip()
            clauses = self._parse_clauses(art)

            if not clauses:
                self._new_chunk(
                    text=article_full,
                    article=art_title,
                    article_full=article_full,
                )
                continue

            for clause_text in clauses:
                clause_title = clause_text.split("\n")[0].strip()
                clause_full = clause_text.strip()
                points = self._parse_points(clause_text)

                if not points:
                    self._new_chunk(
                        text=f"{art_title}\n{clause_full}",
                        article=art_title,
                        clause=clause_title,
                        article_full=article_full,
                        clause_full=clause_full,
                    )
                else:
                    for point_text in points:
                        behavior = (
                            point_text
                            .lstrip("abcdefghijklmnopqrstuvwxyzđ)")
                            .strip()
                            .rstrip(";")
                        )
                        self._new_chunk(
                            text=behavior,
                            article=art_title,
                            clause=clause_title,
                            point=point_text,
                            article_full=article_full,
                            clause_full=clause_full,
                        )

        return self.chunks

    @staticmethod
    def _parse_clauses(article_text: str) -> List[str]:
        return [
            c.strip()
            for c in re.findall(
                r"(\n\d+\.\s.*?)(?=\n\d+\.|\Z)",
                article_text,
                flags=re.S,
            )
        ]

    @staticmethod
    def _parse_points(clause_text: str) -> List[str]:
        return re.findall(
            r"(?:^|\n)([a-zđ]\)\s+[^\n]+)",
            clause_text
        )


# =========================================================
# Factory
# =========================================================

def create_chunks(file_path: str, doc_type: str):
    if doc_type == "luat":
        paragraphs = load_docx_paragraphs(file_path)
        parser = LawParser(source_file=file_path)
        return parser.parse(paragraphs)

    elif doc_type in ["thong_tu", "nghi_dinh"]:
        parser = CircularParser(source_file=file_path)
        raw_text = load_docx_text(file_path)
        return parser.parse(raw_text)

    else:
        raise ValueError("Unsupported document type")


# =========================================================
# AdmissionConsultingParser
# =========================================================

import pdfplumber


class AdmissionConsultingParser(BaseDocumentParser):
    """
    Parser cho tài liệu tư vấn tuyển sinh (PDF).

    Structure:
        Roman (I, II, III)
            → Section (1, 2)
                → Subsection (1.1)

    KHÔNG phải legal document.
    """

    ROMAN_PATTERN = r'^\s*([IVX]+)\.\s+'
    SECTION_PATTERN = r'^\s*(\d+)\.\s+'
    SUBSECTION_PATTERN = r'^\s*(\d+\.\d+)\.?\s+'

    def __init__(self, source_file: str = ""):
        super().__init__(
            doc_type="tu_van_tuyen_sinh",
            source_file=source_file
        )

    def parse(self, pdf_path: str):
        self.chunks = []
        blocks = self._extract_blocks(pdf_path)

        last_chunk_index = None

        for block in blocks:

            if block["type"] == "text":

                text_chunks = self._split_text_block(block["content"])

                for content, meta in text_chunks:
                    self._new_chunk(
                        text=content.strip(),
                        chapter=meta["roman"],
                        section=meta["section"],
                        clause=meta["subsection"],
                        article=f"Page {block['page']}",
                        article_full=content.strip()
                    )
                    last_chunk_index = len(self.chunks) - 1

            elif block["type"] == "table":

                sentences = self._table_to_semantic_sentences(
                    block["content"]
                )

                if sentences and last_chunk_index is not None:
                    self.chunks[last_chunk_index].text += (
                        "\n" + "\n".join(sentences)
                    )

        return self.chunks
        # =========================================================
    # Internal Helpers
    # =========================================================

    @staticmethod
    def _clean(text):
        if not text:
            return ""
        return str(text).strip().replace("\n", " ")

    @staticmethod
    def _is_page_number(line: str) -> bool:
        return bool(re.match(r"^\s*\d+\s*$", line.strip()))

    # ===============================
    # TABLE → SEMANTIC SENTENCES
    # ===============================

    def _table_to_semantic_sentences(self, table):

        if not table or len(table) < 2:
            return []

        headers = [self._clean(h) for h in table[0]]
        rows = table[1:]

        merged_rows = []
        current_row = None

        for row in rows:
            row = [self._clean(c) for c in row]

            if len(row) < len(headers):
                row += [""] * (len(headers) - len(row))

            is_new = row[0] != ""

            if is_new:
                if current_row:
                    merged_rows.append(current_row)
                current_row = row
            else:
                if current_row:
                    for i in range(len(row)):
                        if row[i]:
                            if current_row[i]:
                                current_row[i] += " " + row[i]
                            else:
                                current_row[i] = row[i]

        if current_row:
            merged_rows.append(current_row)

        sentences = []

        for row in merged_rows:
            row_dict = dict(zip(headers, row))

            parts = []
            for h in headers:
                if row_dict.get(h):
                    parts.append(f"{h}: {row_dict[h]}")

            if parts:
                sentences.append(". ".join(parts) + ".")

        return sentences

    # ===============================
    # EXTRACT BLOCKS (TEXT + TABLE)
    # ===============================

    def _extract_blocks(self, pdf_path: str):

        blocks = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):

                tables = page.find_tables()
                tables = sorted(tables, key=lambda t: t.bbox[1])

                last_bottom = 0

                for table in tables:

                    x0, top, x1, bottom = table.bbox

                    # ===== TEXT TRÊN TABLE =====
                    if top > last_bottom:
                        upper = page.crop((0, last_bottom, page.width, top))
                        text = upper.extract_text()
                        if text:
                            blocks.append({
                                "type": "text",
                                "content": text,
                                "page": page_num
                            })

                    # ===== TABLE =====
                    blocks.append({
                        "type": "table",
                        "content": table.extract(),
                        "page": page_num
                    })

                    last_bottom = bottom

                # ===== TEXT DƯỚI TABLE =====
                if last_bottom < page.height:
                    lower = page.crop((0, last_bottom, page.width, page.height))
                    text = lower.extract_text()
                    if text:
                        blocks.append({
                            "type": "text",
                            "content": text,
                            "page": page_num
                        })

        return blocks

    # ===============================
    # SPLIT TEXT THEO HIERARCHY
    # ===============================

    def _split_text_block(self, text):

        lines = text.split("\n")

        chunks = []
        current_chunk = ""
        current_meta = {
            "roman": None,
            "section": None,
            "subsection": None
        }

        for line in lines:

            if self._is_page_number(line):
                continue

            roman_match = re.match(self.ROMAN_PATTERN, line)
            section_match = re.match(self.SECTION_PATTERN, line)
            subsection_match = re.match(self.SUBSECTION_PATTERN, line)

            if roman_match:
                if current_chunk.strip():
                    chunks.append((current_chunk.strip(), current_meta.copy()))

                current_meta["roman"] = roman_match.group(1)
                current_meta["section"] = None
                current_meta["subsection"] = None
                current_chunk = line + "\n"

            elif subsection_match:
                if current_chunk.strip():
                    chunks.append((current_chunk.strip(), current_meta.copy()))

                current_meta["subsection"] = subsection_match.group(1)
                current_chunk = line + "\n"

            elif section_match:
                if current_chunk.strip():
                    chunks.append((current_chunk.strip(), current_meta.copy()))

                current_meta["section"] = section_match.group(1)
                current_meta["subsection"] = None
                current_chunk = line + "\n"

            else:
                current_chunk += line + "\n"

        if current_chunk.strip():
            chunks.append((current_chunk.strip(), current_meta.copy()))

        return chunks


# =========================================================
# TourParser
# =========================================================

class TourParser(BaseDocumentParser):
    """
    Parser cho dữ liệu tour du lịch (JSON).

    Mỗi tour sinh ra nhiều LegalChunk để tìm kiếm ngữ nghĩa:
      - 1 summary chunk  : tiêu đề + điểm đến + loại tour + thời gian + giá
      - N itinerary chunk: mỗi ngày trong lịch trình (tiêu đề + địa điểm + mô tả)

    Mapping sang các field của LegalChunk:
      point        ← tour_id  (dedicated lookup key — reliable, not truncated)
      article      ← tour_id  (kept for legacy compatibility)
      chapter      ← tour_type   (tour miền bắc / nam / trung)
      section      ← title
      clause       ← "summary" hoặc "day_<N>"
      clause_full  ← compact JSON metadata:
                     summary chunk: {p0, p1, d, n, nd, tr, dc, ds}
                     day chunk    : {meals, overnight}
      article_full ← (empty — not used for tours)
    """

    def __init__(self, source_file: str = "extracted_data.json"):
        super().__init__(doc_type="tour", source_file=source_file)

    @staticmethod
    def _fmt_price(amount: int) -> str:
        return f"{amount:,}".replace(",", ".")

    def parse(self, tours: list) -> List[LegalChunk]:
        self.chunks = []
        for tour in tours:
            self._parse_tour(tour)
        return self.chunks

    def _parse_tour(self, tour: dict) -> None:
        """
        Sinh các chunks từ một tour. Mỗi loại thông tin → 1 chunk riêng
        để retrieval chính xác hơn:

          summary      : tổng quan + giá range + điểm đến
          departures   : toàn bộ lịch khởi hành + giá theo ngày/hạng khách sạn
          services     : dịch vụ bao gồm & không bao gồm
          policies     : hủy tour, trẻ em, thanh toán, ghi chú
          day_N        : lịch trình từng ngày (đầy đủ description + địa điểm + bữa ăn)
        """
        import json as _json

        tid       = tour["tour_id"]
        title     = tour["title"]
        tour_type = tour.get("tour_type", "")
        dur       = tour.get("duration", {})
        pr        = tour.get("price_range", {})
        dests     = ", ".join(tour.get("destinations", []))
        transport = ", ".join(tour.get("transport", []))
        deps      = tour.get("departures", [])
        services  = tour.get("services", {})
        policies  = tour.get("policies", {})

        duration_str = f"{dur.get('days', '?')} ngày {dur.get('nights', '?')} đêm"
        price_str    = (
            f"{self._fmt_price(pr.get('min_price', 0))} – "
            f"{self._fmt_price(pr.get('max_price', 0))} VND"
        )

        # Compact metadata lưu trong clause_full để _run_get_tour_detail dùng
        summary_meta = _json.dumps({
            "p0": pr.get("min_price", 0),
            "p1": pr.get("max_price", 0),
            "d":  dur.get("days"),
            "n":  dur.get("nights"),
            "nd": deps[0]["date"] if deps else "",
            "tr": transport,
            "dc": tour.get("departure", {}).get("city", ""),
            "ds": tour.get("destinations", []),
            # Thêm mới: lưu tên dịch vụ và chính sách vào metadata để get_tour_detail dùng
            "inc": services.get("included", ""),
            "exc": services.get("excluded", ""),
            "pol_cancel":   policies.get("cancellation_policy", ""),
            "pol_children": policies.get("children_policy", ""),
            "pol_payment":  policies.get("payment_policy", ""),
            "pol_notes":    policies.get("notes", ""),
        }, ensure_ascii=False)

        # ── 1. Summary chunk ─────────────────────────────────────────────
        summary_text = (
            f"{title}. "
            f"Loại tour: {tour_type}. "
            f"Điểm đến: {dests}. "
            f"Thời gian: {duration_str}. "
            f"Phương tiện: {transport}. "
            f"Giá từ: {price_str}."
        )
        self._new_chunk(
            text=summary_text,
            chapter=tour_type,
            section=title,
            article=tid,
            clause="summary",
            point=tid,
            clause_full=summary_meta,
        )

        # ── 2. Departures chunk ──────────────────────────────────────────
        # Toàn bộ lịch khởi hành + giá theo ngày + hạng khách sạn
        if deps:
            dep_lines = [f"{title} – Lịch khởi hành và giá chi tiết:"]
            # Group theo ngày để dễ đọc
            from collections import defaultdict
            by_date: dict = defaultdict(list)
            for d in deps:
                by_date[d.get("date", "")].append(d)
            for date, entries in sorted(by_date.items()):
                for e in entries:
                    hotel  = e.get("hotel_standard", "")
                    price  = self._fmt_price(e.get("price", 0))
                    slots  = e.get("available_slots", 0)
                    dep_lines.append(
                        f"  Ngày {date} – {hotel}: {price} VND"
                        + (f" (còn {slots} chỗ)" if slots else "")
                    )
            self._new_chunk(
                text="\n".join(dep_lines),
                chapter=tour_type,
                section=title,
                article=tid,
                clause="departures",
                point=tid,
                clause_full=summary_meta,
            )

        # ── 3. Services chunk ────────────────────────────────────────────
        inc = services.get("included", "")
        exc = services.get("excluded", "")
        if inc or exc:
            svc_parts = [f"{title} – Dịch vụ:"]
            if inc:
                svc_parts.append(f"Bao gồm: {inc}")
            if exc:
                svc_parts.append(f"Không bao gồm: {exc}")
            self._new_chunk(
                text="\n".join(svc_parts),
                chapter=tour_type,
                section=title,
                article=tid,
                clause="services",
                point=tid,
                clause_full=summary_meta,
            )

        # ── 4. Policies chunk ────────────────────────────────────────────
        pol_cancel   = policies.get("cancellation_policy", "")
        pol_children = policies.get("children_policy", "")
        pol_payment  = policies.get("payment_policy", "")
        pol_notes    = policies.get("notes", "")
        if any([pol_cancel, pol_children, pol_payment, pol_notes]):
            pol_parts = [f"{title} – Chính sách:"]
            if pol_cancel:
                pol_parts.append(f"Hủy tour: {pol_cancel}")
            if pol_children:
                pol_parts.append(f"Trẻ em: {pol_children}")
            if pol_payment:
                pol_parts.append(f"Thanh toán: {pol_payment}")
            if pol_notes:
                pol_parts.append(f"Lưu ý: {pol_notes}")
            self._new_chunk(
                text="\n".join(pol_parts),
                chapter=tour_type,
                section=title,
                article=tid,
                clause="policies",
                point=tid,
                clause_full=summary_meta,
            )

        # ── 5. Itinerary chunks (one per day) ────────────────────────────
        for day in tour.get("itinerary", []):
            day_num   = day.get("day", 0)
            day_title = day.get("title", "")
            desc      = day.get("description", "")
            locs      = ", ".join(day.get("locations", []))
            meals     = ", ".join(day.get("meals", []))
            overnight = day.get("overnight") or ""

            day_meta = _json.dumps({
                "meals":     day.get("meals", []),
                "overnight": overnight,
                "day_label": f"Ngày {day_num}: {day_title}",
                "locations": day.get("locations", []),
            }, ensure_ascii=False)

            # Text đầy đủ: tiêu đề + địa điểm + mô tả chi tiết + bữa ăn + ngủ đêm
            day_parts = [f"{title} – Ngày {day_num}: {day_title}."]
            if locs:
                day_parts.append(f"Địa điểm: {locs}.")
            if desc:
                day_parts.append(desc)
            if meals:
                day_parts.append(f"Bữa ăn: {meals}.")
            if overnight:
                day_parts.append(f"Nghỉ đêm tại: {overnight}.")

            self._new_chunk(
                text=" ".join(day_parts),
                chapter=tour_type,
                section=title,
                article=tid,
                clause=f"day_{day_num}",
                point=tid,
                clause_full=day_meta,
            )


# =========================================================
# Chroma Indexer
# =========================================================

def index_chunks(chunks: List[LegalChunk], collection, batch_size: int = 5000):
    """
    Index list[LegalChunk] vào ChromaDB collection.

    - Tự động chia batch
    - Không phụ thuộc loại document
    """

    if not chunks:
        print("⚠ No chunks to index.")
        return

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        cid, doc, meta = chunk.to_chromadb()
        ids.append(cid)
        documents.append(doc)
        metadatas.append(meta)

    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"✅ Indexed {len(ids)} chunks.")