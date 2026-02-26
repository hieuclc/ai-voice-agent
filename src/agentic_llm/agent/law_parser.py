"""
law_parser.py — Parser văn bản pháp luật Việt Nam thành chunks có trích dẫn rõ ràng.

Chiến lược parse
----------------
Thay vì dùng regex trên full text (dễ bị nhầm), parser xử lý theo từng PARAGRAPH
của docx (hoặc từng dòng của txt/pdf). Mỗi paragraph được phân loại:

  - ARTICLE : "Điều 1. Phạm vi..."
  - CLAUSE  : "1. Nghị định này..."
  - POINT   : "a) Xử phạt..."
  - TEXT    : nội dung thường

Kết quả: mỗi chunk được gắn header trích dẫn chuẩn:
  [TRÍCH DẪN: Nghị định 168/2024/NĐ-CP | Điều 3 | Khoản 2 | Điểm a]

Usage
-----
    from law_parser import parse_file

    # Docx (chính xác nhất):
    chunks = parse_file("luat.docx", law_name="Nghị định 168/2024/NĐ-CP")

    # PDF / TXT:
    chunks = parse_file("luat.pdf", law_name="...", chunk_at="clause")

CLI
---
    python law_parser.py path/to/file.docx
    python law_parser.py path/to/file.docx --law-name "Nghị định 168/2024/NĐ-CP" --chunk-at clause
"""

from __future__ import annotations

import argparse
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_RE_ARTICLE = re.compile(r'^(?:Điều|ĐIỀU)\s+(\d+[a-zA-Z]?)\s*[.\-:]\s*(.*)', re.DOTALL)
_RE_CLAUSE  = re.compile(r'^(\d{1,3})\.\s+(.*)', re.DOTALL)
_RE_POINT   = re.compile(r'^([a-zđ])\)\s+(.*)', re.DOTALL)
_RE_LAW_TITLE = re.compile(
    r'((?:LUẬT|NGHỊ ĐỊNH|THÔNG TƯ|QUYẾT ĐỊNH|PHÁP LỆNH|NGHỊ QUYẾT)[^\n]{5,150})',
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class _T:
    ARTICLE = "article"
    CLAUSE  = "clause"
    POINT   = "point"
    TEXT    = "text"
    SKIP    = "skip"


def _classify(text: str):
    t = text.strip()
    if not t:
        return _T.SKIP, None
    m = _RE_ARTICLE.match(t)
    if m:
        return _T.ARTICLE, m
    m = _RE_CLAUSE.match(t)
    if m and 1 <= int(m.group(1)) <= 30:
        return _T.CLAUSE, m
    m = _RE_POINT.match(t)
    if m:
        return _T.POINT, m
    return _T.TEXT, None


# ---------------------------------------------------------------------------
# Chunk builder
# ---------------------------------------------------------------------------

@dataclass
class _Chunk:
    law_name: str
    article_num: str
    article_title: str
    clause_num: Optional[str]
    point_label: Optional[str]
    lines: list[str] = field(default_factory=list)

    def citation_header(self) -> str:
        parts = [f"Điều {self.article_num}"]
        if self.clause_num:
            parts.append(f"Khoản {self.clause_num}")
        if self.point_label:
            parts.append(f"Điểm {self.point_label}")
        return f"[TRÍCH DẪN: {self.law_name} | {' | '.join(parts)}]"

    def to_text(self) -> str:
        body = "\n".join(self.lines).strip()
        return f"{self.citation_header()}\n\n{body}" if body else ""


_MAX_ARTICLE_CHARS = 3000  # split article chunk nếu vượt ngưỡng này


def _split_article_chunk(chunk: _Chunk, max_chars: int) -> list[str]:
    """
    Khi một Điều quá dài (nhiều khoản), split thành các sub-chunk nhỏ hơn.
    Mỗi sub-chunk vẫn giữ header "Điều X. <tên>" ở đầu để LightRAG
    biết ngữ cảnh, và đánh dấu "(phần Y/Z)" để không mất trích dẫn.

    Chiến lược: gom các khoản liên tiếp vào từng sub-chunk sao cho
    tổng độ dài <= max_chars. Khoản tham chiếu nhau (ví dụ khoản 13
    reference khoản 12) luôn nằm cùng sub-chunk bằng cách giữ overlap 1 khoản.
    """
    header = f"Điều {chunk.article_num}. {chunk.article_title}"
    citation_base = f"[TRÍCH DẪN: {chunk.law_name} | Điều {chunk.article_num}]"

    # Tách các khoản ra (mỗi dòng bắt đầu bằng "Khoản N:")
    clause_blocks: list[list[str]] = []
    current_block: list[str] = []
    for line in chunk.lines:
        if line.startswith("Khoản ") and current_block:
            clause_blocks.append(current_block)
            current_block = [line]
        else:
            current_block.append(line)
    if current_block:
        clause_blocks.append(current_block)

    if len(clause_blocks) <= 1:
        # Không split được, trả về nguyên vẹn
        return [chunk.to_text()]

    results: list[str] = []
    bucket: list[list[str]] = []
    bucket_len = len(header) + len(citation_base)

    for i, block in enumerate(clause_blocks):
        block_text = "\n".join(block)
        if bucket and bucket_len + len(block_text) > max_chars:
            # flush bucket thành 1 sub-chunk
            body = header + "\n" + "\n".join(
                line for b in bucket for line in b
            )
            results.append(f"{citation_base}\n\n{body}")
            # overlap: giữ lại khoản cuối của bucket để khoản kế tiếp
            # (có thể reference nó) vẫn thấy được ngữ cảnh
            bucket = [bucket[-1]]
            bucket_len = len(header) + len(citation_base) + len("\n".join(bucket[-1]))
        bucket.append(block)
        bucket_len += len(block_text)

    if bucket:
        body = header + "\n" + "\n".join(line for b in bucket for line in b)
        results.append(f"{citation_base}\n\n{body}")

    return results


def _paragraphs_to_chunks(
    paragraphs: list[str],
    law_name: str,
    chunk_at: str,
) -> list[str]:
    chunks: list[str] = []
    preamble_lines: list[str] = []

    cur_article_num: Optional[str] = None
    cur_article_title: str = ""
    cur_clause_num: Optional[str] = None
    cur_point_label: Optional[str] = None
    buf: list[str] = []

    def flush():
        nonlocal buf
        if not buf or cur_article_num is None:
            buf = []
            return
        c = _Chunk(
            law_name=law_name,
            article_num=cur_article_num,
            article_title=cur_article_title,
            clause_num=cur_clause_num if chunk_at != "article" else None,
            point_label=cur_point_label if chunk_at == "point" else None,
            lines=list(buf),
        )
        text = c.to_text()
        if text and len(text) > 40:
            if chunk_at == "article" and len(text) > _MAX_ARTICLE_CHARS:
                # Điều dài: split thành sub-chunks có overlap 1 khoản
                chunks.extend(_split_article_chunk(c, _MAX_ARTICLE_CHARS))
            else:
                chunks.append(text)
        buf = []

    for raw in paragraphs:
        ltype, match = _classify(raw)
        text = raw.strip()

        if ltype == _T.SKIP:
            if cur_article_num is None:
                preamble_lines.append(text)
            else:
                buf.append(text)
            continue

        if ltype == _T.ARTICLE:
            flush()
            cur_article_num = match.group(1)
            cur_article_title = match.group(2).strip()
            cur_clause_num = None
            cur_point_label = None
            buf = [f"Điều {cur_article_num}. {cur_article_title}"]

        elif ltype == _T.CLAUSE:
            if cur_article_num is None:
                preamble_lines.append(text)
                continue
            if chunk_at != "article":
                flush()
                buf = [f"Điều {cur_article_num}. {cur_article_title}"]
            cur_clause_num = match.group(1)
            cur_point_label = None
            buf.append(f"Khoản {cur_clause_num}: {match.group(2).strip()}")

        elif ltype == _T.POINT:
            if cur_article_num is None:
                preamble_lines.append(text)
                continue
            if chunk_at == "point":
                flush()
                buf = [f"Điều {cur_article_num}. {cur_article_title}"]
                if cur_clause_num:
                    buf.append(f"Khoản {cur_clause_num}:")
            cur_point_label = match.group(1)
            buf.append(f"Điểm {cur_point_label}: {match.group(2).strip()}")

        else:  # TEXT
            if cur_article_num is None:
                preamble_lines.append(text)
            else:
                buf.append(text)

    flush()

    preamble = "\n".join(l for l in preamble_lines if l).strip()
    if preamble and len(preamble) > 40:
        chunks.insert(0, f"[TRÍCH DẪN: {law_name} | Phần mở đầu]\n\n{preamble}")

    logger.info("'%s': %d chunks (chunk_at='%s')", law_name, len(chunks), chunk_at)
    return chunks


# ---------------------------------------------------------------------------
# Law name detection
# ---------------------------------------------------------------------------

_RE_DOC_NUMBER = re.compile(r'(\d+/\d{4}/[\w\-CP]+)', re.IGNORECASE)
_RE_KIND = re.compile(
    r'^(LUẬT|NGHỊ ĐỊNH|THÔNG TƯ|QUYẾT ĐỊNH|PHÁP LỆNH|NGHỊ QUYẾT)$',
    re.IGNORECASE
)


def _detect_law_name(items: list[str]) -> str:
    """
    Detect tên văn bản từ header docx. Xử lý 2 trường hợp phổ biến:

    Case 1 - Nghị định / Thông tư (tên nằm trong bảng header):
      paragraph[i]   = "NGHỊ ĐỊNH"
      paragraph[i+1] = "QUY ĐỊNH XỬ PHẠT..."   ← tên/trích yếu
      Số văn bản nằm trong bảng dạng "Số: 168/2024/NĐ-CP | Hà Nội..."

    Case 2 - Luật (tên nằm rải 2 dòng):
      paragraph[i]   = "LUẬT"
      paragraph[i+1] = "GIAO THÔNG ĐƯỜNG BỘ"
      Số văn bản ở dòng "Luật Giao thông đường bộ số 23/2008/QH12..."
    """
    # Lấy tối đa 40 paragraph đầu, lọc trắng
    header = [p.strip() for p in items[:40] if p.strip()]

    # Tìm số văn bản (vd: 168/2024/NĐ-CP, 23/2008/QH12)
    doc_num = ""
    for line in header:
        m = _RE_DOC_NUMBER.search(line)
        if m:
            doc_num = m.group(1)
            break

    # Tìm loại văn bản và tên
    for i, line in enumerate(header):
        if _RE_KIND.match(line):
            kind = line.strip().title()  # "Nghị Định", "Luật"...
            # Dòng tiếp theo là tên/trích yếu (nếu không phải heading khác)
            name_part = ""
            if i + 1 < len(header):
                next_line = header[i + 1]
                # Không phải heading tổ chức (CỘNG HÒA, Độc lập, Căn cứ...)
                if not re.match(r'^(CỘNG HÒA|Độc lập|Căn cứ|Theo đề nghị|Chính phủ|Quốc hội)', next_line, re.IGNORECASE):
                    name_part = next_line.title()

            if name_part:
                result = f"{kind} {name_part}"
            else:
                result = kind

            if doc_num:
                result += f" {doc_num}"
            return result

    # Fallback: dùng regex cũ trên toàn text header
    full_text = "\n".join(header)
    matches = _RE_LAW_TITLE.findall(full_text)
    if matches:
        # Lọc bỏ các match từ dòng "Căn cứ..."
        for m in matches:
            if not m.strip().lower().startswith("căn cứ"):
                full = " ".join(m.split())
                if doc_num and doc_num not in full:
                    full += " " + doc_num
                return full[:100]

    return "Văn bản pháp luật"


# ---------------------------------------------------------------------------
# Docx paragraph extractor
# ---------------------------------------------------------------------------

def _extract_paragraphs_docx(path: Path) -> list[str]:
    from docx import Document
    doc = Document(str(path))
    result: list[str] = []
    for block in doc.element.body:
        tag = block.tag.split("}")[-1]
        if tag == "p":
            from docx.oxml.ns import qn
            text = "".join(
                n.text or "" for n in block.iter() if n.tag == qn("w:t")
            )
            result.append(text)
        elif tag == "tbl":
            ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            for row in block.findall(f".//{{{ns}}}tr"):
                cells = []
                for cell in row.findall(f".//{{{ns}}}tc"):
                    ct = "".join(n.text or "" for n in cell.iter() if n.tag == f"{{{ns}}}t")
                    cells.append(ct.strip())
                line = " | ".join(cells)
                if line.strip():
                    result.append(line)
    return result


# ---------------------------------------------------------------------------
# PDF text extractor
# ---------------------------------------------------------------------------

def _extract_pdf_text(path: Path) -> str:
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
        return "\n".join(pages)
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except ImportError:
        raise ImportError("pip install pdfplumber  # hoặc pip install pypdf")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_citation_chunks_from_paragraphs(
    path: str | Path,
    law_name: Optional[str] = None,
    chunk_at: str = "article",
) -> list[str]:
    """Parse docx theo paragraph → citation chunks. Chính xác nhất cho docx."""
    paragraphs = _extract_paragraphs_docx(Path(path))
    if not law_name:
        law_name = _detect_law_name(paragraphs)
    return _paragraphs_to_chunks(paragraphs, law_name, chunk_at)


def build_citation_chunks(
    text: str,
    law_name: Optional[str] = None,
    chunk_at: str = "article",
) -> list[str]:
    """Parse text thuần (pdf/txt) → citation chunks."""
    lines = text.splitlines()
    if not law_name:
        law_name = _detect_law_name(lines)
    return _paragraphs_to_chunks(lines, law_name, chunk_at)


def parse_file(
    path: str | Path,
    law_name: Optional[str] = None,
    chunk_at: str = "article",
) -> list[str]:
    """
    Entry point duy nhất. Tự chọn parser theo extension:
      .docx/.doc → paragraph-based (chính xác)
      .pdf       → extract text → line-based
      .txt       → line-based
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".docx", ".doc"):
        return build_citation_chunks_from_paragraphs(path, law_name=law_name, chunk_at=chunk_at)
    if ext == ".pdf":
        text = _extract_pdf_text(path)
        return build_citation_chunks(text, law_name=law_name, chunk_at=chunk_at)
    if ext == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return build_citation_chunks(text, law_name=law_name, chunk_at=chunk_at)
    raise ValueError(f"Unsupported: {ext}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Parse văn bản pháp luật → citation chunks")
    parser.add_argument("file", help=".docx / .pdf / .txt")
    parser.add_argument("--law-name", default=None, help="Tên văn bản (tự detect nếu bỏ qua)")
    parser.add_argument("--chunk-at", default="article",
                        choices=["article", "clause", "point"],
                        help="Mức chunk (mặc định: clause)")
    parser.add_argument("--show", type=int, default=5, help="Số chunk in ra (mặc định: 5)")
    args = parser.parse_args()

    chunks = parse_file(args.file, law_name=args.law_name, chunk_at=args.chunk_at)
    print(f"\n=== Tổng số chunks: {len(chunks)} ===\n")
    for i, ch in enumerate(chunks[:args.show], 1):
        print(f"{'─'*60}")
        print(f"Chunk {i}:")
        print(ch[:600])
        if len(ch) > 600:
            print(f"  ... [{len(ch)-600} ký tự còn lại]")
        print()
    if len(chunks) > args.show:
        print(f"... và {len(chunks) - args.show} chunks nữa (--show N để xem thêm).")


if __name__ == "__main__":
    _cli()