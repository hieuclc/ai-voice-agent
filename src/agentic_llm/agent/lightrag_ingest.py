"""
lightrag_ingest.py — Ingest .pdf và .docx văn bản pháp luật vào LightRAG,
với pre-processing để nhúng trích dẫn (Điều/Khoản/Điểm) vào từng chunk.

Cơ chế hoạt động
-----------------
1. Extract text từ PDF/DOCX.
2. Nếu là văn bản luật → dùng law_parser.build_citation_chunks() để tách
   thành chunks, mỗi chunk có header:
       [TRÍCH DẪN: <Tên luật> | Điều X | Khoản Y]
3. Insert từng chunk riêng lẻ vào LightRAG.

LightRAG sẽ học được pattern "[TRÍCH DẪN: ...]" và khi trả lời sẽ tự động
trích dẫn đúng tên văn bản, số điều, số khoản.

Usage examples
--------------
# Ingest file luật (tự động detect tên luật):
    python lightrag_ingest.py path/to/luat_giao_thong.pdf

# Ingest với tên văn bản rõ ràng (khuyến nghị):
    python lightrag_ingest.py luat.pdf --law-name "Luật Giao thông đường bộ 2008"

# Ingest folder (tất cả PDF/DOCX):
    python lightrag_ingest.py docs/ --law-name-map law_names.json

# Dry-run:
    python lightrag_ingest.py docs/ --dry-run

# Chunk ở mức điều (luật ngắn) thay vì mức khoản:
    python lightrag_ingest.py luat.pdf --chunk-at article

Environment variables
---------------------
  OPENAI_API_KEY          Required cho LightRAG LLM.
  LIGHTRAG_WORKING_DIR    Override working directory (default: ./lightrag_data).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LightRAG imports
# ---------------------------------------------------------------------------

from lightrag import LightRAG, QueryParam  # noqa: E402
from lightrag.llm.openai import gpt_4o_mini_complete  # noqa: E402
from lightrag.utils import setup_logger as lightrag_setup_logger  # noqa: E402
from embedding_service import VIETNAMESE_EMBEDDING_FUNC, get_embedding_service  # noqa: E402
from law_parser import parse_file  # noqa: E402

lightrag_setup_logger("lightrag", level="WARNING")

# ---------------------------------------------------------------------------
# Text extraction helpers (giữ nguyên từ bản cũ)
# ---------------------------------------------------------------------------


def extract_text_from_pdf(path: Path) -> str:
    try:
        import pdfplumber
        pages: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError(
            "No PDF library found. Install one of: pdfplumber, pypdf\n"
            "  pip install pdfplumber   # recommended\n"
            "  pip install pypdf"
        )


def extract_text_from_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required: pip install python-docx")

    doc = Document(str(path))
    parts: list[str] = []

    for block in doc.element.body:
        tag = block.tag.split("}")[-1]
        if tag == "p":
            from docx.oxml.ns import qn
            text = "".join(node.text or "" for node in block.iter() if node.tag == qn("w:t"))
            if text.strip():
                parts.append(text)
        elif tag == "tbl":
            from docx.oxml.ns import qn
            rows: list[str] = []
            for row in block.findall(
                ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr"
            ):
                cells: list[str] = []
                for cell in row.findall(
                    ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc"
                ):
                    cell_text = "".join(
                        node.text or ""
                        for node in cell.iter()
                        if node.tag == "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
                    )
                    cells.append(cell_text.strip())
                rows.append(" | ".join(cells))
            parts.append("\n".join(rows))

    return "\n\n".join(parts)


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(path)
    elif ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# LightRAG initialisation
# ---------------------------------------------------------------------------


async def build_rag(working_dir: str) -> LightRAG:
    os.makedirs(working_dir, exist_ok=True)
    await get_embedding_service()
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=VIETNAMESE_EMBEDDING_FUNC,
    )
    await rag.initialize_storages()
    return rag


# ---------------------------------------------------------------------------
# Core ingestion logic
# ---------------------------------------------------------------------------


async def ingest_file(
    rag: LightRAG,
    path: Path,
    law_name: str | None = None,
    chunk_at: str = "article",
) -> bool:
    """
    Parse văn bản luật, tạo citation chunks và insert từng chunk vào LightRAG.
    Trả về True nếu thành công.

    Parameters
    ----------
    rag       : LightRAG instance.
    path      : Đường dẫn đến file PDF/DOCX/TXT.
    law_name  : Tên văn bản pháp luật (vd: "Luật Giao thông đường bộ 2008").
                Nếu None, tự động detect từ nội dung.
    chunk_at  : "article" | "clause" | "point"
    """
    logger.info("Processing: %s", path)

    # Parse và tạo citation chunks (parse_file tự extract text)
    try:
        chunks = parse_file(path, law_name=law_name, chunk_at=chunk_at)
    except Exception as exc:
        logger.error("  ✗ Law parsing failed: %s", exc)
        # Fallback: extract text thô và ingest nguyên vẹn
        try:
            raw_text = extract_text(path)
            chunks = [raw_text] if raw_text.strip() else []
            if chunks:
                logger.warning("  → Fallback: ingest toàn bộ text không qua parser.")
        except Exception:
            return False

    if not chunks:
        logger.warning("  ✗ Parser không tạo được chunk nào từ %s.", path.name)
        return False

    logger.info("  Tạo được %d chunks. Bắt đầu insert vào LightRAG…", len(chunks))

    # 3. Insert từng chunk
    succeeded = 0
    failed = 0
    for i, chunk in enumerate(chunks, 1):
        try:
            await rag.ainsert(chunk, file_paths=[str(path)])
            succeeded += 1
            if i % 10 == 0 or i == len(chunks):
                logger.info("    Chunk %d/%d done.", i, len(chunks))
        except Exception as exc:
            logger.error("    ✗ Chunk %d insert failed: %s", i, exc)
            failed += 1

    if succeeded > 0:
        logger.info(
            "  ✓ Ingested %s: %d chunks OK, %d failed.",
            path.name, succeeded, failed
        )
        return True
    else:
        logger.error("  ✗ Tất cả chunks đều failed cho %s.", path.name)
        return False


def collect_files(source: Path) -> list[Path]:
    supported = {".pdf", ".docx", ".doc", ".txt"}
    if source.is_file():
        if source.suffix.lower() in supported:
            return [source]
        else:
            logger.error("Unsupported file type: %s", source)
            return []
    files: list[Path] = []
    for p in sorted(source.rglob("*")):
        if p.is_file() and p.suffix.lower() in supported:
            files.append(p)
    return files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def async_main(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        logger.error("Source path does not exist: %s", source)
        sys.exit(1)

    working_dir = args.working_dir or os.environ.get("LIGHTRAG_WORKING_DIR", "./lightrag_data")

    # Load law name map (JSON: {"filename.pdf": "Tên luật đầy đủ"})
    law_name_map: dict[str, str] = {}
    if args.law_name_map:
        try:
            with open(args.law_name_map, encoding="utf-8") as f:
                law_name_map = json.load(f)
            logger.info("Loaded law name map: %d entries", len(law_name_map))
        except Exception as exc:
            logger.error("Cannot load law_name_map: %s", exc)

    files = collect_files(source)
    if not files:
        logger.warning("No supported files found under: %s", source)
        return

    logger.info("Found %d file(s) to ingest:", len(files))
    for f in files:
        logger.info("  %s", f)

    if args.dry_run:
        logger.info("Dry-run mode — no files were ingested.")
        return

    rag = await build_rag(working_dir)

    succeeded = 0
    failed = 0
    try:
        for path in files:
            # Resolve law_name: CLI arg > map > None (auto-detect)
            law_name = args.law_name or law_name_map.get(path.name) or law_name_map.get(str(path))
            ok = await ingest_file(rag, path, law_name=law_name, chunk_at=args.chunk_at)
            if ok:
                succeeded += 1
            else:
                failed += 1
    finally:
        await rag.finalize_storages()

    logger.info(
        "\n========================================\n"
        "  Ingestion complete.\n"
        "  Succeeded : %d\n"
        "  Failed    : %d\n"
        "  Working dir: %s\n"
        "========================================",
        succeeded,
        failed,
        working_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest văn bản pháp luật vào LightRAG với citation chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source", help="Đường dẫn file hoặc thư mục.")
    parser.add_argument(
        "--law-name",
        default=None,
        help="Tên đầy đủ văn bản pháp luật (dùng cho tất cả file trong lần chạy này). "
             "Vd: 'Luật Giao thông đường bộ số 23/2008/QH12'",
    )
    parser.add_argument(
        "--law-name-map",
        default=None,
        help="Đường dẫn JSON map: {filename: law_name}. "
             "Dùng khi ingest nhiều file luật khác nhau cùng lúc.",
    )
    parser.add_argument(
        "--chunk-at",
        default="article",
        choices=["article", "clause", "point"],
        help=(
            "Mức chunk: "
            "'article'=mỗi Điều 1 chunk (luật ngắn), "
            "'clause'=mỗi Khoản 1 chunk (mặc định, khuyến nghị), "
            "'point'=mỗi Điểm 1 chunk (luật rất chi tiết)."
        ),
    )
    parser.add_argument(
        "--working-dir",
        default=None,
        help="LightRAG working directory (default: $LIGHTRAG_WORKING_DIR hoặc ./lightrag_data).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Liệt kê file sẽ ingest mà không thực sự insert.",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()