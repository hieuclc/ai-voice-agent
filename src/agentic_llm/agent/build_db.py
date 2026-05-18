"""
build_db.py — Indexing pipeline cho toàn bộ dữ liệu vào Qdrant (hybrid: dense + sparse).

Collections:
  - law        : văn bản pháp luật (Luật, Nghị định, Thông tư)
  - admission  : tư vấn tuyển sinh
  - tours      : dữ liệu tour du lịch

Hybrid strategy:
  - Dense vector  : AITeamVN/Vietnamese_Embedding_v2 (BGE-M3 fine-tuned, dim=1024)
  - Sparse vector : lexical weights từ cùng BGE-M3 model (SPLADE-style)
  - Distance      : COSINE cho dense, default sparse index

Run:
    python build_db.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pdfplumber
import torch
from docx import Document
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_HOST      = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT      = int(os.environ.get("QDRANT_PORT", "6333"))
EMBEDDING_MODEL  = os.environ.get("EMBEDDING_MODEL_NAME", "AITeamVN/Vietnamese_Embedding_v2")
DEVICE           = os.environ.get("DEVICE", "cpu")
DENSE_DIM        = 1024
ENCODE_BATCH     = 32
UPSERT_BATCH     = 256

LAW_COLLECTION       = "law"
ADMISSION_COLLECTION = "admission"
TOUR_COLLECTION      = "tours"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

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

    def to_metadata(self) -> dict:
        return {
            "doc_type":    self.doc_type,
            "source_file": self.source_file,
            "source":      ".".join(self.source_file.split("/")[-1].split(".")[:-1]),
            "chapter":     self.chapter or "",
            "section":     self.section or "",
            "article":     self.article or "",
            "clause":      self.clause or "",
            "point":       self.point or "",
            "article_full": self.article_full,
            "clause_full":  self.clause_full,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_docx_paragraphs(path: str) -> List[str]:
    doc = Document(path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


def load_docx_text(path: str) -> str:
    return "\n".join(load_docx_paragraphs(path))


def make_id(prefix: str = "") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

RE_ARTICLE = re.compile(r"^(Điều\s+\d+[a-z]?\.\s*.*)$")
RE_CHAPTER = re.compile(r"^(Chương\s+[IVXLCDM\d]+)$", re.IGNORECASE)
RE_SECTION = re.compile(r"^(Mục\s+\d+[a-z]?)$", re.IGNORECASE)
RE_CLAUSE  = re.compile(r"^(\d+)\.\s+(.+)")
RE_POINT   = re.compile(r"^([a-zđ])\)\s+(.+)")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

class LawParser:
    """
    Luật: Chương → [Mục] → Điều → Khoản → Điểm
    Granularity: chunk mỗi Khoản (hoặc Điểm nếu có).
    """

    def __init__(self, source_file: str = ""):
        self.source_file = source_file
        self.chunks: List[LegalChunk] = []

    def _new_chunk(self, text: str, **kwargs) -> None:
        self.chunks.append(LegalChunk(
            id=make_id("luat"),
            text=text,
            doc_type="luat",
            source_file=self.source_file,
            **kwargs,
        ))

    def parse(self, paragraphs: List[str]) -> List[LegalChunk]:
        self.chunks = []
        chapter = section = article_title = None
        article_lines: List[str] = []

        def flush():
            if article_title and article_lines:
                self._process_article(article_title, article_lines, chapter, section)

        for para in paragraphs:
            if RE_CHAPTER.match(para):
                flush(); chapter = para; section = article_title = None; article_lines = []
            elif RE_SECTION.match(para):
                flush(); section = para; article_title = None; article_lines = []
            elif RE_ARTICLE.match(para):
                flush(); article_title = para; article_lines = []
            elif article_title:
                article_lines.append(para)

        flush()
        return self.chunks

    def _process_article(self, title, lines, chapter, section):
        article_full = title + "\n" + "\n".join(lines)
        clauses = self._split_clauses(lines)

        if not clauses:
            self._new_chunk(text=article_full, chapter=chapter, section=section,
                            article=title, article_full=article_full)
            return

        for num, clause_lines in clauses.items():
            clause_title = f"Khoản {num}"
            clause_full  = "\n".join(clause_lines)
            points = self._split_points(clause_lines)

            if not points:
                self._new_chunk(
                    text=f"{title}\n{clause_full}",
                    chapter=chapter, section=section,
                    article=title, clause=clause_title,
                    article_full=article_full, clause_full=clause_full,
                )
            else:
                for letter, point_text in points.items():
                    self._new_chunk(
                        text=f"{title} – {clause_title}\n{point_text}",
                        chapter=chapter, section=section,
                        article=title, clause=clause_title, point=f"Điểm {letter}",
                        article_full=article_full, clause_full=clause_full,
                    )

    @staticmethod
    def _split_clauses(lines: List[str]) -> Dict[str, List[str]]:
        clauses: Dict[str, List[str]] = {}
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
        return {RE_POINT.match(line).group(1): line
                for line in lines if RE_POINT.match(line)}


class CircularParser:
    """
    Thông tư / Nghị định: Điều → Khoản → Điểm
    Granularity: chunk ở cấp Điểm; nếu không có điểm → chunk Khoản.
    """

    def __init__(self, source_file: str = ""):
        self.source_file = source_file
        self.chunks: List[LegalChunk] = []

    def _new_chunk(self, text: str, **kwargs) -> None:
        self.chunks.append(LegalChunk(
            id=make_id("thong_tu"),
            text=text,
            doc_type="thong_tu",
            source_file=self.source_file,
            **kwargs,
        ))

    def parse(self, raw_text: str) -> List[LegalChunk]:
        self.chunks = []
        articles = re.findall(
            r"(Điều\s+\d+\.?.*?)(?=Điều\s+\d+\.|$)", raw_text, flags=re.S
        )
        for art in articles:
            art_title    = art.split("\n")[0].strip()
            article_full = art.strip()
            clauses      = self._parse_clauses(art)

            if not clauses:
                self._new_chunk(text=article_full, article=art_title, article_full=article_full)
                continue

            for clause_text in clauses:
                clause_title = clause_text.split("\n")[0].strip()
                clause_full  = clause_text.strip()
                points       = self._parse_points(clause_text)

                if not points:
                    self._new_chunk(
                        text=f"{art_title}\n{clause_full}",
                        article=art_title, clause=clause_title,
                        article_full=article_full, clause_full=clause_full,
                    )
                else:
                    for point_text in points:
                        behavior = point_text.lstrip("abcdefghijklmnopqrstuvwxyzđ)").strip().rstrip(";")
                        self._new_chunk(
                            text=behavior,
                            article=art_title, clause=clause_title, point=point_text,
                            article_full=article_full, clause_full=clause_full,
                        )
        return self.chunks

    @staticmethod
    def _parse_clauses(article_text: str) -> List[str]:
        return [c.strip() for c in re.findall(r"(\n\d+\.\s.*?)(?=\n\d+\.|\Z)", article_text, flags=re.S)]

    @staticmethod
    def _parse_points(clause_text: str) -> List[str]:
        return re.findall(r"(?:^|\n)([a-zđ]\)\s+[^\n]+)", clause_text)


class AdmissionConsultingParser:
    """
    Parser cho tài liệu tư vấn tuyển sinh (PDF).
    Structure: Roman → Section → Subsection
    """

    ROMAN_PATTERN      = r"^\s*([IVX]+)\.\s+"
    SECTION_PATTERN    = r"^\s*(\d+)\.\s+"
    SUBSECTION_PATTERN = r"^\s*(\d+\.\d+)\.?\s+"

    def __init__(self, source_file: str = ""):
        self.source_file = source_file
        self.chunks: List[LegalChunk] = []

    def _new_chunk(self, text: str, **kwargs) -> None:
        self.chunks.append(LegalChunk(
            id=make_id("tu_van_tuyen_sinh"),
            text=text,
            doc_type="tu_van_tuyen_sinh",
            source_file=self.source_file,
            **kwargs,
        ))

    def parse(self, pdf_path: str) -> List[LegalChunk]:
        self.chunks = []
        blocks = self._extract_blocks(pdf_path)
        last_chunk_index = None

        for block in blocks:
            if block["type"] == "text":
                for content, meta in self._split_text_block(block["content"]):
                    self._new_chunk(
                        text=content.strip(),
                        chapter=meta["roman"],
                        section=meta["section"],
                        clause=meta["subsection"],
                        article=f"Page {block['page']}",
                        article_full=content.strip(),
                    )
                    last_chunk_index = len(self.chunks) - 1
            elif block["type"] == "table":
                sentences = self._table_to_semantic_sentences(block["content"])
                if sentences and last_chunk_index is not None:
                    self.chunks[last_chunk_index].text += "\n" + "\n".join(sentences)

        return self.chunks

    @staticmethod
    def _clean(text) -> str:
        return str(text).strip().replace("\n", " ") if text else ""

    @staticmethod
    def _is_page_number(line: str) -> bool:
        return bool(re.match(r"^\s*\d+\s*$", line.strip()))

    def _table_to_semantic_sentences(self, table) -> List[str]:
        if not table or len(table) < 2:
            return []

        headers = [self._clean(h) for h in table[0]]
        rows    = table[1:]

        merged_rows = []
        current_row = None
        for row in rows:
            row = [self._clean(c) for c in row]
            if len(row) < len(headers):
                row += [""] * (len(headers) - len(row))

            if row[0]:
                if current_row:
                    merged_rows.append(current_row)
                current_row = row
            elif current_row:
                for i in range(len(row)):
                    if row[i]:
                        current_row[i] = (current_row[i] + " " + row[i]).strip()

        if current_row:
            merged_rows.append(current_row)

        sentences = []
        for row in merged_rows:
            row_dict = dict(zip(headers, row))
            parts = [f"{h}: {row_dict[h]}" for h in headers if row_dict.get(h)]
            if parts:
                sentences.append(". ".join(parts) + ".")
        return sentences

    def _extract_blocks(self, pdf_path: str) -> list:
        blocks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables    = sorted(page.find_tables(), key=lambda t: t.bbox[1])
                last_bottom = 0

                for table in tables:
                    x0, top, x1, bottom = table.bbox
                    if top > last_bottom:
                        upper = page.crop((0, last_bottom, page.width, top))
                        text  = upper.extract_text()
                        if text:
                            blocks.append({"type": "text", "content": text, "page": page_num})
                    blocks.append({"type": "table", "content": table.extract(), "page": page_num})
                    last_bottom = bottom

                if last_bottom < page.height:
                    lower = page.crop((0, last_bottom, page.width, page.height))
                    text  = lower.extract_text()
                    if text:
                        blocks.append({"type": "text", "content": text, "page": page_num})
        return blocks

    def _split_text_block(self, text: str) -> list:
        lines   = text.split("\n")
        chunks  = []
        current_chunk = ""
        current_meta  = {"roman": None, "section": None, "subsection": None}

        for line in lines:
            if self._is_page_number(line):
                continue

            if re.match(self.ROMAN_PATTERN, line):
                if current_chunk.strip():
                    chunks.append((current_chunk.strip(), current_meta.copy()))
                current_meta = {"roman": re.match(self.ROMAN_PATTERN, line).group(1),
                                "section": None, "subsection": None}
                current_chunk = line + "\n"
            elif re.match(self.SUBSECTION_PATTERN, line):
                if current_chunk.strip():
                    chunks.append((current_chunk.strip(), current_meta.copy()))
                current_meta["subsection"] = re.match(self.SUBSECTION_PATTERN, line).group(1)
                current_chunk = line + "\n"
            elif re.match(self.SECTION_PATTERN, line):
                if current_chunk.strip():
                    chunks.append((current_chunk.strip(), current_meta.copy()))
                current_meta["section"]    = re.match(self.SECTION_PATTERN, line).group(1)
                current_meta["subsection"] = None
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"

        if current_chunk.strip():
            chunks.append((current_chunk.strip(), current_meta.copy()))
        return chunks


class TourParser:
    """
    Parser cho dữ liệu tour du lịch (JSON).
    Mỗi tour → summary + departures + services + policies + day_N chunks.

    Mapping sang LegalChunk:
      point        ← tour_id
      article      ← tour_id (legacy)
      chapter      ← tour_type
      section      ← title
      clause       ← "summary" | "departures" | "services" | "policies" | "day_N"
      clause_full  ← compact JSON metadata
    """

    def __init__(self, source_file: str = "extracted_data.json"):
        self.source_file = source_file
        self.chunks: List[LegalChunk] = []

    @staticmethod
    def _fmt_price(amount: int) -> str:
        return f"{amount:,}".replace(",", ".")

    def _new_chunk(self, text: str, **kwargs) -> None:
        self.chunks.append(LegalChunk(
            id=make_id("tour"),
            text=text,
            doc_type="tour",
            source_file=self.source_file,
            **kwargs,
        ))

    def parse(self, tours: list) -> List[LegalChunk]:
        self.chunks = []
        for tour in tours:
            self._parse_tour(tour)
        return self.chunks

    def _parse_tour(self, tour: dict) -> None:
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

        summary_meta = json.dumps({
            "p0": pr.get("min_price", 0),
            "p1": pr.get("max_price", 0),
            "d":  dur.get("days"),
            "n":  dur.get("nights"),
            "nd": deps[0]["date"] if deps else "",
            "tr": transport,
            "dc": tour.get("departure", {}).get("city", ""),
            "ds": tour.get("destinations", []),
            "inc": services.get("included", ""),
            "exc": services.get("excluded", ""),
            "pol_cancel":   policies.get("cancellation_policy", ""),
            "pol_children": policies.get("children_policy", ""),
            "pol_payment":  policies.get("payment_policy", ""),
            "pol_notes":    policies.get("notes", ""),
        }, ensure_ascii=False)

        # 1. Summary chunk
        self._new_chunk(
            text=(
                f"{title}. Loại tour: {tour_type}. Điểm đến: {dests}. "
                f"Thời gian: {duration_str}. Phương tiện: {transport}. Giá từ: {price_str}."
            ),
            chapter=tour_type, section=title, article=tid,
            clause="summary", point=tid, clause_full=summary_meta,
        )

        # 2. Departures chunk
        if deps:
            from collections import defaultdict
            by_date: dict = defaultdict(list)
            for d in deps:
                by_date[d.get("date", "")].append(d)
            dep_lines = [f"{title} – Lịch khởi hành và giá chi tiết:"]
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
                chapter=tour_type, section=title, article=tid,
                clause="departures", point=tid, clause_full=summary_meta,
            )

        # 3. Services chunk
        inc = services.get("included", "")
        exc = services.get("excluded", "")
        if inc or exc:
            svc_parts = [f"{title} – Dịch vụ:"]
            if inc: svc_parts.append(f"Bao gồm: {inc}")
            if exc: svc_parts.append(f"Không bao gồm: {exc}")
            self._new_chunk(
                text="\n".join(svc_parts),
                chapter=tour_type, section=title, article=tid,
                clause="services", point=tid, clause_full=summary_meta,
            )

        # 4. Policies chunk
        pol_cancel   = policies.get("cancellation_policy", "")
        pol_children = policies.get("children_policy", "")
        pol_payment  = policies.get("payment_policy", "")
        pol_notes    = policies.get("notes", "")
        if any([pol_cancel, pol_children, pol_payment, pol_notes]):
            pol_parts = [f"{title} – Chính sách:"]
            if pol_cancel:   pol_parts.append(f"Hủy tour: {pol_cancel}")
            if pol_children: pol_parts.append(f"Trẻ em: {pol_children}")
            if pol_payment:  pol_parts.append(f"Thanh toán: {pol_payment}")
            if pol_notes:    pol_parts.append(f"Lưu ý: {pol_notes}")
            self._new_chunk(
                text="\n".join(pol_parts),
                chapter=tour_type, section=title, article=tid,
                clause="policies", point=tid, clause_full=summary_meta,
            )

        # 5. Itinerary chunks (one per day)
        for day in tour.get("itinerary", []):
            day_num   = day.get("day", 0)
            day_title = day.get("title", "")
            desc      = day.get("description", "")
            locs      = ", ".join(day.get("locations", []))
            meals     = ", ".join(day.get("meals", []))
            overnight = day.get("overnight") or ""

            day_meta = json.dumps({
                "meals":     day.get("meals", []),
                "overnight": overnight,
                "day_label": f"Ngày {day_num}: {day_title}",
                "locations": day.get("locations", []),
            }, ensure_ascii=False)

            day_parts = [f"{title} – Ngày {day_num}: {day_title}."]
            if locs:     day_parts.append(f"Địa điểm: {locs}.")
            if desc:     day_parts.append(desc)
            if meals:    day_parts.append(f"Bữa ăn: {meals}.")
            if overnight: day_parts.append(f"Nghỉ đêm tại: {overnight}.")

            self._new_chunk(
                text=" ".join(day_parts),
                chapter=tour_type, section=title, article=tid,
                clause=f"day_{day_num}", point=tid, clause_full=day_meta,
            )


# ---------------------------------------------------------------------------
# Embedding engine
# ---------------------------------------------------------------------------

class _EmbeddingEngine:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = DEVICE):
        logger.info("Loading embedding model: %s on %s", model_name, device)
        self.model = BGEM3FlagModel(model_name, use_fp16=(device == "cuda"), device=device)
        self.model.model.eval()
        self.tokenizer = self.model.tokenizer
        self.model.encode(["warmup"], batch_size=1, max_length=32)
        logger.info("Embedding model ready.")

    @torch.no_grad()
    def encode_batch(self, texts: list[str], max_length: int = 512) -> tuple[np.ndarray, list[dict]]:
        results = self.model.encode(
            texts,
            batch_size=len(texts),
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense  = np.array(results["dense_vecs"], dtype=np.float32)
        sparse = [self._clean_sparse(lw) for lw in results["lexical_weights"]]
        return dense, sparse

    def _clean_sparse(self, lexical_weights: dict) -> dict[int, float]:
        specials = {
            self.tokenizer.cls_token_id, self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id, self.tokenizer.unk_token_id,
        }
        result: dict[int, float] = {}
        for k, w in lexical_weights.items():
            tid = int(k); fw = float(w)
            if tid not in specials and fw > 0:
                result[tid] = max(result.get(tid, 0.0), fw)
        return result


def _to_qdrant_sparse(d: dict[int, float]) -> SparseVector:
    idx = sorted(d.keys())
    return SparseVector(indices=idx, values=[d[i] for i in idx])


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def _ensure_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        logger.info("Dropping existing collection '%s'", name)
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))},
    )
    logger.info("Created collection '%s' (dense=%d + sparse)", name, DENSE_DIM)


# ---------------------------------------------------------------------------
# Core ingest
# ---------------------------------------------------------------------------

def ingest_chunks(
    chunks: List[LegalChunk],
    collection_name: str,
    engine: _EmbeddingEngine,
    client: QdrantClient,
) -> None:
    if not chunks:
        logger.warning("No chunks for '%s', skipping.", collection_name)
        return

    texts = [c.text for c in chunks]
    logger.info("Encoding %d chunks for '%s'...", len(texts), collection_name)

    all_dense:  list[np.ndarray] = []
    all_sparse: list[dict]       = []

    for i in tqdm(range(0, len(texts), ENCODE_BATCH), desc=f"Encoding [{collection_name}]"):
        batch = texts[i : i + ENCODE_BATCH]
        d, s  = engine.encode_batch(batch)
        all_dense.append(d)
        all_sparse.extend(s)

    dense_matrix = np.vstack(all_dense)

    points: list[PointStruct] = []
    for chunk, dense_vec, sparse_dict in zip(chunks, dense_matrix, all_sparse):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense":  dense_vec.tolist(),
                "sparse": _to_qdrant_sparse(sparse_dict),
            },
            payload={
                "page_content": chunk.text,
                "metadata":     chunk.to_metadata(),
            },
        ))

    logger.info("Upserting %d points to '%s'...", len(points), collection_name)
    for i in tqdm(range(0, len(points), UPSERT_BATCH), desc=f"Upserting [{collection_name}]"):
        client.upsert(
            collection_name=collection_name,
            points=points[i : i + UPSERT_BATCH],
            wait=False,
        )

    client.get_collection(collection_name)
    logger.info("Ingested %d chunks into '%s'", len(chunks), collection_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    LAW_PATH      = os.environ.get("LAW_PATH",      "../data/Luật giao thông đường bộ.docx")
    CIRCULAR_PATH = os.environ.get("CIRCULAR_PATH", "../data/Nghị định 168 năm 2024.docx")
    ADMISSION_TXT = os.environ.get("ADMISSION_TXT", "../data/admission.txt")
    TOUR_JSON     = os.environ.get("TOUR_JSON",      "../data/extracted_data.json")

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    engine = _EmbeddingEngine()

    # 1. Law
    _ensure_collection(qdrant, LAW_COLLECTION)
    law_chunks: List[LegalChunk] = []

    if Path(LAW_PATH).exists():
        p = LawParser(source_file=LAW_PATH)
        law_chunks += p.parse(load_docx_paragraphs(LAW_PATH))
        logger.info("Law docx: %d chunks", len(law_chunks))

        for file_name in [
            "../data/Nghị định 151 năm 2024.docx",
            "../data/Nghị định 160 năm 2024.docx",
        ]:
            p = LawParser(source_file=file_name)
            law_chunks += p.parse(load_docx_paragraphs(file_name))
            logger.info("Law docx: %d chunks total", len(law_chunks))

    if Path(CIRCULAR_PATH).exists():
        p  = CircularParser(source_file=CIRCULAR_PATH)
        cc = p.parse(load_docx_text(CIRCULAR_PATH))
        law_chunks += cc
        logger.info("Circular docx: %d chunks (running total: %d)", len(cc), len(law_chunks))

    ingest_chunks(law_chunks, LAW_COLLECTION, engine, qdrant)

    # 2. Admission
    _ensure_collection(qdrant, ADMISSION_COLLECTION)
    admission_chunks: List[LegalChunk] = []

    if Path(ADMISSION_TXT).exists():
        ap = AdmissionConsultingParser(source_file=ADMISSION_TXT)
        with open(ADMISSION_TXT, "r", encoding="utf-8") as f:
            raw = f.read()
        for block in [b.strip() for b in raw.split("\n\n") if b.strip()]:
            ap._new_chunk(
                text=block,
                chapter=None, section=None, clause=None,
                article="Tài liệu tư vấn tuyển sinh UET 2026",
                article_full=block,
            )
        admission_chunks = ap.chunks
        logger.info("Admission: %d chunks", len(admission_chunks))

    ingest_chunks(admission_chunks, ADMISSION_COLLECTION, engine, qdrant)

    # 3. Tours
    _ensure_collection(qdrant, TOUR_COLLECTION)
    tour_chunks: List[LegalChunk] = []

    if Path(TOUR_JSON).exists():
        with open(TOUR_JSON, "r", encoding="utf-8") as f:
            tours = json.load(f)
        tp = TourParser(source_file=TOUR_JSON)
        tour_chunks = tp.parse(tours)
        logger.info("Tours: %d chunks", len(tour_chunks))

    ingest_chunks(tour_chunks, TOUR_COLLECTION, engine, qdrant)

    # Summary
    print("\n=== Ingest Summary ===")
    for col in [LAW_COLLECTION, ADMISSION_COLLECTION, TOUR_COLLECTION]:
        info = qdrant.get_collection(col)
        print(f"  {col:12s}: {info.points_count or 0:>6d} vectors")
    print("✅ Done.")