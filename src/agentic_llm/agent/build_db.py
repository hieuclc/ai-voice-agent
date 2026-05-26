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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
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

QDRANT_HOST     = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.environ.get("QDRANT_PORT", "6333"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL_NAME", "AITeamVN/Vietnamese_Embedding_v2")
DEVICE          = os.environ.get("DEVICE", "cpu")
DENSE_DIM       = 1024
ENCODE_BATCH    = 32
UPSERT_BATCH    = 256

LAW_COLLECTION       = "law"
ADMISSION_COLLECTION = "admission"
TOUR_COLLECTION      = "tours"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
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
            "doc_type":     self.doc_type,
            "source_file":  self.source_file,
            "source":       ".".join(self.source_file.split("/")[-1].split(".")[:-1]),
            "chapter":      self.chapter or "",
            "section":      self.section or "",
            "article":      self.article or "",
            "clause":       self.clause or "",
            "point":        self.point or "",
            "article_full": self.article_full,
            "clause_full":  self.clause_full,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_docx_paragraphs(path: str) -> List[str]:
    doc = Document(path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


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

class LegalDocParser:
    """
    Parser chung cho Luật, Nghị định, Thông tư.
    Cấu trúc: [Chương] → [Mục] → Điều → Khoản → Điểm
    Chương/Mục là optional — Nghị định/Thông tư thường không có.
    """

    DOC_TYPE_MAP = {
        "luat":      re.compile(r"Luật", re.IGNORECASE),
        "nghi_dinh": re.compile(r"Nghị\s*định", re.IGNORECASE),
        "thong_tu":  re.compile(r"Thông\s*tư", re.IGNORECASE),
    }

    def __init__(self, source_file: str = "", doc_type: str = "luat"):
        self.source_file = source_file
        self.doc_type    = doc_type
        self.chunks: List[Chunk] = []

    @classmethod
    def from_file(cls, path: str) -> "LegalDocParser":
        """Auto-detect doc_type từ tên file."""
        name = Path(path).name
        for dtype, pattern in cls.DOC_TYPE_MAP.items():
            if pattern.search(name):
                return cls(source_file=path, doc_type=dtype)
        return cls(source_file=path, doc_type="luat")

    def add_chunk(self, text: str, **kwargs) -> None:
        self.chunks.append(Chunk(
            id=make_id(self.doc_type),
            text=text,
            doc_type=self.doc_type,
            source_file=self.source_file,
            **kwargs,
        ))

    def parse(self, source: str | List[str]) -> List[Chunk]:
        """Nhận raw string hoặc list of paragraphs — normalize thành list."""
        if isinstance(source, str):
            paragraphs = [line.strip() for line in source.splitlines() if line.strip()]
        else:
            paragraphs = [p for p in source if p]

        self.chunks = []
        chapter = section = article_title = None
        article_lines: List[str] = []

        def flush() -> None:
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

    def _process_article(
        self,
        title: str,
        lines: List[str],
        chapter: Optional[str],
        section: Optional[str],
    ) -> None:
        article_full = title + "\n" + "\n".join(lines)
        clauses = self._split_clauses(lines)

        if not clauses:
            self.add_chunk(
                text=article_full,
                chapter=chapter, section=section,
                article=title, article_full=article_full,
            )
            return

        for num, clause_lines in clauses.items():
            clause_title = f"Khoản {num}"
            clause_full  = "\n".join(clause_lines)
            points       = self._split_points(clause_lines)

            if not points:
                self.add_chunk(
                    text=f"{title}\n{clause_full}",
                    chapter=chapter, section=section,
                    article=title, clause=clause_title,
                    article_full=article_full, clause_full=clause_full,
                )
            else:
                for letter, point_text in points.items():
                    self.add_chunk(
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
        return {
            RE_POINT.match(line).group(1): line
            for line in lines
            if RE_POINT.match(line)
        }


class AdmissionParser:
    """
    Parser cho tài liệu tư vấn tuyển sinh (plain text).
    Mỗi paragraph (cách nhau bằng dòng trống) → 1 chunk.
    """

    def __init__(self, source_file: str = ""):
        self.source_file = source_file
        self.chunks: List[Chunk] = []

    def add_chunk(self, text: str, **kwargs) -> None:
        self.chunks.append(Chunk(
            id=make_id("tu_van_tuyen_sinh"),
            text=text,
            doc_type="tu_van_tuyen_sinh",
            source_file=self.source_file,
            **kwargs,
        ))

    def parse(self, raw: str) -> List[Chunk]:
        self.chunks = []
        for block in [b.strip() for b in raw.split("\n\n") if b.strip()]:
            self.add_chunk(
                text=block,
                chapter=None, section=None, clause=None,
                article="Tài liệu tư vấn tuyển sinh UET 2026",
                article_full=block,
            )
        return self.chunks


class TourParser:
    """
    Parser cho dữ liệu tour du lịch (JSON).
    Mỗi tour → summary + departures + services + policies + day_N chunks.

    Mapping sang Chunk:
      point        ← tour_id
      article      ← tour_id (legacy)
      chapter      ← tour_type
      section      ← title
      clause       ← "summary" | "departures" | "services" | "policies" | "day_N"
      clause_full  ← compact JSON metadata
    """

    def __init__(self, source_file: str = "extracted_data.json"):
        self.source_file = source_file
        self.chunks: List[Chunk] = []

    @staticmethod
    def _fmt_price(amount: int) -> str:
        return f"{amount:,}".replace(",", ".")

    def add_chunk(self, text: str, **kwargs) -> None:
        self.chunks.append(Chunk(
            id=make_id("tour"),
            text=text,
            doc_type="tour",
            source_file=self.source_file,
            **kwargs,
        ))

    def parse(self, tours: list) -> List[Chunk]:
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
        self.add_chunk(
            text=(
                f"{title}. Loại tour: {tour_type}. Điểm đến: {dests}. "
                f"Thời gian: {duration_str}. Phương tiện: {transport}. Giá từ: {price_str}."
            ),
            chapter=tour_type, section=title, article=tid,
            clause="summary", point=tid, clause_full=summary_meta,
        )

        # 2. Departures chunk
        if deps:
            by_date: dict = defaultdict(list)
            for d in deps:
                by_date[d.get("date", "")].append(d)
            dep_lines = [f"{title} – Lịch khởi hành và giá chi tiết:"]
            for date, entries in sorted(by_date.items()):
                for e in entries:
                    hotel = e.get("hotel_standard", "")
                    price = self._fmt_price(e.get("price", 0))
                    slots = e.get("available_slots", 0)
                    dep_lines.append(
                        f"  Ngày {date} – {hotel}: {price} VND"
                        + (f" (còn {slots} chỗ)" if slots else "")
                    )
            self.add_chunk(
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
            self.add_chunk(
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
            self.add_chunk(
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
            if locs:      day_parts.append(f"Địa điểm: {locs}.")
            if desc:      day_parts.append(desc)
            if meals:     day_parts.append(f"Bữa ăn: {meals}.")
            if overnight: day_parts.append(f"Nghỉ đêm tại: {overnight}.")

            self.add_chunk(
                text=" ".join(day_parts),
                chapter=tour_type, section=title, article=tid,
                clause=f"day_{day_num}", point=tid, clause_full=day_meta,
            )


# ---------------------------------------------------------------------------
# Embedding engine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
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


def to_qdrant_sparse(d: dict[int, float]) -> SparseVector:
    idx = sorted(d.keys())
    return SparseVector(indices=idx, values=[d[i] for i in idx])


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def ensure_collection(client: QdrantClient, name: str) -> None:
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
    chunks: List[Chunk],
    collection_name: str,
    engine: EmbeddingEngine,
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
                "sparse": to_qdrant_sparse(sparse_dict),
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
    LAW_FILES = [
        "../data/Luật giao thông đường bộ.docx",
        "../data/Nghị định 168 năm 2024.docx",
        "../data/Nghị định 151 năm 2024.docx",
        "../data/Nghị định 160 năm 2024.docx",
    ]
    ADMISSION_TXT = "../data/admission.txt"
    TOUR_JSON     = "../data/extracted_data.json"

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    engine = EmbeddingEngine()

    # 1. Law
    ensure_collection(qdrant, LAW_COLLECTION)
    law_chunks: List[Chunk] = []

    for file_path in LAW_FILES:
        if not Path(file_path).exists():
            logger.warning("File not found, skipping: %s", file_path)
            continue
        p      = LegalDocParser.from_file(file_path)
        chunks = p.parse(load_docx_paragraphs(file_path))
        law_chunks += chunks
        logger.info("%s → %d chunks (total: %d)", Path(file_path).name, len(chunks), len(law_chunks))

    ingest_chunks(law_chunks, LAW_COLLECTION, engine, qdrant)

    # 2. Admission
    ensure_collection(qdrant, ADMISSION_COLLECTION)
    admission_chunks: List[Chunk] = []

    if Path(ADMISSION_TXT).exists():
        with open(ADMISSION_TXT, "r", encoding="utf-8") as f:
            raw = f.read()
        admission_chunks = AdmissionParser(source_file=ADMISSION_TXT).parse(raw)
        logger.info("Admission: %d chunks", len(admission_chunks))

    ingest_chunks(admission_chunks, ADMISSION_COLLECTION, engine, qdrant)

    # 3. Tours
    ensure_collection(qdrant, TOUR_COLLECTION)
    tour_chunks: List[Chunk] = []

    if Path(TOUR_JSON).exists():
        with open(TOUR_JSON, "r", encoding="utf-8") as f:
            tours = json.load(f)
        tour_chunks = TourParser(source_file=TOUR_JSON).parse(tours)
        logger.info("Tours: %d chunks", len(tour_chunks))

    ingest_chunks(tour_chunks, TOUR_COLLECTION, engine, qdrant)

    # Summary
    print("\n=== Ingest Summary ===")
    for col in [LAW_COLLECTION, ADMISSION_COLLECTION, TOUR_COLLECTION]:
        info = qdrant.get_collection(col)
        print(f"  {col:12s}: {info.points_count or 0:>6d} vectors")
    print("✅ Done.")