"""
build_db.py — Indexing pipeline cho toàn bộ dữ liệu vào Qdrant (hybrid: dense + sparse).

Collections:
  - law        : văn bản pháp luật (Luật, Nghị định, Thông tư)  ← MIGRATE từ ChromaDB
  - admission  : tư vấn tuyển sinh                               ← MIGRATE từ ChromaDB
  - tours      : dữ liệu tour du lịch                            ← MIGRATE từ ChromaDB

Hybrid strategy:
  - Dense vector  : AITeamVN/Vietnamese_Embedding_v2 (BGE-M3 fine-tuned, dim=1024)
  - Sparse vector : lexical weights từ cùng BGE-M3 model (SPLADE-style)
  - Distance      : COSINE cho dense, default sparse index

Thay đổi so với build_db.py cũ:
  - Backend: ChromaDB → Qdrant
  - Chunking law: giữ nguyên LawParser + CircularParser
  - Thêm sparse vectors cho hybrid search
  - Batch upsert với wait=False để tăng tốc ingest

Run:
    python build_db.py
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

import numpy as np
import torch
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

# Import toàn bộ parsers từ file gốc (đã đổi tên để tránh conflict)
# Lưu ý: đổi tên file build_db gốc thành build_db_parsers.py
from build_db_parsers import (
    LawParser,
    CircularParser,
    AdmissionConsultingParser,
    TourParser,
    LegalChunk,
    load_docx_paragraphs,
    load_docx_text,
)

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
# Embedding engine
# ---------------------------------------------------------------------------

class _EmbeddingEngine:
    """
    Wrapper quanh BGEM3FlagModel (AITeamVN/Vietnamese_Embedding_v2).
    Một lần encode → cả dense lẫn sparse, tránh double-inference.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = DEVICE):
        logger.info("Loading embedding model: %s on %s", model_name, device)
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=(device == "cuda"),
            device=device,
        )
        self.model.model.eval()
        self.tokenizer = self.model.tokenizer
        # Warmup để compile CUDA kernels nếu có GPU
        self.model.encode(["warmup"], batch_size=1, max_length=32)
        logger.info("Embedding model ready.")

    @torch.no_grad()
    def encode_batch(self, texts: list[str], max_length: int = 512) -> tuple[np.ndarray, list[dict]]:
        """
        Encode một batch.

        Returns:
            dense  : np.ndarray shape (N, 1024), float32
            sparse : list[dict[int, float]]  — {token_id: weight}
        """
        results = self.model.encode(
            texts,
            batch_size=len(texts),
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.array(results["dense_vecs"], dtype=np.float32)
        sparse = [self._clean_sparse(lw) for lw in results["lexical_weights"]]
        return dense, sparse

    def _clean_sparse(self, lexical_weights: dict) -> dict[int, float]:
        """
        BGE-M3 trả về {str(token_id): weight}.
        Convert về {int: float}, filter special tokens, deduplicate với max.
        """
        specials = {
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        }
        result: dict[int, float] = {}
        for k, w in lexical_weights.items():
            tid = int(k)
            fw  = float(w)
            if tid not in specials and fw > 0:
                result[tid] = max(result.get(tid, 0.0), fw)
        return result


def _to_qdrant_sparse(d: dict[int, float]) -> SparseVector:
    idx = sorted(d.keys())
    val = [d[i] for i in idx]
    return SparseVector(indices=idx, values=val)


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def _ensure_collection(client: QdrantClient, name: str) -> None:
    """Xoá rồi tạo lại collection với dual-vector (dense + sparse)."""
    if client.collection_exists(name):
        logger.info("Dropping existing collection '%s'", name)
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
    logger.info("Created collection '%s' (dense=%d + sparse)", name, DENSE_DIM)


# ---------------------------------------------------------------------------
# Core ingest
# ---------------------------------------------------------------------------

def ingest_chunks(
    chunks: list[LegalChunk],
    collection_name: str,
    engine: _EmbeddingEngine,
    client: QdrantClient,
) -> None:
    """
    Encode + upsert list[LegalChunk] vào Qdrant collection.

    Payload format:
        {
          "page_content": str,          # text của chunk
          "metadata": dict,             # toàn bộ metadata từ LegalChunk.to_chromadb()
        }
    """
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
        _, text, meta = chunk.to_chromadb()
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense":  dense_vec.tolist(),
                    "sparse": _to_qdrant_sparse(sparse_dict),
                },
                payload={
                    "page_content": text,
                    "metadata":     meta,
                },
            )
        )

    logger.info("Upserting %d points to '%s'...", len(points), collection_name)
    for i in tqdm(range(0, len(points), UPSERT_BATCH), desc=f"Upserting [{collection_name}]"):
        client.upsert(
            collection_name=collection_name,
            points=points[i : i + UPSERT_BATCH],
            wait=False,  # async write cho throughput cao hơn
        )

    # Đảm bảo flush trước khi kết thúc
    client.get_collection(collection_name)
    logger.info("Ingested %d chunks into '%s'", len(chunks), collection_name)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    LAW_PATH      = os.environ.get("LAW_PATH",      "../data/Luật giao thông đường bộ.docx")
    CIRCULAR_PATH = os.environ.get("CIRCULAR_PATH", "../data/Nghị định 168 năm 2024.docx")
    ADMISSION_TXT = os.environ.get("ADMISSION_TXT", "../data/admission.txt")
    TOUR_JSON     = os.environ.get("TOUR_JSON",      "../data/extracted_data.json")

    qdrant  = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    engine  = _EmbeddingEngine()

    # ── 1. Law ───────────────────────────────────────────────────────────
    _ensure_collection(qdrant, LAW_COLLECTION)
    law_chunks: list[LegalChunk] = []

    if Path(LAW_PATH).exists():
        p = LawParser(source_file=LAW_PATH)
        law_chunks += p.parse(load_docx_paragraphs(LAW_PATH))
        logger.info("Law docx: %d chunks", len(law_chunks))

        for file_name in ["../data/Nghị định 151 năm 2024.docx", "../data/Nghị định 160 năm 2024.docx"]:
            p = LawParser(source_file=file_name)
            law_chunks += p.parse(load_docx_paragraphs(file_name))
            logger.info("Law docx: %d chunks", len(law_chunks))

    if Path(CIRCULAR_PATH).exists():
        p   = CircularParser(source_file=CIRCULAR_PATH)
        cc  = p.parse(load_docx_text(CIRCULAR_PATH))
        law_chunks += cc
        logger.info("Circular docx: %d chunks (running total: %d)", len(cc), len(law_chunks))

    ingest_chunks(law_chunks, LAW_COLLECTION, engine, qdrant)

    # ── 2. Admission ─────────────────────────────────────────────────────
    _ensure_collection(qdrant, ADMISSION_COLLECTION)
    admission_chunks: list[LegalChunk] = []

    if Path(ADMISSION_TXT).exists():
        ap = AdmissionConsultingParser(source_file=ADMISSION_TXT)
        with open(ADMISSION_TXT, "r", encoding="utf-8") as f:
            raw = f.read()
        for block in [b.strip() for b in raw.split("\n\n") if b.strip()]:
            ap._new_chunk(
                text=block,
                chapter=None,
                section=None,
                clause=None,
                article="Tài liệu tư vấn tuyển sinh UET 2026",
                article_full=block,
            )
        admission_chunks = ap.chunks
        logger.info("Admission: %d chunks", len(admission_chunks))

    ingest_chunks(admission_chunks, ADMISSION_COLLECTION, engine, qdrant)

    # ── 3. Tours ─────────────────────────────────────────────────────────
    _ensure_collection(qdrant, TOUR_COLLECTION)
    tour_chunks: list[LegalChunk] = []

    if Path(TOUR_JSON).exists():
        with open(TOUR_JSON, "r", encoding="utf-8") as f:
            tours = json.load(f)
        tp = TourParser(source_file=TOUR_JSON)
        tour_chunks = tp.parse(tours)
        logger.info("Tours: %d chunks", len(tour_chunks))

    ingest_chunks(tour_chunks, TOUR_COLLECTION, engine, qdrant)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== Ingest Summary ===")
    for col in [LAW_COLLECTION, ADMISSION_COLLECTION, TOUR_COLLECTION]:
        info = qdrant.get_collection(col)
        print(f"  {col:12s}: {info.points_count or 0:>6d} vectors")
    print("✅ Done.")