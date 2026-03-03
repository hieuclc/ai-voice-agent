"""
reranker.py — Cross-encoder reranker for Vietnamese legal RAG pipeline.

Model: Alibaba-NLP/gte-multilingual-reranker-base
  - Multilingual, handles Vietnamese well
  - Fast inference with torch.inference_mode + argpartition

Usage:
    reranker = LawReranker()          # lazy-loaded, call once at startup
    reranker.startup()
    top_docs = reranker.rerank(query, documents, top_k=5)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Model mặc định — có thể override qua env RERANKER_MODEL
DEFAULT_RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL",
    "Alibaba-NLP/gte-multilingual-reranker-base",
)


class LawReranker:
    """
    Cross-encoder reranker. Singleton — khởi tạo một lần tại startup.

    Tối ưu hiệu năng:
    - torch.inference_mode() để giảm overhead autograd
    - argpartition O(n) thay vì argsort O(n log n)
    - fp16 trên CUDA để tăng throughput
    - Warmup CUDA kernels lần đầu để tránh latency cold-start

    Args:
        model_name: HuggingFace model id.
        max_length: Số token tối đa cho mỗi query-doc pair.
        batch_size: Số pairs xử lý mỗi batch.
        device: "cuda" / "cpu" / None (auto-detect).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        max_length: int = 1024,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._device_arg = device
        self._model: Optional[CrossEncoder] = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def startup(self) -> None:
        """Load model. Idempotent — safe to call multiple times."""
        if self._initialized:
            return

        logger.info("Loading reranker: %s", self.model_name)
        t0 = time.time()

        # Device detection
        if self._device_arg:
            device = self._device_arg
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        cuda = device == "cuda"
        if cuda:
            try:
                gpu = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info("CUDA: %s (%.1f GB)", gpu, mem)
                # Allow TF32 on Ampere+ for ~2x speed with no quality loss
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                logger.info("CUDA available")
        else:
            logger.warning("Reranker running on CPU — will be slower")

        model_kwargs = {"dtype": torch.float16} if cuda else {}

        self._model = CrossEncoder(
            self.model_name,
            device=device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        self._model.max_length = self.max_length

        # Warmup: compile CUDA kernels on first real batch
        if cuda:
            logger.info("Warming up reranker CUDA kernels...")
            warmup_pairs = [["warmup query", "warmup document"]] * min(self.batch_size, 4)
            with torch.inference_mode():
                self._model.predict(warmup_pairs, batch_size=len(warmup_pairs), show_progress_bar=False)

        self._initialized = True
        logger.info("Reranker ready in %.2fs (device=%s)", time.time() - t0, device)

    def rerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        """
        Rerank a list of document dicts.

        Args:
            query: User query string.
            documents: List of dicts, each must have key "text" containing the passage.
            top_k: Number of top documents to return.

        Returns:
            Top-k documents sorted by relevance (highest first).
            Each dict retains all original keys plus a new "_rerank_score" key.
        """
        if not self._initialized or self._model is None:
            raise RuntimeError("Call startup() before rerank()")

        if not documents:
            return []

        if len(documents) <= top_k:
            return documents

        t0 = time.time()
        pairs = [[query, doc["text"]] for doc in documents]

        with torch.inference_mode():
            scores: np.ndarray = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_tensor=False,
            )

        scores = np.array(scores, dtype=np.float32)

        # O(n) partial sort: faster than full argsort for large candidate sets
        if len(scores) > top_k:
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        else:
            top_idx = np.argsort(scores)[::-1][:top_k]

        result = []
        for i in top_idx:
            doc = dict(documents[i])
            doc["_rerank_score"] = float(scores[i])
            result.append(doc)

        logger.info(
            "Reranked %d → %d docs in %.3fs | top score=%.4f",
            len(documents), top_k, time.time() - t0,
            float(scores[top_idx[0]]) if len(top_idx) else 0,
        )
        return result


# ---------------------------------------------------------------------------
# Module-level singleton — shared across the whole process
# ---------------------------------------------------------------------------

_reranker_instance: Optional[LawReranker] = None


def get_reranker() -> LawReranker:
    """Return the process-wide LawReranker singleton (lazy-init)."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = LawReranker()
        _reranker_instance.startup()
    return _reranker_instance