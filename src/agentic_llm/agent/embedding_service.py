"""
embedding_service.py — Singleton embedding service dùng AITeamVN/Vietnamese_Embedding.

Model được load MỘT LẦN DUY NHẤT khi module được import lần đầu (eager init),
hoặc lazily qua get_embedding_service() nếu muốn kiểm soát thời điểm load.

Usage
-----
    from embedding_service import get_embedding_service, VIETNAMESE_EMBEDDING_FUNC

    # Dùng trực tiếp với LightRAG:
    rag = LightRAG(
        ...
        embedding_func=VIETNAMESE_EMBEDDING_FUNC,
    )

    # Hoặc encode thủ công:
    svc = await get_embedding_service()
    vectors = await svc.embed(["câu một", "câu hai"])   # np.ndarray (2, 1024)
"""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME: str = os.environ.get(
    "EMBEDDING_MODEL_NAME", "AITeamVN/Vietnamese_Embedding"
)
DEVICE: str = os.environ.get("DEVICE", "cpu")
EMBEDDING_DIM: int = 1024
EMBEDDING_MAX_TOKENS: int = 512


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------


class EmbeddingService:
    """
    Thin wrapper around SentenceTransformer that:
    - loads the model exactly once
    - exposes an async encode() method (runs in thread-pool)
    - is safe to call from multiple concurrent coroutines
    """

    def __init__(self, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._ready = asyncio.Event()
        self._load_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load the model in a thread-pool (non-blocking). Idempotent."""
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            loop = asyncio.get_event_loop()
            logger.info(
                "EmbeddingService: loading model '%s' on device='%s'…",
                self._model_name,
                self._device,
            )
            self._model = await loop.run_in_executor(None, self._load_model)
            self._ready.set()
            logger.info("EmbeddingService: model ready.")

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self._model_name, device=self._device)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    async def embed(self, texts: list[str]) -> np.ndarray:
        """
        Encode *texts* and return an (N, EMBEDDING_DIM) numpy array.
        Waits for the model to be ready if initialize() is still running.
        """
        await self._ready.wait()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, normalize_embeddings=True),
        )

    # ------------------------------------------------------------------
    # LightRAG-compatible async callable
    # ------------------------------------------------------------------

    async def __call__(self, texts: list[str]) -> np.ndarray:
        """Allows the service instance to be passed directly as EmbeddingFunc.func."""
        return await self.embed(texts)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_service_instance: EmbeddingService | None = None
_service_lock = asyncio.Lock()


async def get_embedding_service() -> EmbeddingService:
    """Return the shared EmbeddingService, initializing it on first call."""
    global _service_instance
    if _service_instance is not None:
        return _service_instance
    async with _service_lock:
        if _service_instance is not None:
            return _service_instance
        svc = EmbeddingService(EMBEDDING_MODEL_NAME, DEVICE)
        await svc.initialize()
        _service_instance = svc
    return _service_instance


# ---------------------------------------------------------------------------
# Pre-built EmbeddingFunc for LightRAG
# ---------------------------------------------------------------------------
# This is constructed lazily — the actual model isn't loaded until the first
# call to the func, which triggers get_embedding_service().

from lightrag.utils import EmbeddingFunc  # noqa: E402


async def _embed_func(texts: list[str]) -> np.ndarray:
    svc = await get_embedding_service()
    return await svc.embed(texts)


VIETNAMESE_EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=EMBEDDING_DIM,
    max_token_size=EMBEDDING_MAX_TOKENS,
    func=_embed_func,
)