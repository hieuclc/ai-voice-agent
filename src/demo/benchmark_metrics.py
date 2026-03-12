"""
benchmark_metrics.py - Thread-safe in-memory store for per-session pipeline metrics.

Each pipeline event is keyed by session_id. The server populates this store
via collect_metric(); the benchmark client reads it via /benchmark/session/{id}.
"""

import time
from collections import defaultdict
from threading import Lock
from typing import Any

_lock = Lock()

# session_id -> list of metric dicts
_store: dict[str, list[dict]] = defaultdict(list)

# session_id -> list of text dicts {stage, key, value, ts}
_text_store: dict[str, list[dict]] = defaultdict(list)


def collect_metric(session_id: str, stage: str, metric: str, value: float, unit: str = "ms"):
    entry = {
        "ts": time.time(),
        "stage": stage,
        "metric": metric,
        "value": value,
        "unit": unit,
    }
    with _lock:
        _store[session_id].append(entry)


def collect_text(session_id: str, stage: str, key: str, value: str):
    """Store a text output (STT transcript, LLM response, TTS text)."""
    entry = {
        "ts": time.time(),
        "stage": stage,
        "key": key,
        "value": value,
    }
    with _lock:
        _text_store[session_id].append(entry)


def get_metrics(session_id: str) -> list[dict]:
    with _lock:
        return list(_store.get(session_id, []))


def get_texts(session_id: str) -> list[dict]:
    with _lock:
        return list(_text_store.get(session_id, []))


def clear_metrics(session_id: str):
    with _lock:
        _store.pop(session_id, None)
        _text_store.pop(session_id, None)