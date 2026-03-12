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


def collect_metric(session_id: str, stage: str, metric: str, value: float, unit: str = "ms"):
    """
    Record a single metric event for a session.

    stage: 'stt' | 'llm' | 'tts'
    metric: 'ttfb' | 'processing_time' | 'usage_chars' | ...
    value: numeric value
    unit: 'ms' | 'chars' | ...
    """
    entry = {
        "ts": time.time(),
        "stage": stage,
        "metric": metric,
        "value": value,
        "unit": unit,
    }
    with _lock:
        _store[session_id].append(entry)


def get_metrics(session_id: str) -> list[dict]:
    with _lock:
        return list(_store.get(session_id, []))


def clear_metrics(session_id: str):
    with _lock:
        _store.pop(session_id, None)