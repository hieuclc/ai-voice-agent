"""
benchmark_log_sink.py - singleton, import once in server.py
"""

import re
import contextvars
from datetime import datetime
from collections import defaultdict
from benchmark_metrics import collect_metric, collect_text

current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_session_id", default=None
)

_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"
_TS_RE  = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")


def _parse_ts(line: str) -> "datetime | None":
    m = _TS_RE.search(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), _TS_FMT)
    except ValueError:
        return None


_IMMEDIATE_PATTERNS = [
    (re.compile(r"LLMService[^T]*?TTFB:\s*([\d.]+)", re.I),           "llm", "ttfb",            "ms",     1000),
    (re.compile(r"LLMService.*?usage.*?tokens.*?(\d+)", re.I),         "llm", "usage_tokens",    "tokens", 1),
    (re.compile(r"STTService.*?TTFB:\s*([\d.]+)", re.I),               "stt", "ttfb",            "ms",     1000),
    (re.compile(r"STTService.*?processing time:\s*([\d.]+)", re.I),    "stt", "processing_time", "ms",     1000),
    (re.compile(r"LLMService[^T]*?processing time:\s*([\d.]+)", re.I), "llm", "processing_time", "ms",     1000),
]

_TTS_GEN_RE   = re.compile(r"Generating TTS \[(.+?)\]", re.I)
_TTS_PROC_RE  = re.compile(r"TTS\w*\s*(?:#\d+\s*)?processing time:\s*([\d.]+)", re.I)
_TTS_CHARS_RE = re.compile(r"TTS\w*\s*(?:#\d+\s*)?usage characters:\s*([\d.]+)", re.I)
_LLM_TTFB_RE  = re.compile(r"LLMService[^T]*?TTFB:", re.I)

_TEXT_PATTERNS = [
    (re.compile(r"TranscriptionFrame.*?text=['\"](.+?)['\"]", re.I), "stt", "transcript"),
    (re.compile(r"Transcription:\s*\[?(.+?)\]?\s*$", re.I),          "stt", "transcript"),
    (re.compile(r"LLMFullResponseEnd.*?text=['\"](.+?)['\"]", re.I), "llm", "response"),
]


class BenchmarkLogSink:
    def __init__(self):
        self._in_turn:       dict = defaultdict(bool)
        self._llm_ttfb_ts:   dict = defaultdict(lambda: None)
        self._tts_first_ts:  dict = defaultdict(lambda: None)
        self._tts_proc_acc:  dict = defaultdict(float)
        self._tts_chars_acc: dict = defaultdict(float)
        self._tts_texts:     dict = defaultdict(list)
        self._flushed:       dict = defaultdict(bool)

    def _reset_turn(self, sid):
        self._in_turn[sid]       = False
        self._llm_ttfb_ts[sid]   = None
        self._tts_first_ts[sid]  = None
        self._tts_proc_acc[sid]  = 0.0
        self._tts_chars_acc[sid] = 0.0
        self._tts_texts[sid]     = []
        self._flushed[sid]       = False

    def flush_session(self, sid: str):
        """Call from GET /benchmark/session/{sid} before returning metrics."""
        if self._in_turn[sid] and not self._flushed[sid]:
            self._flush_tts(sid)
            self._flushed[sid] = True

    def __call__(self, message):
        line = str(message)
        sid  = current_session_id.get()
        if not sid:
            return

        ts = _parse_ts(line)

        for pattern, stage, metric, unit, mult in _IMMEDIATE_PATTERNS:
            m = pattern.search(line)
            if m:
                try:
                    collect_metric(sid, stage, metric, float(m.group(1)) * mult, unit)
                except (IndexError, ValueError):
                    pass

        if _LLM_TTFB_RE.search(line):
            if self._in_turn[sid] and not self._flushed[sid]:
                self._flush_tts(sid)
            self._reset_turn(sid)
            self._in_turn[sid]     = True
            self._llm_ttfb_ts[sid] = ts

        m = _TTS_GEN_RE.search(line)
        if m:
            text = m.group(1).strip()
            self._tts_texts[sid].append(text)
            if self._in_turn[sid] and self._tts_first_ts[sid] is None and ts is not None:
                self._tts_first_ts[sid] = ts

        m = _TTS_PROC_RE.search(line)
        if m:
            try:
                self._tts_proc_acc[sid] += float(m.group(1))
            except ValueError:
                pass

        m = _TTS_CHARS_RE.search(line)
        if m:
            try:
                self._tts_chars_acc[sid] += float(m.group(1))
            except ValueError:
                pass

        for pattern, stage, key in _TEXT_PATTERNS:
            m = pattern.search(line)
            if m:
                collect_text(sid, stage, key, m.group(1).strip())

    def _flush_tts(self, sid: str):
        first_ts = self._tts_first_ts[sid]
        llm_ts   = self._llm_ttfb_ts[sid]
        if first_ts and llm_ts:
            ttfb_ms = (first_ts - llm_ts).total_seconds() * 1000
            if ttfb_ms >= 0:
                collect_metric(sid, "tts", "ttfb", ttfb_ms, "ms")

        proc_s = self._tts_proc_acc[sid]
        if proc_s > 0:
            collect_metric(sid, "tts", "processing_time", proc_s * 1000, "ms")

        chars = self._tts_chars_acc[sid]
        if chars > 0:
            collect_metric(sid, "tts", "usage_chars", chars, "chars")

        texts = self._tts_texts[sid]
        if texts:
            collect_text(sid, "tts", "text", " | ".join(texts))


# Singleton — import this in server.py only, never in bot.py
benchmark_sink = BenchmarkLogSink()