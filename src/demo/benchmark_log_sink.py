"""
benchmark_log_sink.py - singleton, import once in server.py

Metrics collected per turn:
  llm / llm_time          : LLM call start → first "Generating TTS" line  (ms)
  tts / first_sentence    : processing time of the very first TTS chunk     (ms)
  tts / send_audio_time   : first TTS chunk done → "Bot started speaking"  (ms)
  tts / total_time        : first "Generating TTS" → last TTS chunk done   (ms)
  stt / ttfb              : STT TTFB as reported by pipecat                 (ms)
  stt / processing_time   : STT processing time as reported by pipecat      (ms)
  llm / usage_tokens      : LLM token usage                                 (tokens)
  tts / usage_chars       : TTS character usage                             (chars)
"""

import re
import contextvars
from datetime import datetime
from collections import defaultdict
from benchmark_metrics import collect_metric, collect_text

current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_session_id", default=None
)

_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.(\d+)")


def _parse_ts(line: str) -> "datetime | None":
    """Parse timestamp from pipecat log lines.
    Handles both 3-digit ms  (07:51:48.546)
    and 6-digit µs formats   (07:51:48.546123).
    """
    m = _TS_RE.search(line)
    if not m:
        return None
    try:
        base = m.group(1)
        frac = m.group(2).ljust(6, "0")[:6]  # normalise to exactly 6 digits
        return datetime.strptime(f"{base}.{frac}", "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None


# ── Immediate single-value metrics ───────────────────────────────────────────
_IMMEDIATE_PATTERNS = [
    (re.compile(r"LLMService.*?usage.*?tokens.*?(\d+)", re.I), "llm", "usage_tokens", "tokens", 1),
    (re.compile(r"STTService.*?TTFB:\s*([\d.]+)", re.I),       "stt", "ttfb",          "ms",    1000),
    (re.compile(r"STTService.*?processing time:\s*([\d.]+)", re.I), "stt", "processing_time", "ms", 1000),
]

# ── Regex for turn events ─────────────────────────────────────────────────────
_LLM_TTFB_RE       = re.compile(r"LLMService[^T]*?TTFB:", re.I)
# Matches ttsv2:run_tts lines: "ZipVoiceTTS: [text]" — fired once per TTS chunk
_TTS_GEN_RE        = re.compile(r"ZipVoiceTTS\w*:\s*\[(.+?)\]", re.I)
_TTS_PROC_RE       = re.compile(r"TTS\w*\s*(?:#\d+\s*)?processing time:\s*([\d.]+)", re.I)
_TTS_CHARS_RE      = re.compile(r"TTS\w*\s*(?:#\d+\s*)?usage characters:\s*([\d.]+)", re.I)
_BOT_SPEAKING_RE   = re.compile(r"Bot started speaking", re.I)

# ── Text capture patterns ─────────────────────────────────────────────────────
_TEXT_PATTERNS = [
    (re.compile(r"TranscriptionFrame.*?text=['\"](.+?)['\"]", re.I), "stt", "transcript"),
    (re.compile(r"Transcription:\s*\[?(.+?)\]?\s*$", re.I),          "stt", "transcript"),
    (re.compile(r"LLMFullResponseEnd.*?text=['\"](.+?)['\"]", re.I), "llm", "response"),
]


class BenchmarkLogSink:
    def __init__(self):
        self._in_turn:             dict = defaultdict(bool)
        self._llm_call_ts:         dict = defaultdict(lambda: None)  # LLM TTFB log timestamp
        self._tts_first_gen_ts:    dict = defaultdict(lambda: None)  # timestamp of first "Generating TTS"
        self._tts_first_proc_done_ts: dict = defaultdict(lambda: None)  # timestamp when first TTS proc finishes
        self._tts_first_proc_ms:   dict = defaultdict(lambda: None)  # value of first TTS processing_time (ms)
        self._tts_proc_count:      dict = defaultdict(int)
        self._tts_proc_acc_s:      dict = defaultdict(float)         # accumulated seconds
        self._tts_chars_acc:       dict = defaultdict(float)
        self._tts_texts:           dict = defaultdict(list)
        self._flushed:             dict = defaultdict(bool)

    def _reset_turn(self, sid):
        self._in_turn[sid]               = False
        self._llm_call_ts[sid]           = None
        self._tts_first_gen_ts[sid]      = None
        self._tts_first_proc_done_ts[sid] = None
        self._tts_first_proc_ms[sid]     = None
        self._tts_proc_count[sid]        = 0
        self._tts_proc_acc_s[sid]        = 0.0
        self._tts_chars_acc[sid]         = 0.0
        self._tts_texts[sid]             = []
        self._flushed[sid]               = False

    def flush_session(self, sid: str):
        """Call from GET /benchmark/session/{sid} before returning metrics."""
        if self._in_turn[sid] and not self._flushed[sid]:
            self._flush_turn(sid)
            self._flushed[sid] = True

    def __call__(self, message):
        line = str(message)
        sid  = current_session_id.get()
        if not sid:
            return

        ts = _parse_ts(line)

        # ── Immediate single-value metrics ────────────────────────────────
        for pattern, stage, metric, unit, mult in _IMMEDIATE_PATTERNS:
            m = pattern.search(line)
            if m:
                try:
                    collect_metric(sid, stage, metric, float(m.group(1)) * mult, unit)
                except (IndexError, ValueError):
                    pass

        # ── LLM TTFB → marks start of a new turn ─────────────────────────
        if _LLM_TTFB_RE.search(line):
            if self._in_turn[sid] and not self._flushed[sid]:
                self._flush_turn(sid)
            self._reset_turn(sid)
            self._in_turn[sid]     = True
            self._llm_call_ts[sid] = ts

        # ── First TTS gen line → end of LLM streaming lag ────────────────
        # "ZipVoiceTTS: [text]" is logged by ttsv2:run_tts once per chunk
        tts_gen_match = _TTS_GEN_RE.search(line)
        if tts_gen_match:
            text = tts_gen_match.group(1).strip()
            self._tts_texts[sid].append(text)
            if self._in_turn[sid] and self._tts_first_gen_ts[sid] is None and ts is not None:
                self._tts_first_gen_ts[sid] = ts

        # ── TTS processing time lines ─────────────────────────────────────
        m = _TTS_PROC_RE.search(line)
        if m:
            try:
                val_s = float(m.group(1))
                self._tts_proc_acc_s[sid] += val_s
                self._tts_proc_count[sid] += 1

                # First TTS chunk finished
                if self._in_turn[sid] and self._tts_first_proc_done_ts[sid] is None and ts is not None:
                    self._tts_first_proc_done_ts[sid] = ts
                    self._tts_first_proc_ms[sid] = val_s * 1000
            except ValueError:
                pass

        # ── TTS character usage ───────────────────────────────────────────
        m = _TTS_CHARS_RE.search(line)
        if m:
            try:
                self._tts_chars_acc[sid] += float(m.group(1))
            except ValueError:
                pass

        # ── Bot started speaking → send_audio_time can be calculated ─────
        if _BOT_SPEAKING_RE.search(line):
            if self._in_turn[sid] and not self._flushed[sid]:
                first_proc_done_ts = self._tts_first_proc_done_ts[sid]
                if first_proc_done_ts is not None and ts is not None:
                    send_audio_ms = (ts - first_proc_done_ts).total_seconds() * 1000
                    if send_audio_ms >= 0:
                        collect_metric(sid, "tts", "send_audio_time", send_audio_ms, "ms")

        # ── Text patterns ─────────────────────────────────────────────────
        for pattern, stage, key in _TEXT_PATTERNS:
            m = pattern.search(line)
            if m:
                collect_text(sid, stage, key, m.group(1).strip())

    def _flush_turn(self, sid: str):
        llm_ts        = self._llm_call_ts[sid]
        first_gen_ts  = self._tts_first_gen_ts[sid]

        # llm_time = LLM TTFB timestamp → first "Generating TTS" timestamp
        if llm_ts is not None and first_gen_ts is not None:
            llm_time_ms = (first_gen_ts - llm_ts).total_seconds() * 1000
            if llm_time_ms >= 0:
                collect_metric(sid, "llm", "llm_time", llm_time_ms, "ms")

        # first_sentence = processing time of the very first TTS chunk
        first_proc_ms = self._tts_first_proc_ms[sid]
        if first_proc_ms is not None:
            collect_metric(sid, "tts", "first_sentence", first_proc_ms, "ms")

        # total_time = from first "Generating TTS" to last TTS proc done
        # We approximate as the sum of all TTS processing times (they run
        # mostly sequentially in pipecat's ZipVoiceTTS pipeline).
        proc_acc_s = self._tts_proc_acc_s[sid]
        if proc_acc_s > 0:
            collect_metric(sid, "tts", "total_time", proc_acc_s * 1000, "ms")

        # chars
        chars = self._tts_chars_acc[sid]
        if chars > 0:
            collect_metric(sid, "tts", "usage_chars", chars, "chars")

        # full TTS text
        texts = self._tts_texts[sid]
        if texts:
            collect_text(sid, "tts", "text", " | ".join(texts))


# Singleton — import this in server.py only, never in bot.py
benchmark_sink = BenchmarkLogSink()