import re
from typing import List
import time
import contextvars
from datetime import datetime
from collections import defaultdict
from threading import Lock


def split_text_into_chunks(text: str, max_chars: int = 256) -> List[str]:
    """
    Split raw text into chunks no longer than max_chars.
    """
    # 1. First split by newlines - each line/paragraph is handled independently
    paragraphs = re.split(r"[\r\n]+", text.strip())
    final_chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # 2. Split current paragraph into sentences
        sentences = re.split(r"(?<=[\.\!\?\…])\s+", para)
        
        buffer = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If sentence itself is longer than max_chars, we must split it by minor punctuation or words
            if len(sentence) > max_chars:
                # Flush buffer before handling a giant sentence
                if buffer:
                    final_chunks.append(buffer)
                    buffer = ""
                
                # Split giant sentence by minor punctuation (, ; : -)
                sub_parts = re.split(r"(?<=[\,\;\:\-\–\—])\s+", sentence)
                for part in sub_parts:
                    part = part.strip()
                    if not part: continue
                    
                    if len(buffer) + 1 + len(part) <= max_chars:
                        buffer = (buffer + " " + part) if buffer else part
                    else:
                        if buffer: final_chunks.append(buffer)
                        buffer = part
                        
                        # If even a sub-part is too long, split by spaces (words)
                        if len(buffer) > max_chars:
                            words = buffer.split()
                            current = ""
                            for word in words:
                                if current and len(current) + 1 + len(word) > max_chars:
                                    final_chunks.append(current)
                                    current = word
                                else:
                                    current = (current + " " + word) if current else word
                            buffer = current
            else:
                # Normal sentence: check if it fits in current buffer
                if buffer and len(buffer) + 1 + len(sentence) > max_chars:
                    final_chunks.append(buffer)
                    buffer = sentence
                else:
                    buffer = (buffer + " " + sentence) if buffer else sentence
        
        # End of paragraph: flush whatever is in buffer
        if buffer:
            final_chunks.append(buffer)
            buffer = ""

    return [c.strip() for c in final_chunks if c.strip()]

SPECIAL_MAP = {
    "sjc": "ét di xi",
    "pnj": "pi en di",
    "fifa": "phi pha",
    "usd": "đô la mỹ",
    "vnd": "việt nam đồng",
    "vietcombank": "việt com bank",
    "vietinbank": "việt tin bank",
    "vcb": "việt com bank",
    "tcb": "tech com bank",
    "huyndai": "huyn đai",
    "phẩy": "phảy",
    "cccd": "căn cước công dân",
    "bhxh": "bảo hiểm xã hội",
    "bhyt": "bảo hiểm y tế",
    "hđnd": "hội đồng nhân dân",
    "ubnd": "ủy ban nhân dân",
    "json": "di sơn",
    "xml": "ích em eo",
    "html": "ết ti em eo",
    "css": "xê ét ét",
    "iot": "ai ô ti",
    "zalo": "da lô"
}

LETTER_MAP = {
    "A": "ây",
    "B": "bi",
    "C": "xi",
    "D": "đi",
    "E": "i",
    "F": "ép",
    "G": "di",
    "H": "ết",
    "I": "ai",
    "J": "dây",
    "K": "cây",
    "L": "eo",
    "M": "em",
    "N": "en",
    "O": "ô",
    "P": "pi",
    "Q": "kiu",
    "R": "a",
    "S": "ét",
    "T": "ti",
    "U": "iu",
    "V": "vi",
    "W": "đắp bồ liu",
    "X": "ích",
    "Y": "oai",
    "Z": "dét",
}



ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")

def normalize_special_terms(text: str) -> str:
    for key, spoken in SPECIAL_MAP.items():
        text = re.sub(
            key,
            spoken,
            text,
            flags=re.IGNORECASE
        )
    return text

def read_acronym(word: str) -> str:
    w = word.lower()

    if w in SPECIAL_MAP:
        return SPECIAL_MAP[w]

    return " ".join(
        LETTER_MAP.get(ch, ch)
        for ch in word
    )

def normalize_acronyms(text: str) -> str:
    def replacer(match):
        return read_acronym(match.group(0))

    return ACRONYM_RE.sub(replacer, text)

def normalize_sentence(text: str) -> str:
    text = normalize_special_terms(text)
    text = normalize_acronyms(text)
    return text.strip().lower() + "  "

"""
utils.py - Benchmark utilities: in-memory metric store + log sink (singleton).

Metrics collected per turn:
  llm / llm_time          : LLM call start → first "Generating TTS" line  (ms)
  tts / first_sentence    : processing time of the very first TTS chunk     (ms)
  tts / send_audio_time   : first TTS chunk done → "Bot started speaking"  (ms)
  tts / total_time        : first "Generating TTS" → last TTS chunk done   (ms)
  stt / ttfb              : STT TTFB as reported by pipecat                 (ms)
  stt / processing_time   : STT processing time as reported by pipecat      (ms)
  llm / usage_tokens      : LLM token usage                                 (tokens)
  tts / usage_chars       : TTS character usage                             (chars)

Usage in server.py:
    from utils import benchmark_sink, current_session_id
    from utils import get_metrics, get_texts, clear_metrics
    logger.add(benchmark_sink, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}")
"""

# ── ContextVar ────────────────────────────────────────────────────────────────
current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_session_id", default=None
)

# =============================================================================
# Metric store  (ex benchmark_metrics.py)
# =============================================================================

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


# =============================================================================
# Log sink  (ex benchmark_log_sink.py)
# =============================================================================

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
        frac = m.group(2).ljust(6, "0")[:6]   # normalise to exactly 6 digits
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
_LLM_TTFB_RE     = re.compile(r"LLMService[^T]*?TTFB:", re.I)
_TTS_GEN_RE      = re.compile(r"ZipVoiceTTS\w*:\s*\[(.+?)\]", re.I)
_TTS_PROC_RE     = re.compile(r"TTS\w*\s*(?:#\d+\s*)?processing time:\s*([\d.]+)", re.I)
_TTS_CHARS_RE    = re.compile(r"TTS\w*\s*(?:#\d+\s*)?usage characters:\s*([\d.]+)", re.I)
_BOT_SPEAKING_RE = re.compile(r"Bot started speaking", re.I)

# ── Text capture patterns ─────────────────────────────────────────────────────
_TEXT_PATTERNS = [
    (re.compile(r"TranscriptionFrame.*?text=['\"](.+?)['\"]", re.I), "stt", "transcript"),
    (re.compile(r"Transcription:\s*\[?(.+?)\]?\s*$", re.I),          "stt", "transcript"),
    (re.compile(r"LLMFullResponseEnd.*?text=['\"](.+?)['\"]", re.I), "llm", "response"),
]


class BenchmarkLogSink:
    def __init__(self):
        self._in_turn:                dict = defaultdict(bool)
        self._llm_call_ts:            dict = defaultdict(lambda: None)
        self._tts_first_gen_ts:       dict = defaultdict(lambda: None)
        self._tts_first_proc_done_ts: dict = defaultdict(lambda: None)
        self._tts_first_proc_ms:      dict = defaultdict(lambda: None)
        self._tts_proc_count:         dict = defaultdict(int)
        self._tts_proc_acc_s:         dict = defaultdict(float)
        self._tts_chars_acc:          dict = defaultdict(float)
        self._tts_texts:              dict = defaultdict(list)
        self._flushed:                dict = defaultdict(bool)
        self._send_audio_collected:   dict = defaultdict(bool)

    def _reset_turn(self, sid):
        self._in_turn[sid]                = False
        self._llm_call_ts[sid]            = None
        self._tts_first_gen_ts[sid]       = None
        self._tts_first_proc_done_ts[sid] = None
        self._tts_first_proc_ms[sid]      = None
        self._tts_proc_count[sid]         = 0
        self._tts_proc_acc_s[sid]         = 0.0
        self._tts_chars_acc[sid]          = 0.0
        self._tts_texts[sid]              = []
        self._flushed[sid]                = False
        self._send_audio_collected[sid]   = False

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

        # ── Bot started speaking → send_audio_time (chỉ collect 1 lần/turn) ─
        if _BOT_SPEAKING_RE.search(line):
            if (self._in_turn[sid]
                    and not self._flushed[sid]
                    and not self._send_audio_collected[sid]):
                first_proc_done_ts = self._tts_first_proc_done_ts[sid]
                if first_proc_done_ts is not None and ts is not None:
                    send_audio_ms = (ts - first_proc_done_ts).total_seconds() * 1000
                    if 0 <= send_audio_ms < 5000:   # sanity check: bỏ qua nếu > 5s
                        collect_metric(sid, "tts", "send_audio_time", send_audio_ms, "ms")
                        self._send_audio_collected[sid] = True

        # ── Text patterns ─────────────────────────────────────────────────
        for pattern, stage, key in _TEXT_PATTERNS:
            m = pattern.search(line)
            if m:
                collect_text(sid, stage, key, m.group(1).strip())

    def _flush_turn(self, sid: str):
        llm_ts       = self._llm_call_ts[sid]
        first_gen_ts = self._tts_first_gen_ts[sid]

        if llm_ts is not None and first_gen_ts is not None:
            llm_time_ms = (first_gen_ts - llm_ts).total_seconds() * 1000
            if llm_time_ms >= 0:
                collect_metric(sid, "llm", "llm_time", llm_time_ms, "ms")

        first_proc_ms = self._tts_first_proc_ms[sid]
        if first_proc_ms is not None:
            collect_metric(sid, "tts", "first_sentence", first_proc_ms, "ms")

        proc_acc_s = self._tts_proc_acc_s[sid]
        if proc_acc_s > 0:
            collect_metric(sid, "tts", "total_time", proc_acc_s * 1000, "ms")

        chars = self._tts_chars_acc[sid]
        if chars > 0:
            collect_metric(sid, "tts", "usage_chars", chars, "chars")

        texts = self._tts_texts[sid]
        if texts:
            collect_text(sid, "tts", "text", " | ".join(texts))


# Singleton — import this in server.py only, never in bot.py
benchmark_sink = BenchmarkLogSink()