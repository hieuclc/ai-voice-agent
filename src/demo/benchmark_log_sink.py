"""
benchmark_log_sink.py

A loguru sink that parses pipecat's existing log lines and feeds
benchmark_metrics.collect_metric().

Usage — add to bot.py or server.py at startup:

    from benchmark_log_sink import BenchmarkLogSink
    from loguru import logger

    sink = BenchmarkLogSink()
    logger.add(sink, format="{message}")

The sink is session-aware: it matches session_id from a thread-local context
that run_bot() sets before starting the pipeline.

Simpler approach: pass session_id explicitly via contextvars (see below).
"""

import re
import contextvars
from benchmark_metrics import collect_metric, collect_text

# Set this contextvar inside run_bot() so the sink knows which session owns
# the current log line.
current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_session_id", default=None
)

_PATTERNS = [
    (re.compile(r"(OpenAI)?LLMService.*?TTFB:\s*([\d.]+)", re.I),      "llm", "ttfb",            "ms",    1000),
    (re.compile(r"(OpenAI)?STTService.*?TTFB:\s*([\d.]+)", re.I),      "stt", "ttfb",            "ms",    1000),
    (re.compile(r"(OpenAI)?TTSService.*?TTFB:\s*([\d.]+)", re.I),      "tts", "ttfb",            "ms",    1000),
    (re.compile(r"(OpenAI)?LLMService.*?processing time:\s*([\d.]+)", re.I), "llm", "processing_time", "ms", 1000),
    (re.compile(r"(OpenAI)?STTService.*?processing time:\s*([\d.]+)", re.I), "stt", "processing_time", "ms", 1000),
    (re.compile(r"(OpenAI)?TTSService.*?processing time:\s*([\d.]+)", re.I), "tts", "processing_time", "ms", 1000),
    (re.compile(r"(OpenAI)?TTSService.*?usage characters:\s*([\d.]+)", re.I), "tts", "usage_chars", "chars", 1),
    (re.compile(r"(OpenAI)?LLMService.*?usage.*?tokens.*?(\d+)", re.I), "llm", "usage_tokens", "tokens", 1),
]

# Text patterns — capture transcript/response content
# Adjust these regexes to match your actual pipecat log format.
# Common pipecat patterns:
#   TranscriptionFrame: "Hello how are you"
#   LLMFullResponseEndFrame or similar aggregator output
_TEXT_PATTERNS = [
    # STT transcript — pipecat logs TranscriptionFrame text
    (re.compile(r"TranscriptionFrame.*?text=['\"](.+?)['\"]", re.I), "stt", "transcript"),
    (re.compile(r"Transcription:\s*(.+)", re.I),                     "stt", "transcript"),
    # LLM full response — pipecat logs aggregated LLM text
    (re.compile(r"LLMFullResponseEnd.*?text=['\"](.+?)['\"]", re.I), "llm", "response"),
    (re.compile(r"Generating TTS \[(.+?)\]", re.I),                  "tts", "text"),
]


class BenchmarkLogSink:
    """Callable loguru sink."""

    def __call__(self, message):
        text = str(message)
        session_id = current_session_id.get()
        if not session_id:
            return

        for pattern, stage, metric, unit, mult in _PATTERNS:
            m = pattern.search(text)
            if m:
                try:
                    raw = float(m.group(2))
                    collect_metric(session_id, stage, metric, raw * mult, unit)
                except (IndexError, ValueError):
                    pass

        for pattern, stage, key in _TEXT_PATTERNS:
            m = pattern.search(text)
            if m:
                collect_text(session_id, stage, key, m.group(1).strip())