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
from benchmark_metrics import collect_metric

# Set this contextvar inside run_bot() so the sink knows which session owns
# the current log line.
current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_session_id", default=None
)

# Patterns from actual pipecat log output:
#   OpenAILLMService#2 TTFB: 0.009444713592529297
#   OpenAISTTService#2 processing time: 0.567857027053833
#   OpenAITTSService#2 TTFB: 0.7891077995300293
#   OpenAITTSService#2 usage characters: 49

_PATTERNS = [
    # (regex, stage, metric, unit, value_multiplier)
    (re.compile(r"(OpenAI)?LLMService.*?TTFB:\s*([\d.]+)", re.I),      "llm", "ttfb",            "ms",    1000),
    (re.compile(r"(OpenAI)?STTService.*?TTFB:\s*([\d.]+)", re.I),      "stt", "ttfb",            "ms",    1000),
    (re.compile(r"(OpenAI)?TTSService.*?TTFB:\s*([\d.]+)", re.I),      "tts", "ttfb",            "ms",    1000),
    (re.compile(r"(OpenAI)?LLMService.*?processing time:\s*([\d.]+)", re.I), "llm", "processing_time", "ms", 1000),
    (re.compile(r"(OpenAI)?STTService.*?processing time:\s*([\d.]+)", re.I), "stt", "processing_time", "ms", 1000),
    (re.compile(r"(OpenAI)?TTSService.*?processing time:\s*([\d.]+)", re.I), "tts", "processing_time", "ms", 1000),
    (re.compile(r"(OpenAI)?TTSService.*?usage characters:\s*([\d.]+)", re.I), "tts", "usage_chars", "chars", 1),
    (re.compile(r"(OpenAI)?LLMService.*?usage.*?tokens.*?(\d+)", re.I), "llm", "usage_tokens", "tokens", 1),
]


class BenchmarkLogSink:
    """Callable loguru sink."""

    def __call__(self, message):
        text = str(message)
        session_id = current_session_id.get()
        if not session_id:
            return  # no active session, skip

        for pattern, stage, metric, unit, mult in _PATTERNS:
            m = pattern.search(text)
            if m:
                try:
                    raw = float(m.group(2))
                    collect_metric(session_id, stage, metric, raw * mult, unit)
                except (IndexError, ValueError):
                    pass