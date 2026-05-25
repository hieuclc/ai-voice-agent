"""
tts_chunker.py
──────────────
Đặt giữa `llm` và `thinking_processor` trong pipeline.

Split text thành các câu theo dấu chấm câu. Phù hợp khi TTS cần câu hoàn chỉnh
để phát âm tự nhiên (ZipVoice, local TTS...).

Dùng trong bot.py:
    from tts_chunker import TTSChunkerProcessor

    tts_chunker = TTSChunkerProcessor()
"""

import re
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TTSSpeakFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

# ── Tune these if needed ──────────────────────────────────────────────────────
MIN_WORDS_PER_CHUNK   = 4   # chunk < N từ sẽ bị ghép với chunk kế tiếp
SENTENCE_ENDS         = re.compile(r'(?<=[^.!?])([.!?]+)\s*')
# ─────────────────────────────────────────────────────────────────────────────


def _word_count(text: str) -> int:
    return len(text.split())


def _split_on_punctuation(text: str) -> list[str]:
    """
    Split text thành các câu theo dấu . ! ?
    Giữ dấu chấm ở cuối mỗi phần, phần chưa có dấu để nguyên.
    "nhé. hôm nay bạn có khỏe không. tôi sẽ"
    → ["nhé.", "hôm nay bạn có khỏe không.", "tôi sẽ"]
    """
    parts = []
    remaining = text
    while True:
        m = SENTENCE_ENDS.search(remaining)
        if not m:
            if remaining.strip():
                parts.append(remaining.strip())
            break
        parts.append(remaining[:m.end()].strip())
        remaining = remaining[m.end():]
    return parts


def _merge_short_chunks(chunks: list[str], min_words: int) -> list[str]:
    """Ghép chunk đầu vào chunk tiếp nếu chunk đầu < min_words từ."""
    if not chunks:
        return chunks
    merged = list(chunks)
    if _word_count(merged[0]) < min_words and len(merged) > 1:
        merged[1] = merged[0] + " " + merged[1]
        merged.pop(0)
    return merged


class TTSChunkerProcessor(FrameProcessor):
    """
    Custom chunker thay thế SimpleTextAggregator cho TTS latency thấp.
    Chỉ emit khi gặp dấu chấm câu để TTS phát âm tự nhiên.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reset()
        logger.info("TTSChunkerProcessor initialized")

    def _reset(self):
        self._buffer: str = ""

    async def _emit(self, text: str, direction: FrameDirection):
        text = text.strip()
        if text:
            logger.debug("TTSChunker → TTS: %r", text[:80])
            await self.push_frame(TTSSpeakFrame(text), direction)

    # ── Shared: extract completed sentences from buffer, keep remainder ───────

    async def _drain_sentences(self, direction: FrameDirection):
        """Emit mọi câu hoàn chỉnh trong buffer, giữ phần chưa có dấu."""
        buf = self._buffer
        chunks: list[str] = []

        while True:
            m = SENTENCE_ENDS.search(buf)
            if not m:
                break
            chunks.append(buf[:m.end()].strip())
            buf = buf[m.end():]

        self._buffer = buf

        if not chunks:
            return

        chunks = _merge_short_chunks(chunks, MIN_WORDS_PER_CHUNK)
        for chunk in chunks:
            await self._emit(chunk, direction)

    async def _flush_buffer(self, direction: FrameDirection):
        """Flush toàn bộ buffer khi response kết thúc."""
        text = self._buffer.strip()
        self._buffer = ""
        if not text:
            return
        chunks = _split_on_punctuation(text)
        chunks = _merge_short_chunks(chunks, MIN_WORDS_PER_CHUNK)
        for chunk in chunks:
            await self._emit(chunk, direction)

    # ── FrameProcessor interface ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._reset()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame):
            self._buffer += frame.text
            await self._drain_sentences(direction)

        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._flush_buffer(direction)
            self._reset()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)