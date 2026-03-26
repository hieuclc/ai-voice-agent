"""
tts_chunker.py
──────────────
Đặt giữa `llm` và `thinking_processor` trong pipeline.

Hai mode:
  ChunkMode.PUNCT_ONLY  — chỉ split khi gặp dấu chấm câu. Không emit sớm
                          theo word count. Phù hợp khi TTS cần câu hoàn chỉnh
                          để phát âm tự nhiên (ZipVoice, local TTS...).

  ChunkMode.EARLY_FIRST — mode mặc định: emit chunk đầu tiên sau khi đủ 5 từ
                          HOẶC gặp dấu chấm, các chunk sau split theo dấu chấm.
                          Phù hợp khi muốn giảm latency (OpenAI TTS...).

Dùng trong bot.py:
    from tts_chunker import TTSChunkerProcessor, ChunkMode

    tts_chunker = TTSChunkerProcessor()                           # EARLY_FIRST
    tts_chunker = TTSChunkerProcessor(mode=ChunkMode.PUNCT_ONLY)  # chỉ dấu chấm
"""

import re
from enum import Enum
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
FIRST_CHUNK_MIN_WORDS = 5   # EARLY_FIRST: emit chunk đầu sau khi đủ N từ
MIN_WORDS_PER_CHUNK   = 4   # chunk < N từ sẽ bị ghép với chunk kế tiếp
SENTENCE_ENDS         = re.compile(r'(?<=[^.!?])([.!?]+)\s*')
# ─────────────────────────────────────────────────────────────────────────────


class ChunkMode(str, Enum):
    PUNCT_ONLY  = "punct_only"   # chỉ split khi gặp dấu chấm
    EARLY_FIRST = "early_first"  # emit sớm 5 từ đầu + split dấu chấm


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

    Args:
        mode: ChunkMode.EARLY_FIRST (default) hoặc ChunkMode.PUNCT_ONLY
    """

    def __init__(self, mode: ChunkMode = ChunkMode.EARLY_FIRST, **kwargs):
        super().__init__(**kwargs)
        self._mode = mode
        self._reset()
        logger.info("TTSChunkerProcessor mode=%s", self._mode.value)

    def _reset(self):
        self._buffer: str = ""
        self._first_chunk_done: bool = False

    async def _emit(self, text: str, direction: FrameDirection):
        text = text.strip()
        if text:
            logger.debug("TTSChunker [%s] → TTS: %r", self._mode.value, text[:80])
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

    # ── PUNCT_ONLY mode ───────────────────────────────────────────────────────

    async def _handle_punct_only(self, direction: FrameDirection):
        """Chỉ emit khi gặp dấu chấm — không emit sớm theo word count."""
        await self._drain_sentences(direction)

    # ── EARLY_FIRST mode ──────────────────────────────────────────────────────

    async def _try_emit_first_chunk(self, direction: FrameDirection):
        """
        Emit chunk đầu tiên sớm nhất có thể:
          - Gặp dấu chấm → emit đến dấu chấm đó
          - Đủ FIRST_CHUNK_MIN_WORDS từ → emit N từ đầu
        """
        buf = self._buffer

        m = SENTENCE_ENDS.search(buf)
        if m:
            await self._emit(buf[:m.end()], direction)
            self._buffer = buf[m.end():]
            self._first_chunk_done = True
            await self._drain_sentences(direction)
            return

        words = buf.split()
        if len(words) >= FIRST_CHUNK_MIN_WORDS:
            await self._emit(" ".join(words[:FIRST_CHUNK_MIN_WORDS]), direction)
            self._buffer = " ".join(words[FIRST_CHUNK_MIN_WORDS:])
            self._first_chunk_done = True

    async def _handle_early_first(self, direction: FrameDirection):
        if not self._first_chunk_done:
            await self._try_emit_first_chunk(direction)
        else:
            await self._drain_sentences(direction)

    # ── FrameProcessor interface ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._reset()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame):
            self._buffer += frame.text

            if self._mode == ChunkMode.PUNCT_ONLY:
                await self._handle_punct_only(direction)
            else:
                await self._handle_early_first(direction)

        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._flush_buffer(direction)
            self._reset()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)