from typing import AsyncGenerator, Dict, Literal, Optional

from loguru import logger
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

import numpy as np
from utils import split_text_into_chunks

ValidVoice = Literal[
    "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"
]

VALID_VOICES: Dict[str, ValidVoice] = {
    "alloy": "alloy",
    "ash": "ash",
    "ballad": "ballad",
    "coral": "coral",
    "echo": "echo",
    "fable": "fable",
    "onyx": "onyx",
    "nova": "nova",
    "sage": "sage",
    "shimmer": "shimmer",
    "verse": "verse",
}

import numpy as np
from typing import Generator
import requests

import aiohttp

# async def infer_stream_custom_async(text, voice, frame_size=1200):
#     frame_size = 10000
#     async with aiohttp.ClientSession() as session:
#         async with session.post(
#             "http://localhost:8001/synthesize",
#             json={
#                 "text": text,
#                 "voice_id": voice,
#                 "num_step": 16,
#                 "speed": 1.0
#             }
#         ) as resp:
#             remain = b""

#             async for chunk in resp.content.iter_chunked(4096):
#                 chunk = remain + chunk

#                 n = (len(chunk) // 2) * 2   # số byte chẵn
#                 if n == 0:
#                     remain = chunk
#                     continue

#                 pcm = np.frombuffer(chunk[:n], dtype=np.int16)
#                 remain = chunk[n:]

#                 yield pcm
import aiohttp
import numpy as np

_prev_need_pause = False

async def infer_stream_custom_async(text, voice, frame_size=1200):
    global _prev_need_pause

    sample_rate = 24000
    pause_ms = 100
    fade_ms = 20

    pause_len = int(sample_rate * pause_ms / 1000)
    fade_len = int(sample_rate * fade_ms / 1000)

    # buffer PCM theo sample
    frame_buffer = np.empty(0, dtype=np.int16)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8001/synthesize",
            json={
                "text": text,
                "voice_id": voice,
                "num_step": 16,
                "speed": 1.0
            }
        ) as resp:
            remain = b""
            first_chunk = True

            async for chunk in resp.content.iter_chunked(4096):
                chunk = remain + chunk
                n = (len(chunk) // 2) * 2
                if n == 0:
                    remain = chunk
                    continue

                pcm = np.frombuffer(chunk[:n], dtype=np.int16)
                remain = chunk[n:]

                # ----- đầu câu -----
                if first_chunk and _prev_need_pause:
                    # fade-in từ 0
                    pcm = pcm.copy()
                    L = min(fade_len, len(pcm))
                    fade = np.linspace(0, 1, L, endpoint=False)
                    pcm[:L] = (pcm[:L] * fade).astype(np.int16)
                    first_chunk = False
                else:
                    first_chunk = False

                # add vào buffer
                frame_buffer = np.concatenate([frame_buffer, pcm])

                # yield theo frame_size
                while len(frame_buffer) >= frame_size:
                    yield frame_buffer[:frame_size]
                    frame_buffer = frame_buffer[frame_size:]

            # ----- cuối câu -----
            # fade-out về 0
            if len(frame_buffer) > 0:
                L = min(fade_len, len(frame_buffer))
                fade = np.linspace(1, 0, L, endpoint=False)
                frame_buffer[-L:] = (frame_buffer[-L:] * fade).astype(np.int16)

            # chèn silence
            silence = np.zeros(pause_len, dtype=np.int16)
            frame_buffer = np.concatenate([frame_buffer, silence])

            # flush buffer theo frame_size
            while len(frame_buffer) >= frame_size:
                yield frame_buffer[:frame_size]
                frame_buffer = frame_buffer[frame_size:]

            _prev_need_pause = True


class OpenAITTSService(TTSService):
    """OpenAI Text-to-Speech service that generates audio from text.

    This service uses the OpenAI TTS API to generate PCM-encoded audio at 24kHz.
    Supports multiple voice models and configurable parameters for high-quality
    speech synthesis with streaming audio output.
    """

    OPENAI_SAMPLE_RATE = 24000  # OpenAI TTS always outputs at 24kHz

    class InputParams(BaseModel):
        """Input parameters for OpenAI TTS configuration.

        Parameters:
            instructions: Instructions to guide voice synthesis behavior.
            speed: Voice speed control (0.25 to 4.0, default 1.0).
        """

        instructions: Optional[str] = None
        speed: Optional[float] = None

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        voice: str = "alloy",
        model: str = "gpt-4o-mini-tts",
        sample_rate: Optional[int] = None,
        instructions: Optional[str] = None,
        speed: Optional[float] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize OpenAI TTS service.

        Args:
            api_key: OpenAI API key for authentication. If None, uses environment variable.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            voice: Voice ID to use for synthesis. Defaults to "alloy".
            model: TTS model to use. Defaults to "gpt-4o-mini-tts".
            sample_rate: Output audio sample rate in Hz. If None, uses OpenAI's default 24kHz.
            instructions: Optional instructions to guide voice synthesis behavior.
            speed: Voice speed control (0.25 to 4.0, default 1.0).
            params: Optional synthesis controls (acting instructions, speed, ...).
            **kwargs: Additional keyword arguments passed to TTSService.

                .. deprecated:: 0.0.91
                        The `instructions` and `speed` parameters are deprecated, use `InputParams` instead.
        """
        if sample_rate and sample_rate != self.OPENAI_SAMPLE_RATE:
            logger.warning(
                f"OpenAI TTS only supports {self.OPENAI_SAMPLE_RATE}Hz sample rate. "
                f"Current rate of {sample_rate}Hz may cause issues."
            )
        super().__init__(sample_rate=sample_rate, **kwargs)

        self.set_model_name(model)
        self.set_voice(voice)
        self.current_voice = voice

        
        

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as OpenAI TTS service supports metrics generation.
        """
        return True

    async def set_model(self, model: str):
        """Set the TTS model to use.

        Args:
            model: The model name to use for text-to-speech synthesis.
        """
        logger.info(f"Switching TTS model to: [{model}]")
        self.set_model_name(model)

    async def start(self, frame: StartFrame):
        """Start the OpenAI TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        if self.sample_rate != self.OPENAI_SAMPLE_RATE:
            logger.warning(
                f"OpenAI TTS requires {self.OPENAI_SAMPLE_RATE}Hz sample rate. "
                f"Current rate of {self.sample_rate}Hz may cause issues."
            )

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using OpenAI's TTS API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech data.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:

                yield TTSStartedFrame()
                silence = np.zeros(1200, dtype=np.int16)
                yield TTSAudioRawFrame(silence.tobytes(), 24000, 1)
                async for pcm in infer_stream_custom_async(
                    text = text,
                    voice = self.current_voice,
                    frame_size = 480
                ):
                    yield TTSAudioRawFrame(pcm.tobytes(), 24000, 1)
                yield TTSStoppedFrame()
        except BadRequestError as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")