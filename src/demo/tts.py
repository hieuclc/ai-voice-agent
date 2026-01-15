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

from vieneu import FastVieNeuTTS
from pathlib import Path
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

def infer_stream_custom(
    tts,
    text: str,
    voice,
    temperature = 0.7,
    top_k = 50,
    frame_size = 50000
) -> Generator[np.ndarray, None, None]:
    """
    Streaming by text chunks (fake streaming but low latency per chunk).
    Yields PCM int16 frames.
    """

    text_chunks = split_text_into_chunks(text, max_chars=256)

    for chunk_text in text_chunks:
        wav = tts.infer(
          text=chunk_text,
          voice=voice,
          temperature=temperature,
          top_k=top_k
      )

        if wav is None or len(wav) == 0:
            continue

        wav = wav.astype(np.float32).flatten()

        # 2. chia audio thành frame & yield ngay
        for i in range(0, len(wav), frame_size):
            frame = wav[i:i + frame_size]

            # pad frame cuối
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))

            # float → PCM16
            frame = np.clip(frame, -1.0, 1.0)
            pcm = (frame * 32767).astype(np.int16)

            yield pcm



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
        self._tts_model = FastVieNeuTTS(backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B", backbone_device = "cuda")
        sample_audio = Path("/teamspace/studios/this_studio/example.wav")
        sample_text = "ví dụ 2. tính trung bình của dãy số."
        
        self.current_voice = self._tts_model.clone_voice(
            audio_path=sample_audio,
            text=sample_text
        )
            

        if instructions or speed:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The `instructions` and `speed` parameters are deprecated, use `InputParams` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        self._settings = {
            "instructions": params.instructions if params else instructions,
            "speed": params.speed if params else speed,
        }

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
                for chunk in infer_stream_custom(
                    tts = self._tts_model,
                    text = text,
                    voice = self.current_voice,
                    temperature = 0.7,
                    top_k = 50,
                    frame_size = 50000
                ):
                    frame = TTSAudioRawFrame(
                        chunk.tobytes(),
                        24000,
                        1
                    )
                    yield frame
                # for chunk in self._tts_model.infer_stream(
                #     text=text,
                #     voice=self.current_voice,
                #     temperature=0.5,
                #     top_k=50,
                # ):
                #     if chunk is None or len(chunk) == 0:
                #         continue

                #     # chunk: float32 [-1, 1]
                #     pcm16 = (chunk * 32767.0).astype(np.int16)

                #     frame = TTSAudioRawFrame(
                #         pcm16.tobytes(),   # 🔥 BYTES
                #         24000,  # 16000
                #         1                  # mono
                #     )
                #     yield frame
                yield TTSStoppedFrame()
        except BadRequestError as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")




# from typing import AsyncGenerator, Dict, Literal, Optional

# from loguru import logger
# from openai import AsyncOpenAI, BadRequestError
# from pydantic import BaseModel

# from pipecat.frames.frames import (
#     ErrorFrame,
#     Frame,
#     StartFrame,
#     TTSAudioRawFrame,
#     TTSStartedFrame,
#     TTSStoppedFrame,
# )
# from pipecat.services.tts_service import TTSService
# from pipecat.utils.tracing.service_decorators import traced_tts

# ValidVoice = Literal[
#     "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse", "dave"
# ]

# VALID_VOICES: Dict[str, ValidVoice] = {
#     "alloy": "alloy",
#     "ash": "ash",
#     "ballad": "ballad",
#     "coral": "coral",
#     "echo": "echo",
#     "fable": "fable",
#     "onyx": "onyx",
#     "nova": "nova",
#     "sage": "sage",
#     "shimmer": "shimmer",
#     "verse": "verse",
#     "dave": "dave"
# }


# class OpenAITTSService(TTSService):
#     """OpenAI Text-to-Speech service that generates audio from text.

#     This service uses the OpenAI TTS API to generate PCM-encoded audio at 24kHz.
#     Supports multiple voice models and configurable parameters for high-quality
#     speech synthesis with streaming audio output.
#     """

#     OPENAI_SAMPLE_RATE = 24000  # OpenAI TTS always outputs at 24kHz

#     class InputParams(BaseModel):
#         """Input parameters for OpenAI TTS configuration.

#         Parameters:
#             instructions: Instructions to guide voice synthesis behavior.
#             speed: Voice speed control (0.25 to 4.0, default 1.0).
#         """

#         instructions: Optional[str] = None
#         speed: Optional[float] = None

#     def __init__(
#         self,
#         *,
#         api_key: Optional[str] = None,
#         base_url: Optional[str] = None,
#         voice: str = "alloy",
#         model: str = "gpt-4o-mini-tts",
#         sample_rate: Optional[int] = None,
#         instructions: Optional[str] = None,
#         speed: Optional[float] = None,
#         params: Optional[InputParams] = None,
#         **kwargs,
#     ):
#         """Initialize OpenAI TTS service.

#         Args:
#             api_key: OpenAI API key for authentication. If None, uses environment variable.
#             base_url: Custom base URL for OpenAI API. If None, uses default.
#             voice: Voice ID to use for synthesis. Defaults to "alloy".
#             model: TTS model to use. Defaults to "gpt-4o-mini-tts".
#             sample_rate: Output audio sample rate in Hz. If None, uses OpenAI's default 24kHz.
#             instructions: Optional instructions to guide voice synthesis behavior.
#             speed: Voice speed control (0.25 to 4.0, default 1.0).
#             params: Optional synthesis controls (acting instructions, speed, ...).
#             **kwargs: Additional keyword arguments passed to TTSService.

#                 .. deprecated:: 0.0.91
#                         The `instructions` and `speed` parameters are deprecated, use `InputParams` instead.
#         """
#         if sample_rate and sample_rate != self.OPENAI_SAMPLE_RATE:
#             logger.warning(
#                 f"OpenAI TTS only supports {self.OPENAI_SAMPLE_RATE}Hz sample rate. "
#                 f"Current rate of {sample_rate}Hz may cause issues."
#             )
#         super().__init__(sample_rate=sample_rate, **kwargs)

#         self.set_model_name(model)
#         self.set_voice(voice)
#         self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

#         if instructions or speed:
#             import warnings

#             with warnings.catch_warnings():
#                 warnings.simplefilter("always")
#                 warnings.warn(
#                     "The `instructions` and `speed` parameters are deprecated, use `InputParams` instead.",
#                     DeprecationWarning,
#                     stacklevel=2,
#                 )

#         self._settings = {
#             "instructions": params.instructions if params else instructions,
#             "speed": params.speed if params else speed,
#         }

#     def can_generate_metrics(self) -> bool:
#         """Check if this service can generate processing metrics.

#         Returns:
#             True, as OpenAI TTS service supports metrics generation.
#         """
#         return True

#     async def set_model(self, model: str):
#         """Set the TTS model to use.

#         Args:
#             model: The model name to use for text-to-speech synthesis.
#         """
#         logger.info(f"Switching TTS model to: [{model}]")
#         self.set_model_name(model)

#     async def start(self, frame: StartFrame):
#         """Start the OpenAI TTS service.

#         Args:
#             frame: The start frame containing initialization parameters.
#         """
#         await super().start(frame)
#         if self.sample_rate != self.OPENAI_SAMPLE_RATE:
#             logger.warning(
#                 f"OpenAI TTS requires {self.OPENAI_SAMPLE_RATE}Hz sample rate. "
#                 f"Current rate of {self.sample_rate}Hz may cause issues."
#             )

#     @traced_tts
#     async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
#         """Generate speech from text using OpenAI's TTS API.

#         Args:
#             text: The text to synthesize into speech.

#         Yields:
#             Frame: Audio frames containing the synthesized speech data.
#         """
#         logger.debug(f"{self}: Generating TTS [{text}]")
#         try:
#             await self.start_ttfb_metrics()

#             # Setup API parameters
#             create_params = {
#                 "input": text,
#                 "model": self.model_name,
#                 "voice": VALID_VOICES[self._voice_id],
#                 "response_format": "pcm",
#             }

#             if self._settings["instructions"]:
#                 create_params["instructions"] = self._settings["instructions"]

#             if self._settings["speed"]:
#                 create_params["speed"] = self._settings["speed"]

#             async with self._client.audio.speech.with_streaming_response.create(
#                 **create_params
#             ) as r:
#                 if r.status_code != 200:
#                     error = await r.text()
#                     logger.error(
#                         f"{self} error getting audio (status: {r.status_code}, error: {error})"
#                     )
#                     yield ErrorFrame(
#                         error=f"Error getting audio (status: {r.status_code}, error: {error})"
#                     )
#                     return

#                 await self.start_tts_usage_metrics(text)

#                 CHUNK_SIZE = 1024

#                 yield TTSStartedFrame()
#                 async for chunk in r.iter_bytes(CHUNK_SIZE):
#                     if len(chunk) > 0:
#                         await self.stop_ttfb_metrics()
#                         frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
#                         yield frame
#                 yield TTSStoppedFrame()
#         except BadRequestError as e:
#             yield ErrorFrame(error=f"Unknown error occurred: {e}")