from typing import AsyncGenerator, Optional

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import np_to_triton_dtype
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame, Frame, StartFrame,
    TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

from utils import normalize_sentence


SAMPLE_RATE = 24000
FRAME_SIZE  = 240    # 10ms @ 24kHz
FADE_MS     = 20
PAUSE_MS    = 150


def _fade_in(pcm: np.ndarray, fade_len: int) -> np.ndarray:
    pcm = pcm.copy()
    L = min(fade_len, len(pcm))
    pcm[:L] = (pcm[:L] * np.linspace(0, 1, L, endpoint=False)).astype(np.int16)
    return pcm


def _fade_out(pcm: np.ndarray, fade_len: int) -> np.ndarray:
    pcm = pcm.copy()
    L = min(fade_len, len(pcm))
    pcm[-L:] = (pcm[-L:] * np.linspace(1, 0, L, endpoint=False)).astype(np.int16)
    return pcm


class ZipVoiceTTSService(TTSService):

    SAMPLE_RATE = SAMPLE_RATE

    class InputParams(BaseModel):
        pass

    def __init__(
        self,
        *,
        triton_url: str = "localhost:8002",
        model_name: str = "zipvoice",
        **kwargs,
    ):
        super().__init__(sample_rate=self.SAMPLE_RATE, **kwargs)
        self._triton_url = triton_url
        self._model_name = model_name
        self._client: Optional[grpcclient.InferenceServerClient] = None
        self._prev_need_fade_in = False

        self._fade_len = int(SAMPLE_RATE * FADE_MS  / 1000)
        self._silence  = np.zeros(int(SAMPLE_RATE * PAUSE_MS / 1000), dtype=np.int16)

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._client = grpcclient.InferenceServerClient(url=self._triton_url, verbose=False)
        logger.info(f"ZipVoiceTTS: connected to Triton gRPC @ {self._triton_url}")

    async def stop(self, frame):
        await super().stop(frame)
        if self._client:
            await self._client.close()
            self._client = None

    async def _infer(self, text: str) -> np.ndarray:
        """Gọi Triton, trả về PCM int16 @ 24kHz."""
        inp = np.array([[normalize_sentence(text)]], dtype=object)
        inputs = [grpcclient.InferInput("target_text", inp.shape, np_to_triton_dtype(inp.dtype))]
        inputs[0].set_data_from_numpy(inp)

        result = await self._client.infer(
            model_name=self._model_name,
            inputs=inputs,
            outputs=[grpcclient.InferRequestedOutput("waveform")],
        )

        wav = result.as_numpy("waveform").flatten().astype(np.float32)
        return (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

    @traced_tts
    async def run_tts(self, text: str, language: str = "vi") -> AsyncGenerator[Frame, None]:
        logger.debug(f"ZipVoiceTTS: [{text}]")
        try:
            yield TTSStartedFrame()

            pcm = await self._infer(text)

            if self._prev_need_fade_in:
                pcm = _fade_in(pcm, self._fade_len)
            pcm = _fade_out(pcm, self._fade_len)

            for i in range(0, len(pcm), FRAME_SIZE):
                yield TTSAudioRawFrame(pcm[i:i + FRAME_SIZE].tobytes(), SAMPLE_RATE, 1)

            yield TTSAudioRawFrame(self._silence.tobytes(), SAMPLE_RATE, 1)
            self._prev_need_fade_in = True

            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"ZipVoiceTTS error: {e}")
            yield ErrorFrame(error=str(e))