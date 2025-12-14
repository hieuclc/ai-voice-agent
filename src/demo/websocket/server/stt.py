from pipecat.services.whisper.base_stt import BaseWhisperSTTService, Transcription
from openai.resources.audio.transcriptions import Transcription

from faster_whisper import WhisperModel
from io import BytesIO
import wave
from typing import Tuple
import numpy as np
import sherpa_onnx
import asyncio

def read_wave(wave_bytes) -> Tuple[np.ndarray, int]:
  recognizer = None
  with wave.open(BytesIO(wave_bytes), 'rb') as f:
      assert f.getnchannels() == 1, f.getnchannels()
      assert f.getsampwidth() == 2, f.getsampwidth()
      num_samples = f.getnframes()
      samples = f.readframes(num_samples)
      samples_int16 = np.frombuffer(samples, dtype=np.int16)
      samples_float32 = samples_int16.astype(np.float32)

      samples_float32 = samples_float32 / 32768
      return samples_float32, f.getframerate()

class OpenAISTTService(BaseWhisperSTTService):
    def __init__(
        self,
        model,
        api_key,
        encoder,
        decoder,
        joiner,
        tokens,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        blank_penalty=0.0,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            **kwargs,
        )

        self.recognizer = WhisperModel("/teamspace/studios/this_studio/phowhisper-ct2", device="cuda", compute_type="float32")

    async def _transcribe(self, audio: bytes) -> Transcription:
        samples = audio
        sr = None
        text = await asyncio.to_thread(self._run_recognition, samples, sr)

        return Transcription(text=text.lower(), usage=None)

    def _run_recognition(self, samples, sample_rate):
        audio_input = BytesIO(samples)
        segments, info = self.recognizer.transcribe(audio_input, language = "vi")
        return " ".join([segment.text.strip().lower() for segment in segments]).strip()
            