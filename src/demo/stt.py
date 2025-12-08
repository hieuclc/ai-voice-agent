from pipecat.services.whisper.base_stt import BaseWhisperSTTService, Transcription
from pipecat.transcriptions.language import Language
from openai.resources.audio.transcriptions import Transcription

import time
import wave
from pathlib import Path
from typing import List, Tuple, Optional
import io
import numpy as np
import sherpa_onnx
import asyncio


def read_wave(wave_bytes) -> Tuple[np.ndarray, int]:
  recognizer = None
  with wave.open(io.BytesIO(wave_bytes), 'rb') as f:
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
        model="zipformer",
        api_key="my-api-key",
        encoder="/content/zipformer/encoder-epoch-20-avg-10.int8.onnx",
        decoder="/content/zipformer/decoder-epoch-20-avg-10.int8.onnx",
        joiner="/content/zipformer/joiner-epoch-20-avg-10.int8.onnx",
        tokens="/content/zipformer/tokens.txt",
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
            # base_url=base_url,
            # language=language,
            # prompt=prompt,
            # temperature=temperature,
            **kwargs,
        )

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=num_threads,
            sample_rate=sample_rate,
            feature_dim=feature_dim,
            decoding_method=decoding_method,
            blank_penalty=blank_penalty
        )

    async def _transcribe(self, audio: bytes) -> Transcription:
        samples, sr = read_wave(audio)

        text = await asyncio.to_thread(self._run_recognition, samples, sr)

        return Transcription(text=text.lower(), usage=None)

    def _run_recognition(self, samples, sample_rate):
        s = self.recognizer.create_stream()
        s.accept_waveform(sample_rate, samples)
        self.recognizer.decode_stream(s)

        return s.result.text.strip()