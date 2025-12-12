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

        if model == "zipformer":
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
        
        if model == "whisper-large-v3":
            self.recognizer = WhisperModel("large-v3", device="cuda", compute_type="float16")

    async def _transcribe(self, audio: bytes) -> Transcription:
        if self.model_name == "zipformer":
            samples, sr = read_wave(audio)
        if self.model_name == "whisper-v3-large":
            samples = audio
            sr = None
        text = await asyncio.to_thread(self._run_recognition, samples, sr)

        return Transcription(text=text.lower(), usage=None)

    def _run_recognition(self, samples, sample_rate):
        if self.model_name == "zipformer":
            s = self.recognizer.create_stream()
            s.accept_waveform(sample_rate, samples)
            self.recognizer.decode_stream(s)
            return s.result.text.strip()

        if self.model_name == "whisper-large-v3":
            audio_input = BytesIO(samples)
            segments, info = self.model.transcribe(audio_input, language = "vi", multilingual = True)
            return " ".join([segment.text.lower() for segment in segments]).strip()
            