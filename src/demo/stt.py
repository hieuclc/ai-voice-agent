from pipecat.services.whisper.base_stt import BaseWhisperSTTService, Transcription
from openai.resources.audio.transcriptions import Transcription

import asyncio
from chunkformer import ChunkFormerModel

class OpenAISTTService(BaseWhisperSTTService):
    def __init__(
        self,
        model,
        api_key,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            **kwargs,
        )

        self.recognizer = ChunkFormerModel.from_pretrained("khanhld/chunkformer-rnnt-large-vie")

    async def _transcribe(self, audio: bytes) -> Transcription:
        text = await asyncio.to_thread(self._run_recognition(audio))

        return Transcription(text=text.lower(), usage=None)

    def _run_recognition(self, audio):
        transcription = self.recognizer.endless_decode(
            audio_bytes=audio,
            chunk_size=64,
            left_context_size=128,
            right_context_size=128,
            total_batch_duration=14400,  # in seconds
            return_timestamps=False
        )
        return transcription
                    