"""
stt_server.py — ChunkFormer STT, latency-optimized for fast GPU inference

Design rationale:
  - Inference = 37ms → batching adds wait overhead, not throughput
  - asyncio.to_thread = ~10-30ms overhead per call → replaced with
    a dedicated ThreadPoolExecutor (1 thread) so the GPU thread is
    always warm and never recreated
  - Queue + future round-trip removed for the common single-request case
  - ThreadPoolExecutor(1): GPU calls serialize naturally, no CUDA conflict
  - For true concurrency (multiple simultaneous callers), increase
    max_workers — CUDA will time-slice but 37ms inference means
    queuing is still fast
"""

import asyncio
import argparse
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import Optional

# =========================
# CONFIG
# =========================
REQUEST_TIMEOUT = 20
# 1 thread = GPU calls serialize (safe, no CUDA conflict)
# Increase to 2-3 if you have multiple simultaneous callers and
# are willing to trade VRAM for concurrency
GPU_THREADS = 1

_model = None
_executor: Optional[ThreadPoolExecutor] = None
_loop: Optional[asyncio.AbstractEventLoop] = None


# =========================
# INFERENCE
# =========================

def _infer(audio: bytes) -> str:
    """Runs in the dedicated GPU thread. No asyncio, no overhead."""
    t0 = time.perf_counter()
    text = _model.endless_decode(
        audio_bytes=audio,
        chunk_size=64,
        left_context_size=128,
        right_context_size=128,
        total_batch_duration=14400,
        return_timestamps=False,
    )
    ms = (time.perf_counter() - t0) * 1000
    print(f"[infer] {ms:.1f}ms → {repr(text[:60])}")
    return text


# =========================
# LIFESPAN
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _executor, _loop

    _loop = asyncio.get_running_loop()

    print("Loading ChunkFormer model on GPU...")
    from chunkformer import ChunkFormerModel
    _model = ChunkFormerModel.from_pretrained(
        "khanhld/chunkformer-ctc-large-vie"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    _model.eval()
    torch.set_grad_enabled(False)

    # Warm up — first inference is always slower due to CUDA JIT
    print("Warming up model...")
    import wave, struct, io
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack('<1600h', *([0] * 1600)))  # 0.1s silence
    _infer(buf.getvalue())
    print("Model ready ✓")

    # Single persistent thread keeps the GPU context warm
    _executor = ThreadPoolExecutor(max_workers=GPU_THREADS, thread_name_prefix="gpu")

    yield

    _executor.shutdown(wait=False)


# =========================
# FASTAPI APP
# =========================

app = FastAPI(title="ChunkFormer STT Server", version="4.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# ENDPOINT
# =========================

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file:       UploadFile = File(...),
    model_name: str        = Form(None),
    language:   str        = Form(None),
):
    t0 = time.perf_counter()
    audio = await file.read()
    t1 = time.perf_counter()

    try:
        # Run inference in the dedicated GPU thread
        # loop.run_in_executor with a pre-warmed ThreadPoolExecutor
        # avoids the ~10-30ms thread-spawn overhead of asyncio.to_thread
        text = await asyncio.wait_for(
            _loop.run_in_executor(_executor, _infer, audio),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return {"error": "request timeout"}
    except Exception as e:
        return {"error": str(e)}

    t2 = time.perf_counter()
    print(f"[request] read={1000*(t1-t0):.1f}ms infer={1000*(t2-t1):.1f}ms total={1000*(t2-t0):.1f}ms")

    return {"text": text}


@app.get("/health")
def health():
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}


# =========================
# ENTRY POINT
# =========================

def main():
    global GPU_THREADS

    parser = argparse.ArgumentParser()
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--port",    type=int, default=8004)
    parser.add_argument("--threads", type=int, default=GPU_THREADS)
    args = parser.parse_args()

    GPU_THREADS = args.threads

    print(f"Starting STT server on port {args.port}, GPU_THREADS={GPU_THREADS}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()