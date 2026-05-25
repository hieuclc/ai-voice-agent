import asyncio
import argparse
import logging
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.getLogger("root").addFilter(
    lambda r: "missing tensor" not in r.getMessage()
              and "unexpected tensor" not in r.getMessage()
)
logger = logging.getLogger("stt_server")


REQUEST_TIMEOUT = 20

_model = None
_executor: Optional[ThreadPoolExecutor] = None
_device: str = "cpu"  # resolved at startup, used everywhere


def _resolve_device(requested: str) -> str:
    """
    Resolve the actual device to use based on what was requested and what's available.
    Logs a clear warning if the requested device is unavailable and falls back to CPU.
    """
    if requested == "cuda":
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info("GPU detected: %s (%.1f GB VRAM) — running on CUDA", name, vram)
            return "cuda"
        else:
            logger.warning(
                "Device 'cuda' requested but no CUDA-capable GPU found — falling back to CPU. "
                "Inference will be significantly slower."
            )
            return "cpu"

    if requested == "cpu":
        logger.info("Device set to CPU (explicitly requested)")
        return "cpu"

    # "auto" or anything else: prefer GPU, fall back to CPU
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info("GPU detected: %s (%.1f GB VRAM) — running on CUDA", name, vram)
        return "cuda"
    else:
        logger.info("No GPU detected — running on CPU")
        return "cpu"

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
    logger.info("infer: %.1fms → %r", ms, text[:60])
    return text

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _executor

    logger.info("Loading ChunkFormer model...")
    from chunkformer import ChunkFormerModel
    _model = ChunkFormerModel.from_pretrained("khanhld/chunkformer-ctc-large-vie").to(_device)
    _model.eval()
    torch.set_grad_enabled(False)

    logger.info("Model loaded on %s", _device.upper())
    logger.info("Warming up model...")

    import wave, struct, io
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack('<16000h', *([0] * 16000)))
        _infer(buf.getvalue())
    logger.info("Model ready ✓")

    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")

    yield

    _executor.shutdown(wait=False)
    logger.info("Executor shut down")


app = FastAPI(title="ChunkFormer STT Server", version="4.2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file:       UploadFile = File(...),
    model_name: str        = Form(None),
    language:   str        = Form(None),
):
    t0 = time.perf_counter()
    audio = await file.read()
    t1 = time.perf_counter()

    loop = asyncio.get_running_loop()

    try:
        text = await asyncio.wait_for(
            loop.run_in_executor(_executor, _infer, audio),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Request timed out after %ss (audio=%d bytes)", REQUEST_TIMEOUT, len(audio))
        raise HTTPException(status_code=504, detail="Inference timed out")
    except Exception as e:
        logger.exception("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    t2 = time.perf_counter()
    logger.info(
        "request: read=%.1fms infer=%.1fms total=%.1fms",
        1000 * (t1 - t0),
        1000 * (t2 - t1),
        1000 * (t2 - t0),
    )

    return {"text": text}


@app.get("/health")
def health():
    logger.debug("Health check — device=%s", _device)
    return {"status": "ok", "device": _device}

def main():
    global _device

    parser = argparse.ArgumentParser()
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--port",    type=int, default=8005)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on. 'auto' picks GPU if available, else CPU. (default: auto)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())

    # Resolve + log device before anything else starts
    _device = _resolve_device(args.device)

    logger.info(
        "Starting STT server — host=%s port=%d device=%s",
        args.host, args.port, _device.upper(),
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
        workers=1,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()