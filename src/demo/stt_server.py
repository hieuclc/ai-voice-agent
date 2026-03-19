"""
stt_server.py — ChunkFormer STT server with dual concurrency strategy

  GPU  →  1 process, N asyncio batch-worker tasks (pipeline trick)
           CUDA serializes kernel calls anyway; spawning more processes
           only wastes VRAM without adding throughput.

  CPU  →  N worker processes via multiprocessing (true parallelism)
           Each process owns a full model copy and a real OS thread/core.
           mp.Queue bridges HTTP process ↔ inference workers.
"""

import asyncio
import time
import argparse
import uuid
import torch
import multiprocessing as mp
from multiprocessing import Process
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import List, Optional

# =========================
# SHARED CONFIG
# =========================
MAX_BATCH_SIZE   = 8
MAX_BATCH_WAIT   = 0.02   # seconds
QUEUE_SIZE       = 200
REQUEST_TIMEOUT  = 20
NUM_WORKERS      = 4      # async tasks (GPU) OR inference processes (CPU)

IS_GPU = torch.cuda.is_available()


# =========================
# INFERENCE CORE  (shared by both strategies)
# =========================

def _run_batch(model, audios: List[bytes]) -> List[str]:
    results = []
    for audio in audios:
        text = model.endless_decode(
            audio_bytes=audio,
            chunk_size=64,
            left_context_size=128,
            right_context_size=128,
            total_batch_duration=14400,
            return_timestamps=False,
        )
        results.append(text)
    return results


# =========================
# ── STRATEGY A: GPU ──
#    N asyncio tasks, 1 shared model
# =========================

# Module-level so lifespan and endpoint can both access
_gpu_model   = None
_gpu_queue: Optional[asyncio.Queue] = None


async def _gpu_batch_worker(worker_id: int):
    print(f"[GPU worker {worker_id}] started")
    while True:
        batch:   List[bytes]          = []
        futures: List[asyncio.Future] = []
        start = time.time()

        while len(batch) < MAX_BATCH_SIZE:
            timeout = MAX_BATCH_WAIT - (time.time() - start)
            if timeout <= 0:
                break
            try:
                req = await asyncio.wait_for(_gpu_queue.get(), timeout=timeout)
                batch.append(req["audio"])
                futures.append(req["future"])
            except asyncio.TimeoutError:
                break

        if not batch:
            continue

        try:
            # to_thread releases the event loop while GPU is busy
            texts = await asyncio.to_thread(_run_batch, _gpu_model, batch)
            for fut, text in zip(futures, texts):
                if not fut.done():
                    fut.set_result(text)
        except Exception as exc:
            for fut in futures:
                if not fut.done():
                    fut.set_exception(exc)


# =========================
# ── STRATEGY B: CPU ──
#    N processes, each with its own model copy
#    mp.Queue for IPC, asyncio Future resolved by a dispatcher task
# =========================

# Set in __main__ before any fork/spawn
_cpu_request_queue: Optional[mp.Queue] = None
_cpu_result_queue:  Optional[mp.Queue] = None

# Maps request_id → asyncio.Future (main process only)
_pending = {}


def _cpu_inference_worker(worker_id: int, req_q: mp.Queue, res_q: mp.Queue):
    """Runs in its own OS process. No asyncio, no GPU."""
    from chunkformer import ChunkFormerModel

    print(f"[CPU worker {worker_id}] loading model...")
    model = ChunkFormerModel.from_pretrained("khanhld/chunkformer-ctc-large-vie")
    model.eval()
    print(f"[CPU worker {worker_id}] ready")

    while True:
        batch_ids:    List[str]   = []
        batch_audios: List[bytes] = []
        start = time.time()

        while len(batch_ids) < MAX_BATCH_SIZE:
            timeout = MAX_BATCH_WAIT - (time.time() - start)
            if timeout <= 0:
                break
            try:
                item = req_q.get(timeout=max(timeout, 0.001))
                batch_ids.append(item[0])
                batch_audios.append(item[1])
            except Exception:
                break   # queue.Empty

        if not batch_ids:
            continue

        try:
            texts = _run_batch(model, batch_audios)
            for rid, text in zip(batch_ids, texts):
                res_q.put((rid, text, None))
        except Exception as exc:
            for rid in batch_ids:
                res_q.put((rid, None, exc))


async def _cpu_result_dispatcher():
    """Polls the cross-process result queue and resolves asyncio Futures."""
    while True:
        try:
            rid, text, exc = _cpu_result_queue.get_nowait()
            fut = _pending.pop(rid, None)
            if fut and not fut.done():
                if exc:
                    fut.set_exception(exc)
                else:
                    fut.set_result(text)
        except Exception:
            pass    # queue.Empty
        await asyncio.sleep(0.005)


# =========================
# LIFESPAN
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gpu_model, _gpu_queue

    if IS_GPU:
        # ── GPU strategy ──────────────────────────────────────────────────
        print(f"[GPU mode] Loading model on cuda...")
        from chunkformer import ChunkFormerModel
        _gpu_queue  = asyncio.Queue(maxsize=QUEUE_SIZE)
        _gpu_model  = ChunkFormerModel.from_pretrained(
            "khanhld/chunkformer-ctc-large-vie"
        ).to("cuda")
        _gpu_model.eval()
        torch.set_grad_enabled(False)
        print(f"[GPU mode] Spawning {NUM_WORKERS} async batch workers")
        tasks = [asyncio.create_task(_gpu_batch_worker(i)) for i in range(NUM_WORKERS)]

        yield

        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    else:
        # ── CPU strategy ──────────────────────────────────────────────────
        print(f"[CPU mode] Spawning {NUM_WORKERS} inference processes")
        procs = []
        for i in range(NUM_WORKERS):
            p = Process(
                target=_cpu_inference_worker,
                args=(i, _cpu_request_queue, _cpu_result_queue),
                daemon=True,
            )
            p.start()
            procs.append(p)

        dispatcher = asyncio.create_task(_cpu_result_dispatcher())

        yield

        dispatcher.cancel()
        for p in procs:
            p.terminate()


# =========================
# FASTAPI APP
# =========================

app = FastAPI(title="ChunkFormer STT Server", version="3.0", lifespan=lifespan)
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
    audio = await file.read()
    loop  = asyncio.get_running_loop()
    fut   = loop.create_future()   # register BEFORE enqueue → no race condition

    if IS_GPU:
        req = {"audio": audio, "future": fut}
        try:
            await asyncio.wait_for(_gpu_queue.put(req), timeout=5)
        except asyncio.TimeoutError:
            return {"error": "server overloaded"}
    else:
        rid = str(uuid.uuid4())
        _pending[rid] = fut
        try:
            _cpu_request_queue.put_nowait((rid, audio))
        except Exception:
            _pending.pop(rid, None)
            return {"error": "server overloaded"}

    try:
        text = await asyncio.wait_for(fut, timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        return {"error": "request timeout"}
    except Exception as e:
        return {"error": str(e)}

    return {"text": text}


@app.get("/health")
def health():
    return {"status": "ok", "device": "cuda" if IS_GPU else "cpu"}


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--port",    type=int, default=8003)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help="async tasks (GPU) or processes (CPU)")
    args = parser.parse_args()

    NUM_WORKERS = args.workers   # propagate CLI override

    if not IS_GPU:
        # CPU: queues must exist before any Process is forked
        mp.set_start_method("spawn", force=True)
        _cpu_request_queue = mp.Queue(maxsize=QUEUE_SIZE)
        _cpu_result_queue  = mp.Queue(maxsize=QUEUE_SIZE * NUM_WORKERS)

    device_label = "cuda" if IS_GPU else f"cpu ({NUM_WORKERS} workers)"
    print(f"Starting STT server on {device_label}")

    uvicorn.run(
        "stt_server:app",
        host=args.host,
        port=args.port,
        reload=False,
        workers=1,  # HTTP concurrency = asyncio, not uvicorn forks
    )