#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import sys
from contextlib import asynccontextmanager

import uvicorn
from bot import run_bot
from dotenv import load_dotenv
import os
from fastapi import BackgroundTasks, FastAPI, Request, HTTPException, Query
from fastapi.responses import FileResponse
from loguru import logger
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from aiortc import RTCIceServer
from fastapi.middleware.cors import CORSMiddleware

from transcription_handler import TranscriptHandler
from uuid import uuid4

from utils import get_metrics, get_texts, clear_metrics, benchmark_sink

# Thêm sau khi tạo app:
logger.add(benchmark_sink, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}")

# Load environment variables
load_dotenv(override=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await small_webrtc_handler.close()


app = FastAPI(lifespan = lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize the SmallWebRTC request handler
small_webrtc_handler = SmallWebRTCRequestHandler(
    ice_servers=[
        RTCIceServer(
            urls=os.getenv("STUN_URL"),
        ),
        RTCIceServer(
            urls=os.getenv("TURN_URL"),
            username=os.getenv("TURN_USERNAME"),
            credential=os.getenv("TURN_CREDENTIAL"),
        )
    ]
)

transcript_handler = TranscriptHandler(session_id = None, mongo_uri = os.getenv("MONGO_URI"), database_name = os.getenv("DATABASE_NAME"), collection_name = os.getenv("COLLECTION_NAME"))

@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks, session_id = Query(...)):
    async def webrtc_connection_callback(connection):
        background_tasks.add_task(run_bot, connection, session_id)

    answer = await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=webrtc_connection_callback,
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    logger.debug(f"Received patch request: {request}")
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


@app.get("/api/chat-sessions")
async def load_chat_sessions():
    logger.debug(f"Getting previous sessions")
    chat_sessions = await transcript_handler.get_chat_history()
    return {"chat_sessions": chat_sessions}

@app.get("/api/chat-sessions/{session_id}")
async def load_chat_session(session_id: str):
    logger.debug(f"Getting chat session '{session_id}'")
    if session_id == "new":
        return []
    session = await transcript_handler.get_chat_history(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Chat session '{session_id}' not found"
        )

    return session

@app.delete("/api/chat-sessions/{session_id}")
async def delete_chat_session(session_id: str):
    logger.debug(f"Deleting chat session '{session_id}'")
    await transcript_handler.clear_session_by_id(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.post("/api/chat-sessions/create")
async def create_chat_session():
    session_id = uuid4()
    logger.debug(f"Creating chat session '{session_id}'")
    
    return {"session_id": session_id}

@app.get("/benchmark/session/{session_id}")
async def get_benchmark_metrics(session_id: str):
    benchmark_sink.flush_session(session_id)
    metrics = get_metrics(session_id)
    texts   = get_texts(session_id)
    return {"session_id": session_id, "metrics": metrics, "texts": texts}

@app.delete("/benchmark/session/{session_id}")
async def clear_benchmark_metrics(session_id: str):
    clear_metrics(session_id)
    return {"status": "cleared"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC demo")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logger.remove(0)
    if args.verbose:
        logger.add(sys.stderr, level="TRACE")
    else:
        logger.add(sys.stderr, level="DEBUG")

    uvicorn.run(app, host=args.host, port=args.port)