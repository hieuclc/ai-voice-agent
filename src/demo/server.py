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

# Load environment variables
load_dotenv(override=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    await small_webrtc_handler.close()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     yield
#     logger.warning("FastAPI shutting down...")

#     # 1️⃣ Cancel all bot tasks
#     for task in list(active_bot_tasks):
#         task.cancel()

#     await asyncio.gather(*active_bot_tasks, return_exceptions=True)

#     # 2️⃣ Close WebRTC handler (aiortc)
#     await small_webrtc_handler.close()

#     logger.warning("Shutdown complete")


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
            urls="stun:ss-turn1.xirsys.com",
        ),
        RTCIceServer(
            urls="turn:ss-turn1.xirsys.com:3478",
            username="ynuYRR28qU2zB-hB60HmYV6ulUr4Vxn3a08Fti13c0aGS-msyw6Iws7G22TbgQmgAAAAAGlBhJ9oaWV1bGNsY2JnYm4xMjM=",
            credential="d6c8b274-da99-11f0-ba29-0242ac140004",
        )
    ]
)

transcript_handler = TranscriptHandler(session_id = None)

import asyncio

active_bot_tasks: set[asyncio.Task] = set()

@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks, session_id = Query(...)):
    """Handle WebRTC offer requests via SmallWebRTCRequestHandler."""
    # Prepare runner arguments with the callback to run your bot
    async def webrtc_connection_callback(connection):
        background_tasks.add_task(run_bot, connection, session_id)
    # async def webrtc_connection_callback(connection):
    #     task = asyncio.create_task(run_bot(connection))
    #     active_bot_tasks.add(task)

    #     def _done(t: asyncio.Task):
    #         active_bot_tasks.discard(t)

    #     task.add_done_callback(_done)

    # Delegate handling to SmallWebRTCRequestHandler
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
from uuid import uuid4
@app.post("/api/chat-sessions/create")
async def create_chat_session():
    session_id = uuid4()
    logger.debug(f"Creating chat session '{session_id}'")
    
    return {"session_id": session_id}

# @app.get("/")
# async def serve_index():
#     return FileResponse("index.html")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC demo")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
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