import os

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.openai.tts import OpenAITTSService
# from pipecat.services.openai.stt import OpenAISTTService
from stt import OpenAISTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.serializers.protobuf import ProtobufFrameSerializer

from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from mcp_service import MCPClient
from mcp.client.session_group import StreamableHttpParameters
logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(websocket_client):
    ws_transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=ProtobufFrameSerializer(),
        ),
    )
    logger.info(f"Starting bot")

    # stt = OpenAISTTService(
    #     model="gpt-4o-mini-transcribe",
    #     api_key=os.getenv("OPENAI_API_KEY"),
    # )

    stt = OpenAISTTService(
        model="whisper-large-v3",
        api_key="my-api-key",
        encoder="/home/hieuclc/kltn/ai-voice-agent/zipformer/encoder-epoch-20-avg-10.int8.onnx",
        decoder="/home/hieuclc/kltn/ai-voice-agent/zipformer/decoder-epoch-20-avg-10.int8.onnx",
        joiner="/home/hieuclc/kltn/ai-voice-agent/zipformer/joiner-epoch-20-avg-10.int8.onnx",
        tokens="/home/hieuclc/kltn/ai-voice-agent/zipformer/tokens.txt",
    )

    tts = OpenAITTSService(
        model = "gpt-4o-mini-tts",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    llm = OpenAILLMService(model = "gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    # Initialize and configure MCPClient with MCP SSE server url
    server_params = StreamableHttpParameters(
        url = os.getenv("MCP_SERVER"),
        timeout = 20
    )
    mcp = MCPClient(server_params=server_params)
    mcp._tools_dict = {
        "google_search": {
            "text_content": "Tôi đang thực hiện tìm kiếm thông tin, vui lòng chờ trong giây lát...",
            "run_llm": True
        }
    }
    tools = await mcp.register_tools(llm)

    llm_system_prompt = "You are an friendly AI assistant speaking in Vietnamese"
    messages = [
        {
            "role": "system",
            "content": llm_system_prompt
        },

    ]

    context = LLMContext(messages, tools = tools)
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            ws_transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            ws_transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself in Vietnamese."})
        await task.queue_frames([LLMRunFrame()])

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)

