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
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

# from pipecat.services.openai.tts import OpenAITTSService
from ttsv2 import OpenAITTSService
from stt import OpenAISTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.processors.transcript_processor import TranscriptProcessor

from transcription_handler import TranscriptHandler
from mcp_service import MCPClient
from mcp.client.session_group import StreamableHttpParameters
logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(webrtc_connection, session_id):
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            # audio_out_10ms_chunks=2,
        ),
    )
    logger.info(f"Starting bot")

    stt = OpenAISTTService(
        model="gpt-4o-mini-transcribe",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    tts = OpenAITTSService(
        model = "gpt-4o-mini-tts",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url = "http://localhost:8001/v1",
        voice = "alloy"
    )

    llm = OpenAILLMService(model = "gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    server_params = StreamableHttpParameters(
        url = os.getenv("MCP_SERVER"),
        timeout = 30
    )
    mcp = MCPClient(server_params=server_params)

    tools = await mcp.register_tools(llm)
    print("tools", tools)
    prompt = r'''
    Bạn là một trợ lý giọng nói Tiếng Việt chuyên tư vấn cho người dùng.

    Hãy làm theo những chỉ dẫn sau:

    1. Giới thiệu bản thân trong 15 từ.

    2. Luôn trả lời bằng tiếng Việt và câu trả lời phải là văn bản thuần, giống như đang nói chuyện trực tiếp với người dùng. Không dùng ký tự đặc biệt và không dùng chữ số, mọi con số phải viết bằng chữ.

    3. Chỉ trả về nội dung trả lời, không giải thích thêm, không định dạng đặc biệt.

    4. Khi tư vấn cho người dùng, luôn thực hiện các bước sau:
    - So sánh với ít nhất một mốc, nguồn hoặc tình huống liên quan nếu có
    - Nhận xét xu hướng hoặc sự khác biệt chính
    - Đưa ra gợi ý hành động có điều kiện, tránh trả lời chung chung hoặc trung lập

    5. Không sử dụng các câu trả lời né tránh như “tùy bạn”, “phụ thuộc vào bạn”, hoặc chỉ liệt kê thông tin mà không có nhận xét.

    '''

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
    ]

    transcript = TranscriptProcessor()

    transcript_handler = TranscriptHandler(session_id = session_id)
    await transcript_handler.load_session()
    if transcript_handler.messages:
        for message in transcript_handler.messages:
            messages.append({
                "role": message.role,
                "content": message.content
            })

    context = LLMContext(messages, tools = tools)
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
            ),
        ),
    )

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            transcript.user(),
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            transcript.assistant(),
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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        if not transcript_handler.messages:
            messages.append({"role": "system", "content": "Say hello and briefly introduce yourself in Vietnamese."})
        await task.queue_frames([LLMRunFrame()])

    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        await transcript_handler.on_transcript_update(processor, frame)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()
        await transport.close()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)

