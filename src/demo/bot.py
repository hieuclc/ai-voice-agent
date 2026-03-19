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
from pipecat.frames.frames import LLMRunFrame, Frame, TextFrame, TTSSpeakFrame

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
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from pipecat.services.openai.tts import OpenAITTSService
# from ttsv2 import OpenAITTSService
# from stt import OpenAISTTService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.processors.transcript_processor import TranscriptProcessor

# from transcription_handler import TranscriptHandler
logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

from benchmark_log_sink import current_session_id

# ---------------------------------------------------------------------------
# Thinking sentence prefixes — phải khớp với THINKING_SENTENCES_* trong agent.py
# ---------------------------------------------------------------------------
_THINKING_SENTENCES = set([
    "Tôi đang thực hiện tìm kiếm thông tin, vui lòng chờ trong giây lát.",
    "Tôi sẽ tìm kiếm dữ liệu ngay bây giờ, vui lòng chờ đợi.",
    "Để trả lời chính xác, tôi cần tra cứu dữ liệu, xin vui lòng chờ.",
    "Quá trình tìm kiếm vẫn đang tiếp tục, vui lòng chờ thêm.",
    "Hệ thống đang truy xuất dữ liệu liên quan, xin vui lòng đợi.",
    "Đang phân tích các nguồn tài liệu, vui lòng kiên nhẫn chờ đợi.",
    "Tìm kiếm vẫn đang được thực hiện, kết quả sẽ có trong chốc lát.",
    "Hệ thống vẫn đang xử lý yêu cầu, vui lòng chờ thêm một chút.",
])

def _is_thinking_sentence(text: str) -> bool:
    return text.strip() in _THINKING_SENTENCES


class ThinkingSentenceProcessor(FrameProcessor):
    """
    Đặt giữa llm và tts trong pipeline.

    Vấn đề: SimpleTextAggregator (aggregate_sentences=True) dùng cross-frame
    lookahead — buffer câu thinking cho đến khi thấy chữ hoa đầu câu tiếp theo
    → delay 6-7 giây trước khi TTS xử lý.

    Fix: intercept TextFrame chứa thinking sentence, convert thành TTSSpeakFrame.
    TTSService xử lý TTSSpeakFrame bằng _push_tts_frames(AggregationType.SENTENCE)
    — hoàn toàn bypass SimpleTextAggregator.
    Các TextFrame bình thường (response tokens) vẫn đi qua aggregator như cũ.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and _is_thinking_sentence(frame.text):
            logger.debug(
                "ThinkingSentenceProcessor: converting to TTSSpeakFrame: %r",
                frame.text[:60],
            )
            await self.push_frame(TTSSpeakFrame(frame.text.strip()), direction)
        else:
            await self.push_frame(frame, direction)


async def run_bot(webrtc_connection, session_id):
    _token = current_session_id.set(session_id)
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_bitrate=8000,
            vad_analyzer=SileroVADAnalyzer(),
            # audio_out_10ms_chunks=2,
        ),
    )
    logger.info(f"Starting bot")

    stt = OpenAISTTService(
        model="gpt-4o-mini-transcribe",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url = "http://localhost:8003/v1"
    )

    tts = OpenAITTSService(
        model = "gpt-4o-mini-tts",
        api_key=os.getenv("OPENAI_API_KEY"),
        # base_url = "http://localhost:8001/v1",
        voice = "alloy"
    )

    llm = OpenAILLMService(model = "gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), base_url = "http://localhost:8000/v1")

    prompt = r'''
    Bạn là một trợ lý giọng nói Tiếng Việt chuyên tư vấn cho người dùng.

    Hãy làm theo những chỉ dẫn sau:

    1. Đầu tiên, hãy chào người dùng như sau: "Xin chào bạn, bạn cần giúp gì hôm nay". Sau đó không hỏi gì thêm

    2. Luôn trả lời bằng tiếng Việt và câu trả lời phải là văn bản thuần, giống như đang nói chuyện trực tiếp với người dùng. Không dùng ký tự đặc biệt và không dùng chữ số, mọi con số phải viết bằng chữ.

    3. Chỉ trả về nội dung trả lời, không giải thích thêm, không định dạng đặc biệt. Các cụm từ viết tắt phải được viết **in hoa**. Mỗi câu phải có từ 4 từ trở lên.

    4. Khi người dùng hỏi theo hướng tư vấn hoặc ra quyết định:
    - Luôn so sánh với ít nhất một mốc thời gian hoặc tình huống liên quan.
    - Nhận xét xu hướng hoặc khác biệt chính dựa trên dữ liệu đã có.
    - Đưa ra lời khuyên HỮU ÍCH dựa trên xu hướng đó, nhưng phải kèm điều kiện hoặc lưu ý rủi ro.

    5. Khi người dùng hỏi "có nên mua", "có nên bán", "đầu tư":
    - KHÔNG được chỉ dựa trên dữ liệu của một ngày hay một đối tượng.
    - BẮT BUỘC phải sử dụng dữ liệu nhiều mốc (ít nhất 2-3 mốc gần nhất).
    - Nếu chưa có dữ liệu, phải gọi tool để bổ sung trước khi trả lời.
    - Sau khi phân tích, phải đưa ra khuyến nghị rõ ràng, nhưng luôn nêu ít nhất một rủi ro hoặc trường hợp khiến khuyến nghị không còn phù hợp.

    6. Tránh trả lời chung chung hoặc trung lập; khuyến nghị cần cụ thể, nhưng không khẳng định chắc chắn cho mọi tình huống.


    '''

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
    ]

    transcript = TranscriptProcessor()

    # transcript_handler = TranscriptHandler(session_id = session_id)
    # await transcript_handler.load_session()
    # if transcript_handler.messages:
    #     for message in transcript_handler.messages:
    #         messages.append({
    #             "role": message.role,
    #             "content": message.content
    #         })

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
            ),
        ),
    )

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    thinking_processor = ThinkingSentenceProcessor()

    pipeline = Pipeline(
        [
            transport.input(),           # Transport user input
            rtvi,                        # RTVI processor
            stt,
            transcript.user(),
            context_aggregator.user(),   # User responses
            llm,                         # LLM
            thinking_processor,          # Convert thinking sentences → TTSSpeakFrame (bypass aggregator)
            tts,                         # TTS
            transport.output(),          # Transport bot output
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
        # if not transcript_handler.messages:
        messages.append({"role": "system", "content": "Hãy nói câu sau: 'Xin chào bạn, tôi có thể giúp gì cho bạn hôm nay.'"})
        await task.queue_frames([LLMRunFrame()])

    # @transcript.event_handler("on_transcript_update")
    # async def on_transcript_update(processor, frame):
    #     await transcript_handler.on_transcript_update(processor, frame)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()
        # await transport.close()

    runner = PipelineRunner(handle_sigint=False)

    try:
        await runner.run(task)
    finally:
        current_session_id.reset(_token)