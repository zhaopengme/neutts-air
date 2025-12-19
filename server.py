from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List
import os
import io
import logging
from datetime import datetime
import base64
import aiofiles
from functools import lru_cache

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from neuttsair.neutts import NeuTTSAir
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check dependencies
GGUI_AVAILABLE = False
ONNX_AVAILABLE = False
CUDA_AVAILABLE = False

try:
    import llama_cpp
    GGUI_AVAILABLE = True
    logger.info("✓ llama-cpp-python found - GGUF models supported")
except ImportError:
    logger.warning("✗ llama-cpp-python not found - GGUF models not supported")

try:
    import onnxruntime
    ONNX_AVAILABLE = True
    logger.info("✓ onnxruntime found - ONNX decoder supported")
except ImportError:
    logger.warning("✗ onnxruntime not found - ONNX decoder not supported")

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        logger.info("✓ CUDA available - GPU acceleration supported")
    else:
        logger.info("- CUDA not available - using CPU")
except ImportError:
    logger.warning("✗ PyTorch not found")
    CUDA_AVAILABLE = False

# Security
security = HTTPBearer()

# Models
class TTSRequest(BaseModel):
    model: str = Field(default="tts-1", description="Model to use for TTS")
    input: str = Field(..., min_length=1, max_length=4096, description="Text to synthesize")
    voice: str = Field(default="alloy", description="Voice to use (alloy, echo, fable, onyx, nova, shimmer)")
    response_format: Literal["wav", "json"] = Field(default="wav", description="Audio format: wav or json (json returns base64 audio with alignment)")

class AlignmentInfo(BaseModel):
    characters: List[str]
    characterStartTimesSeconds: List[float]
    characterEndTimesSeconds: List[float]

class TTSJSONResponse(BaseModel):
    audio_base64: str
    alignment: AlignmentInfo

# Configuration
class Config:
    def __init__(self):
        # Server Configuration
        self.port = int(os.getenv("PORT", "8000"))
        self.host = os.getenv("HOST", "0.0.0.0")
        self.api_key = os.getenv("API_KEY")
        self.require_auth = bool(self.api_key)

        # TTS Model Configuration - intelligent defaults based on available dependencies
        if os.getenv("BACKBONE_REPO"):
            self.backbone_repo = os.getenv("BACKBONE_REPO")
        else:
            if GGUI_AVAILABLE:
                self.backbone_repo = "neuphonic/neutts-air-q8-gguf"
                logger.info("Using GGUF model (optimized)")
            else:
                self.backbone_repo = "neuphonic/neutts-air"
                logger.info("Using PyTorch model (fallback)")

        backbone_device_env = os.getenv("BACKBONE_DEVICE", "cpu").lower()
        if backbone_device_env == "cuda" and CUDA_AVAILABLE:
            self.backbone_device = "cuda"
            logger.info("Using GPU for backbone")
        else:
            self.backbone_device = "cpu"
            logger.info("Using CPU for backbone")

        codec_repo_env = os.getenv("CODEC_REPO")
        if codec_repo_env and codec_repo_env.strip():
            # Check if user is trying to use ONNX decoder
            if "onnx" in codec_repo_env.lower():
                logger.warning("⚠️  ONNX decoder cannot encode reference audio. Switching to PyTorch codec.")
                self.codec_repo = "neuphonic/neucodec"
            else:
                self.codec_repo = codec_repo_env
        else:
            # ONNX decoder can't encode reference audio, so we must use PyTorch codec
            # ONNX is only used for decoding in streaming mode
            self.codec_repo = "neuphonic/neucodec"
            logger.info("Using PyTorch codec (required for reference encoding)")

        codec_device_env = os.getenv("CODEC_DEVICE", "cpu").lower()
        if codec_device_env == "cuda" and CUDA_AVAILABLE:
            self.codec_device = "cuda"
            logger.info("Using GPU for codec")
        else:
            self.codec_device = "cpu"
            logger.info("Using CPU for codec")

        # Logging Configuration
        log_level = os.getenv("LOG_LEVEL", "INFO")
        logging.getLogger().setLevel(getattr(logging, log_level.upper()))

config = Config()

# Initialize TTS engine (using defaults)
tts_engine = None

def get_tts_engine():
    global tts_engine
    if tts_engine is None:
        # Use configured models or defaults
        backbone_repo = config.backbone_repo if config.backbone_repo else "neuphonic/neutts-air"
        codec_repo = config.codec_repo if config.codec_repo else "neuphonic/neucodec"

        logger.info(f"Initializing TTS engine with backbone: {backbone_repo}, codec: {codec_repo}")
        logger.info(f"Backbone device: {config.backbone_device}, Codec device: {config.codec_device}")

        tts_engine = NeuTTSAir(
            backbone_repo=backbone_repo,
            backbone_device=config.backbone_device,
            codec_repo=codec_repo,
            codec_device=config.codec_device
        )
        logger.info(f"✓ TTS Engine initialized successfully")
    return tts_engine

# Reference audio and text caching with LRU cache
ref_texts_cache = {}

@lru_cache(maxsize=10)
def get_cached_ref_codes(voice: str):
    """获取缓存的参考音频编码（LRU缓存，最多10个语音）"""
    engine = get_tts_engine()
    ref_audio_path = f"samples/{voice}.wav"
    logger.info(f"Cached reference codes for voice: {voice}")
    return engine.encode_reference(ref_audio_path)

async def get_cached_ref_text(voice: str):
    """获取缓存的参考文本（简单缓存，最多10个语音）"""
    if voice not in ref_texts_cache:
        ref_text_path = f"samples/{voice}.txt"
        async with aiofiles.open(ref_text_path, 'r') as f:
            ref_texts_cache[voice] = await f.read()
        # 如果缓存超过10个，删除最旧的
        if len(ref_texts_cache) > 10:
            # 删除第一个（最旧的）条目
            oldest_key = next(iter(ref_texts_cache))
            del ref_texts_cache[oldest_key]
    return ref_texts_cache[voice]

# Authentication
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if config.require_auth and credentials.credentials != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

def wav_to_base64(wav_array, sample_rate: int = 24000) -> str:
    """Convert numpy array to base64-encoded WAV string"""
    buffer = io.BytesIO()
    sf.write(buffer, wav_array, sample_rate, format='WAV')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def estimate_alignment(text: str, duration: float) -> AlignmentInfo:
    """Estimate character-level timing based on text and audio duration"""
    # Basic estimation - distribute time evenly
    char_duration = duration / len(text) if text else 0

    characters = list(text)
    start_times = []
    end_times = []

    current_time = 0.0
    for char in characters:
        # Adjust duration for punctuation (shorter) and spaces (minimal)
        if char == ' ':
            char_time = char_duration * 0.1  # Spaces take 10% of normal time
        elif char in '.,;:!?':
            char_time = char_duration * 0.5  # Punctuation takes 50% of normal time
        else:
            char_time = char_duration

        start_times.append(current_time)
        end_times.append(current_time + char_time)
        current_time += char_time

    return AlignmentInfo(
        characters=characters,
        characterStartTimesSeconds=start_times,
        characterEndTimesSeconds=end_times
    )

# Voice mapping from OpenAI voices to local samples
VOICE_MAPPING = {
    "alloy": "dave",
    "echo": "dave",
    "fable": "dave",
    "onyx": "jo",
    "nova": "jo",
    "shimmer": "jo",
}

# Initialize FastAPI app
app = FastAPI(
    title="NeuTTS Air API",
    description="OpenAI-compatible Text-to-Speech API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("NeuTTS Air API Server - Performance Optimized")
    logger.info("=" * 60)
    logger.info(f"Server will run on {config.host}:{config.port}")
    logger.info(f"Authentication required: {config.require_auth}")

    # Show dependency status
    logger.info("\n🔧 Dependencies Status:")
    logger.info(f"  ✓ llama-cpp-python: {'Available' if GGUI_AVAILABLE else 'Not Available'}")
    logger.info(f"  ✓ onnxruntime: {'Available' if ONNX_AVAILABLE else 'Not Available (Note: ONNX decoder only supports decoding)'}")
    logger.info(f"  ✓ CUDA: {'Available' if CUDA_AVAILABLE else 'Not Available'}")

    # Show model configuration
    logger.info("\n📦 Model Configuration:")
    logger.info(f"  Backbone: {config.backbone_repo}")
    logger.info(f"  Backbone Device: {config.backbone_device}")
    logger.info(f"  Codec: {config.codec_repo}")
    logger.info(f"  Codec Device: {config.codec_device}")

    # Debug: Show environment variables
    logger.info("\n🔍 Environment Variables Debug:")
    logger.info(f"  CODEC_REPO from env: '{os.getenv('CODEC_REPO')}'")
    logger.info(f"  Final config.codec_repo: '{config.codec_repo}'")

    # Preload model
    logger.info("\n⏳ Preloading TTS engine...")
    engine = get_tts_engine()

    # Preload and cache all voice reference encodings
    logger.info("\n🎵 Preloading voice references...")
    available_voices = []
    for voice in ["dave", "jo"]:
        try:
            get_cached_ref_codes(voice)
            await get_cached_ref_text(voice)
            available_voices.append(voice)
            logger.info(f"  ✓ Preloaded voice: {voice}")
        except Exception as e:
            logger.warning(f"  ✗ Failed to preload voice {voice}: {e}")

    logger.info("\n✅ Server ready!")
    logger.info(f"Available voices: {', '.join(available_voices)}")
    logger.info("\n📌 Usage Examples:")
    logger.info("  curl -X POST http://localhost:8000/v1/audio/speech \\")
    logger.info("    -H 'Content-Type: application/json' \\")
    logger.info("    -d '{\"model\":\"tts-1\",\"input\":\"Hello\",\"voice\":\"alloy\"}'")
    logger.info("=" * 60)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/v1/voices")
async def list_voices(credentials: HTTPAuthorizationCredentials = Depends(verify_api_key if config.require_auth else None)):
    """List available voices"""
    # Return OpenAI-compatible voice list
    voices = [
        {"voice": "alloy", "language": "en"},
        {"voice": "echo", "language": "en"},
        {"voice": "fable", "language": "en"},
        {"voice": "onyx", "language": "en"},
        {"voice": "nova", "language": "en"},
        {"voice": "shimmer", "language": "en"},
    ]
    return {"voices": voices}

@app.post("/v1/audio/speech")
async def create_speech(
    request: TTSRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key if config.require_auth else None)
):
    """
    Generate audio from text using preset voices
    """
    try:
        logger.info(f"Processing TTS request: {request.input[:100]}...")

        # Get TTS engine
        engine = get_tts_engine()

        # Map OpenAI voice to local sample
        if request.voice not in VOICE_MAPPING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Voice '{request.voice}' not supported. Use: {', '.join(VOICE_MAPPING.keys())}"
            )

        local_voice = VOICE_MAPPING[request.voice]

        # Check if reference files exist
        ref_text_path = f"samples/{local_voice}.txt"
        ref_audio_path = f"samples/{local_voice}.wav"
        if not os.path.exists(ref_text_path) or not os.path.exists(ref_audio_path):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Reference files for voice '{request.voice}' not found"
            )

        # Use cached reference data
        ref_codes = get_cached_ref_codes(local_voice)
        ref_text = await get_cached_ref_text(local_voice)

        # Generate audio
        wav = engine.infer(request.input, ref_codes, ref_text)

        # Handle different response formats
        if request.response_format == "json":
            # Return JSON response with base64 audio and alignment
            audio_b64 = wav_to_base64(wav)
            duration = len(wav) / 24000  # Calculate duration from sample rate
            alignment = estimate_alignment(request.input, duration)

            return TTSJSONResponse(audio_base64=audio_b64, alignment=alignment)
        else:
            # Return audio stream (default WAV format)
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, wav, 24000, format='WAV')
            audio_buffer.seek(0)

            return StreamingResponse(
                io.BytesIO(audio_buffer.read()),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                }
            )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error processing TTS request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating speech: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)