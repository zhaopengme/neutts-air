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

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from neuttsair.neutts import NeuTTSAir
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        # TTS Model Configuration
        self.backbone_repo = os.getenv("BACKBONE_REPO")  # None means use default
        self.backbone_device = os.getenv("BACKBONE_DEVICE", "cpu")
        self.codec_repo = os.getenv("CODEC_REPO")  # None means use default
        self.codec_device = os.getenv("CODEC_DEVICE", "cpu")

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
    return tts_engine

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
    logger.info("=" * 50)
    logger.info("Starting NeuTTS Air API server...")
    logger.info(f"Server will run on {config.host}:{config.port}")
    logger.info(f"Authentication required: {config.require_auth}")
    if config.require_auth:
        logger.info("API key authentication is enabled")
    else:
        logger.info("API key authentication is disabled (open access)")
    logger.info("=" * 50)
    # Preload TTS engine
    get_tts_engine()

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
        ref_text_path = f"samples/{local_voice}.txt"
        ref_audio_path = f"samples/{local_voice}.wav"

        # Check if reference files exist
        if not os.path.exists(ref_text_path) or not os.path.exists(ref_audio_path):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Reference files for voice '{request.voice}' not found"
            )

        # Read reference text
        with open(ref_text_path, "r") as f:
            ref_text = f.read().strip()

        # Encode reference audio
        ref_codes = engine.encode_reference(ref_audio_path)

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