# NeuTTS Air ☁️

HuggingFace 🤗: [Model](https://huggingface.co/neuphonic/neutts-air), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-air-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-air-q4-gguf) [Spaces](https://huggingface.co/spaces/neuphonic/neutts-air)

[Demo Video](https://github.com/user-attachments/assets/020547bc-9e3e-440f-b016-ae61ca645184)

_Created by [Neuphonic](http://neuphonic.com/) - building faster, smaller, on-device voice AI_

State-of-the-art Voice AI has been locked behind web APIs for too long. NeuTTS Air is the world’s first super-realistic, on-device, TTS speech language model with instant voice cloning. Built off a 0.5B LLM backbone, NeuTTS Air brings natural-sounding speech, real-time performance, built-in security and speaker cloning to your local device - unlocking a new category of embedded voice agents, assistants, toys, and compliance-safe apps.

## Key Features

- 🗣Best-in-class realism for its size - produces natural, ultra-realistic voices that sound human
- 📱Optimised for on-device deployment - provided in GGML format, ready to run on phones, laptops, or even Raspberry Pis
- 👫Instant voice cloning - create your own speaker with as little as 3 seconds of audio
- 🚄Simple LM + codec architecture built off a 0.5B backbone - the sweet spot between speed, size, and quality for real-world applications

> [!CAUTION]
> Websites like neutts.com are popping up and they're not affliated with Neuphonic, our github or this repo.
>
> We are on neuphonic.com only. Please be careful out there! 🙏

## Model Details

NeuTTS Air is built off Qwen 0.5B - a lightweight yet capable language model optimised for text understanding and generation - as well as a powerful combination of technologies designed for efficiency and quality:

- **Supported Languages**: English
- **Audio Codec**: [NeuCodec](https://huggingface.co/neuphonic/neucodec) - our 50hz neural audio codec that achieves exceptional audio quality at low bitrates using a single codebook
- **Context Window**: 2048 tokens, enough for processing ~30 seconds of audio (including prompt duration)
- **Format**: Available in GGML format for efficient on-device inference
- **Responsibility**: Watermarked outputs
- **Inference Speed**: Real-time generation on mid-range devices
- **Power Consumption**: Optimised for mobile and embedded devices

## Get Started

> [!NOTE]
> We have added a [streaming example](examples/basic_streaming_example.py) using the `llama-cpp-python` library as well as a [finetuning script](examples/finetune.py). For finetuning, please refer to the [finetune guide](TRAINING.md) for more details.

1. **Clone Git Repo**

   ```bash
   git clone https://github.com/neuphonic/neutts-air.git
   cd neutts-air
   ```

2. **Install `espeak` (required dependency)**

   Please refer to the following link for instructions on how to install `espeak`:

   https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md

   ```bash
   # Mac OS
   brew install espeak

   # Ubuntu/Debian
   sudo apt install espeak

   # Windows install
   # via chocolatey (https://community.chocolatey.org/packages?page=1&prerelease=False&moderatorQueue=False&tags=espeak)
   choco install espeak-ng
   # via wingit
   winget install -e --id eSpeak-NG.eSpeak-NG
   # via msi (need to add to path or folow the "Windows users who installed via msi" below)
   # find the msi at https://github.com/espeak-ng/espeak-ng/releases
   ```

   Mac users may need to put the following lines at the top of the neutts.py file.

   ```python
   from phonemizer.backend.espeak.wrapper import EspeakWrapper
   _ESPEAK_LIBRARY = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib'  #use the Path to the library.
   EspeakWrapper.set_library(_ESPEAK_LIBRARY)
   ```

   Windows users who installed via msi / do not have their install on path need to run the following (see https://github.com/bootphon/phonemizer/issues/163)
   ```pwsh
   $env:PHONEMIZER_ESPEAK_LIBRARY = "c:\Program Files\eSpeak NG\libespeak-ng.dll"
   $env:PHONEMIZER_ESPEAK_PATH = "c:\Program Files\eSpeak NG"
   setx PHONEMIZER_ESPEAK_LIBRARY "c:\Program Files\eSpeak NG\libespeak-ng.dll"
   setx PHONEMIZER_ESPEAK_PATH "c:\Program Files\eSpeak NG"
   ```

3. **Install Python dependencies**

   The requirements file includes the dependencies needed to run the model with PyTorch.
   When using an ONNX decoder or a GGML model, some dependencies (such as PyTorch) are no longer required.

   The inference is compatible and tested on `python>=3.11`.

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install Llama-cpp-python to use the `GGUF` models.**

   ```bash
   pip install llama-cpp-python
   ```

   To run llama-cpp with GPU suport (CUDA, MPS) support please refer to:
   https://pypi.org/project/llama-cpp-python/

5. **(Optional) Install onnxruntime to use the `.onnx` decoder.**
   If you want to run the onnxdecoder
   ```bash
   pip install onnxruntime
   ```

## Running the Model

Run the basic example script to synthesize speech:

```bash
python -m examples.basic_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_audio samples/dave.wav \
  --ref_text samples/dave.txt
```

To specify a particular model repo for the backbone or codec, add the `--backbone` argument. Available backbones are listed in [NeuTTS-Air huggingface collection](https://huggingface.co/collections/neuphonic/neutts-air-68cc14b7033b4c56197ef350).

Several examples are available, including a Jupyter notebook in the `examples` folder.

### One-Code Block Usage

```python
from neuttsair.neutts import NeuTTSAir
import soundfile as sf

tts = NeuTTSAir(
   backbone_repo="neuphonic/neutts-air", # or 'neutts-air-q4-gguf' with llama-cpp-python installed
   backbone_device="cpu",
   codec_repo="neuphonic/neucodec",
   codec_device="cpu"
)
input_text = "My name is Dave, and um, I'm from London."

ref_text = "samples/dave.txt"
ref_audio_path = "samples/dave.wav"

ref_text = open(ref_text, "r").read().strip()
ref_codes = tts.encode_reference(ref_audio_path)

wav = tts.infer(input_text, ref_codes, ref_text)
sf.write("test.wav", wav, 24000)
```

### Streaming

Speech can also be synthesised in _streaming mode_, where audio is generated in chunks and plays as generated. Note that this requires pyaudio to be installed. To do this, run: 

```bash
python -m examples.basic_streaming_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_codes samples/dave.pt \
  --ref_text samples/dave.txt
```

Again, a particular model repo can be specified with the `--backbone` argument - note that for streaming the model must be in GGUF format.

## Preparing References for Cloning

NeuTTS Air requires two inputs:

1. A reference audio sample (`.wav` file)
2. A text string

The model then synthesises the text as speech in the style of the reference audio. This is what enables NeuTTS Air’s instant voice cloning capability.

### Example Reference Files

You can find some ready-to-use samples in the `examples` folder:

- `samples/dave.wav`
- `samples/jo.wav`

### Guidelines for Best Results

For optimal performance, reference audio samples should be:

1. **Mono channel**
2. **16-44 kHz sample rate**
3. **3–15 seconds in length**
4. **Saved as a `.wav` file**
5. **Clean** — minimal to no background noise
6. **Natural, continuous speech** — like a monologue or conversation, with few pauses, so the model can capture tone effectively

## Guidelines for minimizing Latency

For optimal performance on-device:

1. Use the GGUF model backbones
2. Pre-encode references
3. Use the [onnx codec decoder](https://huggingface.co/neuphonic/neucodec-onnx-decoder)

Take a look at this example [examples README](examples/README.md###minimal-latency-example) to get started.

## Responsibility

Every audio file generated by NeuTTS Air includes [Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth).

## OpenAI-Compatible Server

NeuTTS Air includes an OpenAI-compatible API server that can be used as a drop-in replacement for OpenAI's TTS service.

### Installation

1. Install server dependencies:
   ```bash
   pip install -r requirements-server.txt
   ```

2. (Optional) Create a `.env` file for configuration:
   ```bash
   cp .env.example .env
   # Edit .env to set PORT and API_KEY
   ```

### Running the Server

Start the server with:
```bash
python server.py
```

The server will start on `http://localhost:8000` by default.

### API Endpoints

#### Create Speech
```http
POST /v1/audio/speech
Authorization: Bearer YOUR_API_KEY (if configured)
Content-Type: application/json

{
  "model": "tts-1",
  "input": "Hello, world!",
  "voice": "alloy",
  "response_format": "wav"
}
```

Available voices (OpenAI-compatible):
- `alloy` - Male voice (maps to samples/dave.wav)
- `echo` - Male voice (maps to samples/dave.wav)
- `fable` - Male voice (maps to samples/dave.wav)
- `onyx` - Female voice (maps to samples/jo.wav)
- `nova` - Female voice (maps to samples/jo.wav)
- `shimmer` - Female voice (maps to samples/jo.wav)

**Note**: Supports `wav` format (audio stream) and `json` format (base64 audio with alignment data).

#### List Voices
```http
GET /v1/voices
Authorization: Bearer YOUR_API_KEY (if configured)
```

#### Health Check
```http
GET /health
```

### Configuration

The server can be configured via environment variables or `.env` file:

#### Server Configuration
- `PORT` - Server port (default: 8000)
- `HOST` - Host to bind the server to (default: 0.0.0.0)
- `API_KEY` - Optional API key for authentication. If not set, no authentication is required.
- `LOG_LEVEL` - Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)

#### TTS Model Configuration (Optional)
If not specified, the default models will be used:
- `BACKBONE_REPO` - HuggingFace model repo for the TTS backbone (default: neuphonic/neutts-air)
  - Can also use GGUF models like 'neutts-air-q4-gguf' (requires llama-cpp-python)
- `BACKBONE_DEVICE` - Device for backbone model: cpu or cuda (default: cpu)
- `CODEC_REPO` - HuggingFace model repo for the audio codec (default: neuphonic/neucodec)
  - Options: neuphonic/neucodec, neuphonic/distill-neucodec, neuphonic/neucodec-onnx-decoder
- `CODEC_DEVICE` - Device for codec model: cpu or cuda (default: cpu, not applicable for onnx decoder)

#### Example .env file
```bash
# Basic configuration
PORT=8080
API_KEY=sk-your-secret-key

# Use GGUF model for faster inference
BACKBONE_REPO=neutts-air-q4-gguf
BACKBONE_DEVICE=cpu

# Use ONNX decoder for CPU optimization
CODEC_REPO=neuphonic/neucodec-onnx-decoder
CODEC_DEVICE=cpu
```

### Example Usage with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",  # Required if server has API_KEY set
    base_url="http://localhost:8000/v1"
)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello, this is a test of NeuTTS Air!"
)

response.stream_to_file("output.wav")
```

#### JSON Response Example

For applications that need alignment data or base64 audio:

```http
POST /v1/audio/speech
Content-Type: application/json

{
  "model": "tts-1",
  "input": "Hello world!",
  "voice": "alloy",
  "response_format": "json"
}
```

Response:
```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAA...",
  "alignment": {
    "characters": ["H", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d", "!"],
    "characterStartTimesSeconds": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525],
    "characterEndTimesSeconds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.55]
  }
}
```

#### JavaScript Example with JSON Format

```javascript
const response = await fetch('http://localhost:8000/v1/audio/speech', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'tts-1',
    input: 'Hello world!',
    voice: 'alloy',
    response_format: 'json'
  })
});

const data = await response.json();
const { audio_base64, alignment } = data;

// Convert base64 to audio blob
const audioBlob = base64ToBlob(audio_base64, 'audio/wav');
const audioUrl = URL.createObjectURL(audioBlob);

// Play the audio
const audio = new Audio(audioUrl);
audio.play();

// Use alignment data for visualization
alignment.characters.forEach((char, i) => {
  const start = alignment.characterStartTimesSeconds[i];
  const end = alignment.characterEndTimesSeconds[i];
  console.log(`"${char}" at ${start.toFixed(2)}s - ${end.toFixed(2)}s`);
});

// Helper function to convert base64 to blob
function base64ToBlob(base64, mimeType) {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: mimeType });
}
```

## Disclaimer

Don't use this model to do bad things… please.

## Developer Requirements

To run the pre commit hooks to contribute to this project run:

```bash
pip install pre-commit
```

Then:

```bash
pre-commit install
```
