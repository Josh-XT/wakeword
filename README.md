# 🎤 WakeWord

**Open source custom wake word training and detection.**

Train your own wake word models in minutes and deploy them anywhere - from cloud servers to ESP32 microcontrollers.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-73%20passed-brightgreen.svg)](tests/)

---

## 🚀 Why WakeWord?

Custom wake words shouldn't cost money or require uploading voice data to third parties. WakeWord solves this by:

- **Training models on-the-fly** using synthetic TTS data (no voice recording needed!)
- **Generating compact models** (~1.7MB PyTorch, ~460KB for GRU variant)
- **Providing a simple REST API** for model management and inference
- **Supporting multiple export formats** (PyTorch, ONNX, TFLite)
- **Including ready-to-use examples** for Python, Android, Dart/Flutter, Web/TypeScript, and ESP32

## 📊 Performance Benchmarks (RTX 4090)

| Metric | Value |
|--------|-------|
| **Training Time** (100 samples, 30 epochs) | ~1 minute |
| **Training Time** (500 samples, 50 epochs) | ~10-15 minutes |
| **Single Inference Latency** | ~2.9ms |
| **Batch Throughput** (batch=32) | **122,609 samples/sec** |
| **GPU Memory Usage** | ~141 MB peak |
| **Model Accuracy** | 95-100% (depending on wake word) |
| **Wake Word Detection** | 99.99% confidence (true positive) |
| **False Rejection** | 7.5% confidence (non-wake-word audio) |

## 📦 Model Sizes

| Format | CNN Model | GRU Model |
|--------|-----------|-----------|
| **PyTorch (.pt)** | ~1.7 MB | ~460 KB |
| **ONNX (.onnx)** | ~1.2 MB | ~450 KB |
| **TFLite (.tflite)** | ~1.2 MB | ~450 KB |
| **Parameters** | 425,156 | 115,329 |

## 📋 Features

| Feature | Description |
|---------|-------------|
| 🔧 **Automatic Training** | Request any wake word, get a trained model in ~15 minutes |
| 🗣️ **Multi-TTS Generation** | Uses gTTS, Edge TTS, and optionally Chatterbox for diverse samples |
| 🎛️ **Data Augmentation** | Pitch, speed, noise, and reverb variations for robust models |
| 📦 **Multiple Export Formats** | PyTorch (.pt), ONNX (.onnx), TensorFlow Lite (.tflite) |
| 🔌 **REST API** | Simple HTTP API for training, status checks, and inference |
| 🐳 **Docker Ready** | CPU and CUDA Docker images included |
| 📱 **Cross-Platform** | Examples for Python, Android, iOS/Flutter, Web/TypeScript, and ESP32 |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     WakeWord Server                         │
├─────────────────────────────────────────────────────────────┤
│  POST /train         → Start training for a wake word       │
│  GET  /jobs/{id}     → Check training status                │
│  GET  /models/{word} → Download trained model               │
│  POST /predict/{word}→ Run inference on audio               │
├─────────────────────────────────────────────────────────────┤
│                    Training Pipeline                        │
│  1. Generate TTS samples (gTTS, Edge TTS, Chatterbox)       │
│  2. Augment data (pitch, speed, noise, reverb)              │
│  3. Generate negative samples (noise, similar words)        │
│  4. Train CNN model on MFCC features                        │
│  5. Export to PyTorch, ONNX, TFLite                         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌──────────┐         ┌──────────┐
   │  ESP32  │          │ Android  │         │  Python  │
   │ TFLite  │          │  ONNX    │         │  ONNX    │
   └─────────┘          └──────────┘         └──────────┘
```

## 🚦 Quick Start

### Using Docker (Recommended)

```bash
# CPU version
docker-compose up -d

# GPU version (requires NVIDIA Docker)
docker-compose -f docker-compose.cuda.yml up -d
```

### Manual Installation

```bash
# Clone repository
git clone https://github.com/Josh-XT/wakeword.git
cd wakeword

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Copy environment config
cp .env.example .env

# Start server
python -m wakeword.app
```

The server will be available at `http://localhost:8000`

## 📡 API Usage

### Train a Wake Word Model

```bash
# Request training
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"word": "jarvis", "sample_count": 500, "epochs": 50}'
```

Response:
```json
{
  "job_id": "abc123...",
  "word": "jarvis",
  "status": "pending",
  "message": "Training started for 'jarvis'. This typically takes 10-20 minutes.",
  "estimated_minutes": 15,
  "check_status_url": "/jobs/abc123..."
}
```

### Check Training Status

```bash
curl http://localhost:8000/jobs/{job_id}
```

Response:
```json
{
  "job_id": "abc123...",
  "word": "jarvis",
  "status": "training",
  "progress": 65.5,
  "current_stage": "Training model",
  "metrics": {
    "train_loss": 0.12,
    "val_loss": 0.15,
    "train_acc": 0.95,
    "val_acc": 0.93
  }
}
```

### Download Model

```bash
# PyTorch format
curl -o jarvis.pt "http://localhost:8000/models/jarvis?format=pytorch"

# ONNX format (recommended for edge devices)
curl -o jarvis.onnx "http://localhost:8000/models/jarvis?format=onnx"

# TensorFlow Lite format (for ESP32/microcontrollers)
curl -o jarvis.tflite "http://localhost:8000/models/jarvis?format=tflite"
```

### Run Inference

```bash
# Base64 encode your audio file
AUDIO_B64=$(base64 -w0 audio.wav)

# Predict
curl -X POST http://localhost:8000/predict/jarvis \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\": \"$AUDIO_B64\"}"
```

Response (wake word detected):
```json
{
  "detected": true,
  "confidence": 0.9999983,
  "word": "jarvis"
}
```

Response (non-wake-word audio):
```json
{
  "detected": false,
  "confidence": 0.0748,
  "word": "jarvis"
}
```

## 🐍 Python Client

```python
from examples.python_client import WakeWordClient, LocalWakeWordDetector

# Server-based detection
client = WakeWordClient("http://localhost:8000")
client.ensure_model("jarvis")  # Train if needed

detector = client.get_detector("jarvis")
detector.start_listening(
    on_wake_word=lambda: print("Wake word detected!")
)

# Local detection (offline)
local_detector = LocalWakeWordDetector(
    model_path="jarvis.onnx",
    word="jarvis"
)
local_detector.start_listening(
    on_wake_word=lambda: print("Detected offline!")
)
```

## 📱 Android (Kotlin)

```kotlin
val detector = WakeWordDetector(
    context,
    WakeWordConfig(
        word = "jarvis",
        serverUrl = "http://your-server:8000",
        threshold = 0.5f
    )
)

lifecycleScope.launch {
    detector.initialize()
    detector.startListening { confidence ->
        Log.d("WakeWord", "Detected with confidence: $confidence")
    }
}
```

## 🎯 ESP32

See [examples/esp32](examples/esp32) for complete Arduino code.

1. Download your TFLite model from the server
2. Convert to C array: `xxd -i model.tflite > model_data.h`
3. Flash the sketch to your ESP32

## 💡 Flutter/Dart

```dart
final detector = ServerWakeWordDetector(
  config: WakeWordConfig(
    word: 'jarvis',
    serverUrl: 'http://localhost:8000',
  ),
);

await detector.initialize();
await detector.startListening(
  onWakeWord: () => print('Wake word detected!'),
);
```

## 🌐 TypeScript/JavaScript (Browser)

```typescript
import { WakeWordDetector } from '@wakeword/client';

const detector = new WakeWordDetector({
  word: 'jarvis',
  serverUrl: 'http://localhost:8000',
  threshold: 0.5,
  onWakeWord: (confidence) => {
    console.log(`Detected! Confidence: ${(confidence * 100).toFixed(1)}%`);
  },
});

// Start listening (requests microphone permission)
await detector.start();

// Stop when done
detector.stop();
```

See [examples/typescript](examples/typescript) for a complete demo with audio visualization.

## ⚙️ Configuration

Environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `MODELS_DIR` | `./models` | Where to store trained models |
| `SAMPLES_DIR` | `./samples` | Where to store generated samples |
| `TTS_GTTS_ENABLED` | `true` | Enable Google TTS |
| `TTS_EDGE_ENABLED` | `true` | Enable Microsoft Edge TTS |
| `TTS_CHATTERBOX_ENABLED` | `false` | Enable Chatterbox (requires GPU) |
| `DEFAULT_SAMPLE_COUNT` | `500` | Default samples to generate |
| `DEFAULT_EPOCHS` | `50` | Default training epochs |

## 🔬 How It Works

### Sample Generation
1. **Text-to-Speech**: Multiple TTS engines generate the wake word with different voices and accents
2. **Augmentation**: Each sample is augmented with pitch shifts, speed changes, background noise, and reverb
3. **Negative Samples**: Noise and similar-sounding words are generated for robust discrimination

### Model Architecture
- **Input**: MFCC features (40 coefficients, ~101 frames for 1.5s audio at 16kHz)
- **CNN Network**: 3 conv layers + batch norm + 2 FC layers (425K params, ~1.7MB)
- **GRU Network**: 2-layer GRU + FC layer (115K params, ~460KB)
- **Output**: Single sigmoid for wake word probability

### Training
- Binary classification: wake word vs. not-wake-word
- Cross-entropy loss with Adam optimizer
- Typically 50-500 positive samples + equal negative samples
- Converges in 30-50 epochs (~1-15 minutes depending on sample count)

## 📊 Performance

Benchmarked on NVIDIA RTX 4090:

| Metric | Value |
|--------|-------|
| **Training** (100 samples, 30 epochs) | ~1 minute |
| **Single Inference** | 2.9ms |
| **Batch Throughput** | 122K samples/sec |
| **Model Accuracy** | 95-100% |
| **False Positive Rate** | <2% |
| **GPU Memory** | ~141 MB peak |

## 🛣️ Roadmap

- [ ] Real voice recording for improved accuracy
- [ ] Few-shot learning for faster training
- [ ] Web UI for model management
- [ ] Pre-trained models repository
- [ ] Streaming detection API
- [ ] iOS Swift example
- [ ] Raspberry Pi optimized builds

## 🧪 Testing

Run the full test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=wakeword --cov-report=html

# Run only fast tests (skip slow integration tests)
python -m pytest tests/ -v -m "not slow"

# Run benchmarks
python -m pytest tests/test_model.py tests/test_integration.py -v -s
```

**Test Results**: 73 tests passing, covering:
- Configuration and settings
- TTS sample generation and augmentation
- Model architectures (CNN, GRU)
- Training and export pipelines
- Job management and persistence
- REST API endpoints
- GPU performance benchmarks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Edge TTS](https://github.com/rany2/edge-tts) for high-quality Microsoft TTS
- [gTTS](https://github.com/pndurette/gTTS) for Google TTS
- [Chatterbox](https://github.com/resemble-ai/chatterbox) for voice cloning TTS
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers) for edge deployment

---

**Made with ❤️ for the open source community**

*Wake words should be free. Let's make voice assistants accessible to everyone.*