# WakeWord TypeScript/JavaScript Client

Browser-based wake word detection for web applications. Capture audio from the user's microphone and detect custom wake words in real-time.

## Features

- 🎤 **Real-time microphone capture** with Web Audio API
- 🔊 **Audio visualization** support
- 📡 **Server-side inference** via REST API
- ⚛️ **React hook** included (optional)
- 📦 **Zero dependencies** for the core client
- 🔒 **TypeScript** with full type definitions

## Quick Start

### Using the Demo Page

1. Start the WakeWord server:
   ```bash
   cd /path/to/wakeword
   python -m wakeword.app
   ```

2. Open `index.html` in your browser (or serve it with any HTTP server)

3. Configure the wake word and click "Start Listening"

### Using in Your Project

#### Installation (npm)

```bash
npm install @wakeword/client
```

Or copy `wakeword-client.ts` directly into your project.

#### Basic Usage

```typescript
import { WakeWordDetector } from '@wakeword/client';

// Create detector
const detector = new WakeWordDetector({
  word: 'jarvis',
  serverUrl: 'http://localhost:8000',
  threshold: 0.5,
  onWakeWord: (confidence) => {
    console.log(`Wake word detected! Confidence: ${(confidence * 100).toFixed(1)}%`);
    // Trigger your voice assistant UI here
  },
  onStatusChange: (status) => {
    console.log('Status:', status); // 'idle', 'listening', 'processing', 'error'
  },
  onError: (error) => {
    console.error('Error:', error);
  },
});

// Start listening
await detector.start();

// Stop when done
detector.stop();
```

#### Using the API Client Directly

```typescript
import { WakeWordClient, audioToBase64 } from '@wakeword/client';

const client = new WakeWordClient('http://localhost:8000');

// Check server health
const health = await client.health();
console.log(health); // { status: 'healthy' }

// List available models
const models = await client.listModels();
console.log(models);

// Train a new model
const job = await client.train({
  word: 'computer',
  sample_count: 500,
  epochs: 50,
});

// Wait for training to complete
const result = await client.waitForTraining(job.job_id, {
  onProgress: (status) => {
    console.log(`Progress: ${status.progress}% - ${status.current_stage}`);
  },
});

// Run inference
const prediction = await client.predict('jarvis', audioBase64Data);
console.log(prediction); // { detected: true, confidence: 0.95, word: 'jarvis' }
```

#### React Hook

```tsx
import { useState, useEffect, useRef, useCallback } from 'react';
import { WakeWordDetector, WakeWordConfig, DetectorStatus } from '@wakeword/client';

function useWakeWord(config: WakeWordConfig) {
  const [status, setStatus] = useState<DetectorStatus>('idle');
  const [lastDetection, setLastDetection] = useState<{ timestamp: Date; confidence: number } | null>(null);
  const detectorRef = useRef<WakeWordDetector | null>(null);

  useEffect(() => {
    detectorRef.current = new WakeWordDetector({
      ...config,
      onStatusChange: setStatus,
      onWakeWord: (confidence) => {
        setLastDetection({ timestamp: new Date(), confidence });
        config.onWakeWord?.(confidence);
      },
    });

    return () => {
      detectorRef.current?.stop();
    };
  }, [config.word, config.serverUrl, config.threshold]);

  const start = useCallback(async () => {
    await detectorRef.current?.start();
  }, []);

  const stop = useCallback(() => {
    detectorRef.current?.stop();
  }, []);

  return { status, lastDetection, start, stop };
}

// Usage in component
function VoiceAssistant() {
  const { status, start, stop, lastDetection } = useWakeWord({
    word: 'jarvis',
    onWakeWord: (confidence) => {
      console.log('Wake word detected!', confidence);
    },
  });

  return (
    <div>
      <p>Status: {status}</p>
      {lastDetection && (
        <p>Last detected: {lastDetection.timestamp.toLocaleTimeString()} 
           ({(lastDetection.confidence * 100).toFixed(1)}%)</p>
      )}
      <button onClick={start} disabled={status === 'listening'}>
        Start Listening
      </button>
      <button onClick={stop} disabled={status === 'idle'}>
        Stop
      </button>
    </div>
  );
}
```

## API Reference

### `WakeWordDetector`

The main class for real-time wake word detection.

#### Constructor Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `word` | `string` | required | The wake word to detect |
| `serverUrl` | `string` | `'http://localhost:8000'` | WakeWord server URL |
| `threshold` | `number` | `0.5` | Detection confidence threshold (0-1) |
| `sampleRate` | `number` | `16000` | Audio sample rate in Hz |
| `chunkDuration` | `number` | `1.5` | Audio chunk duration in seconds |
| `onWakeWord` | `(confidence: number) => void` | - | Callback when wake word is detected |
| `onError` | `(error: Error) => void` | - | Callback for errors |
| `onStatusChange` | `(status: DetectorStatus) => void` | - | Callback for status changes |

#### Methods

- `start(): Promise<void>` - Start listening for the wake word
- `stop(): void` - Stop listening
- `getStatus(): DetectorStatus` - Get current status
- `getClient(): WakeWordClient` - Get the underlying API client

### `WakeWordClient`

HTTP client for the WakeWord server API.

#### Methods

- `health(): Promise<{ status: string }>` - Check server health
- `listModels(): Promise<{ models: ModelInfo[] }>` - List available models
- `modelExists(word: string): Promise<boolean>` - Check if model exists
- `getModelConfig(word: string): Promise<ModelConfig>` - Get model configuration
- `downloadModel(word: string, format?: string): Promise<Blob>` - Download model file
- `train(request: TrainRequest): Promise<TrainResponse>` - Start training
- `getJobStatus(jobId: string): Promise<JobStatus>` - Get training job status
- `listJobs(): Promise<JobStatus[]>` - List all training jobs
- `predict(word: string, audioBase64: string): Promise<PredictResponse>` - Run inference
- `waitForTraining(jobId: string, options?): Promise<JobStatus>` - Wait for training

### Utility Functions

- `audioBufferToWav(audioBuffer: AudioBuffer): Uint8Array` - Convert AudioBuffer to WAV
- `audioToBase64(audioBuffer: AudioBuffer): string` - Convert AudioBuffer to base64 WAV
- `resampleAudio(audioBuffer: AudioBuffer, targetSampleRate: number): Promise<AudioBuffer>` - Resample audio

## Browser Requirements

- Modern browser with Web Audio API support
- Microphone access permission
- HTTPS (required for microphone access in production)

## CORS Configuration

If running the WakeWord server on a different origin, ensure CORS is configured:

```python
# The WakeWord server already includes CORS middleware
# It allows all origins by default for development
```

For production, configure appropriate CORS settings.

## Building from Source

```bash
cd examples/typescript
npm install
npm run build
```

This creates:
- `dist/wakeword-client.js` - CommonJS bundle
- `dist/wakeword-client.mjs` - ES Module bundle
- `dist/wakeword-client.d.ts` - TypeScript declarations

## License

MIT
