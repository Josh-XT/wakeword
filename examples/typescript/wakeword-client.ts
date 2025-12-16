/**
 * WakeWord TypeScript Client
 *
 * Browser-based wake word detection client that works with the WakeWord server.
 * Captures audio from the microphone, processes it, and sends it for inference.
 *
 * @example
 * ```typescript
 * const detector = new WakeWordDetector({
 *   serverUrl: 'http://localhost:8000',
 *   word: 'jarvis',
 *   onWakeWord: (confidence) => console.log(`Detected with ${confidence}% confidence`),
 * });
 *
 * await detector.start();
 * ```
 */

// ============================================================================
// Types & Interfaces
// ============================================================================

export interface WakeWordConfig {
  /** The wake word to detect */
  word: string;
  /** Server URL (default: http://localhost:8000) */
  serverUrl?: string;
  /** Detection threshold 0-1 (default: 0.5) */
  threshold?: number;
  /** Sample rate in Hz (default: 16000) */
  sampleRate?: number;
  /** Audio chunk duration in seconds (default: 1.5) */
  chunkDuration?: number;
  /** Callback when wake word is detected */
  onWakeWord?: (confidence: number) => void;
  /** Callback for errors */
  onError?: (error: Error) => void;
  /** Callback for status changes */
  onStatusChange?: (status: DetectorStatus) => void;
}

export interface PredictResponse {
  detected: boolean;
  confidence: number;
  word: string;
}

export interface TrainRequest {
  word: string;
  sample_count?: number;
  epochs?: number;
  model_type?: 'cnn' | 'gru';
  export_formats?: ('pytorch' | 'onnx' | 'tflite')[];
}

export interface TrainResponse {
  job_id: string;
  word: string;
  status: string;
  message: string;
  estimated_minutes: number;
  check_status_url: string;
}

export interface JobStatus {
  job_id: string;
  word: string;
  status: 'pending' | 'generating_samples' | 'training' | 'exporting' | 'completed' | 'failed';
  progress: number;
  current_stage: string;
  error_message?: string;
  metrics?: {
    train_loss?: number;
    val_loss?: number;
    train_acc?: number;
    val_acc?: number;
    positive_samples?: number;
    negative_samples?: number;
  };
}

export interface ModelInfo {
  word: string;
  created_at: string;
  formats: string[];
  config: {
    n_mfcc: number;
    sample_rate: number;
    max_length_sec: number;
    model_type: string;
  };
}

export type DetectorStatus = 'idle' | 'initializing' | 'listening' | 'processing' | 'error';

// ============================================================================
// WakeWord API Client
// ============================================================================

/**
 * HTTP client for interacting with the WakeWord server API.
 */
export class WakeWordClient {
  private serverUrl: string;

  constructor(serverUrl: string = 'http://localhost:8000') {
    this.serverUrl = serverUrl.replace(/\/$/, '');
  }

  /**
   * Check server health
   */
  async health(): Promise<{ status: string }> {
    const response = await fetch(`${this.serverUrl}/health`);
    if (!response.ok) throw new Error(`Health check failed: ${response.statusText}`);
    return response.json();
  }

  /**
   * List all available models
   */
  async listModels(): Promise<{ models: ModelInfo[] }> {
    const response = await fetch(`${this.serverUrl}/models`);
    if (!response.ok) throw new Error(`Failed to list models: ${response.statusText}`);
    return response.json();
  }

  /**
   * Check if a model exists for a word
   */
  async modelExists(word: string): Promise<boolean> {
    try {
      const response = await fetch(`${this.serverUrl}/models/${encodeURIComponent(word)}/config`);
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Get model configuration
   */
  async getModelConfig(word: string): Promise<ModelInfo['config']> {
    const response = await fetch(`${this.serverUrl}/models/${encodeURIComponent(word)}/config`);
    if (!response.ok) throw new Error(`Model '${word}' not found`);
    return response.json();
  }

  /**
   * Download a model file
   */
  async downloadModel(
    word: string,
    format: 'pytorch' | 'onnx' | 'tflite' = 'onnx'
  ): Promise<Blob> {
    const response = await fetch(
      `${this.serverUrl}/models/${encodeURIComponent(word)}?format=${format}`
    );
    if (!response.ok) throw new Error(`Failed to download model: ${response.statusText}`);
    return response.blob();
  }

  /**
   * Request training of a new wake word model
   */
  async train(request: TrainRequest): Promise<TrainResponse> {
    const response = await fetch(`${this.serverUrl}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || 'Training request failed');
    }
    return response.json();
  }

  /**
   * Get the status of a training job
   */
  async getJobStatus(jobId: string): Promise<JobStatus> {
    const response = await fetch(`${this.serverUrl}/jobs/${encodeURIComponent(jobId)}`);
    if (!response.ok) throw new Error(`Failed to get job status: ${response.statusText}`);
    return response.json();
  }

  /**
   * List all jobs
   */
  async listJobs(): Promise<JobStatus[]> {
    const response = await fetch(`${this.serverUrl}/jobs`);
    if (!response.ok) throw new Error(`Failed to list jobs: ${response.statusText}`);
    return response.json();
  }

  /**
   * Run inference on audio data
   */
  async predict(word: string, audioBase64: string): Promise<PredictResponse> {
    const response = await fetch(`${this.serverUrl}/predict/${encodeURIComponent(word)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio_base64: audioBase64 }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || 'Prediction failed');
    }
    return response.json();
  }

  /**
   * Wait for a training job to complete
   */
  async waitForTraining(
    jobId: string,
    options: {
      pollInterval?: number;
      timeout?: number;
      onProgress?: (status: JobStatus) => void;
    } = {}
  ): Promise<JobStatus> {
    const { pollInterval = 5000, timeout = 3600000, onProgress } = options;
    const startTime = Date.now();

    while (true) {
      const status = await this.getJobStatus(jobId);
      onProgress?.(status);

      if (status.status === 'completed') {
        return status;
      }
      if (status.status === 'failed') {
        throw new Error(status.error_message || 'Training failed');
      }
      if (Date.now() - startTime > timeout) {
        throw new Error('Training timeout exceeded');
      }

      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }
  }
}

// ============================================================================
// Audio Processing Utilities
// ============================================================================

/**
 * Convert an AudioBuffer to WAV format as a Uint8Array
 */
export function audioBufferToWav(audioBuffer: AudioBuffer): Uint8Array {
  const numChannels = 1; // Mono
  const sampleRate = audioBuffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;

  // Get audio data (use first channel or mix down)
  let audioData: Float32Array;
  if (audioBuffer.numberOfChannels === 1) {
    audioData = audioBuffer.getChannelData(0);
  } else {
    // Mix down to mono
    audioData = new Float32Array(audioBuffer.length);
    for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
      const channelData = audioBuffer.getChannelData(i);
      for (let j = 0; j < audioBuffer.length; j++) {
        audioData[j] += channelData[j] / audioBuffer.numberOfChannels;
      }
    }
  }

  // Create WAV file
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = audioData.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // WAV header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Write audio data
  let offset = 44;
  for (let i = 0; i < audioData.length; i++) {
    const sample = Math.max(-1, Math.min(1, audioData[i]));
    const value = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    view.setInt16(offset, value, true);
    offset += 2;
  }

  return new Uint8Array(buffer);
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

/**
 * Convert audio data to base64-encoded WAV
 */
export function audioToBase64(audioBuffer: AudioBuffer): string {
  const wavData = audioBufferToWav(audioBuffer);
  return btoa(String.fromCharCode(...wavData));
}

/**
 * Resample audio to target sample rate
 */
export async function resampleAudio(
  audioBuffer: AudioBuffer,
  targetSampleRate: number
): Promise<AudioBuffer> {
  if (audioBuffer.sampleRate === targetSampleRate) {
    return audioBuffer;
  }

  const offlineCtx = new OfflineAudioContext(
    1, // mono
    Math.ceil(audioBuffer.duration * targetSampleRate),
    targetSampleRate
  );

  const source = offlineCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offlineCtx.destination);
  source.start(0);

  return offlineCtx.startRendering();
}

// ============================================================================
// Wake Word Detector (Real-time Detection)
// ============================================================================

/**
 * Real-time wake word detector using browser microphone.
 *
 * @example
 * ```typescript
 * const detector = new WakeWordDetector({
 *   word: 'jarvis',
 *   onWakeWord: (confidence) => {
 *     console.log(`Wake word detected with ${(confidence * 100).toFixed(1)}% confidence`);
 *   },
 * });
 *
 * // Start listening
 * await detector.start();
 *
 * // Later, stop listening
 * detector.stop();
 * ```
 */
export class WakeWordDetector {
  private config: Required<WakeWordConfig>;
  private client: WakeWordClient;
  private status: DetectorStatus = 'idle';
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private analyser: AnalyserNode | null = null;
  private audioBuffer: Float32Array[] = [];
  private isProcessing = false;
  private detectionCooldown = false;

  constructor(config: WakeWordConfig) {
    this.config = {
      word: config.word,
      serverUrl: config.serverUrl ?? 'http://localhost:8000',
      threshold: config.threshold ?? 0.5,
      sampleRate: config.sampleRate ?? 16000,
      chunkDuration: config.chunkDuration ?? 1.5,
      onWakeWord: config.onWakeWord ?? (() => {}),
      onError: config.onError ?? ((e) => console.error('WakeWord error:', e)),
      onStatusChange: config.onStatusChange ?? (() => {}),
    };
    this.client = new WakeWordClient(this.config.serverUrl);
  }

  /**
   * Get current detector status
   */
  getStatus(): DetectorStatus {
    return this.status;
  }

  /**
   * Get the underlying API client
   */
  getClient(): WakeWordClient {
    return this.client;
  }

  /**
   * Start listening for the wake word
   */
  async start(): Promise<void> {
    if (this.status === 'listening') {
      return;
    }

    this.setStatus('initializing');

    try {
      // Check if model exists
      const exists = await this.client.modelExists(this.config.word);
      if (!exists) {
        throw new Error(
          `Model for '${this.config.word}' not found. Train it first using client.train()`
        );
      }

      // Request microphone access
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: { ideal: this.config.sampleRate },
          channelCount: { exact: 1 },
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      // Set up audio processing
      this.audioContext = new AudioContext({ sampleRate: this.config.sampleRate });
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);

      // Use ScriptProcessorNode for audio capture (deprecated but widely supported)
      // For production, consider using AudioWorklet
      const bufferSize = 4096;
      this.processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
      this.analyser = this.audioContext.createAnalyser();

      source.connect(this.analyser);
      this.analyser.connect(this.processor);
      this.processor.connect(this.audioContext.destination);

      this.processor.onaudioprocess = (event) => {
        if (this.status !== 'listening') return;

        const inputData = event.inputBuffer.getChannelData(0);
        this.audioBuffer.push(new Float32Array(inputData));

        // Check if we have enough audio
        const totalSamples = this.audioBuffer.reduce((sum, buf) => sum + buf.length, 0);
        const requiredSamples = this.config.sampleRate * this.config.chunkDuration;

        if (totalSamples >= requiredSamples && !this.isProcessing && !this.detectionCooldown) {
          this.processAudioBuffer();
        }
      };

      this.setStatus('listening');
    } catch (error) {
      this.setStatus('error');
      this.config.onError(error as Error);
      throw error;
    }
  }

  /**
   * Stop listening for the wake word
   */
  stop(): void {
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    if (this.analyser) {
      this.analyser.disconnect();
      this.analyser = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    this.audioBuffer = [];
    this.setStatus('idle');
  }

  /**
   * Process accumulated audio buffer and check for wake word
   */
  private async processAudioBuffer(): Promise<void> {
    if (this.isProcessing || this.audioBuffer.length === 0) return;

    this.isProcessing = true;
    this.setStatus('processing');

    try {
      // Concatenate audio chunks
      const totalLength = this.audioBuffer.reduce((sum, buf) => sum + buf.length, 0);
      const combinedAudio = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of this.audioBuffer) {
        combinedAudio.set(chunk, offset);
        offset += chunk.length;
      }

      // Take only the required duration
      const requiredSamples = Math.floor(this.config.sampleRate * this.config.chunkDuration);
      const audioSlice = combinedAudio.slice(0, requiredSamples);

      // Clear buffer but keep overlap for continuous detection
      const overlapSamples = Math.floor(requiredSamples * 0.5); // 50% overlap
      if (combinedAudio.length > overlapSamples) {
        this.audioBuffer = [combinedAudio.slice(-overlapSamples)];
      } else {
        this.audioBuffer = [];
      }

      // Create AudioBuffer for conversion
      const audioBuffer = this.audioContext!.createBuffer(
        1,
        audioSlice.length,
        this.config.sampleRate
      );
      audioBuffer.getChannelData(0).set(audioSlice);

      // Convert to base64 WAV
      const audioBase64 = audioToBase64(audioBuffer);

      // Send for prediction
      const result = await this.client.predict(this.config.word, audioBase64);

      if (result.detected && result.confidence >= this.config.threshold) {
        this.config.onWakeWord(result.confidence);

        // Add cooldown to prevent rapid re-detection
        this.detectionCooldown = true;
        setTimeout(() => {
          this.detectionCooldown = false;
        }, 2000); // 2 second cooldown
      }
    } catch (error) {
      this.config.onError(error as Error);
    } finally {
      this.isProcessing = false;
      if (this.status === 'processing') {
        this.setStatus('listening');
      }
    }
  }

  private setStatus(status: DetectorStatus): void {
    this.status = status;
    this.config.onStatusChange(status);
  }
}

// ============================================================================
// React Hook (Optional - for React applications)
// ============================================================================

/**
 * React hook for wake word detection.
 *
 * @example
 * ```tsx
 * function VoiceAssistant() {
 *   const { status, start, stop, lastDetection } = useWakeWord({
 *     word: 'jarvis',
 *     onWakeWord: (confidence) => {
 *       console.log('Detected!', confidence);
 *     },
 *   });
 *
 *   return (
 *     <div>
 *       <p>Status: {status}</p>
 *       <button onClick={start}>Start Listening</button>
 *       <button onClick={stop}>Stop</button>
 *     </div>
 *   );
 * }
 * ```
 */
export interface UseWakeWordResult {
  status: DetectorStatus;
  lastDetection: { timestamp: Date; confidence: number } | null;
  start: () => Promise<void>;
  stop: () => void;
  detector: WakeWordDetector;
}

// Note: This is a React hook stub. In a real React project, you'd use:
// import { useState, useEffect, useRef, useCallback } from 'react';
//
// export function useWakeWord(config: WakeWordConfig): UseWakeWordResult {
//   const [status, setStatus] = useState<DetectorStatus>('idle');
//   const [lastDetection, setLastDetection] = useState<{ timestamp: Date; confidence: number } | null>(null);
//   const detectorRef = useRef<WakeWordDetector | null>(null);
//
//   useEffect(() => {
//     detectorRef.current = new WakeWordDetector({
//       ...config,
//       onStatusChange: setStatus,
//       onWakeWord: (confidence) => {
//         setLastDetection({ timestamp: new Date(), confidence });
//         config.onWakeWord?.(confidence);
//       },
//     });
//
//     return () => {
//       detectorRef.current?.stop();
//     };
//   }, [config.word, config.serverUrl, config.threshold]);
//
//   const start = useCallback(async () => {
//     await detectorRef.current?.start();
//   }, []);
//
//   const stop = useCallback(() => {
//     detectorRef.current?.stop();
//   }, []);
//
//   return { status, lastDetection, start, stop, detector: detectorRef.current! };
// }

// ============================================================================
// Exports
// ============================================================================

export default WakeWordDetector;
