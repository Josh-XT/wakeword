"""
WakeWord Python Client

Example client for using wake word detection in Python applications.
"""
import asyncio
import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass

import numpy as np
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WakeWordConfig:
    """Configuration for wake word detector."""
    word: str
    server_url: str = "http://localhost:8000"
    threshold: float = 0.5
    sample_rate: int = 16000
    chunk_duration: float = 1.5  # seconds
    
    
class WakeWordClient:
    """
    Client for wake word detection using the WakeWord server.
    
    Example usage:
        client = WakeWordClient("http://localhost:8000")
        
        # Request training if model doesn't exist
        client.ensure_model("jarvis")
        
        # Use for detection
        detector = client.get_detector("jarvis")
        detector.start_listening(on_wake_word=lambda: print("Wake word detected!"))
    """
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip("/")
        self.session = requests.Session()
    
    def list_models(self) -> List[dict]:
        """List available models on the server."""
        response = self.session.get(f"{self.server_url}/models")
        response.raise_for_status()
        return response.json()["models"]
    
    def model_exists(self, word: str) -> bool:
        """Check if a model exists for a word."""
        try:
            response = self.session.get(f"{self.server_url}/models/{word}")
            return response.status_code == 200
        except:
            return False
    
    def request_training(
        self,
        word: str,
        sample_count: int = 500,
        epochs: int = 50,
        wait: bool = False,
        poll_interval: int = 10,
    ) -> dict:
        """
        Request training of a new model.
        
        Args:
            word: The wake word to train
            sample_count: Number of samples to generate
            epochs: Training epochs
            wait: If True, wait for training to complete
            poll_interval: Seconds between status checks when waiting
            
        Returns:
            Training job info
        """
        response = self.session.post(
            f"{self.server_url}/train",
            json={
                "word": word,
                "sample_count": sample_count,
                "epochs": epochs,
            }
        )
        response.raise_for_status()
        result = response.json()
        
        if wait and result["status"] not in ["completed"]:
            job_id = result["job_id"]
            logger.info(f"Waiting for training to complete (job: {job_id})...")
            
            while True:
                time.sleep(poll_interval)
                status = self.get_job_status(job_id)
                logger.info(f"Progress: {status['progress']:.1f}% - {status['current_stage']}")
                
                if status["status"] == "completed":
                    logger.info("Training completed!")
                    return status
                elif status["status"] == "failed":
                    raise RuntimeError(f"Training failed: {status.get('error_message')}")
        
        return result
    
    def get_job_status(self, job_id: str) -> dict:
        """Get status of a training job."""
        response = self.session.get(f"{self.server_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def download_model(
        self,
        word: str,
        output_path: Path,
        format: str = "pytorch",
    ) -> Path:
        """
        Download a trained model.
        
        Args:
            word: The wake word
            output_path: Where to save the model
            format: Model format (pytorch, onnx, tflite)
            
        Returns:
            Path to downloaded model
        """
        response = self.session.get(
            f"{self.server_url}/models/{word}",
            params={"format": format},
            stream=True,
        )
        response.raise_for_status()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Model downloaded to {output_path}")
        return output_path
    
    def ensure_model(
        self,
        word: str,
        sample_count: int = 500,
        epochs: int = 50,
    ) -> bool:
        """
        Ensure a model exists for a word, training if necessary.
        
        Returns True if model is ready, False if training was started.
        """
        if self.model_exists(word):
            return True
        
        result = self.request_training(word, sample_count, epochs, wait=True)
        return result["status"] == "completed"
    
    def predict(self, word: str, audio_bytes: bytes) -> tuple:
        """
        Predict if audio contains the wake word.
        
        Args:
            word: The wake word to detect
            audio_bytes: Raw audio bytes (WAV format)
            
        Returns:
            Tuple of (detected: bool, confidence: float)
        """
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        response = self.session.post(
            f"{self.server_url}/predict/{word}",
            json={"audio_base64": audio_b64},
        )
        response.raise_for_status()
        result = response.json()
        
        return result["detected"], result["confidence"]
    
    def get_detector(self, word: str, threshold: float = 0.5) -> "WakeWordDetector":
        """Get a detector instance for real-time detection."""
        return WakeWordDetector(
            client=self,
            word=word,
            threshold=threshold,
        )


class WakeWordDetector:
    """
    Real-time wake word detector using microphone input.
    
    Example:
        detector = WakeWordDetector(client, "jarvis")
        detector.start_listening(
            on_wake_word=lambda: print("Wake word detected!"),
            on_speech=lambda text: print(f"Heard: {text}"),
        )
    """
    
    def __init__(
        self,
        client: WakeWordClient,
        word: str,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        chunk_duration: float = 1.5,
    ):
        self.client = client
        self.word = word
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        
        self._running = False
        self._stream = None
    
    def start_listening(
        self,
        on_wake_word: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        blocking: bool = True,
    ):
        """
        Start listening for the wake word.
        
        Args:
            on_wake_word: Callback when wake word is detected
            on_error: Callback on error
            blocking: If True, blocks until stop_listening is called
        """
        try:
            import pyaudio
        except ImportError:
            raise ImportError("pyaudio is required for microphone input. Install with: pip install pyaudio")
        
        self._running = True
        audio = pyaudio.PyAudio()
        
        try:
            self._stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
            
            logger.info(f"Listening for wake word '{self.word}'...")
            
            while self._running:
                try:
                    # Read audio chunk
                    data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Convert to WAV bytes
                    wav_bytes = self._pcm_to_wav(data)
                    
                    # Detect wake word
                    detected, confidence = self.client.predict(self.word, wav_bytes)
                    
                    if detected and confidence >= self.threshold:
                        logger.info(f"Wake word detected! (confidence: {confidence:.2f})")
                        if on_wake_word:
                            on_wake_word()
                            
                except Exception as e:
                    if self._running:
                        logger.error(f"Detection error: {e}")
                        if on_error:
                            on_error(e)
        finally:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
            audio.terminate()
    
    def stop_listening(self):
        """Stop listening for wake word."""
        self._running = False
    
    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """Convert raw PCM data to WAV format."""
        import wave
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_data)
        
        buffer.seek(0)
        return buffer.read()


class LocalWakeWordDetector:
    """
    Local wake word detector using downloaded ONNX model.
    
    Runs entirely offline after model download.
    
    Example:
        detector = LocalWakeWordDetector("models/jarvis.onnx", "jarvis")
        detector.start_listening(on_wake_word=lambda: print("Detected!"))
    """
    
    def __init__(
        self,
        model_path: Path,
        word: str,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        chunk_duration: float = 1.5,
        n_mfcc: int = 40,
    ):
        self.model_path = Path(model_path)
        self.word = word
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.n_mfcc = n_mfcc
        
        # Load ONNX model
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime required. Install with: pip install onnxruntime")
        
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        
        self._running = False
    
    def _extract_mfcc(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio."""
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required. Install with: pip install librosa")
        
        # Ensure correct shape
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Normalize
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=512,
            hop_length=160,
            win_length=400,
        )
        
        return mfcc
    
    def predict(self, audio_data: np.ndarray) -> tuple:
        """
        Predict if audio contains wake word.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            Tuple of (detected: bool, confidence: float)
        """
        # Extract features
        mfcc = self._extract_mfcc(audio_data)
        
        # Pad/truncate to expected size
        max_frames = 150  # ~1.5s at 16kHz with hop_length=160
        if mfcc.shape[1] < max_frames:
            padding = np.zeros((mfcc.shape[0], max_frames - mfcc.shape[1]))
            mfcc = np.concatenate([mfcc, padding], axis=1)
        else:
            mfcc = mfcc[:, :max_frames]
        
        # Run inference
        mfcc = mfcc.astype(np.float32)
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
        
        outputs = self.session.run(None, {self.input_name: mfcc})
        confidence = float(outputs[0][0])
        
        return confidence >= self.threshold, confidence
    
    def start_listening(
        self,
        on_wake_word: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """Start listening for wake word using microphone."""
        try:
            import pyaudio
        except ImportError:
            raise ImportError("pyaudio required. Install with: pip install pyaudio")
        
        self._running = True
        audio = pyaudio.PyAudio()
        chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=chunk_samples,
            )
            
            logger.info(f"Listening for wake word '{self.word}' (local mode)...")
            
            while self._running:
                try:
                    # Read audio
                    data = stream.read(chunk_samples, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # Detect
                    detected, confidence = self.predict(audio_data)
                    
                    if detected:
                        logger.info(f"Wake word detected! (confidence: {confidence:.2f})")
                        if on_wake_word:
                            on_wake_word()
                            
                except Exception as e:
                    if self._running:
                        logger.error(f"Detection error: {e}")
                        if on_error:
                            on_error(e)
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
    
    def stop_listening(self):
        """Stop listening."""
        self._running = False


# ============================================================================
# CLI Example
# ============================================================================

def main():
    """Example CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WakeWord Python Client")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--word", required=True, help="Wake word to detect")
    parser.add_argument("--train", action="store_true", help="Train model if not exists")
    parser.add_argument("--local", type=str, help="Path to local ONNX model for offline detection")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    
    args = parser.parse_args()
    
    if args.local:
        # Local detection mode
        detector = LocalWakeWordDetector(
            model_path=args.local,
            word=args.word,
            threshold=args.threshold,
        )
    else:
        # Server-based detection
        client = WakeWordClient(args.server)
        
        if args.train:
            logger.info(f"Ensuring model exists for '{args.word}'...")
            client.ensure_model(args.word)
        
        detector = client.get_detector(args.word, threshold=args.threshold)
    
    def on_wake():
        print(f"\n🎤 Wake word '{args.word}' detected!\n")
    
    print(f"Listening for '{args.word}'... Press Ctrl+C to stop.")
    
    try:
        detector.start_listening(on_wake_word=on_wake)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
