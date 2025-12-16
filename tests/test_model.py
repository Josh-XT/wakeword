"""
Tests for model architecture and training module.
"""
import os
import io
import time
import wave
import pytest
import torch
import numpy as np
from pathlib import Path

from wakeword.model import (
    AudioFeatureExtractor,
    WakeWordCNN,
    WakeWordGRU,
    WakeWordTrainer,
    NegativeSampleGenerator,
    ModelConfig,
    MAX_FRAMES,
    SAMPLE_RATE,
    N_MFCC,
)


def create_audio_bytes(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert numpy audio to wav bytes."""
    audio_int16 = (audio_data * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    buffer.seek(0)
    return buffer.read()


class TestAudioFeatureExtractor:
    """Test MFCC feature extraction."""
    
    def test_extractor_initialization(self):
        """Test feature extractor initializes correctly."""
        extractor = AudioFeatureExtractor()
        assert extractor is not None
        assert extractor.n_mfcc == N_MFCC
        assert extractor.sample_rate == SAMPLE_RATE
    
    def test_extract_features_from_tensor(self, device, benchmark_tracker):
        """Test extracting features from a waveform tensor."""
        extractor = AudioFeatureExtractor()
        
        # Create a test waveform
        duration = 1.5
        num_samples = int(SAMPLE_RATE * duration)
        waveform = torch.randn(1, num_samples)
        
        with benchmark_tracker.benchmark(
            "mfcc_extraction_tensor",
            iterations=100,
            extra_info={"device": str(device)}
        ):
            for _ in range(100):
                features = extractor.extract_features(waveform)
        
        assert features is not None
        assert features.shape[0] == N_MFCC  # n_mfcc
        assert features.shape[1] > 0  # time frames
    
    def test_load_audio_bytes(self, sample_audio, benchmark_tracker):
        """Test loading audio from bytes."""
        audio, sr = sample_audio
        extractor = AudioFeatureExtractor()
        
        # Convert to bytes
        audio_bytes = create_audio_bytes(audio, sr)
        
        with benchmark_tracker.benchmark(
            "load_audio_bytes",
            iterations=100
        ):
            for _ in range(100):
                waveform = extractor.load_audio_bytes(audio_bytes)
        
        assert waveform is not None
        assert waveform.shape[0] == 1  # mono


class TestWakeWordCNN:
    """Test CNN model architecture."""
    
    def test_model_initialization(self, device):
        """Test model initializes correctly."""
        model = WakeWordCNN(n_mfcc=N_MFCC, max_frames=MAX_FRAMES).to(device)
        assert model is not None
    
    def test_model_forward_pass(self, device, benchmark_tracker):
        """Test forward pass through model."""
        model = WakeWordCNN(n_mfcc=N_MFCC, max_frames=MAX_FRAMES).to(device)
        model.eval()
        
        batch_size = 32
        x = torch.randn(batch_size, 1, N_MFCC, MAX_FRAMES).to(device)
        
        # Warmup
        with torch.no_grad():
            _ = model(x)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        with benchmark_tracker.benchmark(
            "cnn_forward_pass",
            iterations=100,
            extra_info={"batch_size": batch_size, "device": str(device)}
        ):
            with torch.no_grad():
                for _ in range(100):
                    output = model(x)
        
        assert output.shape == (batch_size,)
        assert (output >= 0).all() and (output <= 1).all()  # Sigmoid output
    
    def test_model_inference_latency(self, device, benchmark_tracker):
        """Test single sample inference latency."""
        model = WakeWordCNN(n_mfcc=N_MFCC, max_frames=MAX_FRAMES).to(device)
        model.eval()
        
        x = torch.randn(1, 1, N_MFCC, MAX_FRAMES).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        with benchmark_tracker.benchmark(
            "cnn_single_inference",
            iterations=1000,
            extra_info={"device": str(device)}
        ):
            with torch.no_grad():
                for _ in range(1000):
                    output = model(x)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    def test_model_size(self, device):
        """Test model size is within expected bounds."""
        model = WakeWordCNN(n_mfcc=N_MFCC, max_frames=MAX_FRAMES).to(device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        size_mb = num_params * 4 / (1024 * 1024)  # float32 = 4 bytes
        
        print(f"\nCNN Model: {num_params:,} parameters ({size_mb:.2f} MB)")
        
        # Should be under 5MB for edge deployment
        assert size_mb < 5.0, f"Model too large: {size_mb:.2f} MB"


class TestWakeWordGRU:
    """Test GRU model architecture."""
    
    def test_model_initialization(self, device):
        """Test model initializes correctly."""
        model = WakeWordGRU(n_mfcc=N_MFCC, hidden_size=64).to(device)
        assert model is not None
    
    def test_model_forward_pass(self, device, benchmark_tracker):
        """Test forward pass through model."""
        model = WakeWordGRU(n_mfcc=N_MFCC, hidden_size=64).to(device)
        model.eval()
        
        batch_size = 32
        # GRU expects (batch, n_mfcc, time) which gets transposed internally
        x = torch.randn(batch_size, N_MFCC, MAX_FRAMES).to(device)
        
        # Warmup
        with torch.no_grad():
            _ = model(x)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        with benchmark_tracker.benchmark(
            "gru_forward_pass",
            iterations=100,
            extra_info={"batch_size": batch_size, "device": str(device)}
        ):
            with torch.no_grad():
                for _ in range(100):
                    output = model(x)
        
        assert output.shape == (batch_size,)
    
    def test_model_size(self, device):
        """Test GRU model size."""
        model = WakeWordGRU(n_mfcc=N_MFCC, hidden_size=64).to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        size_mb = num_params * 4 / (1024 * 1024)
        
        print(f"\nGRU Model: {num_params:,} parameters ({size_mb:.2f} MB)")
        
        assert size_mb < 5.0, f"Model too large: {size_mb:.2f} MB"


class TestNegativeSampleGenerator:
    """Test negative sample generation."""
    
    def test_generator_initialization(self):
        """Test negative sample generator initializes correctly."""
        generator = NegativeSampleGenerator()
        assert generator is not None
    
    def test_generate_noise_samples(self, benchmark_tracker):
        """Test generating noise samples."""
        generator = NegativeSampleGenerator()
        
        num_samples = 50
        duration = 1.5
        
        with benchmark_tracker.benchmark(
            "generate_noise_samples",
            extra_info={"num_samples": num_samples}
        ):
            samples = generator.generate_noise_samples(
                count=num_samples,
                duration=duration
            )
        
        assert len(samples) == num_samples
        
        # Verify samples are valid audio bytes
        for audio_bytes, metadata in samples:
            assert len(audio_bytes) > 0
            assert "type" in metadata
            assert metadata["type"] == "noise"


class TestWakeWordTrainer:
    """Test model training functionality."""
    
    @pytest.fixture
    def trainer(self, device):
        """Create a trainer instance."""
        config = ModelConfig(word="test", model_type="cnn")
        return WakeWordTrainer(config=config, device=str(device))
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initializes correctly."""
        assert trainer is not None
        assert trainer.model is not None
    
    def test_train_model_quick(self, device, benchmark_tracker):
        """Test quick model training (reduced epochs)."""
        config = ModelConfig(word="test", model_type="cnn")
        trainer = WakeWordTrainer(config=config, device=str(device))
        
        # Create mock samples as bytes
        num_samples = 40
        duration = 1.5
        sample_rate = SAMPLE_RATE
        num_audio_samples = int(sample_rate * duration)
        
        positive_samples = []
        negative_samples = []
        
        # Generate positive samples (sine wave)
        for i in range(num_samples):
            t = np.linspace(0, duration, num_audio_samples)
            freq = 440 + np.random.uniform(-20, 20)
            audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            audio += 0.05 * np.random.randn(num_audio_samples).astype(np.float32)
            audio_bytes = create_audio_bytes(audio, sample_rate)
            positive_samples.append((audio_bytes, {"type": "positive", "index": i}))
        
        # Generate negative samples (noise)
        for i in range(num_samples):
            audio = np.random.randn(num_audio_samples).astype(np.float32) * 0.1
            audio_bytes = create_audio_bytes(audio, sample_rate)
            negative_samples.append((audio_bytes, {"type": "negative", "index": i}))
        
        def progress_cb(epoch, total, metrics):
            print(f"  Epoch {epoch}/{total}: loss={metrics['train_loss']:.4f}, acc={metrics['train_acc']:.4f}")
        
        with benchmark_tracker.benchmark(
            "train_model_5_epochs",
            extra_info={"device": str(device), "samples": num_samples * 2, "epochs": 5}
        ):
            history = trainer.train(
                positive_samples=positive_samples,
                negative_samples=negative_samples,
                epochs=5,
                batch_size=16,
                progress_callback=progress_cb,
            )
        
        assert trainer.model is not None
        assert "train_loss" in history
        assert "val_acc" in history
        
        print(f"\nFinal metrics: train_loss={history['train_loss'][-1]:.4f}, val_acc={history['val_acc'][-1]:.4f}")
    
    def test_export_pytorch(self, device, temp_dir):
        """Test exporting model to PyTorch format."""
        config = ModelConfig(word="test", model_type="cnn")
        trainer = WakeWordTrainer(config=config, device=str(device))
        
        # Quick training
        num_samples = 20
        duration = 1.5
        num_audio_samples = int(SAMPLE_RATE * duration)
        
        positive_samples = []
        negative_samples = []
        
        for i in range(num_samples):
            audio = np.random.randn(num_audio_samples).astype(np.float32) * 0.5
            audio_bytes = create_audio_bytes(audio, SAMPLE_RATE)
            positive_samples.append((audio_bytes, {"type": "positive"}))
            
            audio = np.random.randn(num_audio_samples).astype(np.float32) * 0.1
            audio_bytes = create_audio_bytes(audio, SAMPLE_RATE)
            negative_samples.append((audio_bytes, {"type": "negative"}))
        
        trainer.train(positive_samples, negative_samples, epochs=2)
        
        output_dir = temp_dir / "export_test"
        saved_files = trainer.save(output_dir)
        
        # Check that at least one file was saved
        assert output_dir.exists()
        files = list(output_dir.glob("*.pt"))
        print(f"\nExported files: {files}")


class TestGPUPerformance:
    """GPU-specific performance tests."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_usage(self, device, benchmark_tracker):
        """Test GPU memory usage during inference."""
        torch.cuda.reset_peak_memory_stats()
        
        model = WakeWordCNN(n_mfcc=N_MFCC, max_frames=MAX_FRAMES).to(device)
        model.eval()
        
        batch_sizes = [1, 8, 32, 64, 128]
        
        print("\nGPU Memory Usage by Batch Size:")
        for batch_size in batch_sizes:
            torch.cuda.reset_peak_memory_stats()
            
            x = torch.randn(batch_size, 1, N_MFCC, MAX_FRAMES).to(device)
            
            with torch.no_grad():
                _ = model(x)
            
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            print(f"  Batch {batch_size}: {peak_memory_mb:.2f} MB")
            
            with benchmark_tracker.benchmark(
                f"gpu_batch_{batch_size}",
                iterations=100,
                extra_info={"batch_size": batch_size, "memory_mb": round(peak_memory_mb, 2)}
            ):
                with torch.no_grad():
                    for _ in range(100):
                        _ = model(x)
                        torch.cuda.synchronize()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_throughput(self, device, benchmark_tracker):
        """Test maximum GPU throughput."""
        model = WakeWordCNN(n_mfcc=N_MFCC, max_frames=MAX_FRAMES).to(device)
        model.eval()
        
        batch_size = 64
        x = torch.randn(batch_size, 1, N_MFCC, MAX_FRAMES).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(50):
                _ = model(x)
        torch.cuda.synchronize()
        
        # Benchmark
        iterations = 1000
        start = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        samples_per_second = (batch_size * iterations) / elapsed
        
        print(f"\nGPU Throughput: {samples_per_second:,.0f} samples/second")
        print(f"  Latency per batch: {(elapsed / iterations) * 1000:.3f} ms")
        
        from tests.conftest import BenchmarkResult
        benchmark_tracker.add_result(
            BenchmarkResult(
                name="gpu_throughput",
                duration_seconds=elapsed,
                iterations=iterations * batch_size,
                extra_info={"samples_per_second": round(samples_per_second)}
            )
        )
        
        # RTX 4090 should easily handle 10000+ samples/second
        assert samples_per_second > 10000, f"Throughput too low: {samples_per_second}"
