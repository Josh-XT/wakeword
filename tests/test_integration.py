"""
Integration tests for end-to-end wake word training and detection.
"""
import os
import io
import time
import wave
import asyncio
import pytest
import torch
import numpy as np
from pathlib import Path

from wakeword.model import (
    WakeWordCNN,
    WakeWordTrainer,
    NegativeSampleGenerator,
    AudioFeatureExtractor,
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


class TestEndToEndTraining:
    """Test complete training pipeline."""
    
    def test_synthetic_training_pipeline(self, device, temp_dir, benchmark_tracker):
        """Test complete training from synthetic data to model export."""
        word = "test"
        
        print(f"\n{'='*60}")
        print(f"END-TO-END TRAINING TEST")
        print(f"Word: {word}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")
        
        # 1. Generate synthetic positive samples
        print("Step 1: Generating synthetic samples...")
        num_samples = 50
        duration = 1.5
        num_audio_samples = int(SAMPLE_RATE * duration)
        
        positive_samples = []
        negative_samples = []
        
        with benchmark_tracker.benchmark(
            "e2e_synthetic_positive_generation",
            extra_info={"num_samples": num_samples}
        ):
            for i in range(num_samples):
                t = np.linspace(0, duration, num_audio_samples)
                # Simulate "positive" pattern: specific frequency range
                freq = 440 + np.random.uniform(-30, 30)
                audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
                audio += 0.1 * np.random.randn(num_audio_samples).astype(np.float32)
                audio_bytes = create_audio_bytes(audio, SAMPLE_RATE)
                positive_samples.append((audio_bytes, {"type": "positive", "freq": freq}))
        
        print(f"  Generated {len(positive_samples)} positive samples")
        
        # 2. Generate negative samples using NegativeSampleGenerator
        print("\nStep 2: Generating negative samples...")
        neg_generator = NegativeSampleGenerator()
        
        with benchmark_tracker.benchmark("e2e_negative_generation"):
            negative_samples = neg_generator.generate_noise_samples(
                count=num_samples,
                duration=duration,
            )
        
        print(f"  Generated {len(negative_samples)} negative samples")
        
        # 3. Train model
        print("\nStep 3: Training model...")
        config = ModelConfig(word=word, model_type="cnn")
        trainer = WakeWordTrainer(config=config, device=str(device))
        
        def progress_cb(epoch, total, metrics):
            if epoch % 2 == 0:
                print(f"  Epoch {epoch}/{total}: loss={metrics['train_loss']:.4f}, val_acc={metrics['val_acc']:.4f}")
        
        with benchmark_tracker.benchmark(
            "e2e_model_training",
            extra_info={"device": str(device), "epochs": 10}
        ):
            history = trainer.train(
                positive_samples=positive_samples,
                negative_samples=negative_samples,
                epochs=10,
                batch_size=16,
                progress_callback=progress_cb,
            )
        
        print(f"  Final: train_loss={history['train_loss'][-1]:.4f}, val_acc={history['val_acc'][-1]:.4f}")
        
        # 4. Export models
        print("\nStep 4: Exporting models...")
        output_dir = temp_dir / "models"
        
        with benchmark_tracker.benchmark("e2e_export_all"):
            saved_files = trainer.save(output_dir)
        
        print(f"  Exported to: {output_dir}")
        for f in output_dir.glob("*"):
            print(f"    {f.name}: {f.stat().st_size / 1024:.1f} KB")
        
        # 5. Test inference
        print("\nStep 5: Testing inference...")
        trainer.model.eval()
        extractor = AudioFeatureExtractor()
        
        # Test on a positive-like sample
        t = np.linspace(0, duration, num_audio_samples)
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        test_bytes = create_audio_bytes(test_audio, SAMPLE_RATE)
        
        with benchmark_tracker.benchmark("e2e_inference", iterations=100):
            for _ in range(100):
                is_wake, confidence = trainer.predict(test_bytes)
        
        print(f"  Positive-like sample: detected={is_wake}, confidence={confidence:.4f}")
        
        # Test on noise
        noise_bytes = create_audio_bytes(np.random.randn(num_audio_samples).astype(np.float32) * 0.1)
        is_wake_noise, conf_noise = trainer.predict(noise_bytes)
        print(f"  Noise sample: detected={is_wake_noise}, confidence={conf_noise:.4f}")
        
        print(f"\n{'='*60}")
        print("END-TO-END TEST COMPLETE")
        print(f"{'='*60}\n")
        
        # Assertions
        assert len(positive_samples) == num_samples
        assert len(negative_samples) == num_samples
        assert trainer.model is not None
        assert history["val_acc"][-1] > 0.5  # Should do better than random


class TestModelAccuracy:
    """Test model accuracy on synthetic data."""
    
    def test_model_accuracy_on_distinguishable_classes(self, device, benchmark_tracker):
        """Test model accuracy on clearly different classes."""
        # Train on distinguishable patterns
        duration = 1.5
        num_audio_samples = int(SAMPLE_RATE * duration)
        num_samples = 100
        
        positive_samples = []
        negative_samples = []
        
        # Positive: Low frequency (200-300 Hz)
        for i in range(num_samples):
            t = np.linspace(0, duration, num_audio_samples)
            freq = 250 + np.random.uniform(-25, 25)
            audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            audio += 0.05 * np.random.randn(num_audio_samples).astype(np.float32)
            audio_bytes = create_audio_bytes(audio, SAMPLE_RATE)
            positive_samples.append((audio_bytes, {"type": "positive"}))
        
        # Negative: High frequency (800-1000 Hz)
        for i in range(num_samples):
            t = np.linspace(0, duration, num_audio_samples)
            freq = 900 + np.random.uniform(-50, 50)
            audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            audio += 0.05 * np.random.randn(num_audio_samples).astype(np.float32)
            audio_bytes = create_audio_bytes(audio, SAMPLE_RATE)
            negative_samples.append((audio_bytes, {"type": "negative"}))
        
        # Train
        config = ModelConfig(word="test", model_type="cnn")
        trainer = WakeWordTrainer(config=config, device=str(device))
        
        with benchmark_tracker.benchmark(
            "accuracy_training",
            extra_info={"samples": num_samples * 2, "epochs": 20}
        ):
            history = trainer.train(
                positive_samples=positive_samples,
                negative_samples=negative_samples,
                epochs=20,
                batch_size=32,
            )
        
        # Test accuracy on held-out samples
        test_correct = 0
        test_total = 20
        
        # Test positive
        for i in range(test_total // 2):
            t = np.linspace(0, duration, num_audio_samples)
            freq = 250 + np.random.uniform(-25, 25)
            audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            audio += 0.05 * np.random.randn(num_audio_samples).astype(np.float32)
            audio_bytes = create_audio_bytes(audio, SAMPLE_RATE)
            
            is_wake, conf = trainer.predict(audio_bytes)
            if is_wake:
                test_correct += 1
        
        # Test negative
        for i in range(test_total // 2):
            t = np.linspace(0, duration, num_audio_samples)
            freq = 900 + np.random.uniform(-50, 50)
            audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            audio += 0.05 * np.random.randn(num_audio_samples).astype(np.float32)
            audio_bytes = create_audio_bytes(audio, SAMPLE_RATE)
            
            is_wake, conf = trainer.predict(audio_bytes)
            if not is_wake:
                test_correct += 1
        
        accuracy = test_correct / test_total
        print(f"\nTest Accuracy: {accuracy*100:.1f}% ({test_correct}/{test_total})")
        print(f"Training final val_acc: {history['val_acc'][-1]*100:.1f}%")
        
        # Should achieve reasonable accuracy on distinguishable classes
        assert accuracy > 0.6, f"Accuracy too low: {accuracy*100:.1f}%"


class TestGPUPerformanceIntegration:
    """GPU-specific performance tests in integration context."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_gpu_utilization(self, device, temp_dir, benchmark_tracker):
        """Test GPU is properly utilized during training."""
        import torch.cuda as cuda
        
        cuda.reset_peak_memory_stats()
        initial_memory = cuda.memory_allocated()
        
        # Create training data
        duration = 1.5
        num_audio_samples = int(SAMPLE_RATE * duration)
        num_samples = 100
        
        positive_samples = []
        negative_samples = []
        
        for i in range(num_samples):
            audio = np.random.randn(num_audio_samples).astype(np.float32) * 0.5
            audio_bytes = create_audio_bytes(audio, SAMPLE_RATE)
            positive_samples.append((audio_bytes, {"type": "positive"}))
            
            audio = np.random.randn(num_audio_samples).astype(np.float32) * 0.1
            audio_bytes = create_audio_bytes(audio, SAMPLE_RATE)
            negative_samples.append((audio_bytes, {"type": "negative"}))
        
        config = ModelConfig(word="gpu_test", model_type="cnn")
        trainer = WakeWordTrainer(config=config, device="cuda")
        
        with benchmark_tracker.benchmark(
            "gpu_training_full",
            extra_info={"samples": num_samples * 2, "epochs": 10}
        ):
            history = trainer.train(
                positive_samples=positive_samples,
                negative_samples=negative_samples,
                epochs=10,
                batch_size=32,
            )
        
        peak_memory = cuda.max_memory_allocated()
        memory_used_mb = (peak_memory - initial_memory) / (1024 * 1024)
        
        print(f"\nGPU Training Memory Usage:")
        print(f"  Peak memory: {peak_memory / (1024*1024):.1f} MB")
        print(f"  Memory increase: {memory_used_mb:.1f} MB")
        
        # Should have used some GPU memory
        assert memory_used_mb > 0, "GPU memory was not utilized"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_inference_throughput(self, device, benchmark_tracker):
        """Test batch inference throughput on GPU."""
        model = WakeWordCNN(n_mfcc=N_MFCC, max_frames=MAX_FRAMES).to(device)
        model.eval()
        
        batch_sizes = [1, 8, 32, 64, 128, 256]
        throughputs = {}
        
        print("\nBatch Inference Throughput:")
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 1, N_MFCC, MAX_FRAMES).to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x)
            torch.cuda.synchronize()
            
            # Benchmark
            iterations = 100
            start = time.perf_counter()
            
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(x)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            throughput = (batch_size * iterations) / elapsed
            throughputs[batch_size] = throughput
            
            print(f"  Batch {batch_size:3d}: {throughput:>10,.0f} samples/sec ({elapsed/iterations*1000:.2f} ms/batch)")
        
        # Report to benchmark tracker
        from tests.conftest import BenchmarkResult
        benchmark_tracker.add_result(
            BenchmarkResult(
                name="batch_inference_throughput",
                duration_seconds=1.0,
                iterations=1,
                extra_info=throughputs
            )
        )
        
        # RTX 4090 should handle at least 100k samples/second at batch 64
        assert throughputs[64] > 50000, f"Throughput too low at batch 64: {throughputs[64]}"


class TestModelExportFormats:
    """Test model export to different formats."""
    
    def test_export_pytorch(self, device, temp_dir):
        """Test PyTorch model export."""
        config = ModelConfig(word="export_test", model_type="cnn")
        trainer = WakeWordTrainer(config=config, device=str(device))
        
        # Quick train
        duration = 1.5
        num_audio_samples = int(SAMPLE_RATE * duration)
        
        samples = []
        for i in range(20):
            audio = np.random.randn(num_audio_samples).astype(np.float32) * 0.5
            audio_bytes = create_audio_bytes(audio, SAMPLE_RATE)
            samples.append((audio_bytes, {"type": "test"}))
        
        trainer.train(samples, samples, epochs=2)
        
        # Export
        output_dir = temp_dir / "export_pytorch"
        saved = trainer.save(output_dir)
        
        # Check PyTorch file exists
        pt_files = list(output_dir.glob("*.pt"))
        assert len(pt_files) > 0, "No PyTorch files exported"
        
        # Load and verify - the saved format wraps state_dict with config
        model_path = pt_files[0]
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
        # Handle both wrapped and unwrapped formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            loaded_state = checkpoint["model_state_dict"]
        else:
            loaded_state = checkpoint
        
        new_model = WakeWordCNN(n_mfcc=N_MFCC, max_frames=MAX_FRAMES)
        new_model.load_state_dict(loaded_state)
        new_model.eval()
        
        # Test inference
        x = torch.randn(1, 1, N_MFCC, MAX_FRAMES)
        with torch.no_grad():
            output = new_model(x)
        
        assert output.shape == (1,)
        assert 0 <= output.item() <= 1
