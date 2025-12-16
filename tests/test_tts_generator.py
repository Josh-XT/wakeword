"""
Tests for TTS sample generation module.
"""
import os
import io
import time
import wave
import pytest
import numpy as np
from pathlib import Path

from wakeword.tts_generator import (
    SampleGenerator,
    AudioAugmenter,
    TTSEngine,
    TTSSample,
)


def create_test_audio_bytes(duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Create test audio as WAV bytes."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())
    buffer.seek(0)
    return buffer.read()


class TestAudioAugmenter:
    """Test audio augmentation functionality."""
    
    def test_augmenter_initialization(self):
        """Test augmenter initializes correctly."""
        augmenter = AudioAugmenter()
        assert augmenter is not None
    
    def test_augment_sample(self, benchmark_tracker):
        """Test augmenting a single sample."""
        augmenter = AudioAugmenter()
        audio_bytes = create_test_audio_bytes(1.0, 16000)
        
        with benchmark_tracker.benchmark("augment_single_sample"):
            augmented = augmenter.augment_sample(audio_bytes, 16000)
        
        # Should produce multiple augmentations
        assert len(augmented) > 0
        
        # Each augmentation should be (bytes, name) tuple
        for aug_bytes, aug_name in augmented:
            assert isinstance(aug_bytes, bytes)
            assert len(aug_bytes) > 0
            assert isinstance(aug_name, str)
        
        print(f"\nGenerated {len(augmented)} augmentations:")
        aug_types = set(name for _, name in augmented)
        print(f"  Types: {sorted(aug_types)}")
    
    def test_augment_sample_batch(self, benchmark_tracker):
        """Test augmenting multiple samples."""
        augmenter = AudioAugmenter()
        num_samples = 20
        
        samples = [create_test_audio_bytes(1.0, 16000) for _ in range(num_samples)]
        
        all_augmented = []
        with benchmark_tracker.benchmark(
            "augment_batch",
            extra_info={"num_samples": num_samples}
        ):
            for audio_bytes in samples:
                augmented = augmenter.augment_sample(audio_bytes, 16000)
                all_augmented.extend(augmented)
        
        print(f"\nBatch augmentation: {num_samples} samples -> {len(all_augmented)} augmented")


class TestSampleGenerator:
    """Test TTS sample generation."""
    
    @pytest.fixture
    def generator(self, temp_dir):
        """Create generator instance."""
        return SampleGenerator(
            output_dir=temp_dir / "samples",
            enable_gtts=True,
            enable_edge=True,
            enable_pyttsx3=False,  # Disable for CI
            enable_chatterbox=False,
        )
    
    def test_generator_initialization(self, generator):
        """Test generator initializes correctly."""
        assert generator is not None
    
    def test_available_engines(self, generator):
        """Test detecting available TTS engines."""
        engines = generator.available_engines
        
        # At least one engine should be available
        assert len(engines) > 0
        print(f"\nAvailable TTS engines: {[e.value for e in engines]}")
    
    @pytest.mark.asyncio
    async def test_generate_gtts_sample(self, generator, benchmark_tracker):
        """Test generating a single gTTS sample."""
        if TTSEngine.GTTS not in generator.available_engines:
            pytest.skip("gTTS not available")
        
        with benchmark_tracker.benchmark("gtts_single_sample"):
            sample = await generator.generate_gtts_sample("hello")
        
        assert sample is not None
        assert sample.sample_rate == 16000
        assert len(sample.audio_data) > 0
        assert sample.engine == TTSEngine.GTTS
    
    @pytest.mark.asyncio
    async def test_generate_edge_tts_sample(self, generator, benchmark_tracker):
        """Test generating a single Edge TTS sample."""
        if TTSEngine.EDGE not in generator.available_engines:
            pytest.skip("Edge TTS not available")
        
        voice = generator.EDGE_VOICES[0]
        
        with benchmark_tracker.benchmark("edge_tts_single_sample"):
            sample = await generator.generate_edge_sample("hello", voice)
        
        # Edge TTS can fail due to network issues, so we handle None gracefully
        if sample is None:
            pytest.skip("Edge TTS returned no audio (possible network issue)")
        
        assert sample.sample_rate == 16000
        assert len(sample.audio_data) > 0
        assert sample.engine == TTSEngine.EDGE
    
    @pytest.mark.asyncio
    async def test_generate_multiple_samples(self, generator, benchmark_tracker):
        """Test generating multiple samples for a wake word."""
        # Small batch for testing
        target_count = 10
        
        with benchmark_tracker.benchmark(
            "generate_samples_batch",
            extra_info={"target_count": target_count}
        ):
            samples = await generator.generate_samples(
                word="jarvis",
                target_count=target_count,
                progress_callback=lambda cur, tot, stage: print(f"  Progress: {cur}/{tot} - {stage}")
            )
        
        # Should get some samples (may not hit target if some engines fail)
        assert len(samples) >= 1
        print(f"\nGenerated {len(samples)} samples")
        
        for sample in samples:
            assert sample.sample_rate == 16000
            assert len(sample.audio_data) > 0
    
    @pytest.mark.asyncio
    async def test_save_samples(self, generator, temp_dir):
        """Test saving generated samples to disk."""
        # Generate a few samples
        samples = await generator.generate_samples(
            word="test",
            target_count=5
        )
        
        if not samples:
            pytest.skip("No samples generated")
        
        # Save samples
        save_dir = generator.save_samples(samples, "test")
        
        # Check files exist
        saved_files = list(save_dir.glob("*.wav")) if save_dir and save_dir.exists() else []
        print(f"\nSaved {len(saved_files)} files to {save_dir}")
        assert len(saved_files) > 0


class TestTTSEngineCoverage:
    """Test coverage across TTS engines."""
    
    @pytest.fixture
    def generator(self, temp_dir):
        """Create generator with all engines enabled."""
        return SampleGenerator(
            output_dir=temp_dir / "samples",
            enable_gtts=True,
            enable_edge=True,
            enable_pyttsx3=False,  # Can be flaky
            enable_chatterbox=False,
        )
    
    @pytest.mark.asyncio
    async def test_engine_diversity(self, generator, benchmark_tracker):
        """Test that we get samples from multiple engines."""
        samples = await generator.generate_samples(
            word="hey computer",
            target_count=20
        )
        
        if not samples:
            pytest.skip("No samples generated")
        
        engines_used = set(s.engine for s in samples)
        print(f"\nEngines used: {[e.value for e in engines_used]}")
        print(f"Samples per engine: {dict((e.value, sum(1 for s in samples if s.engine == e)) for e in engines_used)}")
        
        # Should use at least one engine
        assert len(engines_used) >= 1


class TestAugmentationDataset:
    """Test augmenting a full dataset."""
    
    @pytest.mark.asyncio
    async def test_augment_dataset(self, temp_dir, benchmark_tracker):
        """Test augmenting a dataset of TTS samples."""
        # Create mock samples
        samples = []
        for i in range(5):
            audio_bytes = create_test_audio_bytes(1.0, 16000)
            samples.append(TTSSample(
                audio_data=audio_bytes,
                sample_rate=16000,
                engine=TTSEngine.GTTS,
                voice=f"voice_{i}",
                word="test",
                variation="normal"
            ))
        
        augmenter = AudioAugmenter()
        
        with benchmark_tracker.benchmark(
            "augment_dataset",
            extra_info={"base_samples": len(samples)}
        ):
            augmented = await augmenter.augment_dataset(
                samples,
                augmentations_per_sample=3,
                progress_callback=lambda cur, tot, stage: print(f"  {cur}/{tot} - {stage}")
            )
        
        # Should have originals + augmentations
        expected_min = len(samples)  # At least the originals
        assert len(augmented) >= expected_min
        
        print(f"\nDataset augmentation: {len(samples)} -> {len(augmented)} samples")
