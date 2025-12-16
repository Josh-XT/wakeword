"""
Pytest configuration and fixtures for WakeWord tests.
"""
import os
import sys
import time
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import pytest
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wakeword.config import Settings


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single operation."""
    name: str
    duration_seconds: float
    iterations: int = 1
    extra_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def avg_duration_ms(self) -> float:
        return (self.duration_seconds / self.iterations) * 1000
    
    def __str__(self) -> str:
        extras = ", ".join(f"{k}={v}" for k, v in self.extra_info.items())
        return (
            f"{self.name}: {self.avg_duration_ms:.2f}ms avg "
            f"({self.iterations} iterations, {self.duration_seconds:.2f}s total)"
            + (f" [{extras}]" if extras else "")
        )


class BenchmarkTracker:
    """Track and report benchmark results across tests."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.gpu_info: Optional[Dict[str, Any]] = None
        self._collect_gpu_info()
    
    def _collect_gpu_info(self):
        """Collect GPU information if available."""
        if torch.cuda.is_available():
            self.gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            }
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)
    
    def benchmark(self, name: str, iterations: int = 1, **extra_info):
        """Context manager for benchmarking a block of code."""
        class BenchmarkContext:
            def __init__(ctx, tracker, name, iterations, extra_info):
                ctx.tracker = tracker
                ctx.name = name
                ctx.iterations = iterations
                ctx.extra_info = extra_info
                ctx.start_time = None
            
            def __enter__(ctx):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                ctx.start_time = time.perf_counter()
                return ctx
            
            def __exit__(ctx, *args):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                duration = time.perf_counter() - ctx.start_time
                result = BenchmarkResult(
                    name=ctx.name,
                    duration_seconds=duration,
                    iterations=ctx.iterations,
                    extra_info=ctx.extra_info
                )
                ctx.tracker.add_result(result)
        
        return BenchmarkContext(self, name, iterations, extra_info)
    
    def get_report(self) -> str:
        """Generate a formatted benchmark report."""
        lines = [
            "=" * 80,
            "BENCHMARK RESULTS",
            "=" * 80,
        ]
        
        if self.gpu_info:
            lines.extend([
                f"GPU: {self.gpu_info['name']}",
                f"VRAM: {self.gpu_info['memory_total_gb']:.1f} GB",
                f"CUDA: {self.gpu_info['cuda_version']}",
                f"cuDNN: {self.gpu_info['cudnn_version']}",
                "-" * 80,
            ])
        
        for result in self.results:
            lines.append(str(result))
        
        lines.append("=" * 80)
        return "\n".join(lines)


# Global benchmark tracker
_benchmark_tracker = BenchmarkTracker()


@pytest.fixture(scope="session")
def benchmark_tracker():
    """Provide access to the global benchmark tracker."""
    return _benchmark_tracker


@pytest.fixture(scope="session", autouse=True)
def print_benchmark_report(benchmark_tracker):
    """Print benchmark report at end of test session."""
    yield
    if benchmark_tracker.results:
        print("\n")
        print(benchmark_tracker.get_report())


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for the test session."""
    tmp = tempfile.mkdtemp(prefix="wakeword_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="session")
def test_settings(temp_dir):
    """Create test settings with temporary directories."""
    models_dir = temp_dir / "models"
    samples_dir = temp_dir / "samples"
    models_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    return Settings(
        models_dir=str(models_dir),
        samples_dir=str(samples_dir),
        default_sample_count=50,  # Reduced for faster tests
        default_epochs=10,  # Reduced for faster tests
    )


@pytest.fixture(scope="session")
def device():
    """Get the compute device (GPU if available)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def sample_audio():
    """Generate a sample audio waveform for testing."""
    sample_rate = 16000
    duration = 1.5
    num_samples = int(sample_rate * duration)
    
    # Generate a simple test signal (sine wave with some noise)
    t = np.linspace(0, duration, num_samples)
    freq = 440  # A4 note
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    signal += 0.1 * np.random.randn(num_samples)
    
    return signal.astype(np.float32), sample_rate


@pytest.fixture
def sample_audio_batch(sample_audio):
    """Generate a batch of sample audio waveforms."""
    audio, sr = sample_audio
    batch_size = 8
    batch = np.stack([audio + 0.05 * np.random.randn(len(audio)) for _ in range(batch_size)])
    return batch.astype(np.float32), sr
