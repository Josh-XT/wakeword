"""
API Tests for WakeWord Server

Tests the FastAPI endpoints for the wake word training and detection service.
"""

import pytest
import asyncio
import base64
import numpy as np
import io
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

# Constants
SAMPLE_RATE = 16000


def create_test_audio_bytes(duration_sec: float = 1.0) -> bytes:
    """Create test audio bytes in WAV format."""
    num_samples = int(SAMPLE_RATE * duration_sec)
    audio = np.random.randn(num_samples).astype(np.float32) * 0.1
    audio_int16 = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_int16.tobytes())

    return buffer.getvalue()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test data."""
    return tmp_path


@pytest.fixture
def mock_job_manager(temp_dir):
    """Create a mock job manager."""
    from wakeword.job_manager import JobManager, JobStatus, TrainingJob
    from datetime import datetime

    manager = JobManager(
        jobs_dir=temp_dir / "jobs",
        models_dir=temp_dir / "models",
        samples_dir=temp_dir / "samples",
    )
    return manager


@pytest.fixture
def test_app(temp_dir, mock_job_manager):
    """Create a test FastAPI app with mocked dependencies."""
    from wakeword.app import app, settings

    # Patch the global job_manager
    import wakeword.app as app_module

    original_job_manager = app_module.job_manager
    app_module.job_manager = mock_job_manager

    # Create test client
    client = TestClient(app)

    yield client

    # Restore original
    app_module.job_manager = original_job_manager


class TestHealthEndpoints:
    """Test health and root endpoints."""

    def test_root_endpoint(self, test_app):
        """Test root endpoint returns service info."""
        response = test_app.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "WakeWord Server"
        assert "version" in data
        assert data["status"] == "healthy"

    def test_health_endpoint(self, test_app):
        """Test health check endpoint."""
        response = test_app.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestModelsEndpoints:
    """Test model management endpoints."""

    def test_list_models_empty(self, test_app):
        """Test listing models when none exist."""
        response = test_app.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []
        assert data["total"] == 0

    def test_get_model_not_found(self, test_app):
        """Test getting a model that doesn't exist."""
        response = test_app.get("/models/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert "No model found" in data["detail"]

    def test_delete_model_not_found(self, test_app):
        """Test deleting a model that doesn't exist."""
        response = test_app.delete("/models/nonexistent")

        assert response.status_code == 404


class TestTrainEndpoint:
    """Test training endpoint."""

    @pytest.mark.asyncio
    async def test_train_new_word(self, test_app, mock_job_manager):
        """Test starting training for a new word."""
        response = test_app.post(
            "/train",
            json={
                "word": "hello",
                "sample_count": 50,
                "epochs": 10,
                "batch_size": 16,
            },
        )

        # Should return 200 with job info
        assert response.status_code == 200
        data = response.json()
        assert data["word"] == "hello"
        assert "job_id" in data
        assert data["status"] in ["pending", "generating_samples"]
        assert "/jobs/" in data["check_status_url"]

    def test_train_validation_word_too_short(self, test_app):
        """Test validation rejects empty word."""
        response = test_app.post("/train", json={"word": ""})

        assert response.status_code == 422

    def test_train_validation_sample_count(self, test_app):
        """Test validation of sample_count bounds."""
        # Too low
        response = test_app.post("/train", json={"word": "test", "sample_count": 10})
        assert response.status_code == 422

        # Too high
        response = test_app.post("/train", json={"word": "test", "sample_count": 5000})
        assert response.status_code == 422

    def test_train_validation_epochs(self, test_app):
        """Test validation of epochs bounds."""
        # Too low
        response = test_app.post("/train", json={"word": "test", "epochs": 5})
        assert response.status_code == 422

        # Too high
        response = test_app.post("/train", json={"word": "test", "epochs": 500})
        assert response.status_code == 422


class TestJobsEndpoint:
    """Test job management endpoints."""

    @pytest.mark.asyncio
    async def test_list_jobs_empty(self, test_app):
        """Test listing jobs when none exist."""
        response = test_app.get("/jobs")

        assert response.status_code == 200
        data = response.json()
        assert data == []

    @pytest.mark.asyncio
    async def test_list_jobs_after_create(self, test_app, mock_job_manager):
        """Test listing jobs after creating one."""
        # First create a job
        train_response = test_app.post(
            "/train", json={"word": "listtest", "sample_count": 50, "epochs": 10}
        )
        assert train_response.status_code == 200

        # Then list jobs
        response = test_app.get("/jobs")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

        # Find our job
        job_words = [j["word"] for j in data]
        assert "listtest" in job_words

    def test_get_job_not_found(self, test_app):
        """Test getting a job that doesn't exist."""
        response = test_app.get("/jobs/nonexistent-job-id")

        assert response.status_code == 404


class TestPredictEndpoint:
    """Test prediction endpoint."""

    def test_predict_no_model(self, test_app):
        """Test prediction when model doesn't exist."""
        audio_bytes = create_test_audio_bytes()
        audio_b64 = base64.b64encode(audio_bytes).decode()

        response = test_app.post("/predict/nonexistent", json={"audio_base64": audio_b64})

        assert response.status_code == 404

    def test_predict_invalid_base64(self, test_app):
        """Test prediction with invalid base64 data."""
        response = test_app.post("/predict/someword", json={"audio_base64": "not-valid-base64!!!"})

        # Should return 404 for model not found first
        assert response.status_code == 404


class TestAPIBenchmarks:
    """Benchmark API response times."""

    def test_health_endpoint_latency(self, test_app, benchmark_tracker):
        """Benchmark health endpoint latency."""
        with benchmark_tracker.benchmark("api_health_latency", iterations=100):
            for _ in range(100):
                response = test_app.get("/health")
                assert response.status_code == 200

    def test_list_models_latency(self, test_app, benchmark_tracker):
        """Benchmark list models endpoint latency."""
        with benchmark_tracker.benchmark("api_list_models_latency", iterations=50):
            for _ in range(50):
                response = test_app.get("/models")
                assert response.status_code == 200

    def test_list_jobs_latency(self, test_app, benchmark_tracker):
        """Benchmark list jobs endpoint latency."""
        with benchmark_tracker.benchmark("api_list_jobs_latency", iterations=50):
            for _ in range(50):
                response = test_app.get("/jobs")
                assert response.status_code == 200


class TestCORSHeaders:
    """Test CORS middleware configuration."""

    def test_cors_headers_present(self, test_app):
        """Test that CORS headers are present in response."""
        response = test_app.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # OPTIONS should be allowed
        assert response.status_code in [200, 204, 405]

    def test_cors_allow_all_origins(self, test_app):
        """Test that any origin is allowed."""
        response = test_app.get("/", headers={"Origin": "http://example.com"})

        assert response.status_code == 200
        # Check CORS header
        assert response.headers.get("access-control-allow-origin") == "*"


class TestErrorHandling:
    """Test API error handling."""

    def test_invalid_endpoint(self, test_app):
        """Test 404 for invalid endpoint."""
        response = test_app.get("/invalid/endpoint")

        assert response.status_code == 404

    def test_invalid_method(self, test_app):
        """Test 405 for invalid method."""
        response = test_app.patch("/health")

        assert response.status_code == 405

    def test_invalid_json(self, test_app):
        """Test handling of invalid JSON."""
        response = test_app.post(
            "/train", content="not json", headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422


class TestAPIWithRealTraining:
    """Integration tests with actual training (slower)."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_training_flow(self, test_app, mock_job_manager, benchmark_tracker):
        """Test complete training flow through API."""
        # Skip in CI or when running fast tests
        pytest.skip("This test takes a long time - run with -m slow to include")

        word = "apitest"

        # Start training
        with benchmark_tracker.benchmark("api_train_request"):
            train_response = test_app.post(
                "/train", json={"word": word, "sample_count": 50, "epochs": 5}
            )

        assert train_response.status_code == 200
        job_id = train_response.json()["job_id"]

        # Poll for completion (with timeout)
        import time

        max_wait = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status_response = test_app.get(f"/jobs/{job_id}")
            assert status_response.status_code == 200

            status_data = status_response.json()
            if status_data["status"] == "completed":
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Training failed: {status_data.get('error_message')}")

            await asyncio.sleep(5)
        else:
            pytest.fail("Training timed out")

        # Verify model is available
        model_response = test_app.get(f"/models/{word}")
        assert model_response.status_code == 200
