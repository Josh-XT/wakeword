"""
Tests for job manager module.
"""
import os
import time
import asyncio
import pytest
from pathlib import Path
from datetime import datetime, timedelta

from wakeword.job_manager import (
    JobManager,
    TrainingJob,
    JobStatus,
)


class TestTrainingJob:
    """Test TrainingJob data class."""
    
    def test_job_creation(self):
        """Test creating a training job."""
        now = datetime.now()
        job = TrainingJob(
            job_id="test-123",
            word="jarvis",
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        
        assert job.job_id == "test-123"
        assert job.word == "jarvis"
        assert job.status == JobStatus.PENDING
        assert job.progress == 0.0
    
    def test_job_to_dict(self):
        """Test converting job to dictionary."""
        now = datetime.now()
        job = TrainingJob(
            job_id="test-123",
            word="jarvis",
            status=JobStatus.TRAINING,
            created_at=now,
            updated_at=now,
            progress=50.0,
            current_stage="Training model",
        )
        
        data = job.to_dict()
        
        assert data["job_id"] == "test-123"
        assert data["word"] == "jarvis"
        assert data["status"] == "training"
        assert data["progress"] == 50.0
        assert data["current_stage"] == "Training model"
    
    def test_job_from_dict(self):
        """Test creating job from dictionary."""
        now = datetime.now()
        data = {
            "job_id": "test-456",
            "word": "alexa",
            "status": "generating_samples",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "progress": 25.0,
            "current_stage": "Generating samples",
        }
        
        job = TrainingJob.from_dict(data)
        
        assert job.job_id == "test-456"
        assert job.word == "alexa"
        assert job.status == JobStatus.GENERATING_SAMPLES
        assert job.progress == 25.0


class TestJobManager:
    """Test JobManager functionality."""
    
    @pytest.fixture
    def job_manager(self, temp_dir):
        """Create a job manager instance."""
        import uuid
        # Each test gets a fresh temp directory with unique paths
        unique_id = str(uuid.uuid4())[:8]
        return JobManager(
            jobs_dir=temp_dir / f"jobs_{unique_id}",
            models_dir=temp_dir / f"models_{unique_id}",
            samples_dir=temp_dir / f"samples_{unique_id}",
        )
    
    def test_manager_initialization(self, job_manager):
        """Test job manager initializes correctly."""
        assert job_manager is not None
        assert job_manager.jobs_dir.exists()
    
    @pytest.mark.asyncio
    async def test_create_job(self, job_manager):
        """Test creating a new job."""
        job = await job_manager.create_job("hello_test1", sample_count=100, epochs=10)
        
        assert job is not None
        assert job.word == "hello_test1"
        # Job status might be PENDING or GENERATING_SAMPLES depending on async timing
        assert job.status in [JobStatus.PENDING, JobStatus.GENERATING_SAMPLES]
        assert job_manager.get_job(job.job_id) is not None
    
    @pytest.mark.asyncio
    async def test_get_existing_job_for_word(self, job_manager):
        """Test getting existing active job for a word."""
        # Create first job
        job1 = await job_manager.create_job("unique_word_test")
        
        # Try to get active job for same word
        existing = job_manager.get_job_for_word("unique_word_test")
        
        assert existing is not None
        assert existing.job_id == job1.job_id
    
    @pytest.mark.asyncio
    async def test_update_job_progress(self, job_manager):
        """Test updating job progress."""
        job = await job_manager.create_job("progress_test")
        
        # Use internal _update_job method since there's no public update method
        job_manager._update_job(
            job,
            status=JobStatus.TRAINING,
            progress=50.0,
            current_stage="Training model"
        )
        
        updated = job_manager.get_job(job.job_id)
        
        assert updated.status == JobStatus.TRAINING
        assert updated.progress == 50.0
        assert updated.current_stage == "Training model"
    
    @pytest.mark.asyncio
    async def test_complete_job(self, job_manager):
        """Test completing a job."""
        job = await job_manager.create_job("complete_test")
        
        job_manager._update_job(
            job,
            status=JobStatus.COMPLETED,
            progress=100.0,
            model_path="/path/to/model.pt"
        )
        
        completed = job_manager.get_job(job.job_id)
        
        assert completed.status == JobStatus.COMPLETED
        assert completed.progress == 100.0
        assert completed.model_path == "/path/to/model.pt"
    
    @pytest.mark.asyncio
    async def test_fail_job(self, job_manager):
        """Test failing a job."""
        job = await job_manager.create_job("fail_test")
        
        job_manager._update_job(
            job,
            status=JobStatus.FAILED,
            error_message="Test error message"
        )
        
        failed = job_manager.get_job(job.job_id)
        
        assert failed.status == JobStatus.FAILED
        assert failed.error_message == "Test error message"
    
    @pytest.mark.asyncio
    async def test_list_jobs(self, job_manager):
        """Test listing all jobs."""
        await job_manager.create_job("listword1")
        await job_manager.create_job("listword2")
        await job_manager.create_job("listword3")
        
        jobs = job_manager.list_jobs()
        
        assert len(jobs) == 3
    
    @pytest.mark.asyncio
    async def test_list_jobs_by_status(self, job_manager):
        """Test listing jobs filtered by status."""
        job1 = await job_manager.create_job("status_word1")
        job2 = await job_manager.create_job("status_word2")
        job3 = await job_manager.create_job("status_word3")
        
        # Update statuses using internal method
        job_manager._update_job(job1, status=JobStatus.TRAINING)
        job_manager._update_job(job2, status=JobStatus.COMPLETED)
        job_manager._update_job(job3, status=JobStatus.PENDING)
        
        # Filter by status
        training_jobs = job_manager.list_jobs(status=JobStatus.TRAINING)
        completed_jobs = job_manager.list_jobs(status=JobStatus.COMPLETED)
        pending_jobs = job_manager.list_jobs(status=JobStatus.PENDING)
        
        assert len(training_jobs) == 1
        assert len(completed_jobs) == 1
        assert len(pending_jobs) == 1
    
    @pytest.mark.asyncio
    async def test_job_persistence(self, temp_dir):
        """Test that jobs persist across manager instances."""
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        jobs_dir = temp_dir / f"persistent_jobs_{unique_id}"
        models_dir = temp_dir / f"persistent_models_{unique_id}"
        samples_dir = temp_dir / f"persistent_samples_{unique_id}"
        
        # Create job with first manager
        manager1 = JobManager(jobs_dir=jobs_dir, models_dir=models_dir, samples_dir=samples_dir)
        job = await manager1.create_job("persistent_word")
        job_id = job.job_id
        manager1._update_job(job, status=JobStatus.TRAINING, progress=50.0)
        
        # Create new manager and check job exists
        manager2 = JobManager(jobs_dir=jobs_dir, models_dir=models_dir, samples_dir=samples_dir)
        loaded_job = manager2.get_job(job_id)
        
        assert loaded_job is not None
        assert loaded_job.word == "persistent_word"
        assert loaded_job.status == JobStatus.TRAINING
        assert loaded_job.progress == 50.0


class TestJobManagerConcurrency:
    """Test job manager under concurrent access."""
    
    @pytest.fixture
    def job_manager(self, temp_dir):
        """Create a job manager instance."""
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        return JobManager(
            jobs_dir=temp_dir / f"concurrent_jobs_{unique_id}",
            models_dir=temp_dir / f"concurrent_models_{unique_id}",
            samples_dir=temp_dir / f"concurrent_samples_{unique_id}",
        )
    
    @pytest.mark.asyncio
    async def test_sequential_job_creation(self, job_manager, benchmark_tracker):
        """Test creating many jobs sequentially."""
        num_jobs = 50
        
        with benchmark_tracker.benchmark(
            "sequential_job_creation",
            extra_info={"num_jobs": num_jobs}
        ):
            jobs = []
            for i in range(num_jobs):
                job = await job_manager.create_job(f"seqword_{i}")
                jobs.append(job)
        
        assert len(jobs) == num_jobs
        assert len(job_manager.list_jobs()) == num_jobs
    
    @pytest.mark.asyncio
    async def test_sequential_updates(self, job_manager, benchmark_tracker):
        """Test updating jobs rapidly."""
        job = await job_manager.create_job("concurrent_update_test")
        num_updates = 100
        
        with benchmark_tracker.benchmark(
            "sequential_job_updates",
            extra_info={"num_updates": num_updates}
        ):
            for i in range(num_updates):
                job_manager._update_job(job, progress=float(i))
        
        # Final progress should be the last update
        final = job_manager.get_job(job.job_id)
        assert final.progress == float(num_updates - 1)


class TestJobStatusTransitions:
    """Test job status state machine."""
    
    @pytest.fixture
    def job_manager(self, temp_dir):
        """Create a job manager instance."""
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        return JobManager(
            jobs_dir=temp_dir / f"status_jobs_{unique_id}",
            models_dir=temp_dir / f"status_models_{unique_id}",
            samples_dir=temp_dir / f"status_samples_{unique_id}",
        )
    
    @pytest.mark.asyncio
    async def test_full_job_lifecycle(self, job_manager):
        """Test complete job status lifecycle."""
        # Create job
        job = await job_manager.create_job("lifecycle_test")
        # Status may have already transitioned due to background task
        initial_status = job.status
        assert initial_status in [JobStatus.PENDING, JobStatus.GENERATING_SAMPLES]
        
        # Start generating samples
        job_manager._update_job(job, status=JobStatus.GENERATING_SAMPLES, progress=10.0)
        updated_job = job_manager.get_job(job.job_id)
        assert updated_job.status == JobStatus.GENERATING_SAMPLES
        
        # Start augmentation
        job_manager._update_job(job, status=JobStatus.AUGMENTING, progress=30.0)
        updated_job = job_manager.get_job(job.job_id)
        assert updated_job.status == JobStatus.AUGMENTING
        
        # Start training
        job_manager._update_job(job, status=JobStatus.TRAINING, progress=50.0)
        updated_job = job_manager.get_job(job.job_id)
        assert updated_job.status == JobStatus.TRAINING
        
        # Start exporting
        job_manager._update_job(job, status=JobStatus.EXPORTING, progress=90.0)
        updated_job = job_manager.get_job(job.job_id)
        assert updated_job.status == JobStatus.EXPORTING
        
        # Complete
        job_manager._update_job(job, status=JobStatus.COMPLETED, progress=100.0)
        updated_job = job_manager.get_job(job.job_id)
        assert updated_job.status == JobStatus.COMPLETED
        assert updated_job.progress == 100.0
    
    @pytest.mark.asyncio
    async def test_job_failure_at_any_stage(self, job_manager):
        """Test job can fail at any stage."""
        stages = [
            JobStatus.GENERATING_SAMPLES,
            JobStatus.AUGMENTING,
            JobStatus.TRAINING,
            JobStatus.EXPORTING,
        ]
        
        for i, stage in enumerate(stages):
            job = await job_manager.create_job(f"fail_stage_test_{i}")
            job_manager._update_job(job, status=stage, progress=float(i * 25))
            job_manager._update_job(job, status=JobStatus.FAILED, error_message=f"Failed at {stage.value}")
            
            updated_job = job_manager.get_job(job.job_id)
            assert updated_job.status == JobStatus.FAILED
            assert f"Failed at {stage.value}" in job.error_message
