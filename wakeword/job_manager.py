"""
Training Job Manager

Manages background training jobs for wake word models.
Handles job queuing, progress tracking, and prevents duplicate jobs.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    GENERATING_SAMPLES = "generating_samples"
    AUGMENTING = "augmenting"
    TRAINING = "training"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Represents a wake word training job."""

    job_id: str
    word: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress: float = 0.0  # 0-100
    current_stage: str = ""
    error_message: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    model_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "word": self.word,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "progress": self.progress,
            "current_stage": self.current_stage,
            "error_message": self.error_message,
            "estimated_completion": (
                self.estimated_completion.isoformat() if self.estimated_completion else None
            ),
            "model_path": self.model_path,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingJob":
        return cls(
            job_id=data["job_id"],
            word=data["word"],
            status=JobStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            progress=data.get("progress", 0.0),
            current_stage=data.get("current_stage", ""),
            error_message=data.get("error_message"),
            estimated_completion=(
                datetime.fromisoformat(data["estimated_completion"])
                if data.get("estimated_completion")
                else None
            ),
            model_path=data.get("model_path"),
            metrics=data.get("metrics", {}),
        )


class JobManager:
    """
    Manages training jobs with persistence and progress tracking.

    Features:
    - Prevents duplicate jobs for the same word
    - Tracks job progress
    - Persists job state to disk
    - Handles job cancellation
    """

    def __init__(
        self,
        jobs_dir: Path,
        models_dir: Path,
        samples_dir: Path,
        max_concurrent_jobs: int = 2,
    ):
        self.jobs_dir = Path(jobs_dir)
        self.models_dir = Path(models_dir)
        self.samples_dir = Path(samples_dir)

        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.max_concurrent_jobs = max_concurrent_jobs

        # In-memory job tracking
        self.jobs: Dict[str, TrainingJob] = {}
        self.word_to_job: Dict[str, str] = {}  # word -> job_id

        # Job execution
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self.running_tasks: Dict[str, asyncio.Task] = {}

        # Lock for thread safety
        self.lock = threading.Lock()

        # Load existing jobs from disk
        self._load_jobs()

    def _load_jobs(self):
        """Load existing jobs from disk."""
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file, "r") as f:
                    job_data = json.load(f)
                job = TrainingJob.from_dict(job_data)
                self.jobs[job.job_id] = job

                # Track active jobs by word
                if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    self.word_to_job[job.word.lower()] = job.job_id

                logger.info(
                    f"Loaded job {job.job_id} for word '{job.word}' (status: {job.status.value})"
                )
            except Exception as e:
                logger.error(f"Failed to load job from {job_file}: {e}")

    def _save_job(self, job: TrainingJob):
        """Save job state to disk."""
        job_file = self.jobs_dir / f"{job.job_id}.json"
        with open(job_file, "w") as f:
            json.dump(job.to_dict(), f, indent=2)

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def get_job_for_word(self, word: str) -> Optional[TrainingJob]:
        """Get active job for a word."""
        word_lower = word.lower()
        job_id = self.word_to_job.get(word_lower)
        if job_id:
            return self.jobs.get(job_id)
        return None

    def get_model_for_word(self, word: str) -> Optional[Path]:
        """Get the model directory for a word if it exists and is trained."""
        word_lower = word.lower().replace(" ", "_")
        model_dir = self.models_dir / word_lower

        if model_dir.exists() and (model_dir / "model.pt").exists():
            return model_dir
        return None

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> List[TrainingJob]:
        """List jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available trained models."""
        models = []

        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "model.pt").exists():
                # Load config
                config_path = model_dir / "config.json"
                config = {}
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = json.load(f)

                # Get file sizes
                files = {}
                for ext in ["pt", "onnx", "tflite"]:
                    file_path = model_dir / f"model.{ext}"
                    if file_path.exists():
                        files[ext] = {
                            "path": str(file_path),
                            "size_bytes": file_path.stat().st_size,
                        }

                models.append(
                    {
                        "word": config.get("word", model_dir.name),
                        "directory": str(model_dir),
                        "config": config,
                        "files": files,
                        "created_at": datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat(),
                    }
                )

        return models

    async def create_job(
        self,
        word: str,
        sample_count: int = 500,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> TrainingJob:
        """
        Create a new training job for a word.

        Raises:
            ValueError: If a job is already in progress for this word
        """
        word_lower = word.lower()

        with self.lock:
            # Check if job already exists for this word
            existing_job = self.get_job_for_word(word_lower)
            if existing_job and existing_job.status not in [
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ]:
                raise ValueError(
                    f"Job already in progress for word '{word}' "
                    f"(job_id: {existing_job.job_id}, status: {existing_job.status.value})"
                )

            # Create new job
            job = TrainingJob(
                job_id=str(uuid.uuid4()),
                word=word,
                status=JobStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                estimated_completion=datetime.now() + timedelta(minutes=15),
            )

            self.jobs[job.job_id] = job
            self.word_to_job[word_lower] = job.job_id
            self._save_job(job)

        # Start training in background
        task = asyncio.create_task(self._run_training(job, sample_count, epochs, batch_size))
        self.running_tasks[job.job_id] = task

        logger.info(f"Created training job {job.job_id} for word '{word}'")
        return job

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False

        # Cancel the task
        task = self.running_tasks.get(job_id)
        if task:
            task.cancel()

        # Update job status
        job.status = JobStatus.CANCELLED
        job.updated_at = datetime.now()
        self._save_job(job)

        # Remove from word tracking
        word_lower = job.word.lower()
        if self.word_to_job.get(word_lower) == job_id:
            del self.word_to_job[word_lower]

        logger.info(f"Cancelled job {job_id}")
        return True

    def _update_job(
        self,
        job: TrainingJob,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        current_stage: Optional[str] = None,
        error_message: Optional[str] = None,
        model_path: Optional[str] = None,
        metrics: Optional[Dict] = None,
    ):
        """Update job state."""
        if status:
            job.status = status
        if progress is not None:
            job.progress = progress
        if current_stage:
            job.current_stage = current_stage
        if error_message:
            job.error_message = error_message
        if model_path:
            job.model_path = model_path
        if metrics:
            job.metrics.update(metrics)

        job.updated_at = datetime.now()
        self._save_job(job)

    async def _run_training(
        self,
        job: TrainingJob,
        sample_count: int,
        epochs: int,
        batch_size: int,
    ):
        """Run the full training pipeline."""
        try:
            from .tts_generator import SampleGenerator, AudioAugmenter
            from .model import ModelConfig, WakeWordTrainer, NegativeSampleGenerator
            from .config import settings

            word = job.word
            word_dir = word.lower().replace(" ", "_")

            # Stage 1: Generate TTS samples
            self._update_job(
                job,
                status=JobStatus.GENERATING_SAMPLES,
                progress=0,
                current_stage="Generating TTS samples",
            )

            generator = SampleGenerator(
                output_dir=self.samples_dir / word_dir,
                enable_gtts=settings.tts_gtts_enabled,
                enable_edge=settings.tts_edge_enabled,
                enable_pyttsx3=settings.tts_pyttsx3_enabled,
                enable_chatterbox=settings.tts_chatterbox_enabled,
                chatterbox_voices_dir=(
                    settings.voices_dir if settings.tts_chatterbox_enabled else None
                ),
            )

            def sample_progress(current, total, stage):
                progress = (current / total) * 25  # 0-25%
                self._update_job(job, progress=progress)

            samples = await generator.generate_samples(
                word,
                target_count=sample_count,
                progress_callback=sample_progress,
            )

            if len(samples) < 10:
                raise RuntimeError(f"Only generated {len(samples)} samples, need at least 10")

            # Stage 2: Augment samples
            self._update_job(
                job,
                status=JobStatus.AUGMENTING,
                progress=25,
                current_stage="Augmenting samples",
            )

            augmenter = AudioAugmenter()

            def augment_progress(current, total, stage):
                progress = 25 + (current / total) * 15  # 25-40%
                self._update_job(job, progress=progress)

            positive_samples = await augmenter.augment_dataset(
                samples,
                augmentations_per_sample=5,
                progress_callback=augment_progress,
            )

            # Stage 3: Generate negative samples
            self._update_job(
                job,
                progress=40,
                current_stage="Generating negative samples",
            )

            neg_generator = NegativeSampleGenerator()

            # Noise samples
            noise_samples = neg_generator.generate_noise_samples(count=len(positive_samples) // 4)

            # Similar word samples
            similar_samples = await neg_generator.generate_similar_words(
                word,
                generator,
                count=len(positive_samples) // 4,
            )

            negative_samples = noise_samples + similar_samples

            logger.info(
                f"Dataset prepared: {len(positive_samples)} positive, "
                f"{len(negative_samples)} negative samples"
            )

            # Stage 4: Train model
            self._update_job(
                job,
                status=JobStatus.TRAINING,
                progress=50,
                current_stage="Training model",
            )

            config = ModelConfig(word=word)
            trainer = WakeWordTrainer(config)

            def training_progress(epoch, total_epochs, metrics):
                progress = 50 + (epoch / total_epochs) * 40  # 50-90%
                self._update_job(
                    job,
                    progress=progress,
                    metrics=metrics,
                )

            history = trainer.train(
                positive_samples=positive_samples,
                negative_samples=negative_samples,
                epochs=epochs,
                batch_size=batch_size,
                progress_callback=training_progress,
            )

            # Stage 5: Export model
            self._update_job(
                job,
                status=JobStatus.EXPORTING,
                progress=90,
                current_stage="Exporting model",
            )

            model_dir = self.models_dir / word_dir
            saved_files = trainer.save(model_dir)

            # Complete
            final_metrics = {
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "final_train_acc": history["train_acc"][-1],
                "final_val_acc": history["val_acc"][-1],
                "positive_samples": len(positive_samples),
                "negative_samples": len(negative_samples),
                "exported_formats": list(saved_files.keys()),
            }

            self._update_job(
                job,
                status=JobStatus.COMPLETED,
                progress=100,
                current_stage="Complete",
                model_path=str(model_dir),
                metrics=final_metrics,
            )

            # Remove from word tracking
            word_lower = word.lower()
            if self.word_to_job.get(word_lower) == job.job_id:
                del self.word_to_job[word_lower]

            logger.info(f"Training completed for job {job.job_id}")

        except asyncio.CancelledError:
            logger.info(f"Job {job.job_id} was cancelled")
            raise
        except Exception as e:
            logger.error(f"Training failed for job {job.job_id}: {e}")
            self._update_job(
                job,
                status=JobStatus.FAILED,
                error_message=str(e),
            )

            # Remove from word tracking
            word_lower = job.word.lower()
            if self.word_to_job.get(word_lower) == job.job_id:
                del self.word_to_job[word_lower]
