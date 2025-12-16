"""
WakeWord Server - FastAPI Application

REST API for training and serving wake word detection models.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .job_manager import JobManager, JobStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="WakeWord Server",
    description="Train and serve custom wake word detection models",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize job manager
job_manager: Optional[JobManager] = None


@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    global job_manager
    
    job_manager = JobManager(
        jobs_dir=settings.cache_dir / "jobs",
        models_dir=settings.models_dir,
        samples_dir=settings.samples_dir,
    )
    
    logger.info("WakeWord server started")
    logger.info(f"Models directory: {settings.models_dir}")
    logger.info(f"Samples directory: {settings.samples_dir}")


# ============================================================================
# Request/Response Models
# ============================================================================

class TrainRequest(BaseModel):
    """Request to train a new wake word model."""
    word: str = Field(..., description="The wake word to train", min_length=1, max_length=50)
    sample_count: int = Field(default=500, ge=50, le=2000, description="Number of base samples to generate")
    epochs: int = Field(default=50, ge=10, le=200, description="Training epochs")
    batch_size: int = Field(default=32, ge=8, le=128, description="Training batch size")


class TrainResponse(BaseModel):
    """Response after requesting training."""
    job_id: str
    word: str
    status: str
    message: str
    estimated_minutes: Optional[int] = None
    check_status_url: str


class JobStatusResponse(BaseModel):
    """Response with job status details."""
    job_id: str
    word: str
    status: str
    progress: float
    current_stage: str
    error_message: Optional[str] = None
    estimated_completion: Optional[str] = None
    model_path: Optional[str] = None
    metrics: dict = {}
    created_at: str
    updated_at: str


class ModelInfo(BaseModel):
    """Information about an available model."""
    word: str
    directory: str
    config: dict
    files: dict
    created_at: str


class ModelListResponse(BaseModel):
    """Response listing available models."""
    models: List[ModelInfo]
    total: int


class PredictRequest(BaseModel):
    """Request to predict if audio contains wake word."""
    audio_base64: str = Field(..., description="Base64-encoded audio data (WAV format)")


class PredictResponse(BaseModel):
    """Response from prediction."""
    detected: bool
    confidence: float
    word: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API root - health check."""
    return {
        "service": "WakeWord Server",
        "version": "0.1.0",
        "status": "healthy",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# ----------------------------------------------------------------------------
# Model Management
# ----------------------------------------------------------------------------

@app.get("/models", response_model=ModelListResponse)
async def list_models():
    """List all available trained models."""
    models = job_manager.list_available_models()
    return ModelListResponse(
        models=[ModelInfo(**m) for m in models],
        total=len(models),
    )


@app.get("/models/{word}")
async def get_model(word: str, format: str = Query("pytorch", regex="^(pytorch|onnx|tflite)$")):
    """
    Get a trained model for a word.
    
    If the model doesn't exist and no training is in progress, returns 404.
    If training is in progress, returns 202 with job status.
    If model exists, returns the model file.
    """
    word_lower = word.lower().replace(" ", "_")
    
    # Check if model exists
    model_dir = job_manager.get_model_for_word(word)
    
    if model_dir:
        # Model exists - return the file
        format_to_ext = {
            "pytorch": "pt",
            "onnx": "onnx",
            "tflite": "tflite",
        }
        ext = format_to_ext.get(format, "pt")
        model_file = model_dir / f"model.{ext}"
        
        if not model_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model format '{format}' not available for word '{word}'"
            )
        
        return FileResponse(
            path=model_file,
            filename=f"{word_lower}_model.{ext}",
            media_type="application/octet-stream",
        )
    
    # Check if training is in progress
    job = job_manager.get_job_for_word(word)
    if job and job.status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        return JSONResponse(
            status_code=202,
            content={
                "status": "training_in_progress",
                "job_id": job.job_id,
                "progress": job.progress,
                "current_stage": job.current_stage,
                "message": f"Model for '{word}' is being trained. Check back soon.",
                "check_status_url": f"/jobs/{job.job_id}",
            }
        )
    
    # Model doesn't exist
    raise HTTPException(
        status_code=404,
        detail=f"No model found for word '{word}'. Use POST /train to create one."
    )


@app.get("/models/{word}/config")
async def get_model_config(word: str):
    """Get the configuration for a trained model."""
    model_dir = job_manager.get_model_for_word(word)
    
    if not model_dir:
        raise HTTPException(
            status_code=404,
            detail=f"No model found for word '{word}'"
        )
    
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model config not found"
        )
    
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


@app.delete("/models/{word}")
async def delete_model(word: str):
    """Delete a trained model."""
    model_dir = job_manager.get_model_for_word(word)
    
    if not model_dir:
        raise HTTPException(
            status_code=404,
            detail=f"No model found for word '{word}'"
        )
    
    import shutil
    shutil.rmtree(model_dir)
    
    return {"message": f"Model for '{word}' deleted successfully"}


# ----------------------------------------------------------------------------
# Training Jobs
# ----------------------------------------------------------------------------

@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Request training of a new wake word model.
    
    If a model already exists for the word, returns info about it.
    If training is already in progress, returns the job status.
    Otherwise, starts a new training job.
    """
    word = request.word.strip()
    
    # Check if model already exists
    model_dir = job_manager.get_model_for_word(word)
    if model_dir:
        return TrainResponse(
            job_id="",
            word=word,
            status="completed",
            message=f"Model for '{word}' already exists. Use GET /models/{word} to download.",
            check_status_url=f"/models/{word}",
        )
    
    # Check if training is already in progress
    existing_job = job_manager.get_job_for_word(word)
    if existing_job and existing_job.status not in [
        JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED
    ]:
        return TrainResponse(
            job_id=existing_job.job_id,
            word=word,
            status=existing_job.status.value,
            message=f"Training already in progress for '{word}'",
            estimated_minutes=15,
            check_status_url=f"/jobs/{existing_job.job_id}",
        )
    
    # Start new training job
    try:
        job = await job_manager.create_job(
            word=word,
            sample_count=request.sample_count,
            epochs=request.epochs,
            batch_size=request.batch_size,
        )
        
        return TrainResponse(
            job_id=job.job_id,
            word=word,
            status=job.status.value,
            message=f"Training started for '{word}'. This typically takes 10-20 minutes.",
            estimated_minutes=15,
            check_status_url=f"/jobs/{job.job_id}",
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/jobs", response_model=List[JobStatusResponse])
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
):
    """List training jobs."""
    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {[s.value for s in JobStatus]}"
            )
    
    jobs = job_manager.list_jobs(status=status_filter, limit=limit)
    
    return [
        JobStatusResponse(
            job_id=j.job_id,
            word=j.word,
            status=j.status.value,
            progress=j.progress,
            current_stage=j.current_stage,
            error_message=j.error_message,
            estimated_completion=j.estimated_completion.isoformat() if j.estimated_completion else None,
            model_path=j.model_path,
            metrics=j.metrics,
            created_at=j.created_at.isoformat(),
            updated_at=j.updated_at.isoformat(),
        )
        for j in jobs
    ]


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    """Get status of a specific training job."""
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        job_id=job.job_id,
        word=job.word,
        status=job.status.value,
        progress=job.progress,
        current_stage=job.current_stage,
        error_message=job.error_message,
        estimated_completion=job.estimated_completion.isoformat() if job.estimated_completion else None,
        model_path=job.model_path,
        metrics=job.metrics,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running training job."""
    success = await job_manager.cancel_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Could not cancel job (may already be completed or cancelled)"
        )
    
    return {"message": "Job cancelled successfully"}


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------

@app.post("/predict/{word}", response_model=PredictResponse)
async def predict(word: str, request: PredictRequest):
    """
    Predict if audio contains the wake word.
    
    Requires a trained model for the word.
    """
    import base64
    
    model_dir = job_manager.get_model_for_word(word)
    
    if not model_dir:
        raise HTTPException(
            status_code=404,
            detail=f"No model found for word '{word}'. Train one first with POST /train"
        )
    
    try:
        # Decode audio
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Load model and predict
        from .model import WakeWordTrainer
        trainer = WakeWordTrainer.load(model_dir)
        
        detected, confidence = trainer.predict(audio_bytes)
        
        return PredictResponse(
            detected=detected,
            confidence=confidence,
            word=word,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def main():
    """Run the server."""
    import uvicorn
    
    uvicorn.run(
        "wakeword.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
