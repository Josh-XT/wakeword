"""
WakeWord Server - Configuration
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Directories
    models_dir: Path = Path("./models")
    samples_dir: Path = Path("./samples")
    cache_dir: Path = Path("./cache")
    voices_dir: Path = Path("./voices")

    # Training defaults
    default_sample_count: int = 500
    default_epochs: int = 50
    default_batch_size: int = 32

    # TTS engines
    tts_gtts_enabled: bool = True
    tts_edge_enabled: bool = True
    tts_pyttsx3_enabled: bool = True
    tts_chatterbox_enabled: bool = False

    # Redis (optional)
    redis_url: Optional[str] = None

    # GPU
    cuda_visible_devices: str = "0"
    force_cpu: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
