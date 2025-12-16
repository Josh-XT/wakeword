"""
Tests for configuration module.
"""

import os
import pytest
from wakeword.config import Settings


class TestSettings:
    """Test configuration settings."""

    def test_default_settings(self):
        """Test default settings are properly set."""
        settings = Settings()

        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.default_sample_count == 500
        assert settings.default_epochs == 50
        assert settings.tts_gtts_enabled is True
        assert settings.tts_edge_enabled is True

    def test_settings_from_env(self, monkeypatch):
        """Test settings can be loaded from environment variables."""
        monkeypatch.setenv("HOST", "127.0.0.1")
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("DEFAULT_SAMPLE_COUNT", "100")
        monkeypatch.setenv("TTS_GTTS_ENABLED", "false")

        settings = Settings()

        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.default_sample_count == 100
        assert settings.tts_gtts_enabled is False

    def test_settings_directories(self, test_settings, temp_dir):
        """Test directory settings point to temp directories."""
        assert temp_dir.as_posix() in str(test_settings.models_dir)
        assert temp_dir.as_posix() in str(test_settings.samples_dir)
