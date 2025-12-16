FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional TTS dependencies
RUN pip install --no-cache-dir \
    gtts \
    edge-tts \
    pyttsx3

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/samples /app/cache /app/voices

# Expose port
EXPOSE 8000

# Environment variables
ENV MODELS_DIR=/app/models
ENV SAMPLES_DIR=/app/samples
ENV CACHE_DIR=/app/cache
ENV VOICES_DIR=/app/voices
ENV HOST=0.0.0.0
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python", "-m", "uvicorn", "wakeword.app:app", "--host", "0.0.0.0", "--port", "8000"]
