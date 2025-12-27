# ============================================================================
# Hateful Meme Detection - Dockerfile
# ============================================================================
# Multi-stage build for optimized image size

# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# ============================================================================
# Stage 2: Dependencies
# ============================================================================
FROM base as dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# ============================================================================
# Stage 3: Production image
# ============================================================================
FROM dependencies as production

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Create directories for models and data
RUN mkdir -p models data outputs

# Expose port for API (if using)
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "discord_bot/bot.py"]

# ============================================================================
# Stage 4: Development image
# ============================================================================
FROM dependencies as development

# Install development dependencies
RUN pip install pytest pytest-cov black flake8 isort jupyter

# Copy application code
COPY --chown=appuser:appuser . .

USER appuser

# Expose Jupyter port
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

# ============================================================================
# GPU-enabled image (build with: docker build --target gpu -t hateful-meme-gpu .)
# ============================================================================
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 as gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create user and working directory
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install -r requirements.txt

# Copy application
COPY --chown=appuser:appuser . .

USER appuser
RUN mkdir -p models data outputs

EXPOSE 8000
CMD ["python3", "discord_bot/bot.py"]
