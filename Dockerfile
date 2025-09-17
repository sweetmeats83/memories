FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Set environment so CUDA/cuDNN libraries are found
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install system deps (Python, ffmpeg, cuDNN runtime)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        tzdata \
        libcudnn8=8.9.2.* \
        libcudnn8-dev=8.9.2.* \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Pin compatible ctranslate2 BEFORE faster-whisper
RUN pip install --no-cache-dir \
        ctranslate2==4.4.0 \
        faster-whisper

# Copy project files
WORKDIR /app
COPY . /app

# Default container timezone (overridden at runtime by passing TZ env)
ENV TZ=Etc/UTC

# Install Python dependencies from requirements.txt if it exists
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Expose port (change if your app uses a different port)
EXPOSE 8000

# Run the app (adjust to your startup command)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

