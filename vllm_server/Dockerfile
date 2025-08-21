# vLLM Server with LoRA Support
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHON_VERSION=3.10
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-pip \
    python${PYTHON_VERSION}-dev \
    build-essential \
    git \
    wget \
    curl \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/pip${PYTHON_VERSION} /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM and other dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and cache
RUN mkdir -p /app/models /app/lora_cache /app/logs

# Set environment variables for the application
ENV PYTHONPATH=/app
ENV VLLM_CACHE_DIR=/app/lora_cache
ENV HF_HOME=/app/models/.cache
ENV TRANSFORMERS_CACHE=/app/models/.cache

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/ || exit 1

# Run the application
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8001"]