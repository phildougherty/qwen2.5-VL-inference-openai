FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    ninja-build \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python packages in specific order
RUN pip3 install --no-cache-dir packaging && \
    pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch torchvision && \
    pip3 install --no-cache-dir ninja && \
    pip3 install --no-cache-dir flash-attn --no-build-isolation && \
    pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/models /app/webhook_logs

# Copy application code
COPY . .

# Expose port
EXPOSE 9192

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:9192/health || exit 1

# Start the application
CMD ["python3", "app.py"]