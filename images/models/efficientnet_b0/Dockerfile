FROM tensorflow/tensorflow:2.5.0-gpu

# Set working directory
WORKDIR /app

# Fix the NVIDIA GPG key issue
RUN apt-key del A4B469963BF863CC || true \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub | apt-key add -

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3-pip \
        python3-dev \
        git \
        wget \
        unzip \
        vim \
        less && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for datasets, trained models, and logs
RUN mkdir -p /app/data/datasets /app/data/trained_models /app/logs

# Copy model code
COPY . .

# Create an __init__.py file to make this directory a Python package
RUN touch /app/__init__.py

# Make log files writable
RUN touch /app/efficientnet_service.log /app/efficientnet_api.log && \
    chmod 666 /app/efficientnet_service.log /app/efficientnet_api.log

# Expose port for the service
EXPOSE 5100

# Set environment variables
ENV MODEL_PORT=5100
ENV MODEL_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV LOGLEVEL=DEBUG
ENV PYTHONPATH=/app

# Command to run the service with verbose logging
CMD ["bash", "-c", "echo 'Starting EfficientNet-B0 service with verbose logging...' && python3 -u service.py 2>&1 | tee -a /app/logs/service.log"] 