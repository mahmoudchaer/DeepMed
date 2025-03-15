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
        unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model code
COPY . .

# Create directories for datasets and trained models
RUN mkdir -p /app/data/datasets /app/data/trained_models

# Expose port for the service
EXPOSE 5100

# Set environment variables
ENV MODEL_PORT=5100
ENV MODEL_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Command to run the service
CMD ["python3", "service.py"] 