FROM tensorflow/tensorflow:2.5.0-gpu

# Set working directory
WORKDIR /app

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

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy model code
COPY . /app/

# Create directories for data and trained models
RUN mkdir -p /app/data/datasets
RUN mkdir -p /app/data/trained_models

# Expose port
EXPOSE 5100

# Set environment variables
ENV MODEL_PORT=5100
ENV MODEL_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Run the service
CMD ["python3", "service.py"] 