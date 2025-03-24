FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_efficientnet_b0.txt .
RUN pip install --no-cache-dir -r requirements_efficientnet_b0.txt

# Copy application code
COPY model_service.py .

# Expose port
EXPOSE 5021

# Run the application
CMD ["python", "model_service.py"] 