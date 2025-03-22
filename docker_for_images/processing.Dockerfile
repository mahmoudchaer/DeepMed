FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_processing.txt .
RUN pip install --no-cache-dir -r requirements_processing.txt

# Copy application code
COPY data_processing_service.py .

# Expose port
EXPOSE 5012

# Run the application
CMD ["python", "data_processing_service.py"] 