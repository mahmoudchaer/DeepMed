FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_augmentation.txt .
RUN pip install --no-cache-dir -r requirements_augmentation.txt

# Copy application code
COPY data_augmentation_service.py .

# Expose port
EXPOSE 5111

# Run the application
CMD ["python", "data_augmentation_service.py"] 