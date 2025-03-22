FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_eep.txt .
RUN pip install --no-cache-dir -r requirements_eep.txt

# Copy application code
COPY image_eep.py .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "image_eep.py"] 