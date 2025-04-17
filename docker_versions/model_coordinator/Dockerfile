FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir azure-storage-blob python-dotenv

# Copy application code and required files
COPY app.py .
COPY storage.py .
COPY db/ ./db/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5020
ENV IS_DOCKER=true

# Expose the port
EXPOSE ${PORT}

# Run the application
CMD ["python", "app.py"] 