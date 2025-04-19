FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY predictor.py .

# Create log directory
RUN mkdir -p /app/logs

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5050

# Add Azure Blob Storage environment variables - will be passed from docker-compose
ENV AZURE_STORAGE_ACCOUNT=""
ENV AZURE_STORAGE_KEY=""
ENV AZURE_CONTAINER=""

# Expose the port
EXPOSE 5050

# Command to run the service
CMD ["python", "predictor.py"] 