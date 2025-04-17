FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create directories
RUN mkdir -p /app/logs
RUN mkdir -p /app/mlruns

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5040
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV IS_DOCKER=true

# Modified to use environment variables from docker-compose instead of hardcoding credentials - rebuild required

# Expose the port
EXPOSE 5040

# Command to run the service
CMD ["python", "app.py"] 