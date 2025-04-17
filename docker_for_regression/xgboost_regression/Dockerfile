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
RUN mkdir -p /app/saved_models/xgboost_regression

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5046
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV SAVED_MODELS_DIR=/app/saved_models/xgboost_regression

# Expose the port
EXPOSE 5046

# Command to run the service
CMD ["python", "app.py"] 