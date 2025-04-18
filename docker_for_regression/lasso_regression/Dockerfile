FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create directories for models and MLflow
RUN mkdir -p saved_models/lasso_regression mlruns

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_EXPERIMENT_NAME=regression_experiment
ENV PORT=5042
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV SAVED_MODELS_DIR=/app/saved_models/lasso_regression

# Expose the port
EXPOSE 5042

# Command to run the service
CMD ["python", "app.py"] 