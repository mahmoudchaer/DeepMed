FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY test_db_connection.py .

# Create directories for models and MLflow
RUN mkdir -p logs mlruns saved_models

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5040
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV IS_DOCKER=true
ENV MLFLOW_EXPERIMENT_NAME=regression_experiment

# Modified to use environment variables from docker-compose instead of hardcoding credentials - rebuild required

# Expose the port
EXPOSE 5040

# Simple startup script
RUN echo '#!/bin/bash \n\
echo "Testing database connection..." \n\
python test_db_connection.py || echo "Warning: Database connection failed but continuing anyway" \n\
echo "Starting regression model coordinator service..." \n\
exec python app.py' > /app/start.sh

RUN chmod +x /app/start.sh

# Command to run the service
CMD ["/app/start.sh"] 