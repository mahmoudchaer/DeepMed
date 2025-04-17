FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Create directories for models and MLflow
RUN mkdir -p saved_models/svm mlruns

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_EXPERIMENT_NAME=svm_model
ENV PORT=5013

EXPOSE 5013

CMD ["python", "app.py"] 