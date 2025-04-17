FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Create directories for models and MLflow
RUN mkdir -p saved_models/logistic_regression mlruns

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_EXPERIMENT_NAME=logistic_regression_model
ENV PORT=5010

EXPOSE 5010

CMD ["python", "app.py"] 