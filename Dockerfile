FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables for regression services
ENV REGRESSION_DATA_CLEANER_URL=http://regression-data-cleaner:5031
ENV REGRESSION_FEATURE_SELECTOR_URL=http://regression-feature-selector:5032
ENV REGRESSION_MODEL_COORDINATOR_URL=http://regression-model-coordinator:5040
ENV REGRESSION_PREDICTOR_SERVICE_URL=http://regression-predictor-service:5050

# Expose the port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"] 