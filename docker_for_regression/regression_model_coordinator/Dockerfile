FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and test script
COPY app.py .
COPY test_db_connection.py .

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

# Add script to check database connection and run the service
RUN echo '#!/bin/bash \n\
echo "Testing database connection..." \n\
python test_db_connection.py || echo "Warning: Database connection failed but continuing anyway" \n\
echo "Starting regression model coordinator service..." \n\
exec python app.py' > /app/start.sh

RUN chmod +x /app/start.sh

# Command to run the service
CMD ["/app/start.sh"] 