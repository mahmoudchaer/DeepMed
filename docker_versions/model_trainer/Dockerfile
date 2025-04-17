FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create directory for saved models
RUN mkdir -p saved_models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5004

# Expose the port
EXPOSE ${PORT}

# Volume for persistent model storage
VOLUME ["/app/saved_models"]

# Run the application
CMD ["python", "app.py"] 