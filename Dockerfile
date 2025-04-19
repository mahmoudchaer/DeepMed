FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p static/downloads
RUN mkdir -p templates
RUN mkdir -p db
RUN mkdir -p temp_storage

# Copy application files
COPY app_api.py .
COPY app_tabular.py .
COPY app_images.py .
COPY app_others.py .
COPY app_regression.py . 

# Copy db module
COPY db/ db/

# Copy templates and static files
COPY templates/ templates/
COPY static/ static/

# Set appropriate permissions
RUN chmod -R 755 static
RUN chmod -R 755 templates
RUN chmod -R 777 static/downloads
RUN chmod -R 777 temp_storage

# Set environment variables
ENV FLASK_APP=app_api.py
ENV FLASK_ENV=production
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 5000

# Create volume for persistent data
VOLUME ["/app/static/downloads", "/app/temp_storage"]

# Command to run the application
CMD ["python", "app_api.py"] 