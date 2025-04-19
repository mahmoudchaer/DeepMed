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

# Copy application files
COPY app_api.py .
COPY app_tabular.py .
COPY app_images.py .
COPY app_others.py .

# Create necessary directories
RUN mkdir -p static/downloads
RUN mkdir -p db

# Copy db module if it exists (adjust as needed)
COPY db/ db/

# Copy templates and static files if they exist (adjust as needed)
COPY templates/ templates/
COPY static/ static/

# Set environment variables
ENV FLASK_APP=app_api.py
ENV FLASK_ENV=production
ENV PORT=5000

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app_api.py"] 