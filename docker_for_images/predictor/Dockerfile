# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies (unzip is needed to extract ZIP files)
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies for our API
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask application code
COPY predictor.py /app/predictor.py

# Expose port 5100 for the Flask app
EXPOSE 5100

# Run the application (you can change this as needed)
CMD ["python", "predictor.py"]
