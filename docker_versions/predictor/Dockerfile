# Use an official slim Python image
FROM python:3.9-slim

# Install system dependencies; unzip is needed to extract the ZIP archive.
RUN apt-get update && apt-get install -y --no-install-recommends unzip && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies for our predictor API.
# (These packages are for the service itself, not the model package.)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the predictor API code
COPY predictor.py /app/predictor.py

# Expose port 5101 for this tabular prediction service
EXPOSE 5101

# Run the application
CMD ["python", "predictor.py"]
