FROM python:3.9-slim

WORKDIR /app

# Create logs directory
RUN mkdir -p /app/logs

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5001
# Don't set OPENAI_API_KEY here - it will be passed from docker-compose

# Expose the port
EXPOSE ${PORT}

# Run the application
CMD ["python", "app.py"] 