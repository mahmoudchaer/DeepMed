FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create logs directory
RUN mkdir -p logs

# Copy application code
COPY app.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5002

# Expose the port
EXPOSE ${PORT}

# Run the application
CMD ["python", "app.py"] 