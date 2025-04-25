FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY generate_cert.py .

# Create a directory for certificates
RUN mkdir -p /app/certs

# Environment variables
ENV PORT=5025
ENV AUGMENTATION_SERVICE_URL=http://augmentation-service:5023
ENV MODEL_TRAINING_SERVICE_URL=http://model-training-service:5021
ENV SSL_CERT_PATH=/app/certs/cert.pem
ENV SSL_KEY_PATH=/app/certs/key.pem
ENV USE_HTTPS=true

EXPOSE 5025

# Generate certificates and start the app
CMD python generate_cert.py && python app.py 