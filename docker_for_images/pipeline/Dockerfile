FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=5025
ENV AUGMENTATION_SERVICE_URL=http://augmentation-service:5023
ENV MODEL_TRAINING_SERVICE_URL=http://model-training-service:5021

EXPOSE 5025

CMD ["python", "app.py"] 