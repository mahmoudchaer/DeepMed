#!/bin/bash

echo "Stopping existing containers..."
docker-compose down

echo "Rebuilding and starting containers with new configuration..."
docker-compose build
docker-compose up -d

echo "Checking service status..."
echo "Please wait a moment for all services to start up..."
sleep 10

echo "Model Training Service (5100): "
curl -s http://localhost:5100/health || echo "Not responding"

echo "Data Processing Service (5101): "
curl -s http://localhost:5101/health || echo "Not responding"

echo "Data Augmentation Service (5102): "
curl -s http://localhost:5102/health || echo "Not responding"

echo "Image Processing Service (5103): "
curl -s http://localhost:5103/health || echo "Not responding"

echo "All services should be available at these ports:"
echo "- Model Training:     http://localhost:5100"
echo "- Data Processing:    http://localhost:5101"
echo "- Data Augmentation:  http://localhost:5102"
echo "- Image Processing:   http://localhost:5103" 