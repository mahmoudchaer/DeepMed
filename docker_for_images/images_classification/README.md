# Model Training Service Docker Container

This Docker container provides a service for training image classification models with PyTorch and EfficientNet B0.

## Contents

- `model_service.py`: The Flask application that handles model training
- `Dockerfile`: Instructions for building the Docker image
- `docker-compose.yml`: Configuration for running the service
- `requirements.txt`: Python dependencies

## Usage

1. Build and start the service:

```bash
docker-compose up -d
```

2. The service will be available at `http://localhost:5020`

## API Endpoints

### Health Check

- **URL**: `/health`
- **Method**: `GET`
- **Response**: JSON with service status

### Train Model

- **URL**: `/train`
- **Method**: `POST`
- **Form Data Parameters**:
  - `zipFile`: ZIP file containing folders of images (each folder represents a class)
  - `numClasses`: Number of classes to train for (default: 5)
  - `trainingLevel`: Training level from 1-5 (default: 3)
- **Response**: 
  - The trained model file
  - Metrics in the `X-Training-Metrics` header

## Training Levels

The service supports 5 different training levels:

1. **Level 1**: Fast training, lowest accuracy (1 epoch)
2. **Level 2**: Balanced speed/accuracy (2 epochs)
3. **Level 3**: Standard training (3 epochs)
4. **Level 4**: Extended training (5 epochs)
5. **Level 5**: Thorough training, highest accuracy (8 epochs) 