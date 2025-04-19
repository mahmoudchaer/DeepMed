# Image Processing Pipeline Service

## Overview

The Image Processing Pipeline Service is an integrated service that combines data augmentation and model training into a seamless workflow. It provides a unified endpoint that allows users to either:

1. Directly train a model on their dataset
2. First augment their dataset and then train a model on the augmented data

This service acts as a coordinator between the Augmentation Service and Model Training Service, handling the data transfer between them and presenting a simplified interface to the user.

## API Endpoints

### Health Check

`GET /health`

Returns the health status of the pipeline service and its dependent services.

**Response:**
```json
{
  "status": "healthy",
  "service": "pipeline-service",
  "dependencies": {
    "pipeline_service": "healthy",
    "augmentation_service": "healthy",
    "model_training_service": "healthy"
  }
}
```

### Process Pipeline

`POST /pipeline`

Processes a dataset through the pipeline, optionally performing augmentation before training.

**Form Parameters:**
- `zipFile`: ZIP file containing folders of images (one folder per class)
- `performAugmentation`: "true" or "false" to indicate whether to perform augmentation
- `augmentationLevel`: Augmentation level (1-5) which controls both transformation intensity and number of augmentations
- `numClasses`: Number of classes in the dataset
- `trainingLevel`: Training level (1-5) for model training

**Response:**
- Success: Returns the trained model file (PyTorch .pt file)
- Error: Returns a JSON object with an error message

## Docker Setup

The service is containerized using Docker and integrated with the other services through docker-compose.

### Environment Variables

- `PORT`: The port on which the pipeline service runs (default: 5025)
- `AUGMENTATION_SERVICE_URL`: URL of the augmentation service (default: http://augmentation-service:5023)
- `MODEL_TRAINING_SERVICE_URL`: URL of the model training service (default: http://model-training-service:5021)

## Usage Example

```python
import requests

# URL of the pipeline service
url = "http://localhost:5025/pipeline"

# Prepare the form data
files = {
    'zipFile': ('dataset.zip', open('dataset.zip', 'rb'), 'application/zip')
}

data = {
    'performAugmentation': 'true',
    'augmentationLevel': '3',
    'numClasses': '5',
    'trainingLevel': '3'
}

# Send the request
response = requests.post(url, files=files, data=data)

# Check if successful
if response.status_code == 200:
    # Save the model file
    with open('trained_model.pt', 'wb') as f:
        f.write(response.content)
    
    # Get metrics if available
    metrics = None
    if 'X-Training-Metrics' in response.headers:
        metrics = response.headers['X-Training-Metrics']
        print(f"Training metrics: {metrics}")
else:
    # Handle error
    error = response.json()
    print(f"Error: {error['error']}")
```

## Integration with DeepMed

This service is integrated with the DeepMed platform and can be accessed through the Pipeline tab in the user interface. The UI provides a user-friendly way to configure and start the pipeline process. 