# Object Detection Service with YOLOv5

## Overview

The Object Detection Service provides an API for fine-tuning YOLOv5 models with your custom dataset. This service allows you to train object detection models that can identify specific objects in images, with the resulting model being immediately available for download and deployment.

## Features

- Fine-tune pre-trained YOLOv5 models on custom datasets
- Simplified training process with 5 preset levels
- Returns the trained model as a ZIP file with performance metrics
- No persistent storage of your models or data (privacy-focused)
- Automatic configuration file generation

## API Endpoints

### Health Check

`GET /health`

Returns the health status of the object detection service.

**Response:**
```json
{
  "status": "healthy",
  "service": "object-detection-service"
}
```

### Fine-tune YOLOv5 Model

`POST /finetune`

Fine-tunes a YOLOv5 model on your dataset.

**Form Parameters:**
- `zipFile`: ZIP file containing dataset in YOLOv5 format
- `level`: Training level (1-5) with predefined configurations:
  - Level 1: Nano model, 20 epochs, 320px resolution
  - Level 2: Small model, 30 epochs, 416px resolution
  - Level 3: Small model, 50 epochs, 640px resolution
  - Level 4: Medium model, 80 epochs, 640px resolution
  - Level 5: Large model, 100 epochs, 640px resolution

**Response:**
- Success: Returns a ZIP file containing the fine-tuned model and training results
- Error: Returns a JSON object with an error message

## Dataset Format

Your dataset must follow this exact structure:

```
your_dataset.zip
└── dataset_folder/  (can be any name)
    ├── train/
    │   ├── images/  (JPG/PNG files)
    │   └── labels/  (TXT files in YOLO format)
    └── valid/
        ├── images/  (JPG/PNG files)
        └── labels/  (TXT files in YOLO format)
```

**Important notes:**
- The folder name must be exactly "valid" (not "val") for validation data
- No data.yaml file is required - it will be generated automatically
- The service automatically detects class IDs from your label files

Labels should follow the YOLOv5 format (normalized coordinates):
```
<class_id> <x_center> <y_center> <width> <height>
```

Example: `0 0.5 0.5 0.25 0.25` (class 0, center at (0.5, 0.5), width 0.25, height 0.25)

## Docker Setup

The service is containerized using Docker and based on a Python image with YOLOv5 dependencies.

### Environment Variables

- `PORT`: The port on which the service runs (default: 5027)

## Usage Example

```python
import requests

# URL of the object detection service
url = "http://localhost:5027/finetune"

# Prepare the form data
files = {
    'zipFile': ('dataset.zip', open('dataset.zip', 'rb'), 'application/zip')
}

data = {
    'level': '3'  # Choose training level (1-5)
}

# Send the request
response = requests.post(url, files=files, data=data)

# Check if successful
if response.status_code == 200:
    # Save the model zip file
    with open('yolov5_model.zip', 'wb') as f:
        f.write(response.content)
    print("Model downloaded successfully")
else:
    # Handle error
    error = response.json()
    print(f"Error: {error['error']}")
```

## Integration with DeepMed

This service is integrated with the DeepMed platform and can be accessed through the Object Detection tab in the user interface. The UI provides a user-friendly way to configure and start the fine-tuning process. 