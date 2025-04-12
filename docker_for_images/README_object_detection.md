# Object Detection Service with YOLOv5

## Overview

The Object Detection Service provides an API for fine-tuning YOLOv5 models with your custom dataset. This service allows you to train object detection models that can identify specific objects in images, with the resulting model being immediately available for download and deployment.

## Features

- Fine-tune pre-trained YOLOv5 models on custom datasets
- Simplified training process with 5 preset levels
- Returns the trained model as a ZIP file with performance metrics
- No persistent storage of your models or data (privacy-focused)

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

YOLOv5 requires a specific dataset format:

```
dataset/
├── data.yaml     # Dataset configuration
├── train/        # Training images
│   ├── images/   # JPG files
│   └── labels/   # TXT files (YOLO format)
└── val/          # Validation images (used for testing)
    ├── images/   # JPG files
    └── labels/   # TXT files (YOLO format)
```

Note: YOLOv5 uses the validation set as the test set during training, so a separate test directory is not required.

The `data.yaml` file should define:

```yaml
path: ../dataset  # dataset root dir
train: train/images  # train images
val: val/images  # val images
nc: 3  # number of classes
names: ['person', 'car', 'dog']  # class names
```

Labels should follow the YOLOv5 format (normalized coordinates):
```
<class_id> <x_center> <y_center> <width> <height>
```

## Docker Setup

The service is containerized using Docker and based on the official ultralytics/yolov5 image.

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