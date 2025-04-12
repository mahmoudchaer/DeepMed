# Object Detection Service with YOLOv5

## Overview

The Object Detection Service provides an API for fine-tuning YOLOv5 models with your custom dataset. This service allows you to train object detection models that can identify specific objects in images, with the resulting model being immediately available for download and deployment.

## Features

- Fine-tune pre-trained YOLOv5 models on custom datasets
- Supports multiple model sizes (nano, small, medium, large, x)
- Configurable training parameters (epochs, batch size, image size)
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
- `modelSize`: Size of YOLOv5 model ("nano", "small", "medium", "large", "x")
- `epochs`: Number of training epochs (1-300)
- `batchSize`: Training batch size (1-64)
- `imgSize`: Image size for training (e.g., 640)

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
└── val/          # Validation images
    ├── images/   # JPG files
    └── labels/   # TXT files (YOLO format)
```

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
    'modelSize': 'small',
    'epochs': '50',
    'batchSize': '16',
    'imgSize': '640'
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