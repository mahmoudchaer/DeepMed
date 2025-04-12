# Anomaly Detection Service with PyTorch Autoencoder

## Overview

The Anomaly Detection Service provides an API for training custom autoencoder models that can detect anomalies in images. Unlike traditional classification models which require both normal and anomalous samples, this unsupervised approach learns what "normal" looks like and can detect deviations, making it ideal for scenarios where anomalous samples are rare.

The service is built using PyTorch and leverages convolutional autoencoders to learn compressed representations of normal images. During inference, samples that cannot be accurately reconstructed are flagged as potential anomalies.

## Features

- Train custom anomaly detection models on your image datasets
- Configurable training levels to balance speed and accuracy
- Adjustable image size parameters
- Automatic threshold calculation for anomaly detection
- Simple REST API for training and inference

## API Endpoints

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "anomaly-detection-service"
}
```

### `POST /train`

Train an anomaly detection model using an autoencoder.

**Parameters (multipart/form-data):**
- `zipFile`: A zip archive containing normal (non-anomalous) images for training
- `level`: (Optional) Training level from 1-5, affecting number of epochs and complexity (default: 3)
- `image_size`: (Optional) Size to resize images to (default: 256)

**Note on class names:**
The service will look for a YAML file in the uploaded ZIP (e.g., `data.yaml`, `classes.yaml`, or any `.yaml`/`.yml` file) and extract class names from it. The YAML file should contain a field called `names`, `classes`, or `labels` with a list of class names. For example:

```yaml
# Example data.yaml
names:
  - normal
  - tumor
  - polyp
```

If no YAML file is found, generic class names will be used.

**Response:**
Returns a zip file containing:
- `autoencoder.pt`: The trained PyTorch model
- `metadata.json`: Model metadata including threshold values for anomaly detection

### `POST /detect`

Detect anomalies in an image using a trained model.

**Parameters (multipart/form-data):**
- `modelFile`: Trained autoencoder model (.pt file)
- `imageFile`: Image to check for anomalies
- `metadataFile`: (Optional) Model metadata file

**Response:**
```json
{
  "reconstruction_error": 0.025,
  "threshold": 0.02,
  "is_anomaly": true,
  "class_names": ["normal", "tumor", "polyp"]
}
```

## Usage Example

```python
import requests

# URL of the anomaly detection service
ANOMALY_DETECTION_URL = "http://localhost:5029"

# Train a model
with open("normal_samples.zip", "rb") as f:
    files = {
        "zipFile": ("normal_samples.zip", f, "application/zip")
    }
    data = {
        "level": "3",
        "image_size": "256"
    }
    response = requests.post(f"{ANOMALY_DETECTION_URL}/train", files=files, data=data)
    
    # Save the model
    with open("anomaly_model.zip", "wb") as f_out:
        f_out.write(response.content)

# Detect anomalies
with open("anomaly_model.pt", "rb") as model_file, open("test_image.jpg", "rb") as image_file:
    files = {
        "modelFile": ("model.pt", model_file, "application/octet-stream"),
        "imageFile": ("image.jpg", image_file, "image/jpeg")
    }
    response = requests.post(f"{ANOMALY_DETECTION_URL}/detect", files=files)
    result = response.json()
    print(f"Anomaly detected: {result['is_anomaly']}")
    print(f"Class names: {result['class_names']}")
```

## Web Interface

This service is integrated with the DeepMed platform and can be accessed through the Anomaly Detection tab in the user interface. The UI provides a user-friendly way to configure and start the training process.

## Best Practices

1. **Dataset Preparation**:
   - Include only normal (non-anomalous) samples in the training set
   - Use consistent lighting, angle, and scale in your images
   - For best results, use at least 100+ normal samples
   - Include a YAML file with meaningful class names for better interpretation

2. **Training Parameters**:
   - Start with Level 3 (balanced) for most applications
   - Use Level 1 or 2 for quick testing or small datasets
   - Use Level 4 or 5 for production models requiring high accuracy

3. **Image Size**:
   - 256x256 works well for most applications
   - Use smaller sizes (128x128) for faster training and inference
   - Use larger sizes (384x384) for detecting small anomalies