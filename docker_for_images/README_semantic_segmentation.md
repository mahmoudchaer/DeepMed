# Semantic Segmentation Service

This service provides an API for training DeepLabV3 models with a ResNet-50 backbone for semantic segmentation using PyTorch.

## Overview

The semantic segmentation service allows users to:

1. Upload a ZIP file with images and their corresponding segmentation masks
2. Train a DeepLabV3 model with ResNet-50 backbone on the provided data
3. Download the trained model with usage instructions

## Input Format

The service expects a ZIP file with the following structure:

```
dataset.zip
│
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── masks/
│   ├── image1.jpg  (pixel values should correspond to class labels: 0, 1, 2, etc.)
│   ├── image2.jpg
│   └── ...
│
└── labels.txt (optional)
    └── Contains class names, one per line (excluding background class)
```

### Requirements

- The `images` and `masks` folders must exist in the root of the ZIP file
- Image and mask filenames must match exactly
- Masks should be single-channel images with pixel values representing class indices (0 for background, 1 for first class, etc.)
- If `labels.txt` is not provided, binary segmentation is assumed (background and foreground)

## API Endpoints

### Health Check

```
GET /health
```

Returns the service status.

### Train Model

```
POST /train
```

Parameters:
- `zipFile`: The dataset ZIP file (required)
- `level`: Training complexity level from 1-5 (optional, default: 3)
- `image_size`: Size to resize images to (optional, default: 256)

Training levels:
- Level 1: Quick training (2 epochs, batch size 8)
- Level 2: Basic training (5 epochs, batch size 8)
- Level 3: Standard training (10 epochs, batch size 8)
- Level 4: Extended training (20 epochs, batch size 4)
- Level 5: Thorough training (30 epochs, batch size 4)

## Output

The service returns a ZIP file containing:

- `model.pth`: The trained PyTorch model
- `README.txt`: Usage instructions and model information
- `requirements.txt`: Required packages to use the model

## Using the Trained Model

The returned README.txt includes detailed instructions on how to:

1. Load the model in PyTorch
2. Process images for inference
3. Generate segmentation predictions

## Metrics

During training, the service tracks and reports:

- Mean IoU (Intersection over Union)
- Pixel Accuracy
- Loss

These metrics are saved in the model file and included in the README.

## Technical Details

- The model uses DeepLabV3 architecture with a ResNet-50 backbone
- Training is performed on CPU only
- Images are automatically resized and normalized for training
- Memory-efficient data loading ensures the service won't crash with large datasets 