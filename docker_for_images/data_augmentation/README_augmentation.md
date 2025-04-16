# Data Augmentation Service

This service provides data augmentation capabilities for image datasets to help improve model training.

## Overview

The data augmentation service uses advanced image transformation techniques to generate variations of your training data, which can help:

- Increase dataset size
- Improve model generalization
- Reduce overfitting
- Enhance model robustness

## Features

- Multiple augmentation levels (1-5) with increasing transformation intensity
- Customizable number of augmentations per image
- Parallel processing for faster execution
- Support for classification datasets organized in class folders
- Interactive web interface with preview and progress tracking

## Getting Started

### Running with Docker

1. Build the Docker image:
   ```
   docker build -t deepmed-augmentation -f Dockerfile_augment .
   ```

2. Run the container:
   ```
   docker run -p 5023:5023 deepmed-augmentation
   ```

3. Or use docker-compose to start all services:
   ```
   docker-compose up -d
   ```

### API Endpoints

- **Health Check**: `GET /health`
  - Returns service status

- **Augment Dataset**: `POST /augment`
  - Parameters:
    - `zipFile`: ZIP file containing images organized in class folders
    - `level`: Augmentation level (1-5)
    - `numAugmentations`: Number of augmentations per image (1-10)
  - Returns: ZIP file with augmented dataset

## Augmentation Levels

1. **Level 1 - Light**: Basic flips and minimal brightness/contrast changes
2. **Level 2 - Moderate**: Flips, slight rotations, and moderate brightness/contrast
3. **Level 3 - Medium**: More rotation, scaling, shifting, and color adjustments
4. **Level 4 - Strong**: Elastic transforms, noise, and significant geometric changes
5. **Level 5 - Very Strong**: Maximum variation with all possible transforms

## Implementation Details

The service uses Albumentations library to perform image transformations with:
- Fast execution through parallel processing
- Support for image classification data
- Memory-efficient processing for larger datasets

## Input Dataset Format

The input ZIP file should contain folders, where each folder represents a class and contains images of that class.

Example structure:
```
dataset.zip
├── cat/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
├── dog/
│   ├── dog1.jpg
│   ├── dog2.jpg
│   └── ...
└── ...
```

## Output

The output will be a ZIP file with the same structure but containing:
- Original images
- Augmented versions of each image 