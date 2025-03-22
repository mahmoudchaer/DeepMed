# Image Processing and Model Training Services

This directory contains Docker services for image processing, data augmentation, and model training.

## Architecture

The system follows an End-to-End Process (EEP) and Intermediate Engineering Processes (IEP) architecture:

- **End-to-End Process (EEP)**: Acts as a coordinator that receives requests from the main application and forwards them to the appropriate IEP.
- **Intermediate Engineering Processes (IEPs)**: Specialized services that perform specific tasks:
  - **Model Training**: Trains a deep learning model (EfficientNet-B0) on image data.
  - **Data Augmentation**: Creates additional images by applying various transformations to the original dataset.
  - **Data Processing**: Splits dataset into train/validation/test sets and normalizes images.

## Services

### Image EEP Service (5000)

Acts as the main entry point for the Docker services. Receives requests from the main application and forwards them to the appropriate IEP.

- **Endpoints**:
  - `/health`: Health check endpoint
  - `/model_training`: Forwards requests to the Model Training service
  - `/data_augmentation`: Forwards requests to the Data Augmentation service
  - `/data_processing`: Forwards requests to the Data Processing service

### Model Training Service (5010)

Trains an EfficientNet-B0 model on the provided dataset.

- **Endpoints**:
  - `/health`: Health check endpoint
  - `/train`: Trains a model on the provided data

- **Parameters**:
  - `zipFile`: ZIP file containing folders of images (each folder represents a class)
  - `numClasses`: Number of classes (default: 5)
  - `trainingLevel`: Training intensity level from 1-5 (default: 3)

### Data Augmentation Service (5011)

Augments an image dataset by applying various transformations to increase dataset size.

- **Endpoints**:
  - `/health`: Health check endpoint
  - `/augment`: Augments the provided data

- **Parameters**:
  - `zipFile`: ZIP file containing folders of images (each folder represents a class)
  - `augmentationLevel`: Augmentation intensity level from 1-5 (default: 3)

### Data Processing Service (5012)

Processes an image dataset by normalizing images and splitting into train/validation/test sets.

- **Endpoints**:
  - `/health`: Health check endpoint
  - `/process`: Processes the provided data

- **Parameters**:
  - `zipFile`: ZIP file containing folders of images (each folder represents a class)
  - `testSize`: Proportion of data to use for testing (default: 0.2)
  - `valSize`: Proportion of data to use for validation (default: 0.2)

## Building and Running

Build and start all services using Docker Compose:

```bash
docker-compose up -d
```

Stop all services:

```bash
docker-compose down
```

## API Usage

The main application interacts with the EEP service, which then forwards requests to the appropriate IEP.

### Train Model

```
POST http://localhost:5000/model_training
Content-Type: multipart/form-data

Form Data:
- zipFile: [ZIP file]
- numClasses: 5
- trainingLevel: 3
```

### Augment Data

```
POST http://localhost:5000/data_augmentation
Content-Type: multipart/form-data

Form Data:
- zipFile: [ZIP file]
- augmentationLevel: 3
```

### Process Data

```
POST http://localhost:5000/data_processing
Content-Type: multipart/form-data

Form Data:
- zipFile: [ZIP file]
- testSize: 0.2
- valSize: 0.2
```

## Requirements

Each service has its own Docker image and requirements:

- `image_eep_service`: Requires Flask, requests, requests_toolbelt
- `model_training_service`: Requires Flask, PyTorch, torchvision, Pillow
- `data_augmentation_service`: Requires Flask, PyTorch, torchvision, Pillow
- `data_processing_service`: Requires Flask, PyTorch, torchvision, Pillow, scikit-learn 