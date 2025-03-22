# Image Classification Services

This folder contains Docker services for image classification model training with a modular architecture.

## Architecture

The system follows a modular architecture with:

- **End-to-End Process (EEP)**: Coordinates the entire workflow for training image classification models
- **Intermediate Engineering Processes (IEPs)**: Handle specific tasks in the training pipeline

### Components

1. **Image Classification EEP** (Port 5020)
   - Serves as the main entry point for the application
   - Coordinates all the IEPs
   - Returns the final trained model to the client

2. **Data Processing IEP** (Port 5011)
   - Splits the input data into training, validation, and testing sets
   - Organizes data into a structure suitable for model training

3. **Data Augmentation IEP** (Port 5012)
   - Creates additional training samples by applying transformations
   - Increases the effective size and diversity of the training set

4. **Model Training IEP** (Port 5010)
   - Trains an EfficientNet-B0 model on the processed data
   - Provides metrics on training performance

## Setup and Deployment

### Prerequisites

- Docker and Docker Compose
- Python 3.9+

### Building and Running

1. Navigate to this directory:
   ```
   cd docker_for_images
   ```

2. Start all services using Docker Compose:
   ```
   docker-compose up -d
   ```

3. Check the status of all services:
   ```
   docker-compose ps
   ```

## API Endpoints

### EEP Service (Port 5020)

- **GET /health**: Health check
- **POST /train**: Train a model with data preprocessing and optional augmentation

### Data Processing Service (Port 5011)

- **GET /health**: Health check
- **POST /process**: Process and split data into train/val/test sets

### Data Augmentation Service (Port 5012)

- **GET /health**: Health check
- **POST /augment**: Create augmented versions of training images

### Model Training Service (Port 5010)

- **GET /health**: Health check
- **POST /train**: Train a model on the prepared data

## Communication Flow

1. Client sends data to the EEP service
2. EEP sends data to the Data Processing IEP
3. If requested, EEP sends processed data to the Data Augmentation IEP
4. EEP sends prepared data to the Model Training IEP
5. EEP returns the trained model and metrics to the client

## Configuration

The training process can be configured with these parameters:

- **numClasses**: Number of classes in the dataset (default: 5)
- **trainingLevel**: Level of training from 1-5, higher is more thorough (default: 3)
- **useAugmentation**: Whether to apply data augmentation (default: false)
- **validationSplit**: Percentage of data to use for validation (default: 0.2)
- **testSplit**: Percentage of data to use for testing (default: 0.1) 