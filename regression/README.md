# DeepMed Regression System

This directory contains containerized versions of regression model services for the DeepMed platform. Each service runs in its own Docker container and communicates through REST APIs.

## Services Overview

1. **Data Cleaner API (Port 5001)**
   - Handles data cleaning and preprocessing
   - Can use OpenAI for LLM-based cleaning (optional)
   - Reused from the classification system

2. **Feature Selector API (Port 5002)**
   - Selects important features for machine learning
   - Can use OpenAI for LLM-based feature selection (optional)
   - Reused from the classification system

3. **Anomaly Detector API (Port 5003)**
   - Detects anomalies and checks data quality
   - Uses Isolation Forest and other methods
   - Reused from the classification system

4. **Model Coordinator API (Port 6020)**
   - Orchestrates regression model services
   - Provides unified API for training and prediction
   - Selects best models based on different metrics (MSE, RMSE, R²)

## Distributed Regression Model Architecture

The system uses a distributed architecture for regression model training, where each regression model runs in its own Docker container. This approach provides several benefits:

1. **Parallel Training**: Instead of training models sequentially, all models are trained in parallel, significantly reducing overall training time.
2. **Isolation**: Each model has its own isolated environment, reducing interference between different model implementations.
3. **Scalability**: Individual model services can be scaled independently based on resource requirements.
4. **Resiliency**: Failure in one model service doesn't affect the others.

### Regression Model Services

The system includes 5 separate regression model services, each specialized in a specific regression algorithm:

1. **Linear Regression** (Port 6010)
2. **Ridge Regression** (Port 6011)
3. **Lasso Regression** (Port 6012)
4. **Random Forest Regressor** (Port 6013)
5. **Support Vector Regression (SVR)** (Port 6014)

Each service uses MLflow to track experiments and optimize hyperparameters.

### Model Coordinator

The **Model Coordinator** (Port 6020) acts as an orchestrator for all regression model services. It provides:

- A unified API for training and prediction
- Parallel dispatch of requests to all model services
- Selection of the best models based on different metrics (MSE, RMSE, R²)
- Aggregation of results from all models

### API Endpoints

#### Model Coordinator API

- `GET /health`: Check health status of all model services
- `POST /train`: Train all models in parallel
- `POST /predict`: Get predictions from the best models
- `GET /model_info`: Get information about all available models

#### Individual Model Service API

Each model service provides the following endpoints:

- `GET /health`: Check health status
- `POST /train`: Train the specific model
- `POST /predict`: Get predictions from the model
- `GET /model_info`: Get information about the model
- `GET /download_model`: Download the trained model file

### Using the Distributed Architecture

To train models:
```
curl -X POST http://localhost:6020/train \
  -H "Content-Type: application/json" \
  -d '{"data": {...}, "target": [...]}'
```

To make predictions:
```
curl -X POST http://localhost:6020/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {...}, "models": ["r2", "mse", "rmse"]}'
```

To get model information:
```
curl http://localhost:6020/model_info
```

### Launching the System

```bash
docker-compose up -d
```

This will start all model services and the model coordinator.

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed
- OpenAI API key (for services that use it)

### Running the Services

1. Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

2. Build and start all services using Docker Compose:

```bash
docker-compose up -d
```

3. To check if all services are running:

```bash
docker-compose ps
```

## Stopping the Services

To stop all services:

```bash
docker-compose down
```

To stop and remove all containers, networks, and volumes:

```bash
docker-compose down -v
```

## Troubleshooting

- **Service Logs**: View logs for a specific service with `docker-compose logs service-name`
- **API Issues**: Check if the service is running with `/health` endpoint
- **OpenAI Errors**: Verify your API key is correct in the .env file
- **Container Issues**: Try rebuilding with `docker-compose build --no-cache` 