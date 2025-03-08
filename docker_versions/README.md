# DeepMed Docker Services

This directory contains containerized versions of all DeepMed components as separate microservices. Each service runs in its own Docker container and communicates through REST APIs.

## Services Overview

1. **Data Cleaner API (Port 5001)**
   - Handles data cleaning and preprocessing
   - Can use OpenAI for LLM-based cleaning (optional)

2. **Feature Selector API (Port 5002)**
   - Selects important features for machine learning
   - Can use OpenAI for LLM-based feature selection (optional)

3. **Anomaly Detector API (Port 5003)**
   - Detects anomalies and checks data quality
   - Uses Isolation Forest and other methods

4. **Model Trainer API (Port 5004)**
   - Trains multiple machine learning models
   - Provides model evaluation metrics
   - Stores trained models for later use

5. **Medical Assistant API (Port 5005)**
   - AI assistant for medical insights
   - Analyzes data and provides recommendations
   - Conversation capabilities for Q&A

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

### Testing the Services

You can use the following curl commands to test each service:

**Data Cleaner API:**
```bash
curl -X POST http://localhost:5001/health
curl -X POST http://localhost:5001/clean -H "Content-Type: application/json" -d '{"data": {"column1": [1, 2, 3], "column2": [4, 5, 6]}, "target_column": "column1"}'
```

**Feature Selector API:**
```bash
curl -X GET http://localhost:5002/health
curl -X POST http://localhost:5002/select_features -H "Content-Type: application/json" -d '{"data": {"feature1": [1, 2, 3], "feature2": [4, 5, 6]}, "target": [0, 1, 0]}'
```

**Anomaly Detector API:**
```bash
curl -X GET http://localhost:5003/health
curl -X POST http://localhost:5003/detect_anomalies -H "Content-Type: application/json" -d '{"data": {"feature1": [1, 2, 3, 100], "feature2": [4, 5, 6, 7]}}'
```

**Model Trainer API:**
```bash
curl -X GET http://localhost:5004/health
```

**Medical Assistant API:**
```bash
curl -X GET http://localhost:5005/health
curl -X POST http://localhost:5005/chat -H "Content-Type: application/json" -d '{"message": "What machine learning model is best for heart disease prediction?"}'
```

## Integration with Main Application

To integrate these services with your main Flask application, you'll need to replace direct function calls with API calls. Here's an example:

```python
# Instead of:
# cleaned_data = cleaner.clean_data(data, target_column)

# Do this:
import requests
import pandas as pd

def clean_data(data, target_column):
    response = requests.post(
        'http://localhost:5001/clean',
        json={
            'data': data.to_dict(orient='records'),
            'target_column': target_column
        }
    )
    if response.status_code == 200:
        result = response.json()
        return pd.DataFrame.from_dict(result['data'])
    else:
        raise Exception(f"Error from data cleaner API: {response.json().get('error')}")
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