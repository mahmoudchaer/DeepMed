# Anomaly Detector EEP Middleware Service

This service acts as a middleware between the main application and the anomaly detection service. It forwards requests from the application to the anomaly detector and returns responses back to the application.

## Architecture

```
App (app_images.py) -> Anomaly Detector EEP -> Anomaly Detector Service -> Response -> Anomaly Detector EEP -> App
```

## Features

- Transparent proxy for anomaly detection requests
- Health check endpoint to verify service status
- Logging of request/response flow
- Preserves all headers and response data

## Environment Variables

- `PORT`: Port to run the service on (default: 5030)
- `ANOMALY_DETECTION_SERVICE_URL`: URL of the actual anomaly detection service (default: http://anomaly_detector:5029)

## Endpoints

- `/health`: Health check endpoint
- `/train`: Proxy endpoint for training anomaly detection models
- `/detect`: Proxy endpoint for detecting anomalies in images

## Docker Usage

```
docker build -t anomaly_detector_eep -f Dockerfile_eep .
docker run -p 5030:5030 anomaly_detector_eep
```

## Integration

This service should be placed between the main application and the anomaly detection service in the Docker network. 