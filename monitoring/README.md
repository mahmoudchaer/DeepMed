# DeepMed Monitoring System

A comprehensive monitoring system for tracking the health and performance of all DeepMed services.

## Features

- **Real-time Service Monitoring**: Tracks health status of all DeepMed services
- **Service Response Time Tracking**: Monitors response times with historical charts
- **Service Categories**: Groups services by type (Core, Model, Medical AI, Image Processing)
- **Interactive Dashboard**: Modern web interface with detailed service information
- **Test Runner**: Execute tests directly from the dashboard
- **Prometheus Integration**: Built-in metrics for integration with monitoring systems
- **Status Reports**: Export JSON status reports for documentation
- **Endpoint Testing**: Test specific service endpoints with custom requests

## Services Monitored

The system monitors the following DeepMed services:

### Core Services
- Data Cleaner
- Feature Selector
- Anomaly Detector

### Model Services
- Model Coordinator
- Model Training Service

### Medical AI Services
- Medical Assistant
- Pipeline Service

### Image Processing Services
- Augmentation Service
- Object Detection Service
- Anomaly Detection Service
- Semantic Segmentation Service

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.9+

### Installation

1. Navigate to the monitoring directory:
   ```
   cd monitoring
   ```

2. Build and start the Docker container:
   ```
   docker-compose up -d
   ```

3. Access the monitoring dashboard:
   ```
   http://localhost:5432
   ```

## Endpoints

The monitoring service exposes the following API endpoints:

- `/` - Main dashboard UI
- `/health` - Health check for the monitoring service itself
- `/api/services` - Get the status of all services
- `/api/service_history/<service_name>` - Get history data for a specific service
- `/api/test_endpoint/<service_name>` - Test a specific endpoint on a service
- `/api/run_test` - Run a Python test file
- `/api/refresh` - Manually trigger a refresh of all service statuses
- `/api/metrics` - Prometheus metrics endpoint

## Development

### Project Structure

- `app.py` - Main application file with Flask routes and service monitoring
- `templates/` - HTML templates for the UI
- `static/` - Static assets (if any)
- `logs/` - Log files
- `Dockerfile` - Docker configuration
- `docker-compose.yml` - Docker Compose configuration
- `requirements.txt` - Python dependencies

### Adding New Services

To add a new service to monitor:

1. Add the service URL to `docker-compose.yml` environment variables
2. Add the service to the appropriate category in the `SERVICES` dictionary in `app.py`
3. Restart the monitoring service

## QA Methods

This monitoring system is part of the DeepMed QA approach and provides:

1. **Service Availability Monitoring**: Ensures all services are operational
2. **Response Time Tracking**: Identifies performance issues
3. **Test Execution**: Runs automated tests against services
4. **Metrics Collection**: Gathers data for long-term analysis
5. **Endpoint Testing**: Verifies specific API functionality 