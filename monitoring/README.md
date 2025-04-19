# DeepMed Monitoring System

A comprehensive monitoring system for tracking the health and performance of all DeepMed services.

## Features

- **Real-time Service Monitoring**: Tracks health status of all DeepMed services
- **Service Response Time Tracking**: Monitors response times with historical charts
- **Service Categories**: Groups services by type (Core, Model, Medical AI, Image Processing)
- **Interactive Dashboard**: Modern web interface with detailed service information
- **Test Runner**: Execute tests directly from the dashboard
- **Prometheus Integration**: Built-in metrics for integration with monitoring systems
- **Grafana Dashboards**: Visualize metrics with Grafana
- **Status Reports**: Export JSON status reports for documentation
- **Endpoint Testing**: Test specific service endpoints with custom requests
- **Service Logs**: View container logs for each service directly from the dashboard

## Services Monitored

The system monitors the following DeepMed services:

### Core Services
- Data Cleaner
- Feature Selector
- Anomaly Detector

### Model Services
- Model Coordinator
- Model Training Service
- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- KNN
- Naive Bayes

### Medical AI Services
- Medical Assistant
- Pipeline Service

### Image Processing Services
- Augmentation Service
- Anomaly Detection Service
- Semantic Segmentation Service

### Prediction Services
- Tabular Predictor

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.9+

### Installation

1. Navigate to the monitoring directory:
   ```
   cd monitoring
   ```

2. Start all services (Monitoring, Prometheus, and Grafana):
   ```
   docker-compose up -d
   ```

3. Access the services:
   - Monitoring Dashboard: http://localhost:5432
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/deepmed)

## Endpoints

The monitoring service exposes the following API endpoints:

- `/` - Main dashboard UI
- `/health` - Health check for the monitoring service itself
- `/api/services` - Get the status of all services
- `/api/service_history/<service_name>` - Get history data for a specific service
- `/api/test_endpoint/<service_name>` - Test a specific endpoint on a service
- `/api/logs/<service_name>` - Get container logs for a specific service
- `/api/refresh` - Manually trigger a refresh of all service statuses
- `/api/metrics` - Prometheus metrics endpoint
- `/api/prometheus` - Get Prometheus and Grafana URLs

## Monitoring Architecture

### Monitoring Service
- Flask application that checks service health and provides the dashboard UI
- Runs in its own Docker container on port 5432

### Prometheus
- Time-series database for storing metrics
- Scrapes metrics from services and the monitoring service
- Runs in its own Docker container on port 9090

### Grafana
- Visualization tool for metrics
- Connects to Prometheus as a data source
- Provides customizable dashboards
- Runs in its own Docker container on port 3000

## Development

### Project Structure

- `app.py` - Main application file with Flask routes and service monitoring
- `templates/` - HTML templates for the UI
- `static/` - Static assets (if any)
- `logs/` - Log files
- `prometheus/` - Prometheus configuration and Dockerfile
- `grafana/` - Grafana configuration, dashboards, and Dockerfile
- `Dockerfile` - Docker configuration for the monitoring service
- `docker-compose.yml` - Docker Compose configuration for all services
- `requirements.txt` - Python dependencies

### Adding New Services

To add a new service to monitor:

1. Add the service URL to `docker-compose.yml` environment variables
2. Add the service to the appropriate category in the `SERVICES` dictionary in `app.py`
3. Add the service to `prometheus/prometheus.yml` scrape configs
4. Restart the monitoring service

## QA Methods

This monitoring system is part of the DeepMed QA approach and provides:

1. **Service Availability Monitoring**: Ensures all services are operational
2. **Response Time Tracking**: Identifies performance issues
3. **Logs Viewing**: Access service logs directly from the dashboard
4. **Metrics Collection**: Gathers data for long-term analysis via Prometheus
5. **Dashboard Visualization**: Provides insights through Grafana dashboards
6. **Endpoint Testing**: Verifies specific API functionality 