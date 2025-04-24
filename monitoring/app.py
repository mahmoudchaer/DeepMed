from flask import Flask, render_template, jsonify, request
import requests
import os
import json
import time
import threading
import schedule
import yaml
from datetime import datetime
import pytest
import importlib.util
import sys
import pandas as pd
import numpy as np
from prometheus_client import generate_latest, REGISTRY, Gauge, Counter, Histogram
import plotly.graph_objects as go
import plotly.express as px
import plotly
import logging

# Add parent directory to path for importing keyvault
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import keyvault, or create a fallback implementation
try:
    import keyvault
except ImportError:
    # Create a fallback implementation for keyvault
    class KeyVaultFallback:
        @staticmethod
        def get_secret(secret_name, default_value=None):
            return os.getenv(secret_name, default_value)
        
        @staticmethod
        def getenv(key, default=None):
            return os.getenv(key, default)
    
    # Create the module
    keyvault = KeyVaultFallback()
    print("Using fallback keyvault implementation as the module couldn't be imported")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

app = Flask(__name__)

# Use host.docker.internal to access the host from inside Docker
SERVER_IP = os.getenv('SERVER_IP', 'host.docker.internal')
logger.info(f"Using SERVER_IP: {SERVER_IP}")

# Define services URLs with server IP instead of localhost
DATA_CLEANER_URL = f'http://{SERVER_IP}:5001'
FEATURE_SELECTOR_URL = f'http://{SERVER_IP}:5002'
ANOMALY_DETECTOR_URL = f'http://{SERVER_IP}:5003'
MODEL_COORDINATOR_URL = f'http://{SERVER_IP}:5020'
MEDICAL_ASSISTANT_URL = f'http://{SERVER_IP}:5005'
AUGMENTATION_SERVICE_URL = f'http://{SERVER_IP}:5023'
MODEL_TRAINING_SERVICE_URL = f'http://{SERVER_IP}:5021'
PIPELINE_SERVICE_URL = f'http://{SERVER_IP}:5025'
ANOMALY_DETECTION_SERVICE_URL = f'http://{SERVER_IP}:5029'
TABULAR_PREDICTOR_URL = f'http://{SERVER_IP}:5100'

# Model-specific services
LOGISTIC_REGRESSION_URL = f'http://{SERVER_IP}:5010'
DECISION_TREE_URL = f'http://{SERVER_IP}:5011'
RANDOM_FOREST_URL = f'http://{SERVER_IP}:5012'
SVM_URL = f'http://{SERVER_IP}:5013'
KNN_URL = f'http://{SERVER_IP}:5014'
NAIVE_BAYES_URL = f'http://{SERVER_IP}:5015'

# Categorize services for the dashboard
SERVICES = {
    "Core Services": {
        "Data Cleaner": {"url": DATA_CLEANER_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "Cleans and preprocesses incoming data"},
        "Feature Selector": {"url": FEATURE_SELECTOR_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "Performs feature selection on tabular data"},
        "Anomaly Detector": {"url": ANOMALY_DETECTOR_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "Detects anomalies in tabular data"}
    },
    "Model Services": {
        "Model Coordinator": {"url": MODEL_COORDINATOR_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "Coordinates model training and prediction"},
        "Model Training Service": {"url": MODEL_TRAINING_SERVICE_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "Handles model training"},
        "Logistic Regression": {"url": LOGISTIC_REGRESSION_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "Logistic Regression model service"},
        "Decision Tree": {"url": DECISION_TREE_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "Decision Tree model service"},
        "Random Forest": {"url": RANDOM_FOREST_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "Random Forest model service"},
        "SVM": {"url": SVM_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "Support Vector Machine model service"},
        "KNN": {"url": KNN_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "K-Nearest Neighbors model service"},
        "Naive Bayes": {"url": NAIVE_BAYES_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "Naive Bayes model service"}
    },
    "Medical AI Services": {
        "Medical Assistant": {"url": MEDICAL_ASSISTANT_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "Provides medical insights and analysis"},
        "Pipeline Service": {"url": PIPELINE_SERVICE_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "Manages ML pipelines"}
    },
    "Image Processing Services": {
        "Augmentation Service": {"url": AUGMENTATION_SERVICE_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "Performs data augmentation"},
        "Anomaly Detection Service": {"url": ANOMALY_DETECTION_SERVICE_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "PyTorch autoencoder for anomaly detection"}
    },
    "Prediction Services": {
        "Tabular Predictor": {"url": TABULAR_PREDICTOR_URL, "endpoint": "/health", "status": "unknown", "last_check": None, "description": "Prediction service for tabular data"}
    }
}

# Prometheus metrics
service_status = Gauge('service_status', 'Service status (1=up, 0=down)', ['service', 'category'])
service_response_time = Histogram('service_response_time', 'Service response time in seconds', ['service', 'category'])
service_check_counter = Counter('service_check_counter', 'Number of service health checks performed', ['service', 'category'])
test_status = Gauge('test_status', 'Test status (1=pass, 0=fail)', ['test'])

# Data for service history
service_history = {}
for category, services in SERVICES.items():
    for service_name in services:
        service_history[service_name] = {
            "timestamps": [],
            "response_times": [],
            "statuses": []
        }

def check_service_health(category, service_name, service_info):
    """Check health of a specific service and update metrics"""
    url = service_info["url"]
    endpoint = service_info["endpoint"]
    full_url = f"{url}{endpoint}"
    
    start_time = time.time()
    try:
        logger.info(f"Checking health of {service_name} at {full_url}")
        response = requests.get(full_url, timeout=10)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            SERVICES[category][service_name]["status"] = "up"
            service_status.labels(service=service_name, category=category).set(1)
            logger.info(f"{service_name} is up, response time: {response_time:.2f}s")
            
            # Try to parse additional info from the response
            try:
                health_data = response.json()
                SERVICES[category][service_name]["details"] = health_data
            except:
                SERVICES[category][service_name]["details"] = None
        else:
            SERVICES[category][service_name]["status"] = "down"
            service_status.labels(service=service_name, category=category).set(0)
            logger.warning(f"{service_name} returned status code {response.status_code}")
    except Exception as e:
        response_time = time.time() - start_time
        SERVICES[category][service_name]["status"] = "down"
        service_status.labels(service=service_name, category=category).set(0)
        logger.error(f"Error checking {service_name} health: {str(e)}")
    
    # Update metrics and history
    SERVICES[category][service_name]["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    SERVICES[category][service_name]["response_time"] = f"{response_time:.2f}s"
    service_response_time.labels(service=service_name, category=category).observe(response_time)
    service_check_counter.labels(service=service_name, category=category).inc()
    
    # Update history (keep only last 100 points)
    current_time = datetime.now().strftime("%H:%M:%S")
    service_history[service_name]["timestamps"].append(current_time)
    service_history[service_name]["response_times"].append(response_time)
    service_history[service_name]["statuses"].append(1 if SERVICES[category][service_name]["status"] == "up" else 0)
    
    # Trim history if needed
    if len(service_history[service_name]["timestamps"]) > 100:
        service_history[service_name]["timestamps"] = service_history[service_name]["timestamps"][-100:]
        service_history[service_name]["response_times"] = service_history[service_name]["response_times"][-100:]
        service_history[service_name]["statuses"] = service_history[service_name]["statuses"][-100:]

def check_all_services():
    """Check health of all registered services"""
    for category, services in SERVICES.items():
        for service_name, service_info in services.items():
            check_service_health(category, service_name, service_info)

def run_test(test_file):
    """Run a test file using pytest"""
    try:
        # Load test file as a module
        spec = importlib.util.spec_from_file_location("test_module", test_file)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        # Run pytest on the module
        result = pytest.main(["-xvs", test_file])
        
        test_name = os.path.basename(test_file)
        if result == 0:  # 0 means all tests passed
            test_status.labels(test=test_name).set(1)
            return {"status": "pass", "test": test_name}
        else:
            test_status.labels(test=test_name).set(0)
            return {"status": "fail", "test": test_name}
    except Exception as e:
        test_name = os.path.basename(test_file)
        test_status.labels(test=test_name).set(0)
        return {"status": "error", "test": test_name, "error": str(e)}

def scheduler_thread():
    """Background thread for scheduled tasks"""
    schedule.every(1).minutes.do(check_all_services)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

def create_response_time_chart(service_name):
    """Create a plotly chart of response times for a service"""
    history = service_history[service_name]
    
    if not history["timestamps"]:
        return None
    
    fig = go.Figure()
    
    # Add response time line
    fig.add_trace(go.Scatter(
        x=history["timestamps"], 
        y=history["response_times"],
        mode='lines+markers',
        name='Response Time (s)',
        line=dict(color='blue')
    ))
    
    # Add status indicators
    fig.add_trace(go.Scatter(
        x=history["timestamps"],
        y=[max(history["response_times"]) * 1.1 if status == 0 else None for status in history["statuses"]],
        mode='markers',
        name='Down Periods',
        marker=dict(color='red', size=10, symbol='x')
    ))
    
    fig.update_layout(
        title=f"{service_name} Response Time History",
        xaxis_title="Time",
        yaxis_title="Response Time (s)",
        template="plotly_white",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return plotly.utils.PlotlyJSONEncoder().encode(fig)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', services=SERVICES)

@app.route('/health')
def health():
    """Health check endpoint for the monitoring service itself"""
    return jsonify({"status": "healthy"})

@app.route('/api/services')
def get_services():
    """API endpoint to get all service statuses"""
    return jsonify(SERVICES)

@app.route('/api/service_history/<service_name>')
def get_service_history(service_name):
    """API endpoint to get history for a specific service"""
    if service_name in service_history:
        chart_json = create_response_time_chart(service_name)
        return jsonify({
            "history": service_history[service_name],
            "chart": chart_json
        })
    else:
        return jsonify({"error": "Service not found"}), 404

@app.route('/api/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(REGISTRY)

@app.route('/api/test_endpoint/<service_name>', methods=['POST'])
def test_endpoint(service_name):
    """API endpoint to test a specific endpoint of a service"""
    data = request.json
    endpoint = data.get('endpoint', '/health')
    method = data.get('method', 'GET')
    payload = data.get('payload', {})
    
    # Find the service
    service_info = None
    category = None
    
    for cat, services in SERVICES.items():
        if service_name in services:
            service_info = services[service_name]
            category = cat
            break
    
    if not service_info:
        return jsonify({"error": "Service not found"}), 404
    
    # Test the endpoint
    try:
        url = f"{service_info['url']}{endpoint}"
        
        if method.upper() == 'GET':
            response = requests.get(url, timeout=10)
        elif method.upper() == 'POST':
            response = requests.post(url, json=payload, timeout=10)
        else:
            return jsonify({"error": f"Unsupported method: {method}"}), 400
        
        # Return response info
        try:
            response_data = response.json()
        except:
            response_data = {"text": response.text[:500]}
            
        return jsonify({
            "status_code": response.status_code,
            "data": response_data,
            "headers": dict(response.headers)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/run_test', methods=['POST'])
def api_run_test():
    """API endpoint to run a test file"""
    data = request.json
    test_file = data.get('test_file')
    
    if not test_file:
        return jsonify({"error": "No test file specified"}), 400
        
    result = run_test(test_file)
    return jsonify(result)

@app.route('/api/refresh')
def api_refresh():
    """API endpoint to manually refresh all service statuses"""
    check_all_services()
    return jsonify({"status": "refreshed"})

@app.route('/api/prometheus')
def prometheus_info():
    """API endpoint to get Prometheus info"""
    # For external URLs, use the public IP, not the Docker host internal reference
    external_ip = os.getenv('EXTERNAL_IP', '20.119.81.37')
    return jsonify({
        "prometheus_url": f"http://{external_ip}:9090",
        "grafana_url": f"http://{external_ip}:3000"
    })

@app.route('/api/logs/<service_name>')
def get_service_logs(service_name):
    """API endpoint to get logs for a specific service using Docker Python client"""
    try:
        # Find service in the registered services
        service_found = False
        for category, services in SERVICES.items():
            if service_name in services:
                service_found = True
                break
        
        if not service_found:
            return jsonify({"error": f"Service '{service_name}' not found"}), 404
        
        # Try to use Docker client
        try:
            import docker
            client = docker.from_env()
            
            # Convert service name to container name format (lowercase with underscores)
            container_name = service_name.lower().replace(' ', '_').replace('-', '_')
            
            # Try different variations of container names
            container_variations = [
                container_name,
                f"deepmed_{container_name}",
                f"monitoring_{container_name}",
                f"docker_for_images-{container_name}",
                f"docker_for_images_{container_name}",
                f"{container_name}_1",
                f"deepmed_{container_name}_1",
                f"monitoring_{container_name}_1",
                f"docker_for_images-{container_name}_1",
                f"docker_for_images-{container_name}-1",
                f"monitoring-{container_name}-1",
                f"docker_for_images_{container_name}_1"
            ]
            
            # Get a list of all containers to check against
            all_containers = client.containers.list(all=True)
            all_container_names = [c.name for c in all_containers]
            
            container = None
            # First try exact matches
            for name in container_variations:
                if name in all_container_names:
                    container = client.containers.get(name)
                    break
            
            # If no exact match, try partial matching
            if not container:
                for c in all_containers:
                    # Check if the service name is part of the container name
                    if container_name in c.name or service_name.lower().replace('_', '-') in c.name:
                        container = c
                        break
            
            if not container:
                return jsonify({
                    "error": f"Could not find container for {service_name}. Available containers: {', '.join(all_container_names)}"
                }), 404
            
            # Get logs from container
            logs = container.logs(tail=100).decode('utf-8')
            
            if not logs:
                logs = "No logs available for this service."
                
            return jsonify({"logs": logs})
            
        except ImportError:
            return jsonify({"error": "Docker Python client not installed"}), 500
        except docker.errors.DockerException as e:
            logger.error(f"Docker error: {str(e)}")
            return jsonify({"error": f"Docker error: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in logs endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Start the scheduler in a separate thread
    scheduler = threading.Thread(target=scheduler_thread)
    scheduler.daemon = True
    scheduler.start()
    
    # Initial service check
    check_all_services()
    
    app.run(host='0.0.0.0', port=5432, debug=False) 