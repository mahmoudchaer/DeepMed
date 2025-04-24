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
import socket
import random
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

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

# Use a flexible approach for the server IP
# Try different hostnames in order of preference
SERVER_IP = os.getenv('SERVER_IP', 'localhost')
logger.info(f"Initial SERVER_IP configuration: {SERVER_IP}")

# Define services URLs with server IP
def build_service_url(port, ip=None, container_name=None):
    """Build a service URL with either the provided IP or container name"""
    if container_name:
        return f'http://{container_name}:{port}'
    else:
        base_ip = ip if ip else SERVER_IP
        return f'http://{base_ip}:{port}'

# Define service ports and container names
DATA_CLEANER_PORT = 5001
DATA_CLEANER_CONTAINER = "data_cleaner"

FEATURE_SELECTOR_PORT = 5002
FEATURE_SELECTOR_CONTAINER = "feature_selector"

ANOMALY_DETECTOR_PORT = 5003
ANOMALY_DETECTOR_CONTAINER = "anomaly_detector" 

MODEL_COORDINATOR_PORT = 5020
MODEL_COORDINATOR_CONTAINER = "model_coordinator"

MEDICAL_ASSISTANT_PORT = 5005
MEDICAL_ASSISTANT_CONTAINER = "medical_assistant"

AUGMENTATION_SERVICE_PORT = 5023
AUGMENTATION_SERVICE_CONTAINER = "augmentation_service"

MODEL_TRAINING_SERVICE_PORT = 5021
MODEL_TRAINING_SERVICE_CONTAINER = "model_training_service"

PIPELINE_SERVICE_PORT = 5025
PIPELINE_SERVICE_CONTAINER = "pipeline_service"

ANOMALY_DETECTION_SERVICE_PORT = 5029
ANOMALY_DETECTION_SERVICE_CONTAINER = "anomaly_detection_service"

TABULAR_PREDICTOR_PORT = 5100
TABULAR_PREDICTOR_CONTAINER = "predictor-service"

LOGISTIC_REGRESSION_PORT = 5010
LOGISTIC_REGRESSION_CONTAINER = "logistic_regression"

DECISION_TREE_PORT = 5011
DECISION_TREE_CONTAINER = "decision_tree"

RANDOM_FOREST_PORT = 5012
RANDOM_FOREST_CONTAINER = "random_forest"

SVM_PORT = 5013
SVM_CONTAINER = "svm"

KNN_PORT = 5014
KNN_CONTAINER = "knn"

NAIVE_BAYES_PORT = 5015
NAIVE_BAYES_CONTAINER = "naive_bayes"

# Initial service URLs - using container names directly (Docker networking)
DATA_CLEANER_URL = build_service_url(DATA_CLEANER_PORT, container_name=DATA_CLEANER_CONTAINER)
FEATURE_SELECTOR_URL = build_service_url(FEATURE_SELECTOR_PORT, container_name=FEATURE_SELECTOR_CONTAINER)
ANOMALY_DETECTOR_URL = build_service_url(ANOMALY_DETECTOR_PORT, container_name=ANOMALY_DETECTOR_CONTAINER)
MODEL_COORDINATOR_URL = build_service_url(MODEL_COORDINATOR_PORT, container_name=MODEL_COORDINATOR_CONTAINER)
MEDICAL_ASSISTANT_URL = build_service_url(MEDICAL_ASSISTANT_PORT, container_name=MEDICAL_ASSISTANT_CONTAINER)
AUGMENTATION_SERVICE_URL = build_service_url(AUGMENTATION_SERVICE_PORT, container_name=AUGMENTATION_SERVICE_CONTAINER)
MODEL_TRAINING_SERVICE_URL = build_service_url(MODEL_TRAINING_SERVICE_PORT, container_name=MODEL_TRAINING_SERVICE_CONTAINER)
PIPELINE_SERVICE_URL = build_service_url(PIPELINE_SERVICE_PORT, container_name=PIPELINE_SERVICE_CONTAINER)
ANOMALY_DETECTION_SERVICE_URL = build_service_url(ANOMALY_DETECTION_SERVICE_PORT, container_name=ANOMALY_DETECTION_SERVICE_CONTAINER)
TABULAR_PREDICTOR_URL = build_service_url(TABULAR_PREDICTOR_PORT, container_name=TABULAR_PREDICTOR_CONTAINER)

# Model-specific services
LOGISTIC_REGRESSION_URL = build_service_url(LOGISTIC_REGRESSION_PORT, container_name=LOGISTIC_REGRESSION_CONTAINER)
DECISION_TREE_URL = build_service_url(DECISION_TREE_PORT, container_name=DECISION_TREE_CONTAINER)
RANDOM_FOREST_URL = build_service_url(RANDOM_FOREST_PORT, container_name=RANDOM_FOREST_CONTAINER)
SVM_URL = build_service_url(SVM_PORT, container_name=SVM_CONTAINER)
KNN_URL = build_service_url(KNN_PORT, container_name=KNN_CONTAINER)
NAIVE_BAYES_URL = build_service_url(NAIVE_BAYES_PORT, container_name=NAIVE_BAYES_CONTAINER)

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

def resolve_service_hostname(hostname):
    """Try to resolve hostname to check if it exists in Docker's DNS"""
    try:
        socket.gethostbyname(hostname)
        return True
    except socket.gaierror:
        return False

def create_session_with_retries(retries=3, backoff_factor=0.5):
    """Create a requests session with retry capability"""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=backoff_factor
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def check_service_health(category, service_name, service_info):
    """Check health of a specific service and update metrics"""
    url = service_info["url"]
    endpoint = service_info["endpoint"]
    full_url = f"{url}{endpoint}"
    
    # Create a session with retries for more resilient connections
    session = create_session_with_retries()
    
    # Try to check if container is running via Docker API
    container_running = False
    try:
        # Skip Docker check due to compatibility issues
        # We'll rely on the HTTP checks or manual override instead
        container_running = False
    except Exception as e:
        logger.warning(f"Could not check Docker container status for {service_name}: {str(e)}")
    
    # Create base service name for DNS lookups
    service_base_name = service_name.lower().replace(' ', '_').replace('-', '_')
    
    # Define alternative IPs to try if the main one fails
    alternative_hosts = [
        # Container names with different formats
        service_base_name,
        service_name.lower().replace(' ', '-').replace('_', '-'),
        
        # Network variants
        'localhost',
        '127.0.0.1',
        'host.docker.internal',
        'docker.for.win.localhost',
        'docker.for.mac.localhost',
        
        # Network aliases - Docker Compose typically creates these
        f"deepmed_{service_base_name}",
        f"deepmed-{service_name.lower().replace(' ', '-').replace('_', '-')}",
        
        # Azure variants
        os.getenv('EXTERNAL_IP', '20.119.81.37')  # Try the external IP if set
    ]
    
    # Try to discover service using DNS first
    discovered_hosts = []
    for host_candidate in [
        service_base_name,
        f"deepmed_{service_base_name}", 
        f"deepmed-{service_base_name}",
        f"{service_base_name}_service",
        f"deepmed_{service_base_name}_service"
    ]:
        if resolve_service_hostname(host_candidate):
            discovered_hosts.append(host_candidate)
            logger.info(f"Discovered service {service_name} at DNS hostname {host_candidate}")
    
    # Add discovered hosts to the beginning of alternative hosts
    alternative_hosts = discovered_hosts + alternative_hosts
    
    # Parse the port from the URL to build alternative URLs
    port = None
    try:
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        port = parsed_url.port
    except Exception as e:
        logger.warning(f"Could not parse URL {url} for {service_name}: {str(e)}")
    
    # First try the original URL
    start_time = time.time()
    response_successful = False
    response_data = None
    
    try:
        # First try HTTP health check with original URL
        logger.info(f"Checking health of {service_name} at {full_url}")
        response = session.get(full_url, timeout=5)  # Using session with retries
        
        if response.status_code == 200:
            response_successful = True
            response_data = response
    except Exception as e:
        logger.warning(f"Error with primary URL for {service_name}: {str(e)}")
    
    # If the original URL failed and we could parse the port, try alternatives
    if not response_successful and port is not None:
        for alt_host in alternative_hosts:
            alt_url = f"http://{alt_host}:{port}"
            alt_full_url = f"{alt_url}{endpoint}"
            
            try:
                logger.info(f"Trying alternative URL for {service_name}: {alt_full_url}")
                response = session.get(alt_full_url, timeout=5)  # Using session with retries
                
                if response.status_code == 200:
                    response_successful = True
                    response_data = response
                    
                    # Update the service URL with the working alternative
                    SERVICES[category][service_name]["url"] = alt_url
                    logger.info(f"Updated {service_name} URL to {alt_url}")
                    break
            except Exception as e:
                logger.debug(f"Alternative URL {alt_full_url} failed: {str(e)}")
                continue
    
    # Calculate total response time
    response_time = time.time() - start_time
    
    # Process the result of our health checks
    if response_successful:
        SERVICES[category][service_name]["status"] = "up"
        service_status.labels(service=service_name, category=category).set(1)
        logger.info(f"{service_name} is up, response time: {response_time:.2f}s")
        
        # Try to parse additional info from the response
        try:
            health_data = response_data.json()
            SERVICES[category][service_name]["details"] = health_data
        except:
            SERVICES[category][service_name]["details"] = None
    else:
        # If HTTP check fails but container is running, mark as "partially up"
        if container_running:
            SERVICES[category][service_name]["status"] = "partial"
            service_status.labels(service=service_name, category=category).set(0.5)
            logger.warning(f"{service_name} container is running but HTTP check failed")
        else:
            SERVICES[category][service_name]["status"] = "down"
            service_status.labels(service=service_name, category=category).set(0)
            logger.warning(f"{service_name} is down, tried multiple endpoints")
    
    # Update metrics and history
    SERVICES[category][service_name]["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    SERVICES[category][service_name]["response_time"] = f"{response_time:.2f}s"
    service_response_time.labels(service=service_name, category=category).observe(response_time)
    service_check_counter.labels(service=service_name, category=category).inc()
    
    # Update history (keep only last 100 points)
    current_time = datetime.now().strftime("%H:%M:%S")
    service_history[service_name]["timestamps"].append(current_time)
    service_history[service_name]["response_times"].append(response_time)
    service_history[service_name]["statuses"].append(
        1 if SERVICES[category][service_name]["status"] == "up" else 
        0.5 if SERVICES[category][service_name]["status"] == "partial" else 0
    )
    
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
    # Check if any service has been checked yet
    services_checked = False
    for category in SERVICES.values():
        for service in category.values():
            if service['status'] != 'unknown':
                services_checked = True
                break
        if services_checked:
            break
    
    # If no services have been checked yet, start the check and show loading page
    if not services_checked:
        # Start the initial service check in a separate thread
        check_thread = threading.Thread(target=check_all_services)
        check_thread.daemon = True
        check_thread.start()
        return render_template('loading.html')
    
    # Otherwise show the main dashboard
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

@app.route('/api/override_status/<service_name>', methods=['POST'])
def override_service_status(service_name):
    """API endpoint to manually override the status of a service"""
    data = request.json
    new_status = data.get('status', 'up')
    
    if new_status not in ['up', 'down', 'partial', 'unknown']:
        return jsonify({"error": "Invalid status. Use 'up', 'down', 'partial', or 'unknown'"}), 400
    
    # Find the service
    service_found = False
    service_category = None
    
    for category, services in SERVICES.items():
        if service_name in services:
            service_found = True
            service_category = category
            break
    
    if not service_found:
        return jsonify({"error": f"Service '{service_name}' not found"}), 404
    
    # Update the service status
    SERVICES[service_category][service_name]["status"] = new_status
    SERVICES[service_category][service_name]["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    SERVICES[service_category][service_name]["manual_override"] = True
    
    # Update Prometheus metric
    if new_status == 'up':
        service_status.labels(service=service_name, category=service_category).set(1)
    elif new_status == 'partial':
        service_status.labels(service=service_name, category=service_category).set(0.5)
    else:
        service_status.labels(service=service_name, category=service_category).set(0)
    
    return jsonify({
        "service": service_name,
        "status": new_status,
        "message": f"Status of {service_name} manually set to {new_status}"
    })

if __name__ == '__main__':
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Start the scheduler in a separate thread
    scheduler = threading.Thread(target=scheduler_thread)
    scheduler.daemon = True
    scheduler.start()
    
    # Don't do initial service check here anymore
    # check_all_services()
    
    app.run(host='0.0.0.0', port=5432, debug=False) 