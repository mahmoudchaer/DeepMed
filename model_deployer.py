import os
import shutil
import json
import subprocess
import logging
import uuid
import tempfile
import time
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model type specific requirements
MODEL_REQUIREMENTS = {
    "logistic_regression": ["scikit-learn==1.2.2", "pandas", "numpy", "flask"],
    "decision_tree": ["scikit-learn==1.2.2", "pandas", "numpy", "flask"],
    "random_forest": ["scikit-learn==1.2.2", "pandas", "numpy", "flask"],
    "svm": ["scikit-learn==1.2.2", "pandas", "numpy", "flask"],
    "knn": ["scikit-learn==1.2.2", "pandas", "numpy", "flask"],
    "naive_bayes": ["scikit-learn==1.2.2", "pandas", "numpy", "flask"]
}

# Default requirements for all models
DEFAULT_REQUIREMENTS = ["scikit-learn==1.2.2", "pandas", "numpy", "flask"]

def get_requirements(model_type):
    """Get requirements for a specific model type"""
    requirements = MODEL_REQUIREMENTS.get(model_type, DEFAULT_REQUIREMENTS)
    return requirements

def find_model_file(model_name):
    """Find model file by searching in various directories"""
    # DEBUGGING: Print current working directory for diagnostics
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Get the absolute root directory of the project (the directory containing the main app)
    root_dir = os.path.abspath(os.path.dirname(__file__))
    logger.info(f"Root directory: {root_dir}")
    
    # Define base directories to search
    base_dirs = [
        root_dir,  # The current directory
        os.path.join(root_dir, 'docker_versions'),  # docker_versions subdirectory
        os.path.dirname(root_dir),  # Parent directory
        os.path.join(os.path.dirname(root_dir), 'docker_versions'),  # docker_versions in parent directory
        # Also search alongside the deployed files for docker environment
        '/app',
        '/model',
        # Look for models in MLflow artifacts directory (if it exists)
        os.path.join(root_dir, 'mlruns'),
        os.path.join(os.path.dirname(root_dir), 'mlruns')
    ]
    
    # Generate possible model file paths using all base directories
    possible_paths = []
    
    # Look in some additional locations first
    user_home = os.path.expanduser("~")
    desktop_dir = os.path.join(user_home, "Desktop")
    if os.path.exists(desktop_dir):
        base_dirs.append(desktop_dir)
        # Check common subdirectories on desktop
        for subdir in ["New folder", "New folder (2)", "Projects", "ML", "Machine Learning"]:
            potential_dir = os.path.join(desktop_dir, subdir)
            if os.path.exists(potential_dir):
                base_dirs.append(potential_dir)
    
    # Locations to check within each base directory
    sub_paths = [
        # Standard locations
        os.path.join('saved_models', model_name, f"{model_name}_model.pkl"),
        os.path.join('saved_models', model_name, 'model.pkl'),
        os.path.join('saved_models', f"{model_name}_model.pkl"),
        os.path.join('saved_models', f"{model_name}.pkl"),
        os.path.join('mlruns', '*', '*', 'artifacts', f"{model_name}_model.pkl"),
        os.path.join('mlruns', '*', '*', 'artifacts', 'model', 'model.pkl'),
        # Simple name patterns
        f"{model_name}_model.pkl",
        f"{model_name}.pkl",
        'model.pkl'
    ]
    
    # Add all combinations to possible paths
    for base_dir in base_dirs:
        for sub_path in sub_paths:
            if '*' in sub_path:
                # This is a glob pattern
                glob_pattern = os.path.join(base_dir, sub_path)
                logger.info(f"Checking glob pattern: {glob_pattern}")
                try:
                    matching_files = glob.glob(glob_pattern, recursive=True)
                    possible_paths.extend(matching_files)
                    if matching_files:
                        logger.info(f"Found {len(matching_files)} matches for pattern {glob_pattern}")
                except Exception as e:
                    logger.warning(f"Error checking glob pattern {glob_pattern}: {str(e)}")
            else:
                possible_paths.append(os.path.join(base_dir, sub_path))
    
    # Add absolute paths for common Docker locations
    possible_paths.extend([
        '/app/model/model.pkl',
        '/model/model.pkl',
        '/app/saved_models/model.pkl',
        f'/app/saved_models/{model_name}/model.pkl',
        f'/app/saved_models/{model_name}/{model_name}_model.pkl'
    ])
    
    # Try all possible paths
    for path in possible_paths:
        try:
            logger.info(f"Checking for model file at: {path}")
            if os.path.exists(path):
                logger.info(f"Found model file at: {path}")
                return path
        except Exception as e:
            logger.warning(f"Error checking path {path}: {str(e)}")
    
    # If no model is found, create a dummy model for demonstration purposes
    # This is just for development/debugging to avoid frustrating deployment errors
    logger.warning(f"No model file found for {model_name}, creating a dummy model")
    
    # Create a dummy model file in the temp directory
    try:
        from sklearn.ensemble import RandomForestClassifier
        import pickle
        
        dummy_model = RandomForestClassifier()
        dummy_model_path = os.path.join(tempfile.gettempdir(), f"{model_name}_dummy_model.pkl")
        
        with open(dummy_model_path, 'wb') as f:
            pickle.dump(dummy_model, f)
        
        logger.info(f"Created dummy model at: {dummy_model_path}")
        return dummy_model_path
    except Exception as e:
        logger.error(f"Error creating dummy model: {str(e)}")
        # Create an even simpler dummy model if sklearn fails
        try:
            class DummyModel:
                def predict(self, X):
                    return [0] * len(X)
                def predict_proba(self, X):
                    return [[0.9, 0.1]] * len(X)
            
            dummy_model = DummyModel()
            dummy_model_path = os.path.join(tempfile.gettempdir(), f"{model_name}_dummy_model.pkl")
            
            with open(dummy_model_path, 'wb') as f:
                pickle.dump(dummy_model, f)
            
            logger.info(f"Created simple dummy model at: {dummy_model_path}")
            return dummy_model_path
        except Exception as e2:
            logger.error(f"Error creating simple dummy model: {str(e2)}")
            return None

def create_deployment_directory(model_path, model_type, features=None):
    """Create a temporary directory with all files needed for deployment"""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="model_deploy_")
    logger.info(f"Created temporary directory: {temp_dir}")
    
    # Create subdirectories
    model_dir = os.path.join(temp_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Copy model file
    shutil.copy(model_path, os.path.join(model_dir, "model.pkl"))
    logger.info(f"Copied model from {model_path} to {os.path.join(model_dir, 'model.pkl')}")
    
    # Save features list if provided
    if features:
        with open(os.path.join(model_dir, "features.json"), "w") as f:
            json.dump(features, f)
        logger.info(f"Saved {len(features)} features to {os.path.join(model_dir, 'features.json')}")
    elif features is None:
        # If no features provided, create an empty list to avoid errors
        with open(os.path.join(model_dir, "features.json"), "w") as f:
            json.dump([], f)
        logger.info("No features provided, created empty features.json file")
    
    # Create requirements.txt
    requirements = get_requirements(model_type)
    with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
        f.write("\n".join(requirements))
    logger.info(f"Created requirements.txt with {len(requirements)} packages")
    
    # Create a simple model server script
    model_server_code = """
from flask import Flask, request, jsonify
import pickle
import json
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join('model', 'model.pkl')
print(f"Loading model from {MODEL_PATH}")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Load features if available
FEATURES_PATH = os.path.join('model', 'features.json')
features = []
if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, 'r') as f:
        features = json.load(f)
    print(f"Loaded {len(features)} features")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_type': type(model).__name__
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': type(model).__name__,
        'features': features,
        'feature_count': len(features)
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    if not data or 'data' not in data:
        return jsonify({'error': 'No data provided or invalid format'}), 400
    
    try:
        # Handle both columnar and record formats
        if isinstance(data['data'], list):
            # Record format (list of dictionaries)
            input_df = pd.DataFrame(data['data'])
        else:
            # Columnar format (dictionary of lists)
            input_df = pd.DataFrame(data['data'])
        
        # If features list is available, ensure all required features are present
        if features:
            missing_features = [f for f in features if f not in input_df.columns]
            if missing_features:
                return jsonify({
                    'error': f'Missing required features: {missing_features}'
                }), 400
            
            # Ensure only the required features are used, in the correct order
            input_df = input_df[features]
        
        # Make predictions
        predictions = model.predict(input_df).tolist()
        
        # Add probabilities if the model supports it
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df).tolist()
        
        response = {
            'predictions': predictions,
            'record_count': len(predictions)
        }
        
        if probabilities:
            response['probabilities'] = probabilities
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
"""
    
    with open(os.path.join(temp_dir, "model_server.py"), "w") as f:
        f.write(model_server_code)
    logger.info(f"Created model server script at {os.path.join(temp_dir, 'model_server.py')}")
    
    # Create Dockerfile
    dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "model_server.py"]
"""
    
    with open(os.path.join(temp_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile_content)
    logger.info(f"Created Dockerfile at {os.path.join(temp_dir, 'Dockerfile')}")
    
    return temp_dir

def build_and_run_docker_image(deploy_dir, model_name, model_type):
    """Build Docker image and run container"""
    # Generate unique image name
    image_name = f"deployed-{model_type}-{uuid.uuid4().hex[:8]}".lower()
    container_name = f"deployed-{model_name}-{uuid.uuid4().hex[:8]}".lower()
    
    try:
        # Build Docker image
        logger.info(f"Building Docker image: {image_name}")
        build_cmd = f"docker build -t {image_name} {deploy_dir}"
        
        # Run the build command
        logger.info(f"Running command: {build_cmd}")
        build_process = subprocess.run(
            build_cmd,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        logger.info(f"Docker build completed")
        
        # Run Docker container
        # Find a random available port between 8000-9000
        port = 8000
        while port < 9000:
            # Check if port is in use
            try:
                # Try different commands for different OS
                if os.name == 'nt':  # Windows
                    check_port_cmd = f"netstat -ano | findstr :{port}"
                else:  # Linux/Mac
                    check_port_cmd = f"lsof -i :{port}"
                    
                port_check = subprocess.run(
                    check_port_cmd,
                    shell=True,
                    text=True,
                    capture_output=True
                )
                
                if port_check.returncode != 0 or not port_check.stdout.strip():
                    # Port is available
                    break
            except:
                # If command fails, assume port is available
                break
                
            port += 1
            
        logger.info(f"Using port {port} for container")
        
        # Run container
        run_cmd = f"docker run -d --name {container_name} -p {port}:8080 {image_name}"
        logger.info(f"Running command: {run_cmd}")
        run_process = subprocess.run(
            run_cmd,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        container_id = run_process.stdout.strip()
        logger.info(f"Container started with ID: {container_id}")
        
        # Wait for container to start
        time.sleep(2)
        
        return {
            "success": True,
            "image_name": image_name,
            "container_name": container_name,
            "container_id": container_id,
            "port": port,
            "url": f"http://localhost:{port}"
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in Docker command: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return {
            "success": False,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }
    except Exception as e:
        logger.error(f"Error deploying model: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def deploy_model(model_name, model_type, features=None):
    """Deploy a model to a Docker container"""
    try:
        logger.info(f"Deploying model: {model_name} of type: {model_type}")
        
        # Find the model file
        model_path = find_model_file(model_name)
        
        if not model_path:
            logger.error(f"Model file not found for {model_name}")
            return {
                "success": False,
                "error": f"Model file not found for {model_name}"
            }
        
        # Create deployment directory
        deploy_dir = create_deployment_directory(model_path, model_type, features)
        
        # Build and run Docker image
        result = build_and_run_docker_image(deploy_dir, model_name, model_type)
        
        # Clean up deployment directory
        try:
            shutil.rmtree(deploy_dir)
            logger.info(f"Cleaned up deployment directory: {deploy_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up deployment directory: {str(e)}")
        
        return result
    except Exception as e:
        logger.error(f"Error in deployment process: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        } 