#!/usr/bin/env python
"""
MLflow Initialization Script
This script initializes the MLflow directory structure for proper model tracking.
Run this script before starting Docker services to ensure all directories exist.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MLRUNS_DIR = os.path.join(ROOT_DIR, 'docker_versions', 'mlruns')

def initialize_mlflow_directory():
    """Initialize the MLflow directory structure"""
    # Create mlruns directory if it doesn't exist
    if not os.path.exists(MLRUNS_DIR):
        logger.info(f"Creating MLflow tracking directory: {MLRUNS_DIR}")
        Path(MLRUNS_DIR).mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"MLflow tracking directory already exists: {MLRUNS_DIR}")
    
    # Set permissions
    try:
        os.chmod(MLRUNS_DIR, 0o777)  # Full permissions
        logger.info("Set permissions on MLflow directory")
    except Exception as e:
        logger.error(f"Error setting permissions: {str(e)}")
    
    # Create basic structure for MLflow (0 is the default experiment)
    for experiment_id in ['0']:
        experiment_dir = os.path.join(MLRUNS_DIR, experiment_id)
        if not os.path.exists(experiment_dir):
            Path(experiment_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created experiment directory: {experiment_id}")
        
        # Create .meta file
        meta_path = os.path.join(experiment_dir, 'meta.yaml')
        if not os.path.exists(meta_path):
            with open(meta_path, 'w') as f:
                f.write(f"artifact_location: {os.path.join('file://', experiment_dir)}\n")
                f.write(f"experiment_id: {experiment_id}\n")
                f.write("lifecycle_stage: active\n")
                f.write("name: Default\n")
            logger.info(f"Created meta.yaml for experiment {experiment_id}")

def initialize_model_experiments():
    """Initialize experiments for each model type"""
    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    logger.info(f"Set MLflow tracking URI to: file://{MLRUNS_DIR}")
    
    # Create experiments for each model type
    model_types = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "svm",
        "knn",
        "naive_bayes"
    ]
    
    client = MlflowClient()
    experiment_names = [exp.name for exp in client.list_experiments()]
    
    for model_type in model_types:
        experiment_name = f"{model_type}_experiment"
        if experiment_name not in experiment_names:
            try:
                mlflow.create_experiment(experiment_name)
                logger.info(f"Created experiment: {experiment_name}")
            except Exception as e:
                logger.error(f"Error creating experiment {experiment_name}: {str(e)}")
        else:
            logger.info(f"Experiment {experiment_name} already exists")

if __name__ == "__main__":
    print("Initializing MLflow directory structure...")
    initialize_mlflow_directory()
    print("Initializing model experiments...")
    try:
        initialize_model_experiments()
    except Exception as e:
        logger.error(f"Error initializing model experiments: {str(e)}")
    print("MLflow directory structure initialized successfully!")
    
    # Print instructions for next steps
    print("\nNext steps:")
    print("1. Run your Docker services:")
    print("   docker-compose -f docker_versions/docker-compose.yml up -d")
    print("2. Check if all services are running:")
    print("   docker-compose -f docker_versions/docker-compose.yml ps")
    print("3. Start training models through your application") 