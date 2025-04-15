#!/usr/bin/env python3
"""
Script to update metric values for existing models in the database.
This extracts the actual metric values from model_results instead of generating fake values.
"""

import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import json
import logging
from dotenv import load_dotenv
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Import the TrainingModel directly from the users module
from db.users import TrainingModel, TrainingRun

def update_metrics_from_stored_results():
    """Update model metrics using stored model results files"""
    with app.app_context():
        # Look for stored model results in temp directory
        temp_dir = 'static/temp'
        result_files = glob.glob(f"{temp_dir}/**/*.json", recursive=True)
        logger.info(f"Found {len(result_files)} JSON files to check for model results")
        
        updated_models = 0
        metrics_found = 0
        
        # Get all models that need metrics
        models = TrainingModel.query.filter(TrainingModel.metric_value.is_(None)).all()
        logger.info(f"Found {len(models)} models without metric values")
        
        # Group models by run_id for faster processing
        models_by_run = {}
        for model in models:
            if model.run_id not in models_by_run:
                models_by_run[model.run_id] = []
            models_by_run[model.run_id].append(model)
        
        # Process each result file
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Check if this is a model result
                if not isinstance(data, dict):
                    continue
                    
                # Check for model metrics in various formats
                all_models = data.get('all_models', {})
                best_scores = data.get('best_scores', {})
                model_metrics = data.get('model_metrics', {})
                best_models = data.get('best_models', {})
                
                if all_models or best_scores or model_metrics or best_models:
                    logger.info(f"Found model results in {file_path}")
                    metrics_found += 1
                else:
                    continue
                
                # Try to match this result file to runs
                for run_id, run_models in models_by_run.items():
                    for model in run_models:
                        if model.metric_value is not None:
                            continue  # Skip if already has a value
                            
                        # Extract the metric name from model_name
                        metric_name = model.metric_name or model.model_name.replace("best_model_for_", "")
                        if not metric_name:
                            continue
                        
                        # Try to find metric value in the result data
                        metric_value = None
                        
                        # Method 1: Check all_models
                        if metric_name in all_models:
                            model_data = all_models.get(metric_name, {})
                            if 'score' in model_data:
                                metric_value = model_data['score']
                                
                        # Method 2: Check best_scores
                        if metric_value is None and metric_name in best_scores:
                            metric_value = best_scores[metric_name]
                            
                        # Method 3: Check model_metrics
                        if metric_value is None and model_metrics:
                            # Extract model base name from model_name
                            base_model_name = None
                            parts = model.model_name.split('_')
                            if len(parts) > 1:
                                base_model_name = parts[-1]
                                
                            if base_model_name and base_model_name in model_metrics:
                                model_metric_data = model_metrics[base_model_name]
                                if metric_name in model_metric_data:
                                    metric_value = model_metric_data[metric_name]
                        
                        # Method 4: Check best_models
                        if metric_value is None and metric_name in best_models:
                            model_data = best_models[metric_name]
                            if 'score' in model_data:
                                metric_value = model_data['score']
                                
                        # If we found a metric value, update the model
                        if metric_value is not None:
                            model.metric_value = float(metric_value)
                            logger.info(f"Updated model {model.id} with {metric_name} = {metric_value}")
                            updated_models += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        # Commit the changes if any updates were made
        if updated_models > 0:
            db.session.commit()
            logger.info(f"Updated {updated_models} models with actual metric values from {metrics_found} result files")
        else:
            logger.info(f"No models were updated (checked {metrics_found} result files)")

if __name__ == "__main__":
    logger.info("Starting model metric update...")
    update_metrics_from_stored_results()
    logger.info("Model metric update completed!") 