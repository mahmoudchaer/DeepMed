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
import sys

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

def update_metrics_from_stored_results(force_update=False):
    """Update model metrics using stored model results files
    
    Args:
        force_update: If True, will update all models' metrics, not just NULL ones
    """
    with app.app_context():
        # Look for stored model results in multiple directories
        search_paths = [
            'static/temp/**/*.json',
            'static/models/**/*.json',
            'static/results/**/*.json',
            'static/downloads/**/*.json'
        ]
        
        all_files = []
        for path in search_paths:
            all_files.extend(glob.glob(path, recursive=True))
            
        result_files = list(set(all_files))  # Remove duplicates
        logger.info(f"Found {len(result_files)} JSON files to check for model results")
        
        updated_models = 0
        metrics_found = 0
        
        # Get all models that need metrics
        query = TrainingModel.query
        if not force_update:
            query = query.filter(TrainingModel.metric_value.is_(None))
        models = query.all()
        
        # Get all training runs for reference
        all_runs = {run.id: run for run in TrainingRun.query.all()}
        
        logger.info(f"Found {len(models)} models {'without metric values' if not force_update else 'to update'}")
        
        # Group models by run_id for faster processing
        models_by_run = {}
        for model in models:
            if model.run_id not in models_by_run:
                models_by_run[model.run_id] = []
            models_by_run[model.run_id].append(model)
        
        # Also create a lookup by model name
        models_by_name = {}
        for model in models:
            if model.model_name not in models_by_name:
                models_by_name[model.model_name] = []
            models_by_name[model.model_name].append(model)
        
        # Process each result file
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Skip non-dictionary data
                if not isinstance(data, dict):
                    continue
                
                # Check all common structures where metrics might be stored
                metric_structures = [
                    data.get('all_models', {}),
                    data.get('best_scores', {}),
                    data.get('model_metrics', {}),
                    data.get('best_models', {}),
                    data.get('metrics', {}),
                    data.get('results', {}),
                    data.get('model_results', {}),
                    data.get('evaluation', {})
                ]
                
                has_metrics = any(bool(structure) for structure in metric_structures)
                
                if has_metrics:
                    logger.info(f"Found potential model results in {file_path}")
                    metrics_found += 1
                else:
                    # Check if this is a session file that might contain model_results
                    if 'model_results' in data:
                        model_results = data.get('model_results', {})
                        if isinstance(model_results, dict):
                            metric_structures.append(model_results.get('all_models', {}))
                            metric_structures.append(model_results.get('best_scores', {}))
                            metric_structures.append(model_results.get('best_models', {}))
                            has_metrics = any(bool(structure) for structure in metric_structures)
                            if has_metrics:
                                logger.info(f"Found metrics in session data from {file_path}")
                                metrics_found += 1
                
                if not has_metrics:
                    continue
                
                # Extract run_id from filename if possible
                file_run_id = None
                filename = os.path.basename(file_path)
                if '_run_' in filename:
                    try:
                        file_run_id = int(filename.split('_run_')[1].split('_')[0])
                        logger.info(f"Extracted run_id {file_run_id} from filename {filename}")
                    except:
                        pass
                
                # Try to match this result file to runs
                for run_id, run_models in models_by_run.items():
                    # If we extracted a run_id from the filename, only process that run
                    if file_run_id is not None and file_run_id != run_id:
                        continue
                        
                    for model in run_models:
                        if force_update or model.metric_value is None:
                            # Extract the metric name from model_name
                            metric_name = model.metric_name or model.model_name.replace("best_model_for_", "")
                            if not metric_name:
                                continue
                            
                            # Try to find metric value in the result data
                            metric_value = None
                            model_info = None
                            
                            # Check all possible sources of metric values
                            for structure in metric_structures:
                                if not structure:
                                    continue
                                
                                # Method 1: Direct lookup by metric name
                                if metric_name in structure:
                                    value = structure[metric_name]
                                    if isinstance(value, (int, float)):
                                        metric_value = value
                                        logger.info(f"Found direct metric value for {metric_name}: {metric_value}")
                                        break
                                    elif isinstance(value, dict) and 'score' in value:
                                        metric_value = value['score']
                                        logger.info(f"Found score in dict for {metric_name}: {metric_value}")
                                        break
                                
                                # Method 2: Look for model_name in structure
                                base_model_name = None
                                parts = model.model_name.split('_')
                                if len(parts) > 1:
                                    base_model_name = parts[-1]
                                    
                                if base_model_name and base_model_name in structure:
                                    model_data = structure[base_model_name]
                                    if isinstance(model_data, dict):
                                        if metric_name in model_data:
                                            metric_value = model_data[metric_name]
                                            logger.info(f"Found metric in model_data: {metric_value}")
                                            break
                                        elif 'metrics' in model_data and isinstance(model_data['metrics'], dict):
                                            if metric_name in model_data['metrics']:
                                                metric_value = model_data['metrics'][metric_name]
                                                logger.info(f"Found metric in nested metrics: {metric_value}")
                                                break
                            
                            # Look for CV scores if they exist in model_results
                            if 'cv_scores' in data:
                                cv_scores = data.get('cv_scores', {})
                                if metric_name in cv_scores:
                                    cv_data = cv_scores[metric_name]
                                    if isinstance(cv_data, dict):
                                        if 'mean' in cv_data and 'std' in cv_data:
                                            model.cv_score_mean = float(cv_data['mean'])
                                            model.cv_score_std = float(cv_data['std'])
                                            logger.info(f"Updated CV scores for model {model.id}: {model.cv_score_mean} ± {model.cv_score_std}")
                                
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
        
        return updated_models

if __name__ == "__main__":
    logger.info("Starting model metric update...")
    
    # Check if we should force update all models
    force_update = len(sys.argv) > 1 and sys.argv[1] == '--force'
    
    update_metrics_from_stored_results(force_update=force_update)
    logger.info("Model metric update completed!") 