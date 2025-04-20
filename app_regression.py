from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file, copy_current_request_context
from flask_login import login_required, current_user
import os
import pandas as pd
import numpy as np
import logging
import json
import time
from datetime import datetime
import requests
import tempfile
import secrets
import uuid
from werkzeug.utils import secure_filename
import threading
import queue
import multiprocessing
from flask import abort
import shutil
import joblib  # Add joblib import for direct model loading
import io

# Import common components from app_api.py
from app_api import app, MEDICAL_ASSISTANT_URL  # Only import MEDICAL_ASSISTANT_URL
from app_api import is_service_available, get_temp_filepath, safe_requests_post, cleanup_session_files, check_services
from app_api import allowed_file, load_data, clean_data_for_json, SafeJSONEncoder, logger
from app_api import save_to_temp_file, load_from_temp_file, check_session_size

# Import database models
from db.users import db, User, TrainingRun, TrainingModel, PreprocessingData

# Define regression-specific service URLs - Do not rely on default values that use localhost
# Check the environment variables first
REGRESSION_DATA_CLEANER_URL = os.environ.get('REGRESSION_DATA_CLEANER_URL') 
if not REGRESSION_DATA_CLEANER_URL:
    # Use localhost with ports - MODIFIED for direct access
    REGRESSION_DATA_CLEANER_URL = 'http://localhost:5031'
    logger.warning(f"REGRESSION_DATA_CLEANER_URL not found in environment, using default: {REGRESSION_DATA_CLEANER_URL}")

REGRESSION_FEATURE_SELECTOR_URL = os.environ.get('REGRESSION_FEATURE_SELECTOR_URL')
if not REGRESSION_FEATURE_SELECTOR_URL:
    REGRESSION_FEATURE_SELECTOR_URL = 'http://localhost:5032'
    logger.warning(f"REGRESSION_FEATURE_SELECTOR_URL not found in environment, using default: {REGRESSION_FEATURE_SELECTOR_URL}")

REGRESSION_MODEL_COORDINATOR_URL = os.environ.get('REGRESSION_MODEL_COORDINATOR_URL')
if not REGRESSION_MODEL_COORDINATOR_URL:
    REGRESSION_MODEL_COORDINATOR_URL = 'http://localhost:5040'
    logger.warning(f"REGRESSION_MODEL_COORDINATOR_URL not found in environment, using default: {REGRESSION_MODEL_COORDINATOR_URL}")

# Fix: Use the correct predictor service name from docker-compose.yml
REGRESSION_PREDICTOR_SERVICE_URL = os.environ.get('REGRESSION_PREDICTOR_SERVICE_URL')
if not REGRESSION_PREDICTOR_SERVICE_URL:
    REGRESSION_PREDICTOR_SERVICE_URL = 'http://localhost:5050'
    logger.warning(f"REGRESSION_PREDICTOR_SERVICE_URL not found in environment, using default: {REGRESSION_PREDICTOR_SERVICE_URL}")

# Log the actual URLs being used
logger.info(f"Regression Data Cleaner URL: {REGRESSION_DATA_CLEANER_URL}")
logger.info(f"Regression Feature Selector URL: {REGRESSION_FEATURE_SELECTOR_URL}")
logger.info(f"Regression Model Coordinator URL: {REGRESSION_MODEL_COORDINATOR_URL}")
logger.info(f"Regression Predictor Service URL: {REGRESSION_PREDICTOR_SERVICE_URL}")

# Global variables to track regression training status
regression_training_status = {
    'status': 'idle',  # 'idle', 'in_progress', 'complete', 'failed'
    'progress': 0,
    'message': '',
    'model_statuses': {
        'Linear Regression': 'waiting',
        'Lasso Regression': 'waiting',
        'Ridge Regression': 'waiting',
        'Random Forest Regression': 'waiting',
        'K-Nearest Neighbors': 'waiting',
        'XGBoost Regression': 'waiting'
    },
    'error': None,
    'stop_requested': False,
    'training_thread': None
}

def ensure_regression_models_saved(user_id, run_id, model_result):
    """Ensure that regression models are saved to the TrainingModel table.
    
    This function is specifically for regression models and handles the
    regression-specific metrics (r2, rmse, mae, mse).
    """
    try:
        # Check if there are any results to save
        if 'results' not in model_result or not model_result['results']:
            logger.warning("No model results found to save")
            return False
        
        # Get the regression models results
        model_results = model_result['results']
        
        # Sort by R2 score (descending order) to get the best models
        if len(model_results) > 0 and 'r2' in model_results[0]:
            sorted_models = sorted(model_results, key=lambda x: x.get('r2', 0), reverse=True)
            # Limit to top 4 models
            top_models = sorted_models[:4]
        else:
            top_models = model_results[:min(4, len(model_results))]
            
        # Check if models already exist for this run
        existing_models = TrainingModel.query.filter_by(run_id=run_id).count()
        if existing_models >= len(top_models):
            logger.info(f"Found {existing_models} models already saved for run_id {run_id}")
            return True
            
        # Save each model
        for model_info in top_models:
            model_name = model_info.get('name', 'unknown_model')
            model_url = model_info.get('model_url')
            
            if not model_url:
                logger.warning(f"No URL found for model {model_name}")
                continue
                
            # Extract filename from URL
            filename = model_url.split('/')[-1] if '/' in model_url else model_url
            
            # Check if this model is already saved
            existing_model = TrainingModel.query.filter_by(
                run_id=run_id,
                model_name=model_name
            ).first()
            
            if existing_model:
                logger.info(f"Model {model_name} already exists for run_id {run_id}")
                continue
            
            # Get metrics from the model info
            metrics = model_info.get('metrics', {})
            if not metrics and 'r2' in model_info:
                # If metrics not in separate dict but directly in model_info
                metrics = {
                    'r2': model_info.get('r2', 0),
                    'rmse': model_info.get('rmse', 0),
                    'mae': model_info.get('mae', 0),
                    'mse': model_info.get('mse', 0)
                }
            
            # Primary metric for regression is usually R2
            r2_score = metrics.get('r2', 0) if metrics else model_info.get('r2', 0)
            
            # Create and save the model record
            model_record = TrainingModel(
                user_id=user_id,
                run_id=run_id,
                model_name=model_name,
                model_url=model_url,
                file_name=filename,
                metric_name='r2',
                metric_value=r2_score
            )
            
            db.session.add(model_record)
            logger.info(f"Added regression model {model_name} with R2={r2_score} to database for run_id {run_id}")
        
        # Commit all changes
        db.session.commit()
        logger.info(f"Committed regression models to database for run_id {run_id}")
        
        # Verify models were saved
        saved_count = TrainingModel.query.filter_by(run_id=run_id).count()
        logger.info(f"Verified {saved_count} regression models saved for run_id {run_id}")
        
        return True
            
    except Exception as e:
        logger.error(f"Error ensuring regression models are saved: {str(e)}")
        logger.error(f"Error details: {str(e)}", exc_info=True)
        return False

# Add regression services to the services check
def check_regression_services():
    """Check the health of regression-specific services"""
    status = {}
    regression_services = {
        "Regression Data Cleaner": {"url": REGRESSION_DATA_CLEANER_URL, "endpoint": "/health"},
        "Regression Feature Selector": {"url": REGRESSION_FEATURE_SELECTOR_URL, "endpoint": "/health"},
        "Regression Model Coordinator": {"url": REGRESSION_MODEL_COORDINATOR_URL, "endpoint": "/health"},
        "Regression Predictor Service": {"url": REGRESSION_PREDICTOR_SERVICE_URL, "endpoint": "/health"}
    }
    
    for name, service_info in regression_services.items():
        url = service_info["url"]
        endpoint = service_info["endpoint"]
        try:
            logger.info(f"Checking health of {name} at {url}{endpoint}")
            response = requests.get(f"{url}{endpoint}", timeout=5)  # Increased timeout
            if response.status_code == 200:
                status[name] = "healthy"
                # Log more details from the response if available
                try:
                    response_data = response.json()
                    logger.info(f"{name} health check response: {response_data}")
                except:
                    pass
            else:
                status[name] = f"unhealthy - {response.status_code}"
                logger.error(f"{name} returned status code {response.status_code}")
                try:
                    logger.error(f"Response content: {response.text[:200]}")  # Log first 200 chars
                except:
                    pass
        except Exception as e:
            logger.error(f"Error checking {name} health: {str(e)}")
            status[name] = f"unreachable - {str(e)[:100]}"  # Include more error details
    
    # Log the final status of all services
    logger.info(f"Regression services status: {status}")
    return status

@app.route('/upload_regression', methods=['POST'])
@login_required
def upload_regression():
    # Double check authentication - ensure user is logged in
    if not current_user.is_authenticated:
        logger.warning("Regression upload attempted without authentication")
        flash('Please log in to upload files.', 'warning')
        return redirect(url_for('login'))
        
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('train_regression'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('train_regression'))
    
    if file and allowed_file(file.filename):
        # Clean up previous upload if exists
        if 'uploaded_file_regression' in session and os.path.exists(session['uploaded_file_regression']):
            try:
                os.remove(session['uploaded_file_regression'])
            except:
                pass
        
        # Generate unique filename for temporary storage
        filepath = get_temp_filepath(file.filename)
        file.save(filepath)
        
        # Load the data to validate it
        data, result = load_data(filepath)
        if data is None:
            # Clean up invalid file
            if os.path.exists(filepath):
                os.remove(filepath)
            flash(result, 'error')
            return redirect(url_for('train_regression'))
        
        session['uploaded_file_regression'] = filepath
        session['file_stats_regression'] = result
        
        # Store data columns for later use
        session['data_columns_regression'] = data.columns.tolist()
        
        # Redirect to regression training page
        return redirect(url_for('train_regression'))
    
    flash('Invalid file type. Please upload a CSV or Excel file.', 'error')
    return redirect(url_for('train_regression'))

@app.route('/train_regression', methods=['GET', 'POST'])
@login_required
def train_regression():
    """Route for training regression models"""
    # Check if the user is logged in
    if not current_user.is_authenticated:
        flash('Please log in to access the regression training page.', 'info')
        return redirect('/login', code=302)
    
    # Check if we need to reset the dataset (new upload requested)
    if request.args.get('new') == '1':
        # Clear dataset from session
        if 'uploaded_file_regression' in session:
            # Delete the file if it exists
            if os.path.exists(session['uploaded_file_regression']):
                try:
                    os.remove(session['uploaded_file_regression'])
                except:
                    pass
            # Remove from session
            session.pop('uploaded_file_regression', None)
            session.pop('file_stats_regression', None)
            session.pop('data_columns_regression', None)
            
        # Redirect to clean URL
        return redirect(url_for('train_regression'))
    
    filepath = session.get('uploaded_file_regression')
    
    if not filepath:
        # If accessed directly without upload, show the upload interface
        # Check regression services health for status display
        services_status = check_regression_services()
        
        return render_template('train_regression.html', services_status=services_status)
    
    data, _ = load_data(filepath)
    
    # Handle case where data could not be loaded
    if data is None:
        flash('Error loading data from the uploaded file. Please upload a valid file.', 'error')
        services_status = check_regression_services()
        return render_template('train_regression.html', services_status=services_status)
    
    if request.method == 'POST':
        # Get target column from form
        target_column = request.form.get('target_column')
        session['target_column'] = target_column
        
        # Always use fixed test size of 20%
        session['test_size'] = 0.2
        
        try:
            # Check if required services are available directly
            logger.info("Checking required regression services before training:")
            unavailable_services = []
            
            # Check required services directly using custom health check
            regression_service_status = check_regression_services()
            
            for service_name, status in regression_service_status.items():
                if status != "healthy":
                    logger.error(f"Service {service_name} is not available: {status}")
                    unavailable_services.append(service_name)
                else:
                    logger.info(f"Service {service_name} is available")
            
            if unavailable_services:
                error_message = f"The following regression services are not available: {', '.join(unavailable_services)}. Cannot proceed with training."
                logger.error(error_message)
                flash(error_message, 'error')
                return redirect(url_for('train_regression'))
                
            # Reset training status
            global regression_training_status
            regression_training_status = {
                'status': 'in_progress',
                'progress': 5,
                'message': 'Initializing regression model training...',
                'model_statuses': {
                    'Linear Regression': 'waiting',
                    'Lasso Regression': 'waiting',
                    'Ridge Regression': 'waiting',
                    'Random Forest Regression': 'waiting',
                    'K-Nearest Neighbors': 'waiting',
                    'XGBoost Regression': 'waiting'
                },
                'error': None,
                'stop_requested': False,
                'training_thread': None
            }
            
            # Create a copy of necessary session data
            session_data = {
                'filepath': filepath,
                'target_column': target_column,
                'user_id': current_user.id,
                'cleaned_file': session.get('cleaned_file'),
                'selected_features_regression_file': session.get('selected_features_regression_file')
            }
            
            # Wrap the training function with the current request context
            @copy_current_request_context
            def run_training():
                train_regression_models_background(session_data)
            
            # Start training in background thread
            training_thread = threading.Thread(
                target=run_training
            )
            training_thread.daemon = True
            regression_training_status['training_thread'] = training_thread
            training_thread.start()
            
            # Redirect to loading page while training happens in background
            return render_template('loading_regression.html')
            
        except Exception as e:
            logger.error(f"Error starting regression training: {str(e)}", exc_info=True)
            flash(f"Error starting regression training: {str(e)}", 'error')
            return redirect(url_for('train_regression'))
    
    # Get AI recommendations for the dataset (via Medical Assistant API) - OPTIONAL
    ai_recommendations = None
    try:
        # Check if Medical Assistant API is available
        medical_assistant_available = False
        try:
            response = requests.get(f"{MEDICAL_ASSISTANT_URL}/health", timeout=2)
            medical_assistant_available = response.status_code == 200
        except:
            medical_assistant_available = False
            
        if medical_assistant_available and 'ai_recommendations' not in session and 'ai_recommendations_file' not in session:
            logger.info(f"Sending data to Medical Assistant API for regression analysis")
            
            # Convert to simple Python structure
            data_records = data.replace([np.inf, -np.inf], np.nan).where(pd.notnull(data), None).to_dict(orient='records')
            
            # Use our safe request method
            response = safe_requests_post(
                f"{MEDICAL_ASSISTANT_URL}/analyze_data",
                {
                    "data": data_records,
                    "task_type": "regression"  # Specify regression task
                },
                timeout=30
            )
            
            if response.status_code == 200:
                ai_recommendations = response.json()["recommendations"]
                
                # Save to file instead of storing in session
                recommendations_file = save_to_temp_file(ai_recommendations, 'ai_recommendations_regression')
                session['ai_recommendations_regression_file'] = recommendations_file
                logger.info(f"Saved AI recommendations for regression to {recommendations_file}")
            else:
                logger.warning(f"Medical Assistant API returned an error: {response.text}")
        elif 'ai_recommendations_regression_file' in session:
            # Load from file
            ai_recommendations = load_from_temp_file(session['ai_recommendations_regression_file'])
    except Exception as e:
        logger.error(f"Error getting AI recommendations for regression: {str(e)}", exc_info=True)
        # Don't flash this error to avoid confusing the user
        logger.info("Continuing without AI recommendations for regression")
    
    # Make sure session size is under control
    check_session_size()
    
    return render_template('train_regression.html', 
                          data=data.head().to_html(classes='table table-striped'),
                          columns=data.columns.tolist(),
                          file_stats=session.get('file_stats_regression'),
                          ai_recommendations=ai_recommendations)

@app.route('/models_regression')
@login_required
def models_regression():
    """Route for managing regression models"""
    # Get user's regression models from the database
    user_id = current_user.id
    
    try:
        # Query the database for the user's regression models
        regression_model_types = ['linear_regression', 'lasso_regression', 'ridge_regression', 
                                  'random_forest_regression', 'knn_regression', 'xgboost_regression']
        
        with app.app_context():
            # Get regression training runs
            training_runs = db.session.query(TrainingRun).filter_by(user_id=user_id).all()
            
            # Get models for these runs that have regression model types
            regression_models = db.session.query(TrainingModel).filter(
                TrainingModel.user_id == user_id,
                TrainingModel.model_name.in_(regression_model_types)
            ).order_by(TrainingModel.run_id.desc(), TrainingModel.created_at.desc()).all()
            
            # Group models by run
            models_by_run = {}
            for model in regression_models:
                if model.run_id not in models_by_run:
                    models_by_run[model.run_id] = []
                models_by_run[model.run_id].append(model)
            
            # Create runs data with their models
            runs_data = []
            for run in training_runs:
                if run.id in models_by_run:
                    run_models = models_by_run[run.id]
                    
                    # Get preprocessing data for this run
                    preprocessing_data = db.session.query(PreprocessingData).filter_by(run_id=run.id).first()
                    selected_columns = []
                    if preprocessing_data and preprocessing_data.selected_columns:
                        try:
                            selected_columns = json.loads(preprocessing_data.selected_columns)
                        except:
                            pass
                    
                    runs_data.append({
                        'id': run.id,
                        'name': run.run_name,
                        'created_at': run.created_at,
                        'models': run_models,
                        'selected_features': selected_columns
                    })
        
        return render_template('models_regression.html', runs=runs_data)
    except Exception as e:
        logger.error(f"Error fetching regression models: {str(e)}", exc_info=True)
        flash(f"Error fetching regression models: {str(e)}", 'error')
        return render_template('models_regression.html', runs=[])

@app.route('/regression_prediction')
@login_required
def regression_prediction():
    """Route for making predictions with regression models"""
    # Check if the user is logged in
    if not current_user.is_authenticated:
        flash('Please log in to access the regression prediction page.', 'info')
        return redirect('/login', code=302)
    
    # Generate a CSRF token for logout form if needed
    if 'logout_token' not in session:
        session['logout_token'] = secrets.token_hex(16)
    
    # Get available regression models for the user
    user_id = current_user.id
    regression_models = []
    
    try:
        # Define regression model types
        regression_model_types = ['linear_regression', 'lasso_regression', 'ridge_regression', 
                              'random_forest_regression', 'knn_regression', 'xgboost_regression']
        
        # Query database for user's models
        with app.app_context():
            models = db.session.query(TrainingModel).filter(
                TrainingModel.user_id == user_id,
                TrainingModel.model_name.in_(regression_model_types)
            ).order_by(TrainingModel.metric_value.desc()).all()
            
            regression_models = models
            
    except Exception as e:
        logger.error(f"Error fetching regression models for prediction: {str(e)}", exc_info=True)
        flash(f"Error loading regression models: {str(e)}", 'error')
    
    # Check regression services health for status display
    services_status = check_regression_services()
    
    return render_template('regression_prediction.html', 
                           models=regression_models,
                           services_status=services_status, 
                           logout_token=session['logout_token'])

@app.route('/regression_results')
@login_required
def regression_results():
    """Route for displaying regression model training results"""
    try:
        # Check if the user is logged in
        if not current_user.is_authenticated:
            flash('Please log in to access the regression results page.', 'info')
            return redirect('/login', code=302)
        
        # Get model results
        global regression_training_status
        model_result = regression_training_status.get('model_result')
        run_id = regression_training_status.get('last_regression_training_run_id')
        
        # If still no model_result, try to get from database
        if not model_result or not run_id:
            logger.info("No model results found, attempting to fetch from database")
            with app.app_context():
                # Get the user's most recent regression training run
                regression_model_types = ['linear_regression', 'lasso_regression', 'ridge_regression', 
                                          'random_forest_regression', 'knn_regression', 'xgboost_regression']
                
                # First try to get the most recent run ID that has models
                most_recent_run = db.session.query(TrainingRun).join(
                    TrainingModel, TrainingModel.run_id == TrainingRun.id
                ).filter(
                    TrainingRun.user_id == current_user.id,
                    TrainingModel.model_name.in_(regression_model_types)
                ).order_by(TrainingRun.created_at.desc()).first()
                
                if most_recent_run:
                    run_id = most_recent_run.id
                    logger.info(f"Found most recent regression run with ID {run_id}")
                    
                    # Get models for this run
                    regression_models = db.session.query(TrainingModel).filter(
                        TrainingModel.run_id == run_id,
                        TrainingModel.model_name.in_(regression_model_types)
                    ).all()
                    
                    # Build model_result structure
                    results = []
                    for model in regression_models:
                        # Try to parse metrics
                        metrics = {}
                        try:
                            if model.metrics:
                                metrics = json.loads(model.metrics)
                        except:
                            # Fallback to simple metrics
                            metrics = {
                                'r2': model.metric_value if model.metric_value is not None else 0,
                                'rmse': 0,
                                'mae': 0, 
                                'mse': 0
                            }
                        
                        # Create model result structure
                        model_data = {
                            'name': model.model_name,
                            'model_url': model.model_url,
                            'metrics': metrics,
                            'parameters': {},
                            'r2': metrics.get('r2', model.metric_value if model.metric_value is not None else 0),
                            'rmse': metrics.get('rmse', 0),
                            'mae': metrics.get('mae', 0),
                            'mse': metrics.get('mse', 0)
                        }
                        
                        results.append(model_data)
                    
                    if results:
                        model_result = {
                            'run_id': run_id,
                            'results': results,
                            'dataset_size': 'Unknown',
                            'training_time': 'Unknown'
                        }
                        logger.info(f"Built model result from database with {len(results)} models")
                else:
                    logger.warning("No regression training runs found in database")
        
        # If no model result from training status, try from session as fallback
        if not model_result:
            model_result = session.get('regression_model_results')
            run_id = session.get('last_regression_training_run_id')
        
        # If still no model_result, redirect to training page
        if not model_result:
            flash('No recent regression training results found. Please train a model first.', 'warning')
            return redirect(url_for('train_regression'))
        
        # Get the run_id from model_result if not already set
        if not run_id and 'run_id' in model_result:
            run_id = model_result.get('run_id')
            
        # Log what we found    
        logger.info(f"Using run_id: {run_id} with {len(model_result.get('results', []))} models")
        
        # Process model results
        results = model_result.get('results', [])
        
        # Find best model for each metric
        best_r2_model = max(results, key=lambda x: x.get('metrics', {}).get('r2', 0) if isinstance(x.get('metrics'), dict) else x.get('r2', 0), default=None)
        best_rmse_model = min(results, key=lambda x: x.get('metrics', {}).get('rmse', float('inf')) if isinstance(x.get('metrics'), dict) else float('inf'), default=None)
        best_mae_model = min(results, key=lambda x: x.get('metrics', {}).get('mae', float('inf')) if isinstance(x.get('metrics'), dict) else float('inf'), default=None)
        best_mse_model = min(results, key=lambda x: x.get('metrics', {}).get('mse', float('inf')) if isinstance(x.get('metrics'), dict) else float('inf'), default=None)
        
        # Get best models dict if it exists in the results
        best_models = model_result.get('best_models', {
            'r2': best_r2_model,
            'rmse': best_rmse_model,
            'mae': best_mae_model,
            'mse': best_mse_model
        })
        
        # Ensure each model has the metrics values properly set
        for model in results:
            if 'metrics' not in model and 'r2' in model:
                model['metrics'] = {
                    'r2': model.get('r2', 0),
                    'rmse': model.get('rmse', 0),
                    'mae': model.get('mae', 0),
                    'mse': model.get('mse', 0)
                }
            elif 'metrics' in model:
                # Ensure all metrics are present
                if 'r2' not in model['metrics']:
                    model['metrics']['r2'] = model.get('r2', 0)
                if 'rmse' not in model['metrics']:
                    model['metrics']['rmse'] = model.get('rmse', 0)
                if 'mae' not in model['metrics']:
                    model['metrics']['mae'] = model.get('mae', 0)
                if 'mse' not in model['metrics']:
                    model['metrics']['mse'] = model.get('mse', 0)
            
            # Add parameters if missing
            if 'parameters' not in model:
                model['parameters'] = {}
            
            # Check if model is the best in any metric
            model['is_best_r2'] = model == best_models.get('r2', best_r2_model)
            model['is_best_rmse'] = model == best_models.get('rmse', best_rmse_model)
            model['is_best_mae'] = model == best_models.get('mae', best_mae_model)
            model['is_best_mse'] = model == best_models.get('mse', best_mse_model)
            model['is_best'] = model['is_best_r2']  # For backwards compatibility
        
        # Sort models by R² score (descending)
        sorted_models = sorted(results, key=lambda x: x.get('metrics', {}).get('r2', 0) if isinstance(x.get('metrics'), dict) else x.get('r2', 0), reverse=True)
        
        # Get top models (up to 4)
        top_models = sorted_models[:min(4, len(sorted_models))]
        
        # Extract the best model
        best_model = top_models[0] if top_models else None
        
        # Save the session
        session['regression_model_results'] = model_result
        session['last_regression_training_run_id'] = run_id
        
        # Get training info
        training_info = {
            'dataset_size': model_result.get('dataset_size', 0),
            'training_time': model_result.get('training_time', 0),
            'run_id': run_id
        }
        
        # Add metrics information for the template
        metrics_info = {
            'r2': 'Coefficient of determination (R²) measures how well the regression model fits the data. Higher is better, with 1 being perfect prediction.',
            'rmse': 'Root Mean Square Error measures the square root of the average squared differences between predicted and actual values. Lower is better.',
            'mae': 'Mean Absolute Error measures the average of absolute differences between predicted and actual values. Lower is better.',
            'mse': 'Mean Squared Error measures the average of squared differences between predicted and actual values. Lower is better.'
        }
        
        # Pass all models to template, highlighting the best ones
        logger.info(f"Rendering regression_results.html with {len(sorted_models)} models")
        return render_template(
            'regression_results.html',
            models=sorted_models,
            top_models=top_models,
            best_model=best_model,
            best_models=best_models,
            training_info=training_info,
            metrics_info=metrics_info,
            model_results=sorted_models,  # Add alias for models to match template variable
            run_id=run_id
        )
    except Exception as e:
        logger.error(f"Error rendering regression results: {str(e)}", exc_info=True)
        flash(f"Error displaying regression results: {str(e)}", 'error')
        return redirect(url_for('train_regression'))

@app.route('/api/predict_regression', methods=['POST'])
@login_required
def api_predict_regression():
    """API endpoint for regression prediction using custom models"""
    try:
        # Check for file uploads
        if 'file' not in request.files:
            return jsonify({'error': 'No data file uploaded'}), 400
            
        data_file = request.files['file']
        if data_file.filename == '':
            return jsonify({'error': 'No selected data file'}), 400
            
        if not allowed_file(data_file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a CSV or Excel file.'}), 400
            
        # Check for model package upload
        model_package = request.files.get('model_package')
        if not model_package or model_package.filename == '':
            return jsonify({'error': 'Please upload a model package'}), 400
        
        # Verify it's a zip file
        if not model_package.filename.endswith('.zip'):
            return jsonify({'error': 'Model package must be a ZIP file'}), 400
            
        # Save both files temporarily to pass to the predictor service
        model_package_path = get_temp_filepath(model_package.filename)
        data_filepath = get_temp_filepath(data_file.filename)
        
        model_package.save(model_package_path)
        data_file.save(data_filepath)
        
        # Load data for metadata and validation
        data, file_stats = load_data(data_filepath)
        if data is None:
            # Clean up files
            if os.path.exists(model_package_path):
                os.remove(model_package_path)
            return jsonify({'error': file_stats}), 400
        
        model_name = "Custom Model"  # Default name
        
        logger.info(f"Using regression predictor service for prediction")
        try:
            # Create multipart form data for request to predictor service
            files = {
                'model_package': (os.path.basename(model_package_path), open(model_package_path, 'rb'), 'application/zip'),
                'input_file': (os.path.basename(data_filepath), open(data_filepath, 'rb'), 'text/csv')
            }
            
            # Make the request to the predictor service
            logger.info(f"Sending request to regression predictor service at {REGRESSION_PREDICTOR_SERVICE_URL}")
            response = requests.post(
                f"{REGRESSION_PREDICTOR_SERVICE_URL}/predict",
                files=files,
                timeout=300  # 5 minute timeout
            )
            
            # Close the file handlers
            files['model_package'][1].close()
            files['input_file'][1].close()
            
            # Clean up temporary files
            os.remove(model_package_path)
            os.remove(data_filepath)
            
            if response.status_code != 200:
                logger.error(f"Error from regression predictor service: {response.text}")
                return jsonify({'error': f'Prediction service error: {response.text}'}), 500
                
            # Extract the predictions result
            result = response.json()
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 400
                
            if 'output_file' not in result:
                return jsonify({'error': 'No prediction results returned from service'}), 500
                
            # Parse the output CSV
            output_csv = result['output_file']
            
            # Debug the CSV output
            logger.info(f"Received CSV output of length: {len(output_csv)}")
            output_lines = output_csv.strip().split('\n')
            if output_lines:
                logger.info(f"First line of CSV: {output_lines[0]}")
                logger.info(f"Total lines in CSV: {len(output_lines)}")
            
            try:
                # Strip any non-CSV content that might be at the beginning
                # Sometimes debug output gets mixed with CSV output
                csv_start = output_csv.find(',')
                if csv_start > 0 and not output_csv.startswith('"'):
                    possible_header_line = output_csv[:csv_start].strip()
                    if ',' not in possible_header_line and '\n' in output_csv:
                        # This might be a debug line, try to find where the CSV actually starts
                        first_newline = output_csv.find('\n')
                        if first_newline > 0:
                            logger.warning(f"Stripping potential non-CSV content: '{output_csv[:first_newline]}'")
                            output_csv = output_csv[first_newline+1:]
                
                # Parse the CSV
                result_df = pd.read_csv(io.StringIO(output_csv))
                logger.info(f"Parsed DataFrame with shape: {result_df.shape}, columns: {result_df.columns.tolist()}")
                
                # Check if the first line might be an error message
                if len(result_df.columns) == 1 and result_df.shape[0] > 0:
                    first_col = result_df.columns[0]
                    first_val = result_df.iloc[0, 0] if not result_df.empty else None
                    
                    if (first_col.lower() == 'error_message' or 
                        first_col.lower() == 'error' or 
                        'error' in first_col.lower()):
                        # This is likely an error
                        logger.error(f"Error returned from predictor: {first_val}")
                        return jsonify({'error': f"Error in prediction: {first_val}"}), 400
                
                # Extract predictions from the result
                if 'prediction' in result_df.columns:
                    predictions = result_df['prediction'].tolist()
                    logger.info(f"Found 'prediction' column with {len(predictions)} values")
                else:
                    # Try to find the prediction column if it has a different name
                    prediction_columns = [col for col in result_df.columns if 'predict' in col.lower()]
                    if prediction_columns:
                        predictions = result_df[prediction_columns[0]].tolist()
                        logger.info(f"Using '{prediction_columns[0]}' as prediction column")
                    else:
                        # Assume the last column is the prediction
                        predictions = result_df.iloc[:, -1].tolist()
                        last_col = result_df.columns[-1] if result_df.columns.size > 0 else "unknown"
                        logger.info(f"Using last column '{last_col}' as prediction column")
                
                # Verify predictions are numeric
                try:
                    # Check the first few predictions
                    sample_predictions = predictions[:5] if len(predictions) >= 5 else predictions
                    logger.info(f"Sample predictions: {sample_predictions}")
                    
                    # Convert string numbers to float if needed
                    if all(isinstance(p, str) for p in sample_predictions if p is not None):
                        logger.info("Converting string predictions to float")
                        predictions = [float(p) if p and p.strip() and p.strip().isdigit() else p for p in predictions]
                except Exception as e:
                    logger.warning(f"Error validating predictions: {str(e)}")
                
                # Convert to records for response
                # Use clean_data_for_json to handle special values like NaN
                try:
                    result_records = clean_data_for_json(result_df)
                    logger.info(f"Converted DataFrame to {len(result_records)} records")
                except Exception as e:
                    logger.error(f"Error converting DataFrame to records: {str(e)}")
                    # Fallback to simple conversion
                    result_records = result_df.to_dict(orient='records')
                    # Replace any problematic values
                    for record in result_records:
                        for k, v in record.items():
                            if pd.isna(v) or v is None:
                                record[k] = None
                
                # Return predictions
                return jsonify({
                    'success': True,
                    'predictions': predictions,
                    'full_results': result_records,
                    'model_name': model_name,
                    'file_stats': file_stats
                })
                
            except pd.errors.ParserError as pe:
                logger.error(f"CSV parsing error: {str(pe)}")
                logger.error(f"Raw CSV data: {output_csv[:200]}...")  # Log first 200 chars
                return jsonify({'error': f'Error parsing prediction results: {str(pe)}'}), 500
                
            except Exception as e:
                logger.error(f"Error processing prediction results: {str(e)}", exc_info=True)
                return jsonify({'error': f'Error processing prediction results: {str(e)}'}), 500
                
        except requests.RequestException as e:
            logger.error(f"Request to predictor service failed: {str(e)}")
            # Clean up files if they still exist
            if os.path.exists(model_package_path):
                os.remove(model_package_path)
            if os.path.exists(data_filepath):
                os.remove(data_filepath)
            return jsonify({'error': f'Failed to connect to prediction service: {str(e)}'}), 500
            
        except Exception as service_error:
            logger.error(f"Error using predictor service: {str(service_error)}", exc_info=True)
            # Clean up files if they still exist
            if os.path.exists(model_package_path):
                os.remove(model_package_path)
            if os.path.exists(data_filepath):
                os.remove(data_filepath)
            return jsonify({'error': f'Error using predictor service: {str(service_error)}'}), 500
        
    except Exception as e:
        logger.error(f"Error in regression prediction: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Add this new function for background training
def train_regression_models_background(session_data):
    """Train regression models in the background"""
    global regression_training_status
    
    # Extract the session data
    filepath = session_data['filepath']
    target_column = session_data['target_column']
    user_id = session_data['user_id']
    
    try:
        logger.info(f"Starting background regression model training for target: {target_column}")
        data, _ = load_data(filepath)
        test_size = 0.2  # Fixed test size
        
        # 1. CLEAN DATA (via Regression Data Cleaner API)
        regression_training_status['message'] = 'Cleaning and preprocessing data...'
        regression_training_status['progress'] = 10
        logger.info(f"Sending data to Regression Data Cleaner API")
        
        if regression_training_status['stop_requested']:
            regression_training_status['status'] = 'failed'
            regression_training_status['message'] = 'Training was stopped by user'
            return
            
        # Convert data to records - just a plain dictionary
        data_records = data.replace([np.inf, -np.inf], np.nan).where(pd.notnull(data), None).to_dict(orient='records')

        # Additional step to convert NumPy data types to Python native types
        for record in data_records:
            for key, value in record.items():
                if isinstance(value, np.integer):
                    record[key] = int(value)
                elif isinstance(value, np.floating):
                    record[key] = float(value)
                elif isinstance(value, np.ndarray):
                    record[key] = value.tolist()
                elif isinstance(value, np.bool_):
                    record[key] = bool(value)

        # Use our safe request method
        response = safe_requests_post(
            f"{REGRESSION_DATA_CLEANER_URL}/clean",
            {
                "data": data_records,
                "target_column": target_column
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"Regression Data Cleaner API error: {response.json().get('error', 'Unknown error')}")
        
        cleaning_result = response.json()
        cleaned_data = pd.DataFrame.from_dict(cleaning_result["data"])
        
        # Instead of using session, store directly to temporary file
        old_cleaned_file = session_data.get('cleaned_file')
        if old_cleaned_file and os.path.exists(old_cleaned_file):
            try:
                os.remove(old_cleaned_file)
            except:
                pass
                
        cleaned_filepath = get_temp_filepath(extension='.csv')
        cleaned_data.to_csv(cleaned_filepath, index=False)
        # Store file path in app variable instead of session
        regression_training_status['cleaned_file'] = cleaned_filepath
        
        if regression_training_status['stop_requested']:
            regression_training_status['status'] = 'failed'
            regression_training_status['message'] = 'Training was stopped by user'
            return
            
        # 2. FEATURE SELECTION (via Regression Feature Selector API)
        regression_training_status['message'] = 'Selecting important features...'
        regression_training_status['progress'] = 25
        logger.info(f"Sending data to Regression Feature Selector API")
        X = cleaned_data.drop(columns=[target_column])
        y = cleaned_data[target_column]
        
        # Convert X and y to simple Python structures
        X_records = X.replace([np.inf, -np.inf], np.nan).where(pd.notnull(X), None).to_dict(orient='records')
        y_list = y.replace([np.inf, -np.inf], np.nan).where(pd.notnull(y), None).tolist()
        
        # Additional step to convert NumPy data types to Python native types for X_records
        for record in X_records:
            for key, value in record.items():
                if isinstance(value, np.integer):
                    record[key] = int(value)
                elif isinstance(value, np.floating):
                    record[key] = float(value)
                elif isinstance(value, np.ndarray):
                    record[key] = value.tolist()
                elif isinstance(value, np.bool_):
                    record[key] = bool(value)

        # Convert NumPy data types in y_list
        y_list = [int(y) if isinstance(y, np.integer) else 
                  float(y) if isinstance(y, np.floating) else 
                  bool(y) if isinstance(y, np.bool_) else y 
                  for y in y_list]

        # Use our safe request method
        response = safe_requests_post(
            f"{REGRESSION_FEATURE_SELECTOR_URL}/select_features",
            {
                "data": X_records,
                "target": y_list,
                "target_name": target_column
            },
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"Regression Feature Selector API error: {response.json().get('error', 'Unknown error')}")
        
        feature_result = response.json()
        X_selected = pd.DataFrame.from_dict(feature_result["transformed_data"])
        
        # Store selected features
        selected_features = X_selected.columns.tolist()
        
        # Save selected features and feature importance to files without using session
        selected_features_file_json = save_to_temp_file(selected_features, 'selected_features_regression')
        regression_training_status['selected_features_file_json'] = selected_features_file_json
        
        # Store feature importances for visualization
        feature_importance = []
        for feature, importance in feature_result["feature_importances"].items():
            feature_importance.append({'name': feature, 'importance': importance})
        
        # Save feature importance to file without using session
        feature_importance_file = save_to_temp_file(feature_importance, 'feature_importance_regression')
        regression_training_status['feature_importance_file'] = feature_importance_file
        
        # Clean up previous selected features file if exists
        old_features_file = session_data.get('selected_features_regression_file')
        if old_features_file and os.path.exists(old_features_file):
            try:
                os.remove(old_features_file)
            except:
                pass
                
        selected_features_filepath = get_temp_filepath(extension='.csv')
        X_selected.to_csv(selected_features_filepath, index=False)
        regression_training_status['selected_features_file'] = selected_features_filepath
        
        if regression_training_status['stop_requested']:
            regression_training_status['status'] = 'failed'
            regression_training_status['message'] = 'Training was stopped by user'
            return
            
        # Extract original dataset filename from the temporary filepath
        import re
        temp_filepath = filepath
        # The filepath format is typically: UPLOAD_FOLDER/uuid_originalfilename
        # Extract the original filename portion
        original_filename = ""
        if temp_filepath:
            # Match the pattern uuid_originalfilename
            match = re.search(r'[a-f0-9-]+_(.+)$', os.path.basename(temp_filepath))
            if match:
                original_filename = match.group(1)
            else:
                # Fallback to just basename if pattern doesn't match
                original_filename = os.path.basename(temp_filepath)

        # Add direct SQL insert to ensure training_run is populated
        run_name = original_filename if original_filename else f"Regression Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Direct database connection to ensure the training run is saved
        try:
            # Create a temporary app context for database operations
            with app.app_context():
                # Create training run entry
                training_run = TrainingRun(
                    user_id=user_id,
                    run_name=run_name,
                    prompt=None
                )
                db.session.add(training_run)
                db.session.commit()
                
                # Store the run_id for model saving
                local_run_id = training_run.id
                logger.info(f"Directly added regression training run to database with ID {local_run_id}")
                
                # Store preprocessing data for this run
                try:
                    # Prepare data for storage
                    original_columns = list(data.columns)
                    
                    # Save encoding mappings
                    encoding_mappings = cleaning_result.get("encoding_mappings", {})

                    # If there are too many mappings, limit or compress them
                    if len(json.dumps(encoding_mappings)) > 10000:  # Set a reasonable size limit
                        # Option 1: Only keep mappings for columns with fewer than 50 unique values
                        reduced_mappings = {}
                        for col, mapping in encoding_mappings.items():
                            if len(mapping) < 50:
                                reduced_mappings[col] = mapping
                        encoding_mappings = reduced_mappings

                    # Create cleaner_config without encoding_mappings
                    cleaner_config = {
                        "llm_instructions": cleaning_result.get("prompt", ""),
                        "options": cleaning_result.get("options", {}),
                        "handle_missing": True,
                        "handle_outliers": True,
                        "encoding_mappings_summary": {col: len(mapping) for col, mapping in encoding_mappings.items()}  # Just store summary
                    }

                    # Save encoding_mappings to a separate file
                    if encoding_mappings:
                        try:
                            # Create a file path for this run's encoding mappings
                            mappings_dir = os.path.join('static', 'temp', 'mappings')
                            os.makedirs(mappings_dir, exist_ok=True)
                            mappings_file = os.path.join(mappings_dir, f'encoding_mappings_regression_{local_run_id}.json')
                            
                            with open(mappings_file, 'w') as f:
                                json.dump(encoding_mappings, f)
                            
                            # Add file reference to cleaner_config
                            cleaner_config["encoding_mappings_file"] = mappings_file
                            logger.info(f"Saved regression encoding mappings to {mappings_file}")
                        except Exception as e:
                            logger.error(f"Error saving regression encoding mappings to file: {str(e)}")
                            # Continue anyway, we'll just have a less detailed cleaner_config

                    # Create preprocessing data record
                    preprocessing_data = PreprocessingData(
                        run_id=local_run_id,
                        user_id=user_id,
                        original_columns=json.dumps(original_columns),
                        cleaner_config=json.dumps(cleaner_config),
                        feature_selector_config=json.dumps(feature_result.get("options", {})),
                        selected_columns=json.dumps(selected_features),
                        cleaning_report=json.dumps(cleaning_result.get("report", {}))
                    )
                    db.session.add(preprocessing_data)
                    db.session.commit()
                    logger.info(f"Stored regression preprocessing data for run ID {local_run_id}")
                except Exception as pp_error:
                    logger.error(f"Error saving regression preprocessing data: {str(pp_error)}")
                
                # Verify entry was created by checking the database
                verification = db.session.query(TrainingRun).filter_by(id=local_run_id).first()
                if verification:
                    logger.info(f"Verified regression training run in database: {verification.id}, {verification.run_name}")
                else:
                    logger.warning(f"Could not verify regression training run in database after commit")
        except Exception as e:
            logger.error(f"Error adding regression training run directly to database: {str(e)}")
            local_run_id = None

        # 3. MODEL TRAINING (via Regression Model Coordinator API)
        regression_training_status['message'] = 'Training multiple regression models...'
        regression_training_status['progress'] = 40
        logger.info(f"Sending data to Regression Model Coordinator API")
        
        # Update statuses to training
        regression_training_status['model_statuses']['Linear Regression'] = 'training'
        regression_training_status['model_statuses']['Lasso Regression'] = 'training'
        regression_training_status['model_statuses']['Ridge Regression'] = 'training'
        regression_training_status['model_statuses']['Random Forest Regression'] = 'training'
        regression_training_status['model_statuses']['K-Nearest Neighbors'] = 'training'
        regression_training_status['model_statuses']['XGBoost Regression'] = 'training'
        
        if regression_training_status['stop_requested']:
            regression_training_status['status'] = 'failed'
            regression_training_status['message'] = 'Training was stopped by user'
            return
        
        # Prepare data for model coordinator
        X_data = {feature: X_selected[feature].tolist() for feature in selected_features}
        y_data = y.tolist()

        # Convert NumPy types in X_data
        for feature, values in X_data.items():
            X_data[feature] = [int(v) if isinstance(v, np.integer) else 
                              float(v) if isinstance(v, np.floating) else 
                              bool(v) if isinstance(v, np.bool_) else v 
                              for v in values]

        # Convert NumPy types in y_data
        y_data = [int(y) if isinstance(y, np.integer) else 
                  float(y) if isinstance(y, np.floating) else 
                  bool(y) if isinstance(y, np.bool_) else y 
                  for y in y_data]

        # Use our safe request method
        response = safe_requests_post(
            f"{REGRESSION_MODEL_COORDINATOR_URL}/train",
            {
                "data": X_data,
                "target": y_data,
                "test_size": test_size,
                "user_id": user_id,
                "run_name": run_name
            },
            timeout=1800  # Model training can take time
        )
        
        if response.status_code != 200:
            # Update model statuses to failed
            for model_name in regression_training_status['model_statuses']:
                regression_training_status['model_statuses'][model_name] = 'failed'
                
            raise Exception(f"Regression Model Coordinator API error: {response.json().get('error', 'Unknown error')}")
        
        # Store the complete model results
        model_result = response.json()
        regression_training_status['model_result'] = model_result
        
        # Update model statuses based on results
        available_models = {
            'linear_regression': 'Linear Regression',
            'lasso_regression': 'Lasso Regression',
            'ridge_regression': 'Ridge Regression',
            'random_forest_regression': 'Random Forest Regression',
            'knn_regression': 'K-Nearest Neighbors',
            'xgboost_regression': 'XGBoost Regression'
        }
        
        if 'results' in model_result:
            for result in model_result['results']:
                model_name = result.get('name', '')
                if model_name in available_models:
                    display_name = available_models[model_name]
                    regression_training_status['model_statuses'][display_name] = 'complete'
                    
            # Mark missing models as failed
            trained_models = set(result.get('name', '') for result in model_result['results'])
            for model_code, display_name in available_models.items():
                if model_code not in trained_models:
                    regression_training_status['model_statuses'][display_name] = 'failed'
        
        # Ensure models are saved for this run properly
        # Check if we need to ensure models are saved for the coordinator's run_id
        # Use either the coordinator's run_id or our local one
        run_id_to_use = model_result.get('run_id', local_run_id)
        
        # Check if models were successfully trained
        if 'results' in model_result and model_result['results']:
            logger.info(f"Received {len(model_result['results'])} trained regression models")
            
            if run_id_to_use:
                # Ensure all models are properly saved to the database
                with app.app_context():
                    ensure_regression_models_saved(user_id, run_id_to_use, model_result)
                
                # Save the run ID for later use
                regression_training_status['last_regression_training_run_id'] = run_id_to_use
                
                # If we have our local run_id, also ensure models are saved for it
                if local_run_id and local_run_id != run_id_to_use:
                    with app.app_context():
                        ensure_regression_models_saved(user_id, local_run_id, model_result)
        else:
            logger.error("No models were successfully trained by the model coordinator")
            regression_training_status['status'] = 'failed'
            regression_training_status['message'] = 'No models were successfully trained by the model coordinator'
            return
        
        # Mark training as complete
        regression_training_status['status'] = 'complete'
        regression_training_status['progress'] = 100
        regression_training_status['message'] = 'Training complete!'
        logger.info("Regression model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in background regression training: {str(e)}", exc_info=True)
        regression_training_status['status'] = 'failed'
        regression_training_status['message'] = str(e)
        regression_training_status['error'] = str(e)

# Add these new routes for the training status tracking and control
@app.route('/api/regression_training_status')
@login_required
def get_regression_training_status():
    """API endpoint to get the current status of regression model training"""
    global regression_training_status
    
    # Return the current status without the thread object (not JSON serializable)
    status_copy = regression_training_status.copy()
    if 'training_thread' in status_copy:
        del status_copy['training_thread']
    
    return jsonify(status_copy)

@app.route('/stop_regression_training', methods=['POST'])
@login_required
def stop_regression_training():
    """Stop the current regression training process"""
    global regression_training_status
    
    if regression_training_status['status'] == 'in_progress':
        regression_training_status['stop_requested'] = True
        logger.info("Regression training stop requested by user")
        flash("Training process has been stopped.", "info")
    else:
        logger.info("No active regression training to stop")
        flash("No active training process found.", "warning")
    
    return redirect(url_for('train_regression')) 