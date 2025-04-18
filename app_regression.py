from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
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
    
    filepath = session.get('uploaded_file_regression')
    
    if not filepath:
        # If accessed directly without upload, show the upload interface
        # Check regression services health for status display
        services_status = check_regression_services()
        
        return render_template('train_regression.html', services_status=services_status)
    
    data, _ = load_data(filepath)
    
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
            
            # DEBUG: Check which model services are available through the coordinator
            try:
                coordinator_health_response = requests.get(f"{REGRESSION_MODEL_COORDINATOR_URL}/health", timeout=5)
                if coordinator_health_response.status_code == 200:
                    coordinator_health = coordinator_health_response.json()
                    if "model_services" in coordinator_health:
                        for model, status in coordinator_health["model_services"].items():
                            logger.info(f"Regression model service {model}: {status}")
                    else:
                        logger.warning("Regression model services not found in coordinator health response")
            except Exception as e:
                logger.error(f"Error checking regression model services: {str(e)}")
            
            # 1. CLEAN DATA (via Regression Data Cleaner API)
            logger.info(f"Sending data to Regression Data Cleaner API")
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
            
            # Clean up previous cleaned file if exists
            if 'cleaned_file' in session and os.path.exists(session['cleaned_file']):
                try:
                    os.remove(session['cleaned_file'])
                except:
                    pass
                    
            cleaned_filepath = get_temp_filepath(extension='.csv')
            cleaned_data.to_csv(cleaned_filepath, index=False)
            session['cleaned_file'] = cleaned_filepath
            
            # Add logging to verify data being sent to APIs
            logger.info(f"Data being sent to Regression Data Cleaner API: {data_records[:5]}")
            
            # 2. FEATURE SELECTION (via Regression Feature Selector API)
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
            
            # Store selected features in session
            selected_features = X_selected.columns.tolist()
            
            # Save selected features to file to prevent session bloat
            selected_features_file_json = save_to_temp_file(selected_features, 'selected_features_regression')
            session['selected_features_regression_file_json'] = selected_features_file_json
            # Use a reference in session to avoid storing large data
            session['selected_features_regression'] = f"[{len(selected_features)} features]"
            
            # Store feature importances for visualization
            feature_importance = []
            for feature, importance in feature_result["feature_importances"].items():
                feature_importance.append({'Feature': feature, 'Importance': importance})
            
            # Save feature importance to file
            feature_importance_file = save_to_temp_file(feature_importance, 'feature_importance_regression')
            session['feature_importance_regression_file'] = feature_importance_file
            
            # Clean up previous selected features file if exists
            if 'selected_features_regression_file' in session and os.path.exists(session['selected_features_regression_file']):
                try:
                    os.remove(session['selected_features_regression_file'])
                except:
                    pass
                    
            selected_features_filepath = get_temp_filepath(extension='.csv')
            X_selected.to_csv(selected_features_filepath, index=False)
            session['selected_features_regression_file'] = selected_features_filepath
            
            # Extract original dataset filename from the temporary filepath
            import re
            temp_filepath = session.get('uploaded_file_regression', '')
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
            user_id = current_user.id
            
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
            logger.info(f"Sending data to Regression Model Coordinator API")
            
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
                    "test_size": session['test_size'],
                    "user_id": current_user.id,
                    "run_name": run_name
                },
                timeout=1800  # Model training can take time
            )
            
            if response.status_code != 200:
                raise Exception(f"Regression Model Coordinator API error: {response.json().get('error', 'Unknown error')}")
            
            # Store the complete model results in session
            model_result = response.json()
            
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
                    
                    # Save the run ID to session for models_regression page
                    session['last_regression_training_run_id'] = run_id_to_use
                    
                    # If we have our local run_id, also ensure models are saved for it
                    if local_run_id and local_run_id != run_id_to_use:
                        with app.app_context():
                            ensure_regression_models_saved(user_id, local_run_id, model_result)
                else:
                    logger.warning("No run_id available to save regression models")
            else:
                logger.error("No models were successfully trained by the model coordinator")
                flash("No models were successfully trained. Please check the logs for more information.", "error")
                return redirect(url_for('train_regression'))
            
            # Save processed results to session
            session['regression_model_results'] = model_result
            
            return redirect(url_for('regression_results'))
            
        except Exception as e:
            logger.error(f"Error processing regression data: {str(e)}", exc_info=True)
            flash(f"Error processing regression data: {str(e)}", 'error')
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
    
    # Check regression services health for status display
    services_status = check_regression_services()
    
    return render_template('regression_prediction.html', services_status=services_status, logout_token=session['logout_token'])

@app.route('/regression_results')
@login_required
def regression_results():
    """Route for displaying regression model training results"""
    try:
        # Check if the user is logged in
        if not current_user.is_authenticated:
            flash('Please log in to access the regression results page.', 'info')
            return redirect('/login', code=302)
        
        # Get model results from session
        model_result = session.get('regression_model_results')
        
        if not model_result:
            flash('No recent regression training results found.', 'warning')
            return redirect(url_for('train_regression'))
        
        # Get the run_id
        run_id = model_result.get('run_id')
        if not run_id:
            run_id = session.get('last_regression_training_run_id')
        
        # Process model results
        results = model_result.get('results', [])
        
        # Sort models by R² score (descending)
        sorted_models = sorted(results, key=lambda x: x.get('metrics', {}).get('r2', 0) if isinstance(x.get('metrics'), dict) else x.get('r2', 0), reverse=True)
        
        # Get top models (up to 4)
        top_models = sorted_models[:min(4, len(sorted_models))]
        
        # Ensure each model has necessary data structure
        for model in top_models:
            # Add parameters if missing
            if 'parameters' not in model:
                model['parameters'] = {}
            
            # Check if model has the best score (for highlighting)
            model['is_best'] = model == top_models[0] if top_models else False
            
            # Ensure metrics is defined
            if 'metrics' not in model and 'r2' in model:
                model['metrics'] = {
                    'r2': model.get('r2', 0),
                    'rmse': model.get('rmse', 0),
                    'mae': model.get('mae', 0),
                    'mse': model.get('mse', 0)
                }
        
        # Extract the best model
        best_model = top_models[0] if top_models else None
        
        # Get training info
        training_info = {
            'target_column': session.get('target_column', 'Unknown'),
            'dataset_size': model_result.get('dataset_size', 'Unknown'),
            'training_time': model_result.get('training_time', 'Unknown')
        }
        
        # Get feature importance if available
        feature_importance = []
        if 'selected_features_regression_file_json' in session:
            try:
                # Load selected features
                selected_features = load_from_temp_file(session['selected_features_regression_file_json'])
                
                # Create dummy feature importance if actual importance not available
                for feature in selected_features:
                    feature_importance.append({
                        'name': feature,
                        'importance': 1.0 / len(selected_features)  # Equal importance as fallback
                    })
            except Exception as e:
                logger.error(f"Error loading selected features for regression: {str(e)}", exc_info=True)
        
        if 'feature_importance_regression_file' in session:
            try:
                # Load actual feature importance
                feature_importance = load_from_temp_file(session['feature_importance_regression_file'])
            except Exception as e:
                logger.error(f"Error loading feature importance for regression: {str(e)}", exc_info=True)
        
        # Metrics information for explanation
        metrics_info = {
            'r2': 'R² (Coefficient of Determination): Represents the proportion of variance in the dependent variable that is predictable from the independent variables. Values range from 0 to 1, with 1 indicating perfect prediction.',
            'rmse': "RMSE (Root Mean Square Error): Measures the average magnitude of the error. It's the square root of the average of squared differences between predicted and actual values.",
            'mae': 'MAE (Mean Absolute Error): Measures the average magnitude of the errors in a set of predictions, without considering their direction.',
            'mse': 'MSE (Mean Squared Error): The average of the squares of the errors—the average squared difference between the estimated values and the actual value.'
        }
        
        logger.info(f"Rendering regression_results.html with {len(top_models)} models")
        return render_template('regression_results.html', 
                              run_id=run_id,
                              model_results=top_models,
                              best_model=best_model,
                              training_info=training_info,
                              feature_importance=feature_importance,
                              metrics_info=metrics_info)
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
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a CSV or Excel file.'}), 400
            
        # Get model ID from form
        model_id = request.form.get('model_id')
        if not model_id:
            return jsonify({'error': 'No model selected'}), 400
            
        # Save file temporarily
        filepath = get_temp_filepath(file.filename)
        file.save(filepath)
        
        # Load the data
        data, file_stats = load_data(filepath)
        if data is None:
            return jsonify({'error': file_stats}), 400
            
        # Load model metadata from database
        with app.app_context():
            model = db.session.query(TrainingModel).filter_by(id=model_id).first()
            if not model:
                return jsonify({'error': 'Model not found'}), 404
                
            # Check if it's a regression model
            regression_model_types = ['linear_regression', 'lasso_regression', 'ridge_regression', 
                                      'random_forest_regression', 'knn_regression', 'xgboost_regression']
            if model.model_name not in regression_model_types:
                return jsonify({'error': 'Selected model is not a regression model'}), 400
                
            # Get preprocessing data for this run to get selected features
            preprocessing_data = db.session.query(PreprocessingData).filter_by(run_id=model.run_id).first()
            if not preprocessing_data:
                return jsonify({'error': 'Preprocessing data not found for this model'}), 404
                
            # Parse selected features
            selected_features = []
            if preprocessing_data.selected_columns:
                try:
                    selected_features = json.loads(preprocessing_data.selected_columns)
                except:
                    return jsonify({'error': 'Could not parse selected features'}), 500
        
        # Check if all required columns are present
        missing_columns = [col for col in selected_features if col not in data.columns]
        if missing_columns:
            return jsonify({
                'error': f'Input data is missing required columns: {", ".join(missing_columns)}'
            }), 400
            
        # Prepare data for prediction
        X = data[selected_features]
        
        # Clean up any missing data
        for col in X.columns:
            if X[col].dtype.kind in 'ifc':  # integer, float, complex
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "")
        
        # Convert to records for API
        X_records = clean_data_for_json(X)
        
        # Call the regression predictor service - Use the URL defined at the top
        prediction_response = safe_requests_post(
            f"{REGRESSION_PREDICTOR_SERVICE_URL}/predict",
            {
                "data": X_records,
                "model_url": model.url
            },
            timeout=30
        )
        
        if prediction_response.status_code != 200:
            return jsonify({'error': f'Prediction service error: {prediction_response.text}'}), 500
            
        # Extract predictions
        predictions_data = prediction_response.json()
        predictions = predictions_data.get('predictions', [])
        
        # Combine with original data for display
        result_df = data.copy()
        result_df['prediction'] = predictions
        
        # Convert to records for response
        result_records = clean_data_for_json(result_df)
        
        # Return predictions
        return jsonify({
            'success': True,
            'predictions': predictions,
            'full_results': result_records,
            'model_name': model.model_name,
            'file_stats': file_stats
        })
        
    except Exception as e:
        logger.error(f"Error in regression prediction: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 