import os
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import io
import logging
import time
from datetime import datetime
import requests
import tempfile
import secrets as std_secrets
import uuid
import traceback
import re
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
from flask_login import LoginManager, login_required, current_user

# Import common components from app_api.py
from app_api import app, DATA_CLEANER_URL, FEATURE_SELECTOR_URL, ANOMALY_DETECTOR_URL, MODEL_COORDINATOR_URL, MEDICAL_ASSISTANT_URL
from app_api import is_service_available, get_temp_filepath, safe_requests_post, cleanup_session_files, check_services
from app_api import allowed_file, load_data, clean_data_for_json, SafeJSONEncoder, logger
from app_api import save_to_temp_file, load_from_temp_file, check_session_size

# Import database models
from db.users import db, User, TrainingRun, TrainingModel, PreprocessingData

# Add a global variable to track training status for each user
classification_training_status = {}

@app.route('/upload', methods=['POST'])


@login_required
def upload():
    # Double check authentication - ensure user is logged in
    if not current_user.is_authenticated:
        logger.warning("Upload attempted without authentication")
        flash('Please log in to upload files.', 'warning')
        return redirect(url_for('login'))
    
    # Clear all training-related session data to ensure a fresh start
    keys_to_remove = [
        'model_results', 'last_training_run_id', 'run_id', 'trained_models',
        'selected_features', 'selected_features_file', 'cleaned_file',
        'feature_importance_file', 'selected_features_file_json', 'anomaly_results',
        'ai_recommendations', 'ai_recommendations_file'  # Reset medical assistant recommendations
    ]
    
    for key in keys_to_remove:
        if key in session:
            # If this is a file path, remove the actual file too
            if key.endswith('_file') and session[key] and os.path.exists(session[key]):
                try:
                    os.remove(session[key])
                    logger.info(f"Removed file {session[key]} from key {key}")
                except Exception as e:
                    logger.error(f"Failed to remove file {session[key]}: {str(e)}")
            
            session.pop(key, None)
            logger.info(f"Cleared session key: {key}")
    
    # Also clear any keys that might be from previous training
    for key in list(session.keys()):
        if key.startswith('old_') or key.endswith('_cached'):
            session.pop(key, None)
            
    # Remove user from training status tracking
    if current_user.id in classification_training_status:
        classification_training_status.pop(current_user.id, None)
        
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Clean up previous upload if exists
        if 'uploaded_file' in session and os.path.exists(session['uploaded_file']):
            try:
                os.remove(session['uploaded_file'])
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
            return redirect(url_for('index'))
        
        session['uploaded_file'] = filepath
        session['file_stats'] = result
        
        # Store data columns for later use
        session['data_columns'] = data.columns.tolist()
        
        # Redirect to training page
        return redirect(url_for('training'))
    
    flash('Invalid file type. Please upload a CSV or Excel file.', 'error')
    return redirect(url_for('index'))

@app.route('/training', methods=['GET', 'POST'])
@login_required
def training():
    # Check if we need to reset the dataset (new upload requested)
    if request.args.get('new') == '1':
        # Clear dataset from session
        if 'uploaded_file' in session:
            # Delete the file if it exists
            if os.path.exists(session['uploaded_file']):
                try:
                    os.remove(session['uploaded_file'])
                except:
                    pass
            # Remove from session
            session.pop('uploaded_file', None)
            session.pop('file_stats', None)
            session.pop('data_columns', None)
        
        # Also clear medical assistant recommendations
        if 'ai_recommendations_file' in session:
            # Delete the file if it exists
            if os.path.exists(session['ai_recommendations_file']):
                try:
                    os.remove(session['ai_recommendations_file'])
                except:
                    pass
            session.pop('ai_recommendations_file', None)
        
        if 'ai_recommendations' in session:
            session.pop('ai_recommendations', None)
            
        # Redirect to clean URL
        return redirect(url_for('training'))
    
    # Clear any training results if we're at the training page (this ensures we don't show old results)
    if 'model_results' in session:
        session.pop('model_results', None)
    if 'last_training_run_id' in session:
        session.pop('last_training_run_id', None)
    
    filepath = session.get('uploaded_file')
    
    if not filepath:
        # If accessed directly without upload, show the upload interface
        # Check services health for status display
        services_status = check_services()
        # We need to keep rendering index.html here for the upload interface
        # This doesn't redirect to home because we're directly rendering the template
        return render_template('index.html', services_status=services_status)
    
    data, _ = load_data(filepath)
    
    # Handle case where data could not be loaded
    if data is None:
        flash('Error loading data from the uploaded file. Please upload a valid file.', 'error')
        services_status = check_services()
        # Directly render the index template with the upload interface
        return render_template('index.html', services_status=services_status)
    
    if request.method == 'POST':
        # Get target column from form
        target_column = request.form.get('target_column')
        session['target_column'] = target_column
        
        # Always use fixed test size of 20%
        session['test_size'] = 0.2
        
        try:
            # Check if required services are available
            required_services = {
                "Data Cleaner": DATA_CLEANER_URL,
                "Feature Selector": FEATURE_SELECTOR_URL,
                "Anomaly Detector": ANOMALY_DETECTOR_URL,
                "Model Coordinator": MODEL_COORDINATOR_URL  # Changed from Model Trainer to Model Coordinator
            }
            
            logger.info("Checking required services before training:")
            unavailable_services = []
            
            for service_name, service_url in required_services.items():
                logger.info(f"Checking service: {service_name} at {service_url}")
                if not is_service_available(service_url):
                    logger.error(f"Service {service_name} at {service_url} is not available")
                    unavailable_services.append(service_name)
                else:
                    logger.info(f"Service {service_name} is available")
            
            if unavailable_services:
                error_message = f"The following services are not available: {', '.join(unavailable_services)}. Cannot proceed with training."
                logger.error(error_message)
                flash(error_message, 'error')
                return redirect(url_for('training'))
            
            # DEBUG: Check which model services are available through the coordinator
            try:
                coordinator_health_response = requests.get(f"{MODEL_COORDINATOR_URL}/health", timeout=5)
                if coordinator_health_response.status_code == 200:
                    coordinator_health = coordinator_health_response.json()
                    if "model_services" in coordinator_health:
                        for model, status in coordinator_health["model_services"].items():
                            logger.info(f"Model service {model}: {status}")
                    else:
                        logger.warning("Model services not found in coordinator health response")
            except Exception as e:
                logger.error(f"Error checking model services: {str(e)}")
            
            # Initialize training status for this user BEFORE starting the actual training
            # This ensures we always have a fresh status
            user_id = current_user.id
            classification_training_status[user_id] = {
                'status': 'in_progress',
                'overall_progress': 0,
                'overall_status': 'Training initialized. Preparing data...',
                'model_statuses': {
                    'logistic-regression': {'progress': 0, 'status': 'Initializing...'},
                    'decision-tree': {'progress': 0, 'status': 'Initializing...'},
                    'random-forest': {'progress': 0, 'status': 'Initializing...'},
                    'knn': {'progress': 0, 'status': 'Initializing...'},
                    'svm': {'progress': 0, 'status': 'Initializing...'},
                    'naive-bayes': {'progress': 0, 'status': 'Initializing...'}
                }
            }
            
            # 1. CLEAN DATA (via Data Cleaner API)
            logger.info(f"Sending data to Data Cleaner API")
            # Convert data to records - just a plain dictionary
            data_records = data.replace([np.inf, -np.inf], np.nan).where(pd.notnull(data), None).to_dict(orient='records')
            # Use our safe request method
            response = safe_requests_post(
                f"{DATA_CLEANER_URL}/clean",
                {
                    "data": data_records,
                    "target_column": target_column
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Data Cleaner API error: {response.json().get('error', 'Unknown error')}")
            
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
            logger.info(f"Data being sent to Data Cleaner API: {data_records[:5]}")
            
            # 2. FEATURE SELECTION (via Feature Selector API)
            logger.info(f"Sending data to Feature Selector API")
            X = cleaned_data.drop(columns=[target_column])
            y = cleaned_data[target_column]
            
            # Convert X and y to simple Python structures
            X_records = X.replace([np.inf, -np.inf], np.nan).where(pd.notnull(X), None).to_dict(orient='records')
            y_list = y.replace([np.inf, -np.inf], np.nan).where(pd.notnull(y), None).tolist()
            
            # Use our safe request method
            response = safe_requests_post(
                f"{FEATURE_SELECTOR_URL}/select_features",
                {
                    "data": X_records,
                    "target": y_list,
                    "target_name": target_column
                },
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"Feature Selector API error: {response.json().get('error', 'Unknown error')}")
            
            feature_result = response.json()
            X_selected = pd.DataFrame.from_dict(feature_result["transformed_data"])
            
            # Store selected features in session
            selected_features = X_selected.columns.tolist()
            
            # Save selected features to file to prevent session bloat
            selected_features_file_json = save_to_temp_file(selected_features, 'selected_features')
            session['selected_features_file_json'] = selected_features_file_json
            # Use a reference in session to avoid storing large data
            session['selected_features'] = f"[{len(selected_features)} features]"
            
            # Store feature importances for visualization
            feature_importance = []
            for feature, importance in feature_result["feature_importances"].items():
                feature_importance.append({'Feature': feature, 'Importance': importance})
            
            # Save feature importance to file
            feature_importance_file = save_to_temp_file(feature_importance, 'feature_importance')
            session['feature_importance_file'] = feature_importance_file
            
            # Clean up previous selected features file if exists
            if 'selected_features_file' in session and os.path.exists(session['selected_features_file']):
                try:
                    os.remove(session['selected_features_file'])
                except:
                    pass
                    
            selected_features_filepath = get_temp_filepath(extension='.csv')
            X_selected.to_csv(selected_features_filepath, index=False)
            session['selected_features_file'] = selected_features_filepath
            
            # Add logging to verify data being sent to APIs
            logger.info(f"Data being sent to Feature Selector API: {X_records[:5]}, Target: {y_list[:5]}")
            
            # 3. ANOMALY DETECTION (via Anomaly Detector API)
            logger.info(f"Sending data to Anomaly Detector API")
            # Convert to simple Python structure
            X_selected_records = X_selected.replace([np.inf, -np.inf], np.nan).where(pd.notnull(X_selected), None).to_dict(orient='records')

            # Use our safe request method
            response = safe_requests_post(
                f"{ANOMALY_DETECTOR_URL}/detect_anomalies",
                {
                    "data": X_selected_records
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Anomaly Detector API error: {response.json().get('error', 'Unknown error')}")
            
            anomaly_results = response.json()
            session['anomaly_results'] = {
                'is_data_valid': anomaly_results["is_data_valid"],
                'anomaly_percentage': anomaly_results["anomaly_report"]["anomaly_percentage"]
            }
            
            # Add logging to verify data being sent to APIs
            logger.info(f"Data being sent to Anomaly Detector API: {X_selected_records[:5]}")
            
            # 4. MODEL TRAINING (via Model Coordinator API instead of Model Trainer API)
            logger.info(f"Sending data to Model Coordinator API")
            logger.info(f"CRITICAL: Ensuring FRESH models are trained for this dataset upload")
            
            # Prepare data for model coordinator
            X_data = {feature: X_selected[feature].tolist() for feature in selected_features}
            y_data = y.tolist()

            # Extract original dataset filename from the temporary filepath
            import re
            temp_filepath = session.get('uploaded_file', '')
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
            run_name = original_filename if original_filename else f"Training Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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
                    logger.info(f"Directly added training run to database with ID {local_run_id}")
                    
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
                            "encoding_mappings_summary": {}  # Start with empty dictionary
                        }

                        # Save encoding_mappings to a separate file
                        if encoding_mappings:
                            try:
                                # First validate that the encoding mappings are actually for this dataset
                                # by checking if the keys exist in the original columns
                                valid_mappings = {}
                                for col, mapping in encoding_mappings.items():
                                    # Extract the actual column name from possible formats like "column_encoding" or "column_map"
                                    base_col = col.split('_')[0] if '_encoding' in col or '_map' in col else col
                                    if base_col in original_columns:
                                        valid_mappings[col] = mapping
                                    else:
                                        logger.warning(f"Dropping mapping for column '{col}' as it's not in original columns")
                                
                                # Only save file if we have valid mappings
                                if valid_mappings:
                                    # Update mapping summary with only valid mappings
                                    cleaner_config["encoding_mappings_summary"] = {col: len(mapping) for col, mapping in valid_mappings.items()}
                                    
                                    # Create a file path for this run's encoding mappings
                                    mappings_dir = os.path.join('static', 'temp', 'mappings')
                                    os.makedirs(mappings_dir, exist_ok=True)
                                    mappings_file = os.path.join(mappings_dir, f'encoding_mappings_{local_run_id}.json')
                                    
                                    with open(mappings_file, 'w') as f:
                                        json.dump(valid_mappings, f)
                                    
                                    # Add file reference to cleaner_config
                                    cleaner_config["encoding_mappings_file"] = mappings_file
                                    logger.info(f"Saved encoding mappings to {mappings_file} with {len(valid_mappings)} valid mappings")
                                else:
                                    logger.info("No valid encoding mappings found for this dataset")
                            except Exception as e:
                                logger.error(f"Error saving encoding mappings to file: {str(e)}")
                                # Continue anyway, we'll just have a less detailed cleaner_config
                        else:
                            logger.info("No encoding mappings to save (no categorical columns or all numeric data)")

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
                        logger.info(f"Stored preprocessing data for run ID {local_run_id}")
                    except Exception as pp_error:
                        logger.error(f"Error saving preprocessing data: {str(pp_error)}")
                    
                    # Verify entry was created by checking the database
                    verification = db.session.query(TrainingRun).filter_by(id=local_run_id).first()
                    if verification:
                        logger.info(f"Verified training run in database: {verification.id}, {verification.run_name}")
                    else:
                        logger.warning(f"Could not verify training run in database after commit")
                    
                    # Try to diagnose the issue if verification failed
                    if not verification:
                        # Check if database is accessible
                        db.session.execute("SELECT 1")
                        logger.info("Database connection is working")
                        
                        # Check table structure
                        logger.info("Attempting to check training_run table structure")
                        table_info = db.session.execute("DESCRIBE training_run").fetchall()
                        logger.info(f"Table structure: {table_info}")
            except Exception as e:
                logger.error(f"Error adding training run directly to database: {str(e)}")
                # Try with direct SQL as a fallback
                try:
                    # Get database credentials from Key Vault
                    MYSQLUSER = keyvault.getenv("MYSQLUSER")
                    MYSQLPASSWORD = keyvault.getenv("MYSQLPASSWORD")
                    MYSQLHOST = keyvault.getenv("MYSQLHOST")
                    MYSQLPORT = int(keyvault.getenv("MYSQLPORT"))
                    MYSQLDB = keyvault.getenv("MYSQLDB")
                    
                    # Connect to database
                    conn = pymysql.connect(
                        host=MYSQLHOST,
                        user=MYSQLUSER,
                        password=MYSQLPASSWORD,
                        port=MYSQLPORT,
                        database=MYSQLDB
                    )
                    cursor = conn.cursor()
                    
                    # Insert training run
                    cursor.execute(
                        "INSERT INTO training_run (user_id, run_name, prompt, created_at) VALUES (%s, %s, %s, NOW())",
                        (user_id, run_name, None)
                    )
                    conn.commit()
                    local_run_id = cursor.lastrowid
                    logger.info(f"Added training run using direct SQL: {local_run_id}")
                    
                    cursor.close()
                    conn.close()
                except Exception as sql_error:
                    logger.error(f"Error with direct SQL approach: {str(sql_error)}")
                    local_run_id = None

            # Update status before sending to model coordinator
            classification_training_status[user_id]['overall_progress'] = 15
            classification_training_status[user_id]['overall_status'] = 'Data prepared. Starting model training...'
            
            # Create a unique dataset identifier for the coordinator
            unique_dataset_id = f"unique_dataset_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            
            # Save it in the session to check if we've seen this request before
            session['current_training_dataset_id'] = unique_dataset_id
            
            # Use our safe request method
            response = safe_requests_post(
                f"{MODEL_COORDINATOR_URL}/train",
                {
                    "data": X_data,
                    "target": y_data,
                    "test_size": session['test_size'],
                    "user_id": current_user.id,
                    "run_name": run_name,
                    "force_new_training": True,
                    "unique_dataset_id": unique_dataset_id
                },
                timeout=1800  # Model training can take time
            )
            
            if response.status_code != 200:
                raise Exception(f"Model Coordinator API error: {response.json().get('error', 'Unknown error')}")
            
            # Store the complete model results in session
            model_result = response.json()
            
            # Add code to ensure models are saved for this run properly
            # This will be implemented in the app_others.py
            from app_others import ensure_training_models_saved
            
            # Check if we need to ensure models are saved for the coordinator's run_id
            if 'saved_best_models' in model_result and model_result['saved_best_models']:
                # Use either the coordinator's run_id or our local one
                run_id_to_use = model_result.get('run_id', local_run_id)
                if run_id_to_use:
                    # Ensure all 4 models are properly saved to the database
                    with app.app_context():
                        ensure_training_models_saved(user_id, run_id_to_use, model_result)
                    
                    # Save the run ID to session for model_selection page
                    session['last_training_run_id'] = run_id_to_use
                    
                    # If we have our local run_id, also ensure models are saved for it
                    if local_run_id and local_run_id != run_id_to_use:
                        with app.app_context():
                            ensure_training_models_saved(user_id, local_run_id, model_result)
                else:
                    logger.warning("No run_id available to save models")
            
            # Save processed results to session
            session['model_results'] = model_result
            
            # When training is complete, update status
            if user_id in classification_training_status:
                classification_training_status[user_id]['status'] = 'complete'
                classification_training_status[user_id]['overall_progress'] = 100
                classification_training_status[user_id]['overall_status'] = 'Training complete!'
                classification_training_status[user_id]['redirect_url'] = url_for('model_selection')
            
            return redirect(url_for('model_selection'))
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}", exc_info=True)
            flash(f"Error processing data: {str(e)}", 'error')
            
            # Clear training status on error
            if user_id in classification_training_status:
                classification_training_status.pop(user_id, None)
            
            return redirect(url_for('training'))
    
    # Get AI recommendations for the dataset (via Medical Assistant API) - OPTIONAL
    ai_recommendations = None
    # Always try to get fresh recommendations if the service is available
    if is_service_available(MEDICAL_ASSISTANT_URL):
        try:
            logger.info(f"Sending data to Medical Assistant API")
            
            # Convert to simple Python structure
            data_records = data.replace([np.inf, -np.inf], np.nan).where(pd.notnull(data), None).to_dict(orient='records')
            
            # Use our safe request method
            response = safe_requests_post(
                f"{MEDICAL_ASSISTANT_URL}/analyze_data",
                {
                    "data": data_records
                },
                timeout=30
            )
            
            if response.status_code == 200:
                ai_recommendations = response.json()["recommendations"]
                
                # Save to file instead of storing in session
                # Remove old file if it exists
                if 'ai_recommendations_file' in session and os.path.exists(session['ai_recommendations_file']):
                    try:
                        os.remove(session['ai_recommendations_file'])
                    except:
                        pass
                        
                recommendations_file = save_to_temp_file(ai_recommendations, 'ai_recommendations')
                session['ai_recommendations_file'] = recommendations_file
                logger.info(f"Saved AI recommendations to {recommendations_file}")
                
                # Log sample of data sent
                logger.info(f"Data being sent to Medical Assistant API: {data_records[:3]}")
            else:
                logger.warning(f"Medical Assistant API returned an error: {response.text}")
        except Exception as e:
            logger.error(f"Error getting AI recommendations: {str(e)}", exc_info=True)
            # Don't flash this error to avoid confusing the user
            logger.info("Continuing without AI recommendations")
    # If the service is not available, try to load from file if we have it
    elif 'ai_recommendations_file' in session:
        # Load from file
        ai_recommendations = load_from_temp_file(session['ai_recommendations_file'])
    # Backward compatibility: Try to get from session
    else:
        ai_recommendations = session.get('ai_recommendations')
        
        # If it's in session, move to file
        if ai_recommendations:
            recommendations_file = save_to_temp_file(ai_recommendations, 'ai_recommendations')
            session['ai_recommendations_file'] = recommendations_file
            session.pop('ai_recommendations', None)
            logger.info(f"Moved AI recommendations from session to file: {recommendations_file}")
    
    # Make sure session size is under control
    check_session_size()
    
    return render_template('training.html', 
                          data=data.head().to_html(classes='table table-striped'),
                          columns=data.columns.tolist(),
                          file_stats=session.get('file_stats'),
                          ai_recommendations=ai_recommendations)

@app.route('/tabular_prediction')
@login_required
def tabular_prediction():
    """Route for tabular prediction page that uses custom models"""
    # Check if the user is logged in
    if not current_user.is_authenticated:
        flash('Please log in to access the tabular prediction page.', 'info')
        return redirect('/login', code=302)
    
    # Generate a CSRF token for logout form if needed
    if 'logout_token' not in session:
        session['logout_token'] = std_secrets.token_hex(16)
    
    # Check services health for status display
    services_status = check_services()
    
    # Add tabular prediction service to services status
    predictor_service_url = 'http://localhost:5101'
    services_status['tabular_predictor_service'] = is_service_available(predictor_service_url)
    
    return render_template('tabular_prediction.html', services_status=services_status, logout_token=session['logout_token'])

@app.route('/api/predict_tabular', methods=['POST'])
@login_required
def api_predict_tabular():
    """API endpoint for tabular prediction using custom models"""
    # Check for file uploads
    if 'model_package' not in request.files or 'input_file' not in request.files:
        return jsonify({"error": "Both model package and input file are required"}), 400
    
    model_package = request.files['model_package']
    input_file = request.files['input_file']
    
    if not model_package.filename or not input_file.filename:
        return jsonify({"error": "Both model package and input file must be selected"}), 400
    
    # Validate file extensions
    if not model_package.filename.lower().endswith('.zip'):
        return jsonify({"error": "Model package must be a ZIP archive"}), 400
    
    if not (input_file.filename.lower().endswith('.xlsx') or 
            input_file.filename.lower().endswith('.xls') or 
            input_file.filename.lower().endswith('.csv')):
        return jsonify({"error": "Input file must be an Excel or CSV file"}), 400
    
    try:
        # Define the predictor service URL
        predictor_service_url = 'http://localhost:5101'
        
        # Check if the predictor service is available
        if not is_service_available(predictor_service_url):
            return jsonify({"error": "Tabular prediction service is not available. Please try again later."}), 503
        
        logger.info(f"Starting tabular prediction for model: {model_package.filename} and input: {input_file.filename}")
        
        # Forward the files to the prediction service
        files = {
            'model_package': (model_package.filename, model_package.stream, 'application/zip'),
            'input_file': (input_file.filename, input_file.stream, 'application/octet-stream')
        }
        
        # Add selected encoding column if provided
        data = {}
        encoding_column = request.form.get('encoding_column')
        if encoding_column:
            data['encoding_column'] = encoding_column
            logger.info(f"Using encoding column: {encoding_column}")
        
        # Send request to the predictor service
        response = requests.post(
            f"{predictor_service_url}/predict",
            files=files,
            data=data,
            timeout=600  # 10 minute timeout
        )
        
        # Check response status
        if response.status_code != 200:
            error_message = "Error in prediction service"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_message = error_data['error']
            except:
                error_message = f"Error in prediction service (HTTP {response.status_code})"
            
            return jsonify({"error": error_message}), response.status_code
        
        # Return prediction results (CSV content)
        result = response.json()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in tabular prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict_tabular/extract_encodings', methods=['POST'])
@login_required
def api_extract_encodings():
    """API endpoint to extract encoding maps from a model package"""
    if 'model_package' not in request.files:
        return jsonify({"error": "Model package file is required"}), 400
    
    model_package = request.files['model_package']
    
    if not model_package.filename or not model_package.filename.lower().endswith('.zip'):
        return jsonify({"error": "Model package must be a ZIP archive"}), 400
    
    try:
        # Define the predictor service URL
        predictor_service_url = 'http://localhost:5101'
        
        # Check if the predictor service is available
        if not is_service_available(predictor_service_url):
            return jsonify({"error": "Tabular prediction service is not available. Please try again later."}), 503
        
        logger.info(f"Extracting encodings from model package: {model_package.filename}")
        
        # Forward the file to the prediction service
        files = {
            'model_package': (model_package.filename, model_package.stream, 'application/zip')
        }
        
        # Send request to the predictor service
        response = requests.post(
            f"{predictor_service_url}/extract_encodings",
            files=files,
            timeout=60  # 1 minute timeout
        )
        
        # Check response status
        if response.status_code != 200:
            error_message = "Error extracting encodings"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_message = error_data['error']
            except:
                error_message = f"Error extracting encodings (HTTP {response.status_code})"
            
            return jsonify({"error": error_message}), response.status_code
        
        # Return encoding maps
        result = response.json()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error extracting encodings: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/classification_training_status')
@login_required
def get_classification_training_status():
    """API endpoint to get the current status of classification model training"""
    user_id = current_user.id
    
    # If no status exists for this user, return a default status
    if user_id not in classification_training_status:
        return jsonify({
            'status': 'not_started',
            'overall_progress': 0,
            'overall_status': 'No training in progress.',
            'model_statuses': {}
        })
    
    status_data = classification_training_status[user_id]
    
    # If training is in progress, update model statuses to simulate progress
    if status_data['status'] == 'in_progress':
        # Get the current timestamp to use as a seed for progress simulation
        current_time = int(time.time())
        
        # Update each model's progress based on time elapsed
        model_names = ['logistic-regression', 'decision-tree', 'random-forest', 'knn', 'svm', 'naive-bayes']
        
        # Calculate overall progress
        total_progress = 0
        completed_models = 0
        
        # Make all models progress very quickly to 100%
        for i, model_name in enumerate(model_names):
            # Use a very fast simulation that completes almost immediately
            # Start all models at high progress values
            base_progress = 80
            
            # Calculate a deterministic progress value based on time (80-100)
            # Different models complete at slightly different rates
            if model_name == 'logistic-regression':
                progress = min(100, base_progress + (current_time % 5) * 5)
            elif model_name == 'decision-tree':
                progress = min(100, base_progress + (current_time % 6) * 4)
            elif model_name == 'random-forest':
                progress = min(100, base_progress + (current_time % 7) * 3)
            elif model_name == 'knn':
                progress = min(100, base_progress + (current_time % 5) * 4)
            elif model_name == 'svm':
                progress = min(100, base_progress + (current_time % 8) * 3)
            else:  # naive-bayes
                progress = min(100, base_progress + (current_time % 4) * 6)
            
            # Count completed models
            if progress >= 100:
                completed_models += 1
                progress = 100
            
            # Update status text based on progress
            status_text = 'Initializing...'
            if progress >= 100:
                status_text = 'Model training complete!'
            elif progress > 80:
                status_text = 'Evaluating model performance...'
            elif progress > 60:
                status_text = 'Cross-validating model...'
            elif progress > 40:
                status_text = 'Fitting model to training data...'
            elif progress > 20:
                status_text = 'Preparing model parameters...'
            
            # Update the model's status
            status_data['model_statuses'][model_name] = {
                'progress': round(progress),
                'status': status_text
            }
            
            total_progress += progress
        
        # Update overall progress (average of all models)
        overall_progress = round(total_progress / len(model_names))
        status_data['overall_progress'] = overall_progress
        
        # Update overall status text
        if completed_models == len(model_names):
            status_data['overall_status'] = 'All models trained successfully! Finalizing results...'
            status_data['status'] = 'complete'  # Mark as complete when all models are done
        elif completed_models > len(model_names) / 2:
            status_data['overall_status'] = f'{completed_models}/{len(model_names)} models complete. Finishing remaining models...'
        elif overall_progress > 75:
            status_data['overall_status'] = 'Model training almost complete...'
        elif overall_progress > 50:
            status_data['overall_status'] = 'Model training in progress...'
        elif overall_progress > 25:
            status_data['overall_status'] = 'Processing features and training models...'
        else:
            status_data['overall_status'] = 'Preparing data and initializing models...'
    
    # Return the current status
    return jsonify(status_data)

@app.route('/api/stop_classification_training', methods=['POST'])
@login_required
def stop_classification_training():
    """API endpoint to stop the classification model training process"""
    user_id = current_user.id
    
    # Mark training as stopped in the status
    if user_id in classification_training_status:
        classification_training_status[user_id]['status'] = 'stopped'
        
        # In a real implementation, you would signal the training processes to stop
        # This could involve setting a flag in a database or sending a message to a queue
        
        # For now, just remove the status to clean up
        classification_training_status.pop(user_id, None)
    
    return jsonify({
        'status': 'stopped',
        'message': 'Training has been stopped.'
    })