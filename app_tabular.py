from flask import render_template, request, redirect, url_for, session, flash, send_file, jsonify
from flask_login import login_required, current_user
import os
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import io
import logging
from datetime import datetime
import requests
import tempfile
import secrets
import uuid
from werkzeug.utils import secure_filename

# Import common components from app_api.py
from app_api import app, DATA_CLEANER_URL, FEATURE_SELECTOR_URL, ANOMALY_DETECTOR_URL, MODEL_COORDINATOR_URL, MEDICAL_ASSISTANT_URL
from app_api import is_service_available, get_temp_filepath, safe_requests_post, cleanup_session_files, check_services
from app_api import allowed_file, load_data, clean_data_for_json, SafeJSONEncoder, logger
from app_api import save_to_temp_file, load_from_temp_file, check_session_size

# Import database models
from db.users import db, User, TrainingRun, TrainingModel, PreprocessingData

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    # Double check authentication - ensure user is logged in
    if not current_user.is_authenticated:
        logger.warning("Upload attempted without authentication")
        flash('Please log in to upload files.', 'warning')
        return redirect(url_for('login'))
        
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
    filepath = session.get('uploaded_file')
    
    if not filepath:
        # If accessed directly without upload, show the upload interface
        # Check services health for status display
        services_status = check_services()
        return render_template('index.html', services_status=services_status)
    
    data, _ = load_data(filepath)
    
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
                "Model Coordinator": MODEL_COORDINATOR_URL
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
            
            # Extract original dataset filename from the temporary filepath
            import re
            temp_filepath = session.get('uploaded_file', '')
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

            # Prepare run name for the training job
            run_name = original_filename if original_filename else f"Training Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Create database entry for the training run
            local_run_id = None
            try:
                # Create a temporary app context for database operations
                with app.app_context():
                    # Create training run entry
                    training_run = TrainingRun(
                        user_id=current_user.id,
                        run_name=run_name,
                        prompt=None
                    )
                    db.session.add(training_run)
                    db.session.commit()
                    
                    # Store the run_id for model saving
                    local_run_id = training_run.id
                    logger.info(f"Added training run to database with ID {local_run_id}")
            except Exception as e:
                logger.error(f"Error adding training run to database: {str(e)}")
            
            # MODIFIED APPROACH: Send the raw data to the model coordinator with metadata
            # The model coordinator will handle all the preprocessing steps
            logger.info(f"Sending raw data to Model Coordinator for full processing")
            
            # Convert data to records format suitable for API transmission
            data_records = data.replace([np.inf, -np.inf], np.nan).where(pd.notnull(data), None).to_dict(orient='records')
            
            # Send the request with all necessary metadata
            response = safe_requests_post(
                f"{MODEL_COORDINATOR_URL}/train",
                {
                    "raw_data": data_records,
                    "target_column": target_column,
                    "test_size": session['test_size'],
                    "user_id": current_user.id,
                    "run_id": local_run_id,
                    "run_name": run_name,
                    "file_name": original_filename,
                    "handle_preprocessing": True  # Signal to the coordinator to do all preprocessing
                },
                timeout=1800  # Model training can take time
            )
            
            if response.status_code != 200:
                raise Exception(f"Model Coordinator API error: {response.json().get('error', 'Unknown error')}")
            
            # Store the complete model results in session
            model_result = response.json()
            
            # Store the run ID for model_selection page
            session['last_training_run_id'] = model_result.get('run_id', local_run_id)
            
            # Get any preprocessing results that should be stored in session
            if 'preprocessing_results' in model_result:
                pp_results = model_result['preprocessing_results']
                
                # Store preprocessing artifacts if provided
                if 'cleaned_data_summary' in pp_results:
                    session['cleaning_summary'] = pp_results['cleaned_data_summary']
                    
                if 'feature_importance' in pp_results:
                    # Save feature importance to file
                    feature_importance = pp_results['feature_importance']
                    feature_importance_file = save_to_temp_file(feature_importance, 'feature_importance')
                    session['feature_importance_file'] = feature_importance_file
                
                if 'anomaly_results' in pp_results:
                    session['anomaly_results'] = pp_results['anomaly_results']
                
                if 'selected_features' in pp_results:
                    selected_features = pp_results['selected_features']
                    # Save selected features to file to prevent session bloat
                    selected_features_file_json = save_to_temp_file(selected_features, 'selected_features')
                    session['selected_features_file_json'] = selected_features_file_json
                    # Use a reference in session to avoid storing large data
                    session['selected_features'] = f"[{len(selected_features)} features]"
            
            # Save model results to session
            session['model_results'] = model_result
            
            # Add code to ensure models are saved for this run properly
            from app_others import ensure_training_models_saved
            if 'saved_best_models' in model_result and model_result['saved_best_models']:
                run_id_to_use = model_result.get('run_id', local_run_id)
                if run_id_to_use:
                    with app.app_context():
                        ensure_training_models_saved(current_user.id, run_id_to_use, model_result)
            
            return redirect(url_for('model_selection'))
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}", exc_info=True)
            flash(f"Error processing data: {str(e)}", 'error')
            return redirect(url_for('training'))
    
    # Get AI recommendations for the dataset (via Medical Assistant API) - OPTIONAL
    ai_recommendations = None
    if 'ai_recommendations' not in session and 'ai_recommendations_file' not in session and is_service_available(MEDICAL_ASSISTANT_URL):
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
    elif 'ai_recommendations_file' in session:
        # Load from file
        ai_recommendations = load_from_temp_file(session['ai_recommendations_file'])
    else:
        # Try to get from session (backward compatibility)
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
        session['logout_token'] = secrets.token_hex(16)
    
    # Check services health for status display
    services_status = check_services()
    
    # Add tabular prediction service to services status
    predictor_service_url = os.environ.get('TABULAR_PREDICTOR_SERVICE_URL', 'http://localhost:5101')
    services_status['tabular_predictor_service'] = is_service_available(predictor_service_url)
    
    return render_template('tabular_prediction.html', services_status=services_status, logout_token=session['logout_token'])

@app.route('/api/predict_tabular', methods=['POST'])
@login_required
def api_predict_tabular():
    """API endpoint for tabular prediction using custom models"""
    # Simply forward the request to the predictor service
    try:
        # Define the predictor service URL
        predictor_service_url = os.environ.get('TABULAR_PREDICTOR_SERVICE_URL', 'http://localhost:5101')
        
        # Check if the predictor service is available
        if not is_service_available(predictor_service_url):
            return jsonify({"error": "Tabular prediction service is not available. Please try again later."}), 503
        
        # Forward the files and form data directly
        response = requests.post(
            f"{predictor_service_url}/predict",
            files={name: (f.filename, f.stream, f.content_type) 
                  for name, f in request.files.items()},
            data=request.form,
            timeout=600  # 10 minute timeout
        )
        
        # Return the predictor service response directly
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        logger.error(f"Error in tabular prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict_tabular/extract_encodings', methods=['POST'])
@login_required
def api_extract_encodings():
    """API endpoint to extract encoding maps from a model package"""
    # Simply forward the request to the predictor service
    try:
        # Define the predictor service URL
        predictor_service_url = os.environ.get('TABULAR_PREDICTOR_SERVICE_URL', 'http://localhost:5101')
        
        # Check if the predictor service is available
        if not is_service_available(predictor_service_url):
            return jsonify({"error": "Tabular prediction service is not available. Please try again later."}), 503
        
        # Forward the files directly
        response = requests.post(
            f"{predictor_service_url}/extract_encodings",
            files={name: (f.filename, f.stream, f.content_type) 
                  for name, f in request.files.items()},
            timeout=60  # 1 minute timeout
        )
        
        # Return the predictor service response directly
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        logger.error(f"Error extracting encodings: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500