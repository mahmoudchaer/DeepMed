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
            # Check if Model Coordinator service is available
            logger.info("Checking required service: Model Coordinator")
            if not is_service_available(MODEL_COORDINATOR_URL):
                error_message = "The Model Coordinator service is not available. Cannot proceed with training."
                logger.error(error_message)
                flash(error_message, 'error')
                return redirect(url_for('training'))
            else:
                logger.info("Model Coordinator service is available.")

            # Load the original raw data
            raw_data, _ = load_data(filepath)
            # Convert data to records suitable for JSON serialization
            # Handle potential NaNs/Infs that JSON cannot serialize directly
            raw_data_records = raw_data.replace([np.inf, -np.inf], np.nan).where(pd.notnull(raw_data), None).to_dict(orient='records')

            # --- Create Training Run Entry ---
            import re
            temp_filepath = session.get('uploaded_file', '')
            original_filename = ""
            if temp_filepath:
                match = re.search(r'[a-f0-9-]+_(.+)$', os.path.basename(temp_filepath))
                if match:
                    original_filename = match.group(1)
                else:
                    original_filename = os.path.basename(temp_filepath)

            run_name = original_filename if original_filename else f"Training Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            user_id = current_user.id
            local_run_id = None # Initialize run_id

            # Direct database connection to create the training run
            try:
                with app.app_context():
                    training_run = TrainingRun(
                        user_id=user_id,
                        run_name=run_name,
                        prompt=None # Prompt might be set later if applicable
                    )
                    db.session.add(training_run)
                    db.session.commit()
                    local_run_id = training_run.id
                    logger.info(f"Directly added training run to database with ID {local_run_id}")

                    # Verify entry was created
                    verification = db.session.query(TrainingRun).filter_by(id=local_run_id).first()
                    if verification:
                        logger.info(f"Verified training run in database: {verification.id}, {verification.run_name}")
                    else:
                        logger.warning(f"Could not verify training run {local_run_id} in database after commit")

            except Exception as e:
                logger.error(f"Error adding training run directly to database: {str(e)}")
                # Optionally, add fallback or flash error
                flash("Failed to create a training run record in the database. Cannot proceed.", 'error')
                return redirect(url_for('training'))

            if not local_run_id:
                 flash("Failed to obtain a valid training run ID. Cannot proceed.", 'error')
                 return redirect(url_for('training'))

            # --- Call Model Coordinator for Preprocessing and Training ---
            logger.info(f"Sending data to Model Coordinator API for preprocessing and training (Run ID: {local_run_id})")
            # Use our safe request method
            # We assume the Model Coordinator has a new '/preprocess_and_train' endpoint
            coordinator_payload = {
                "data": raw_data_records,
                "target_column": target_column,
                "test_size": session['test_size'],
                "user_id": user_id,
                "run_name": run_name,
                "run_id": local_run_id # Pass the created run_id
            }

            # Log payload size estimate (optional debug)
            try:
                 payload_size_mb = len(json.dumps(coordinator_payload)) / (1024 * 1024)
                 logger.info(f"Payload size to coordinator: {payload_size_mb:.2f} MB")
            except Exception as size_err:
                 logger.warning(f"Could not estimate payload size: {size_err}")


            response = safe_requests_post(
                f"{MODEL_COORDINATOR_URL}/preprocess_and_train", # NEW ENDPOINT
                coordinator_payload,
                timeout=1800  # Allow ample time (30 mins) for full pipeline
            )

            if response.status_code != 200:
                error_msg = f"Model Coordinator API error: {response.json().get('error', 'Unknown error')}"
                logger.error(error_msg)
                # Attempt to roll back the training run? Or mark it as failed?
                try:
                    with app.app_context():
                        run_to_update = db.session.query(TrainingRun).filter_by(id=local_run_id).first()
                        if run_to_update:
                            run_to_update.status = 'FAILED' # Assuming a status field exists
                            run_to_update.status_message = error_msg[:500] # Limit message size
                            db.session.commit()
                            logger.info(f"Marked TrainingRun {local_run_id} as FAILED.")
                except Exception as db_err:
                     logger.error(f"Failed to mark TrainingRun {local_run_id} as FAILED: {db_err}")
                raise Exception(error_msg)

            # Process the combined results from the Model Coordinator
            coordinator_result = response.json()

            # --- Save Preprocessing Data ---
            # Assume coordinator_result contains necessary preprocessing info
            # e.g., selected_features, cleaner_config, feature_selector_config, cleaning_report
            try:
                with app.app_context():
                    # Extract required info - use .get() for safety
                    original_columns = list(raw_data.columns)
                    selected_features = coordinator_result.get("selected_features", [])
                    cleaner_config = coordinator_result.get("cleaner_config", {}) # Expect coordinator to provide this structure
                    feature_selector_config = coordinator_result.get("feature_selector_config", {}) # Expect coordinator to provide this
                    cleaning_report = coordinator_result.get("cleaning_report", {}) # Expect coordinator to provide this
                    # Handle encoding mappings - maybe coordinator saved to file and provides path?
                    # Or provides a summary like before. Assume summary for now.
                    if "encoding_mappings" in cleaner_config:
                         # Assume coordinator handled large mappings and provides summary/path
                         pass # Use the config as provided by coordinator


                    # Create preprocessing data record
                    preprocessing_data = PreprocessingData(
                        run_id=local_run_id,
                        user_id=user_id,
                        original_columns=json.dumps(original_columns),
                        # Safely serialize potentially complex dicts
                        cleaner_config=json.dumps(cleaner_config, cls=SafeJSONEncoder, default=str),
                        feature_selector_config=json.dumps(feature_selector_config, cls=SafeJSONEncoder, default=str),
                        selected_columns=json.dumps(selected_features),
                        cleaning_report=json.dumps(cleaning_report, cls=SafeJSONEncoder, default=str)
                    )
                    db.session.add(preprocessing_data)
                    db.session.commit()
                    logger.info(f"Stored preprocessing data for run ID {local_run_id}")
            except Exception as pp_error:
                # Log error, but don't necessarily stop the process if model results are okay
                logger.error(f"Error saving preprocessing data for run {local_run_id}: {str(pp_error)}")
                # Flash a warning?
                flash(f"Warning: Could not save detailed preprocessing data for run {local_run_id}. Model results might still be available.", 'warning')


            # --- Handle Model Results ---
            # Extract model results part from the coordinator's response
            # Assuming the structure is similar to before, nested within coordinator_result
            model_result = coordinator_result.get("model_training_results", {}) # Adjust key if needed

            # Add code to ensure models are saved for this run properly
            from app_others import ensure_training_models_saved # Keep this import local

            # Check if models were saved (based on the response structure)
            if model_result and model_result.get('saved_best_models'):
                 # Use the local_run_id we created
                 with app.app_context():
                     ensure_training_models_saved(user_id, local_run_id, model_result)
                     logger.info(f"Ensured TrainingModel entries saved for run {local_run_id}")

                 # Save the run ID to session for model_selection page
                 session['last_training_run_id'] = local_run_id
                 logger.info(f"Set last_training_run_id in session: {local_run_id}")
            else:
                 logger.warning(f"No saved models reported by coordinator for run {local_run_id}. Skipping TrainingModel entries.")
                 # Ensure last_training_run_id is still set if we want to show results page anyway
                 session['last_training_run_id'] = local_run_id


            # Save processed results (the whole coordinator response might be useful)
            # Be mindful of session size limit! Storing the full coordinator_result might be too large.
            # Let's only store the model_result part for now.
            # If other parts are needed on the results page, they should be loaded from DB using run_id.
            session['model_results'] = model_result # Store only the model training part

            # --- Clean up intermediate session keys (no longer used) ---
            session.pop('cleaned_file', None)
            session.pop('selected_features', None) # Pop placeholder string if it existed
            session.pop('selected_features_file_json', None)
            session.pop('feature_importance_file', None)
            session.pop('selected_features_file', None)
            session.pop('anomaly_results', None)
            # Keep 'uploaded_file', 'file_stats', 'target_column', 'test_size', 'last_training_run_id', 'model_results'
            # Keep AI recommendations file key if used
            # session.pop('ai_recommendations_file', None) # Keep this as GET uses it

            # Ensure session size is okay after changes
            check_session_size()

            return redirect(url_for('model_selection'))

        except Exception as e:
            logger.error(f"Error processing data via Model Coordinator: {str(e)}", exc_info=True)
            flash(f"Error processing data: {str(e)}", 'error')
            # Redirect back to training page to allow retrying or choosing different options
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