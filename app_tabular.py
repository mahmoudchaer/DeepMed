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
from app_api import app, DATA_CLEANER_URL, FEATURE_SELECTOR_URL, ANOMALY_DETECTOR_URL, MODEL_COORDINATOR_URL
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
        # If accessed directly without upload, show a friendly message
        flash('Please upload a file first to start training.', 'info')
        return redirect(url_for('index'))
    
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
                        
                        # Save cleaning configuration
                        cleaner_config = {
                            "llm_instructions": cleaning_result.get("prompt", ""),
                            "options": cleaning_result.get("options", {}),
                            "handle_missing": True,
                            "handle_outliers": True
                        }
                        
                        # Save feature selection configuration
                        feature_selector_config = {
                            "llm_instructions": feature_result.get("prompt", ""),
                            "options": feature_result.get("options", {}),
                            "method": feature_result.get("method", "auto")
                        }
                        
                        # Create preprocessing data record
                        preprocessing_data = PreprocessingData(
                            run_id=local_run_id,
                            user_id=user_id,
                            cleaner_config=json.dumps(cleaner_config),
                            feature_selector_config=json.dumps(feature_selector_config),
                            original_columns=json.dumps(original_columns),
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
                    import pymysql
                    # Get database credentials from environment variables
                    MYSQL_USER = os.getenv("MYSQL_USER")
                    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
                    MYSQL_HOST = os.getenv("MYSQL_HOST")
                    MYSQL_PORT = int(os.getenv("MYSQL_PORT"))
                    MYSQL_DB = os.getenv("MYSQL_DB")
                    
                    # Connect to database
                    conn = pymysql.connect(
                        host=MYSQL_HOST,
                        user=MYSQL_USER,
                        password=MYSQL_PASSWORD,
                        port=MYSQL_PORT,
                        database=MYSQL_DB
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

            # Use our safe request method
            response = safe_requests_post(
                f"{MODEL_COORDINATOR_URL}/train",
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
                    
                    # If we have our local run_id, also ensure models are saved for it
                    if local_run_id and local_run_id != run_id_to_use:
                        with app.app_context():
                            ensure_training_models_saved(user_id, local_run_id, model_result)
                else:
                    logger.warning("No run_id available to save models")
            
            # Save processed results to session
            session['model_results'] = model_result
            
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

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    """Prediction page where users can upload data to make predictions with selected models"""
    # Check if user is logged in
    if not current_user.is_authenticated:
        flash('Please log in to use the prediction feature.', 'warning')
        return redirect(url_for('login'))
    
    # Get the selected model ID from session or query parameter
    model_id = request.args.get('model_id', session.get('selected_model_id'))
    
    # Store in session if provided
    if model_id:
        session['selected_model_id'] = model_id
    
    # Get model details if we have an ID
    model = None
    if model_id:
        model = TrainingModel.query.filter_by(id=model_id, user_id=current_user.id).first()
        if not model:
            flash('Selected model not found or does not belong to you.', 'warning')
            session.pop('selected_model_id', None)
    
    # Process file upload for prediction
    if request.method == 'POST' and model:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('prediction'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for('prediction'))
        
        if file and allowed_file(file.filename):
            # Save file
            filepath = get_temp_filepath(file.filename)
            file.save(filepath)
            
            # Load data to validate
            data, result = load_data(filepath)
            if data is None:
                if os.path.exists(filepath):
                    os.remove(filepath)
                flash(result, 'error')
                return redirect(url_for('prediction'))
            
            # Store prediction file path in session
            session['prediction_file'] = filepath
            
            # Process the prediction with the model
            try:
                # Get preprocessing data for this model's run
                preprocessing = PreprocessingData.query.filter_by(run_id=model.run_id).first()
                
                # Clean and preprocess the data similar to training
                if preprocessing and preprocessing.selected_columns:
                    selected_columns = json.loads(preprocessing.selected_columns)
                    # Ensure all required columns exist
                    missing_columns = [col for col in selected_columns if col not in data.columns]
                    if missing_columns:
                        flash(f'Missing required columns in uploaded file: {", ".join(missing_columns)}', 'error')
                        return redirect(url_for('prediction'))
                    
                    # Apply preprocessing (feature selection)
                    data = data[selected_columns]
                
                # Convert to records for JSON
                data_json = clean_data_for_json(data)
                
                # Make prediction request to model coordinator
                logger.info(f"Sending prediction request to model coordinator for model ID {model_id}")
                
                response = safe_requests_post(
                    f"{MODEL_COORDINATOR_URL}/predict_with_model",
                    {
                        "model_id": model_id,
                        "data": data_json
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    error_msg = "Error from model service."
                    try:
                        error_data = response.json()
                        if 'error' in error_data:
                            error_msg = error_data['error']
                    except:
                        pass
                    
                    flash(f'Error making prediction: {error_msg}', 'error')
                    return redirect(url_for('prediction'))
                
                # Process prediction results
                prediction_result = response.json()
                
                # Generate result table with pandas
                result_df = data.copy()
                
                # Add prediction column
                if 'predictions' in prediction_result:
                    predictions = prediction_result['predictions']
                    result_df['prediction'] = predictions
                
                # Add probability columns if available
                if 'probabilities' in prediction_result and prediction_result['probabilities']:
                    probs = prediction_result['probabilities']
                    if len(probs) == len(result_df):
                        if isinstance(probs[0], list):
                            for i, class_prob in enumerate(probs[0]):
                                result_df[f'probability_class_{i}'] = [p[i] for p in probs]
                
                # Calculate class distribution
                if 'predictions' in prediction_result:
                    distribution = []
                    value_counts = pd.Series(prediction_result['predictions']).value_counts()
                    total = len(prediction_result['predictions'])
                    
                    for value, count in value_counts.items():
                        distribution.append({
                            'class': str(value),
                            'count': int(count),
                            'percentage': round(100 * count / total, 2)
                        })
                    
                    # Store in session
                    session['prediction_distribution'] = distribution
                
                # Store only first 20 rows of result table for display
                display_df = result_df.head(20)
                html_table = display_df.to_html(classes='table table-striped', index=False)
                
                # Save complete results to file
                results_filepath = get_temp_filepath(extension='.csv')
                result_df.to_csv(results_filepath, index=False)
                session['predictions_file'] = results_filepath
                
                # Render prediction results template
                return render_template('prediction_results.html', 
                                      predictions=html_table, 
                                      distribution=session.get('prediction_distribution'),
                                      model=model)
                
            except Exception as e:
                logger.error(f"Error in prediction process: {str(e)}")
                flash(f'Error processing prediction: {str(e)}', 'error')
                return redirect(url_for('prediction'))
    
    return render_template('prediction.html', model=model)

@app.route('/download_predictions')
@login_required
def download_predictions():
    """Download prediction results as CSV"""
    predictions_filepath = session.get('predictions_file')
    if not predictions_filepath:
        flash('No prediction results available for download', 'error')
        return redirect(url_for('prediction'))
    
    # Create a BytesIO object to serve the file from memory
    file_data = io.BytesIO()
    with open(predictions_filepath, 'rb') as f:
        file_data.write(f.read())
    file_data.seek(0)
    
    return send_file(file_data, as_attachment=True, download_name='prediction_results.csv')
