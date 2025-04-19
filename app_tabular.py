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