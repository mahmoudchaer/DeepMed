from flask import render_template, request, redirect, url_for, session, flash, send_file, jsonify, make_response, Response
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import os
import pandas as pd
import numpy as np
import json
import io
import time
from datetime import datetime
import logging
import tempfile
import uuid
import secrets
from werkzeug.utils import secure_filename
import requests
import urllib.parse
import zipfile
import shutil
import atexit
import glob

# Import common components from app_api.py
from app_api import app, DATA_CLEANER_URL, FEATURE_SELECTOR_URL, MODEL_COORDINATOR_URL, MEDICAL_ASSISTANT_URL
from app_api import UPLOAD_FOLDER, is_service_available, get_temp_filepath, safe_requests_post, logger, SafeJSONEncoder
from app_api import load_data, clean_data_for_json, check_services

# Import database models
from db.users import db, User, TrainingRun, TrainingModel, PreprocessingData

def ensure_training_models_saved(user_id, run_id, model_result):
    """Ensure that all 4 best models are saved to the TrainingModel table"""
    try:
        # Check if we have saved_best_models in the result
        if 'saved_best_models' not in model_result or not model_result['saved_best_models']:
            logger.warning("No saved_best_models found in model result")
            return False
            
        # Get the saved models
        saved_models = model_result['saved_best_models']
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Check if models already exist for this run
        existing_models = TrainingModel.query.filter_by(run_id=run_id).count()
        if existing_models >= 4:
            logger.info(f"Found {existing_models} models already saved for run_id {run_id}")
            return True
            
        # Save each model
        for metric in metrics:
            if metric not in saved_models:
                logger.warning(f"No saved model found for metric {metric}")
                continue
                
            model_info = saved_models[metric]
            
            # Check if we have the URL
            if 'url' not in model_info:
                logger.warning(f"No URL found for saved model {metric}")
                continue
                
            # Create model record
            model_name = f"best_model_for_{metric}"
            model_url = model_info['url']
            
            # Extract filename from URL or use the one from the model_info
            if 'filename' in model_info:
                filename = model_info['filename']
            else:
                # Extract filename from URL: https://accountname.blob.core.windows.net/container/filename
                filename = model_url.split('/')[-1]
            
            # Check if this model is already saved
            existing_model = TrainingModel.query.filter_by(
                run_id=run_id,
                model_name=model_name
            ).first()
            
            if existing_model:
                logger.info(f"Model {model_name} already exists for run_id {run_id}")
                continue
                
            # Create and save the model record
            model_record = TrainingModel(
                user_id=user_id,
                run_id=run_id,
                model_name=model_name,
                model_url=model_url,
                file_name=filename  # Save the filename too
            )
            
            db.session.add(model_record)
            logger.info(f"Added model {model_name} to database for run_id {run_id}")
        
        # Commit all changes
        db.session.commit()
        logger.info(f"Committed training models to database for run_id {run_id}")
        
        # Verify models were saved
        saved_count = TrainingModel.query.filter_by(run_id=run_id).count()
        logger.info(f"Verified {saved_count} models saved for run_id {run_id}")
        
        return True
            
    except Exception as e:
        logger.error(f"Error ensuring training models are saved: {str(e)}")
        return False

def create_utility_script(temp_dir, model_filename, preprocessing_info):
    """Create a utility script for using the model"""
    script_content = '''import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

class ModelPredictor:
    """Utility class for making predictions with the trained model"""
    
    def __init__(self):
        # Load the model
        self.model = self._load_model()
        # Load preprocessing info
        self.preprocessing_info = self._load_preprocessing_info()
        # Store column names
        self.selected_columns = self.preprocessing_info.get('selected_columns', [])
        
    def _load_model(self):
        """Load the model from file"""
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Find model file in the directory (should be a .joblib file)
        model_files = [f for f in os.listdir(script_dir) 
                      if f.endswith('.joblib') or f.endswith('.pkl')]
        
        if not model_files:
            raise ValueError("No model file found in the directory")
        
        model_path = os.path.join(script_dir, model_files[0])
        print(f"Loading model from {model_path}")
        
        # Load the model
        return joblib.load(model_path)
    
    def _load_preprocessing_info(self):
        """Load preprocessing information from the JSON file"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        preprocessing_path = os.path.join(script_dir, "preprocessing_info.json")
        
        if not os.path.exists(preprocessing_path):
            print("Warning: preprocessing_info.json not found")
            return {}
        
        with open(preprocessing_path, 'r') as f:
            return json.load(f)
    
    def preprocess_data(self, df):
        """Apply preprocessing steps to the input data"""
        # Ensure all required columns are present
        missing_columns = [col for col in self.selected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in input data: {missing_columns}")
        
        # Select only the columns used during training
        if self.selected_columns:
            df = df[self.selected_columns].copy()
        
        # Basic preprocessing
        # 1. Handle missing values
        df = df.fillna(df.mean())
        
        # 2. Convert categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes
        
        return df
    
    def predict(self, df):
        """Preprocess data and make predictions"""
        # Preprocess the data
        processed_df = self.preprocess_data(df)
        
        # Make predictions
        try:
            # Try regular predict method first
            predictions = self.model.predict(processed_df)
            prediction_methods = ["predict"]
        except Exception as e:
            print(f"Error using standard predict method: {str(e)}")
            print("Trying compatibility fixes...")
            
            # Try predict_proba method for classifier
            try:
                proba = self.model.predict_proba(processed_df)
                predictions = np.argmax(proba, axis=1)
                prediction_methods = ["predict_proba", "argmax"]
            except Exception as e2:
                print(f"Error with predict_proba: {str(e2)}")
                
                # Last resort: decision function
                try:
                    decision = self.model.decision_function(processed_df)
                    if len(decision.shape) > 1 and decision.shape[1] > 1:
                        predictions = np.argmax(decision, axis=1)
                        prediction_methods = ["decision_function", "argmax"]
                    else:
                        predictions = (decision > 0).astype(int)
                        prediction_methods = ["decision_function", "threshold"]
                except Exception as e3:
                    print(f"All prediction methods failed: {str(e3)}")
                    raise ValueError("Could not make predictions with this model")
        
        # Output prediction statistics
        print(f"\nPrediction Summary:")
        unique, counts = np.unique(predictions, return_counts=True)
        for val, count in zip(unique, counts):
            print(f"  {val}: {count} ({count/len(predictions)*100:.1f}%)")
            
        print(f"Prediction method used: {prediction_methods[0]}")
        return predictions
        
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <data_file.csv>")
        print("The data file should be a CSV file with features.")
        sys.exit(1)
        
    data_file = sys.argv[1]
    
    try:
        # Load the data
        df = pd.read_csv(data_file)
        print(f"Loaded data from {data_file} with shape {df.shape}")
        
        # Create a predictor instance
        predictor = ModelPredictor()
        
        # Make predictions
        predictions = predictor.predict(df)
        
        # Save predictions to file
        output_file = data_file.replace(".csv", "_predictions.csv")
        # If the file doesn't end with .csv, just append _predictions.csv
        if output_file == data_file:
            output_file = data_file + "_predictions.csv"
            
        result_df = df.copy()
        result_df['prediction'] = predictions
        result_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
'''

    # Write the utility script to a file
    script_path = os.path.join(temp_dir, 'predict.py')
    with open(script_path, 'w') as f:
        f.write(script_content)

def create_readme_file(temp_dir, model, preprocessing_info):
    """Create a README file with instructions for using the model."""
    readme_content = f'''# Model Package: {model.model_name}

## Contents

- {model.file_name if model.file_name else model.model_name + ".joblib"}: The trained model file
- preprocessing_info.json: Configuration for data preprocessing
- predict.py: Utility script for making predictions
- README.md: This file

## Quick Start
1. Make sure you have the required packages installed:
   ```
   pip install pandas numpy scikit-learn joblib
   ```

2. Place your data file (CSV format) in the same directory as these files.

3. Run the prediction script:
   ```
   python predict.py your_data.csv
   ```

4. The script will:
   - Load your data
   - Preprocess it using the same steps as during training
   - Make predictions with the model
   - Save the results as your_data_predictions.csv

## Version Compatibility

The included predict.py script is designed to work with different versions of scikit-learn. If you encounter any issues with predictions (such as all predictions being the same value), the script will automatically apply compatibility fixes to ensure accurate results.

This makes the model package robust across different environments and scikit-learn versions. The script will provide detailed logging about which prediction method was used.

## Using in Your Own Code
```python
from predict import ModelPredictor
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize the predictor
predictor = ModelPredictor()

# Preprocess data and get predictions
predictions = predictor.predict(df)

# Or do the steps separately
processed_df = predictor.preprocess_data(df)
predictions = predictor.predict(processed_df)
```

## Model Information
- Model Type: {model.model_name}
- Created: {model.created_at}
- ID: {model.id}

'''

    # Add preprocessing information if available
    if preprocessing_info:
        if 'selected_columns' in preprocessing_info and preprocessing_info['selected_columns']:
            column_count = len(preprocessing_info['selected_columns'])
            readme_content += f"\n## Preprocessing Information\n"
            readme_content += f"- Selected Features: {column_count} features\n"
            if column_count <= 20:  # Only list if not too many
                readme_content += f"- Feature List: {', '.join(preprocessing_info['selected_columns'])}\n"
            
            if 'cleaning_report' in preprocessing_info and preprocessing_info['cleaning_report']:
                readme_content += "- Cleaning Operations Applied:\n"
                for operation, details in preprocessing_info['cleaning_report'].items():
                    if isinstance(details, dict):
                        readme_content += f"  - {operation}: {details.get('description', 'Applied')}\n"
                    else:
                        readme_content += f"  - {operation}: Applied\n"
    
    readme_path = os.path.join(temp_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)

@app.route('/preprocess_new_data/<int:run_id>', methods=['POST'])
@login_required
def preprocess_new_data(run_id):
    """Preprocess a new dataset using the same steps as a previous training run."""
    import pandas as pd
    import json
    import traceback
    
    user_id = current_user.id
    
    # Check if the run exists and belongs to the user
    training_run = TrainingRun.query.filter_by(id=run_id, user_id=user_id).first_or_404()
    
    # Get preprocessing data for this run
    preproc_data = PreprocessingData.query.filter_by(run_id=run_id, user_id=user_id).first_or_404()
    
    if 'dataFile' not in request.files:
        flash("No file was uploaded", "danger")
        return redirect(url_for('my_models'))
    
    file = request.files['dataFile']
    
    if file.filename == '':
        flash("No file was selected", "danger")
        return redirect(url_for('my_models'))
    
    if not allowed_file(file.filename):
        flash("Only CSV and Excel files are supported", "danger")
        return redirect(url_for('my_models'))
    
    try:
        # Save uploaded file temporarily
        temp_path = get_temp_filepath(file.filename)
        file.save(temp_path)
        
        # Load the dataset
        if temp_path.endswith('.csv'):
            df = pd.read_csv(temp_path)
        else:
            df = pd.read_excel(temp_path)
        
        # Load preprocessing configurations
        cleaner_config = json.loads(preproc_data.cleaner_config) if preproc_data.cleaner_config else {}
        feature_selector_config = json.loads(preproc_data.feature_selector_config) if preproc_data.feature_selector_config else {}
        selected_columns = json.loads(preproc_data.selected_columns) if preproc_data.selected_columns else []
        
        # Apply data cleaning using the stored configuration
        cleaned_df = apply_stored_cleaning(df, cleaner_config)
        
        # Apply feature selection using the stored configuration
        if selected_columns:
            # If we have a list of selected columns, use it
            processed_df = cleaned_df[selected_columns].copy()
        else:
            # Otherwise, apply the feature selection config
            processed_df = apply_stored_feature_selection(cleaned_df, feature_selector_config)
        
        # Create a temporary file for the processed data
        processed_file_path = get_temp_filepath(original_filename=file.filename, extension='.csv')
        processed_df.to_csv(processed_file_path, index=False)
        
        # Create a descriptive filename for the download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"preprocessed_data_{run_id}_{timestamp}.csv"
        
        return send_file(
            processed_file_path,
            as_attachment=True,
            download_name=output_filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        flash(f"Error during preprocessing: {str(e)}", "danger")
        return redirect(url_for('my_models'))

def apply_stored_cleaning(df, cleaner_config):
    """Apply the stored cleaning configuration to a dataframe"""
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Apply basic cleaning steps
    return apply_basic_cleaning(cleaned_df, cleaner_config)

def apply_basic_cleaning(df, cleaner_config):
    """Apply basic cleaning steps to a dataframe"""
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # 1. Handle missing values - simple imputation
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
    categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
    
    # For numeric columns, fill with mean
    for col in numeric_cols:
        if cleaned_df[col].isna().any():
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    # For categorical columns, fill with mode
    for col in categorical_cols:
        if cleaned_df[col].isna().any():
            mode_value = cleaned_df[col].mode()[0]
            cleaned_df[col] = cleaned_df[col].fillna(mode_value)
    
    # 2. Handle outliers - replace outliers with column limits
    for col in numeric_cols:
        # Calculate IQR
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace outliers
        cleaned_df[col] = cleaned_df[col].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    
    return cleaned_df

def apply_stored_feature_selection(df, feature_selector_config):
    """Apply the stored feature selection configuration to a dataframe"""
    # If we have specified columns in the config, use them
    if 'selected_columns' in feature_selector_config and feature_selector_config['selected_columns']:
        selected_columns = feature_selector_config['selected_columns']
        # Ensure all columns exist in the dataframe
        available_columns = [col for col in selected_columns if col in df.columns]
        return df[available_columns].copy()
    
    # Otherwise, return the original dataframe
    return df.copy()

@app.route('/my_models')
@login_required
def my_models():
    """Show the user's trained models"""
    # Get the user's training runs with models
    runs_with_models = db.session.query(TrainingRun).\
        join(TrainingModel, TrainingRun.id == TrainingModel.run_id).\
        filter(TrainingRun.user_id == current_user.id).\
        group_by(TrainingRun.id).\
        order_by(TrainingRun.created_at.desc()).all()
    
    # Also get all training runs to show even those without models
    all_runs = TrainingRun.query.filter_by(user_id=current_user.id).\
        order_by(TrainingRun.created_at.desc()).all()
    
    # Get the count of models for each run
    for run in all_runs:
        run.model_count = TrainingModel.query.filter_by(run_id=run.id).count()
    
    return render_template('my_models.html', 
                          training_runs=all_runs,
                          runs_with_models=runs_with_models)

@app.route('/download_model/<int:model_id>')
@login_required
def download_model(model_id):
    """Download a model as a package with utility scripts"""
    # Get the model
    model = TrainingModel.query.filter_by(id=model_id, user_id=current_user.id).first_or_404()
    
    # Get the corresponding training run
    run = TrainingRun.query.filter_by(id=model.run_id).first_or_404()
    
    # Get preprocessing info
    preprocessing_data = PreprocessingData.query.filter_by(run_id=model.run_id).first()
    
    # Prepare preprocessing info for the package
    preprocessing_info = None
    if preprocessing_data:
        preprocessing_info = {
            'selected_columns': json.loads(preprocessing_data.selected_columns) if preprocessing_data.selected_columns else [],
            'cleaning_report': json.loads(preprocessing_data.cleaning_report) if preprocessing_data.cleaning_report else {}
        }
    
    try:
        # Create a temporary directory for the package
        temp_dir = tempfile.mkdtemp()
        
        # Download the model file from storage
        from storage import download_blob
        if model.model_url:
            try:
                logger.info(f"Downloading model from {model.model_url}")
                
                # If a filename is provided, use it; otherwise extract from URL
                if model.file_name:
                    local_model_path = os.path.join(temp_dir, model.file_name)
                else:
                    # Extract filename from URL
                    filename = model.model_url.split('/')[-1]
                    local_model_path = os.path.join(temp_dir, filename)
                
                # Download the model file
                download_blob(model.model_url, local_model_path)
                logger.info(f"Downloaded model to {local_model_path}")
            except Exception as e:
                logger.error(f"Error downloading model: {str(e)}")
                flash(f"Error downloading model: {str(e)}", "danger")
                return redirect(url_for('my_models'))
        else:
            flash("No model URL available for download", "danger")
            return redirect(url_for('my_models'))
        
        # Save preprocessing info to a JSON file
        if preprocessing_info:
            preprocessing_file = os.path.join(temp_dir, 'preprocessing_info.json')
            with open(preprocessing_file, 'w') as f:
                json.dump(preprocessing_info, f, indent=2)
        
        # Create utility script for using the model
        create_utility_script(temp_dir, model.file_name, preprocessing_info)
        
        # Create README file
        create_readme_file(temp_dir, model, preprocessing_info)
        
        # Create ZIP archive
        zip_filename = f"model_{model.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add all files from the temp directory
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Add the file with a relative path inside the ZIP
                    zipf.write(file_path, os.path.relpath(file_path, temp_dir))
        
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        
        # Return the ZIP file
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )
        
    except Exception as e:
        logger.error(f"Error creating model package: {str(e)}")
        flash(f"Error creating model package: {str(e)}", "danger")
        return redirect(url_for('my_models'))

@app.route('/download_cleaned')
@login_required
def download_cleaned():
    cleaned_filepath = session.get('cleaned_file')
    if not cleaned_filepath:
        flash('No cleaned file available', 'error')
        return redirect(url_for('training'))
    
    # Create a BytesIO object to serve the file from memory
    file_data = io.BytesIO()
    with open(cleaned_filepath, 'rb') as f:
        file_data.write(f.read())
    file_data.seek(0)
    
    return send_file(file_data, as_attachment=True, download_name='cleaned_data.csv')

@app.route('/feature_importance')
def feature_importance():
    # First check if we have it in a file
    importance_file = session.get('feature_importance_file')
    importance_data = None
    
    if importance_file:
        importance_data = load_from_temp_file(importance_file)
    else:
        # Try the session (backward compatibility)
        importance_data = session.get('feature_importance')
        
        # If it's in session and large, move to file
        if importance_data and len(importance_data) > 20:
            importance_file = save_to_temp_file(importance_data, 'feature_importance')
            session['feature_importance_file'] = importance_file
            session.pop('feature_importance', None)
            logger.info(f"Moved feature importance from session to file: {importance_file}")
    
    if not importance_data:
        return jsonify({'error': 'No feature importance data available'})
    
    # Sort by importance
    importance_data = sorted(importance_data, key=lambda x: x['Importance'])
    
    # Create Plotly figure
    fig = go.Figure(go.Bar(
        x=[x['Importance'] for x in importance_data],
        y=[x['Feature'] for x in importance_data],
        orientation='h',
        marker_color='rgba(50, 171, 96, 0.6)',
    ))
    
    fig.update_layout(
        title='Feature Importance Scores',
        height=400 + len(importance_data) * 20,
        xaxis_title='Importance Score',
        yaxis_title='Feature Name',
        template='plotly_white'
    )
    
    # Convert to JSON for frontend
    graphJSON = json.dumps(fig, cls=SafeJSONEncoder)
    return jsonify(graphJSON)

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    # Check if Medical Assistant API is available
    if not is_service_available(MEDICAL_ASSISTANT_URL):
        flash('Medical Assistant service is not available.', 'error')
        return render_template('chat.html', messages=[])
    
    # Initialize chat history in session if not present
    if 'messages' not in session:
        session['messages'] = []
    
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        if prompt:
            # Add user message to chat history
            session['messages'].append({
                'role': 'user',
                'content': prompt
            })
            
            # Get AI response via API
            try:
                response = safe_requests_post(
                    f"{MEDICAL_ASSISTANT_URL}/chat",
                    {
                        "message": prompt,
                        "session_id": f"session_{id(session)}"  # Create a unique session ID
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    ai_response = response.json()["response"]
                    
                    # Add assistant response to chat history
                    session['messages'].append({
                        'role': 'assistant',
                        'content': ai_response
                    })
                    # Add logging to verify data being sent to APIs
                    logger.info(f"Data being sent to Medical Assistant Chat API: {prompt}")
                else:
                    flash(f"Error communicating with AI assistant: {response.text}", 'error')
                
            except Exception as e:
                logger.error(f"Error communicating with AI assistant: {str(e)}", exc_info=True)
                flash(f"Error communicating with AI assistant: {str(e)}", 'error')
    
    return render_template('chat.html', messages=session.get('messages', []))

@app.route('/clear_chat')
@login_required
def clear_chat():
    if 'messages' in session:
        session.pop('messages')
        
        # Also clear on the API side
        if is_service_available(MEDICAL_ASSISTANT_URL):
            try:
                safe_requests_post(
                    f"{MEDICAL_ASSISTANT_URL}/clear_chat",
                    {"session_id": f"session_{id(session)}"},
                    timeout=10
                )
            except:
                pass  # Ignore errors in clearing remote chat history
            
    return redirect(url_for('chat'))

@app.route('/container_prediction', methods=['GET', 'POST'])
@login_required
def container_prediction():
    """For making predictions with containerized models"""
    # Check if user is logged in
    if not current_user.is_authenticated:
        flash('Please log in to use the containerized prediction feature.', 'warning')
        return redirect(url_for('login'))
    
    # Get the container URL from session
    container_url = session.get('container_url')
    model_name = session.get('container_model_name', 'Containerized Model')
    
    if not container_url:
        flash('No containerized model is running. Please deploy a model first.', 'warning')
        return redirect(url_for('my_models'))
    
    # Check if the container is still running
    try:
        response = requests.get(f"{container_url}/health", timeout=2)
        if response.status_code != 200:
            flash('The containerized model is not responding. It may have been stopped.', 'warning')
            session.pop('container_url', None)
            session.pop('container_model_name', None)
            return redirect(url_for('my_models'))
    except:
        flash('The containerized model is not available. It may have been stopped.', 'warning')
        session.pop('container_url', None)
        session.pop('container_model_name', None)
        return redirect(url_for('my_models'))
    
    # Process file upload for prediction
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('container_prediction'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for('container_prediction'))
        
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
                return redirect(url_for('container_prediction'))
            
            # Convert to records for JSON
            data_json = clean_data_for_json(data)
            
            # Make prediction request to containerized model
            try:
                response = safe_requests_post(
                    f"{container_url}/predict",
                    {
                        "data": data_json
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    error_msg = "Error from container model service."
                    try:
                        error_data = response.json()
                        if 'error' in error_data:
                            error_msg = error_data['error']
                    except:
                        pass
                    
                    flash(f'Error making prediction: {error_msg}', 'error')
                    return redirect(url_for('container_prediction'))
                
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
                    session['container_prediction_distribution'] = distribution
                
                # Store only first 20 rows of result table for display
                display_df = result_df.head(20)
                html_table = display_df.to_html(classes='table table-striped', index=False)
                
                # Save complete results to file
                results_filepath = get_temp_filepath(extension='.csv')
                result_df.to_csv(results_filepath, index=False)
                session['container_predictions_file'] = results_filepath
                
                # Render prediction results template
                return render_template('container_prediction_results.html', 
                                      predictions=html_table, 
                                      distribution=session.get('container_prediction_distribution'),
                                      model_name=model_name)
                
            except Exception as e:
                logger.error(f"Error in container prediction process: {str(e)}")
                flash(f'Error processing prediction: {str(e)}', 'error')
                return redirect(url_for('container_prediction'))
    
    return render_template('container_prediction.html', 
                         model_name=model_name,
                         container_url=container_url)

@app.route('/download_container_predictions')
@login_required
def download_container_predictions():
    """Download container prediction results as CSV"""
    predictions_filepath = session.get('container_predictions_file')
    if not predictions_filepath:
        flash('No prediction results available for download', 'error')
        return redirect(url_for('container_prediction'))
    
    # Create a BytesIO object to serve the file from memory
    file_data = io.BytesIO()
    with open(predictions_filepath, 'rb') as f:
        file_data.write(f.read())
    file_data.seek(0)
    
    return send_file(file_data, as_attachment=True, download_name='container_prediction_results.csv')

@app.route('/select_model/<model_name>/<metric>')
@login_required
def select_model(model_name, metric):
    """Select a model for prediction based on model name and metric"""
    # Check if the user is logged in
    if not current_user.is_authenticated:
        flash('Please log in to select a model.', 'warning')
        return redirect(url_for('login'))
    
    # Get the training run ID from session
    run_id = session.get('last_training_run_id')
    if not run_id:
        flash('No training run found. Please train models first.', 'warning')
        return redirect(url_for('training'))
    
    # Find the model matching the name and metric
    model = TrainingModel.query.filter_by(
        run_id=run_id,
        model_name=f"best_model_for_{metric}",
        user_id=current_user.id
    ).first()
    
    if not model:
        flash(f'Model not found for {model_name} optimized for {metric}.', 'warning')
        return redirect(url_for('my_models'))
    
    # Store selected model ID in session
    session['selected_model_id'] = model.id
    
    # Redirect to prediction page
    flash(f'Selected {model_name} model optimized for {metric}.', 'success')
    return redirect(url_for('prediction'))

@app.route('/service_status')
def service_status():
    """Check the status of all services"""
    services_status = check_services()
    
    # If model coordinator is available, also check model services
    if services_status.get("Model Coordinator") == "healthy":
        try:
            response = requests.get(f"{MODEL_COORDINATOR_URL}/health", timeout=5)
            if response.status_code == 200:
                coordinator_data = response.json()
                if "model_services" in coordinator_data:
                    for model_name, status in coordinator_data["model_services"].items():
                        services_status[f"Model - {model_name}"] = "healthy" if status == "healthy" else "unhealthy"
        except Exception as e:
            logger.error(f"Error getting model service status: {str(e)}")
    
    return jsonify(services_status)

@app.route('/model_selection')
@app.route('/model_selection/<int:run_id>')
@login_required
def model_selection(run_id=None):
    """Show the model selection page with training results"""
    # Check if the user is logged in
    if not current_user.is_authenticated:
        flash('Please log in to view model selection.', 'warning')
        return redirect(url_for('login'))
    
    # Get the run ID from URL parameter, request args, or session
    if run_id is None:
        run_id = request.args.get('run_id', type=int)
        
    if run_id is None:
        run_id = session.get('last_training_run_id')
    else:
        # If run_id was provided via URL, store it in session for future use
        session['last_training_run_id'] = run_id
    
    if not run_id:
        # If we still don't have a run_id, check if the user has any training runs
        latest_run = TrainingRun.query.filter_by(user_id=current_user.id).order_by(TrainingRun.created_at.desc()).first()
        if latest_run:
            run_id = latest_run.id
            session['last_training_run_id'] = run_id
            flash(f'Using your most recent training run (ID: {run_id}).', 'info')
        else:
            flash('No training run ID found. Please train models first.', 'warning')
            return redirect(url_for('training'))
    
    # Load models directly from the database using the run_id
    db_models = TrainingModel.query.filter_by(run_id=run_id, user_id=current_user.id).all()
    
    if not db_models:
        flash('No models found in the database for this run.', 'warning')
        # Get all training runs to show as options
        all_runs = TrainingRun.query.filter_by(user_id=current_user.id).order_by(TrainingRun.created_at.desc()).all()
        if all_runs:
            return render_template('model_selection_empty.html', 
                                   run_id=run_id,
                                   all_runs=all_runs)
        else:
            return redirect(url_for('training'))
    
    # Convert database models to the format expected by the template
    models = []
    for db_model in db_models:
        # Extract metric from model name (e.g., "best_model_for_accuracy" -> "accuracy")
        metric = db_model.model_name.replace("best_model_for_", "")
        models.append({
            'id': db_model.id,
            'name': db_model.model_name,
            'metric': metric,
            'file_name': db_model.file_name
        })
    
    # Get feature importance from session (or fall back to empty)
    feature_importance = session.get('feature_importance', {})
    
    # Get training run details
    training_run = TrainingRun.query.filter_by(id=run_id).first()
    
    # Create overall metrics dictionary
    overall_metrics = {
        'models_trained': len(models),
        'dataset_name': training_run.dataset_name if training_run and hasattr(training_run, 'dataset_name') else 'Unknown',
        'target_column': training_run.target_column if training_run and hasattr(training_run, 'target_column') else 'Unknown',
        'training_date': training_run.created_at.strftime('%Y-%m-%d %H:%M') if training_run else 'Unknown',
        'total_runs': TrainingRun.query.filter_by(user_id=current_user.id).count()
    }
    
    # Render the model selection template
    return render_template(
        'model_selection.html',
        models=models,
        feature_importance=feature_importance,
        run_id=run_id,
        training_run=training_run,
        overall_metrics=overall_metrics
    )
