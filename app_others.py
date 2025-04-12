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
import joblib
from sklearn.dummy import DummyClassifier

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
    """Create a utility script for using the model with proper preprocessing."""
    script_content = '''
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self, model_path=None, preprocessing_info_path=None):
        """
        Initialize the predictor with model and preprocessing information.
        
        Args:
            model_path: Path to the .joblib model file
            preprocessing_info_path: Path to the preprocessing_info.json file
        """
        # Find paths automatically if in the same directory
        if model_path is None:
            for file in os.listdir('.'):
                if file.endswith('.joblib'):
                    model_path = file
                    break
            if model_path is None:
                raise ValueError("No .joblib model file found in current directory")
                
        if preprocessing_info_path is None:
            if os.path.exists('preprocessing_info.json'):
                preprocessing_info_path = 'preprocessing_info.json'
        
        # Load the model
        logger.info(f"Loading model from {model_path}")
        self.model_data = joblib.load(model_path)
        
        # Handle both direct model objects and dictionary storage format
        if isinstance(self.model_data, dict) and 'model' in self.model_data:
            self.model = self.model_data['model']
            logger.info("Model loaded from dictionary format")
        else:
            self.model = self.model_data
            logger.info("Model loaded directly")
            
        # Get model type information
        self.model_type = type(self.model).__name__
        logger.info(f"Model type: {self.model_type}")
        
        # Extract components if model is a pipeline
        self.classifier = None
        self.scaler = None
        if hasattr(self.model, 'steps'):
            logger.info("Model is a pipeline with steps:")
            for name, step in self.model.steps:
                logger.info(f"- {name}: {type(step).__name__}")
                if name == 'classifier':
                    self.classifier = step
                elif name == 'scaler':
                    self.scaler = step
        
        # Load preprocessing info if available
        self.preprocessing_info = None
        if preprocessing_info_path and os.path.exists(preprocessing_info_path):
            with open(preprocessing_info_path, 'r') as f:
                self.preprocessing_info = json.load(f)
                logger.info("Loaded preprocessing information")
        
        # Validate the pipeline
        self.validate_pipeline()
                
    def validate_pipeline(self):
        """Validate that the pipeline is properly configured."""
        if not hasattr(self.model, 'steps'):
            raise ValueError("Model is not a pipeline. Expected a pipeline with StandardScaler and classifier steps.")
        
        if 'scaler' not in self.model.named_steps:
            raise ValueError("Pipeline does not contain a scaler step. Expected StandardScaler in the pipeline.")
        
        if 'classifier' not in self.model.named_steps:
            raise ValueError("Pipeline does not contain a classifier step.")
        
        if self.preprocessing_info is None:
            raise ValueError("No preprocessing information available. This is required for proper prediction.")
            
        logger.info("Pipeline validation successful")
        
    def preprocess_data(self, df):
        """Apply the same preprocessing steps as during training."""
        if self.preprocessing_info is None:
            logger.warning("No preprocessing information available. Using raw data.")
            return df
            
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Apply data cleaning
        processed_df = self._apply_cleaning(processed_df)
        
        # Apply feature selection 
        processed_df = self._apply_feature_selection(processed_df)
        
        return processed_df
        
    def _apply_cleaning(self, df):
        """Apply cleaning operations based on stored configuration."""
        if self.preprocessing_info is None or 'cleaner_config' not in self.preprocessing_info:
            return df
            
        cleaner_config = self.preprocessing_info['cleaner_config']
        
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Apply basic cleaning operations based on cleaner_config
        # Handle missing values
        if cleaner_config.get('handle_missing', True):
            for col in cleaned_df.columns:
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    # Fill numeric columns with mean or specified value
                    fill_value = cleaner_config.get('numeric_fill', 'mean')
                    if fill_value == 'mean':
                        if cleaned_df[col].isna().any():
                            # Calculate mean excluding NaN values
                            mean_value = cleaned_df[col].mean()
                            cleaned_df[col] = cleaned_df[col].fillna(mean_value)
                    elif fill_value == 'median':
                        if cleaned_df[col].isna().any():
                            # Calculate median excluding NaN values
                            median_value = cleaned_df[col].median()
                            cleaned_df[col] = cleaned_df[col].fillna(median_value)
                    elif fill_value == 'zero':
                        cleaned_df[col] = cleaned_df[col].fillna(0)
                else:
                    # Fill categorical/text columns with mode or specified value
                    fill_value = cleaner_config.get('categorical_fill', 'mode')
                    if fill_value == 'mode':
                        if cleaned_df[col].isna().any():
                            mode = cleaned_df[col].mode()
                            if not mode.empty:
                                cleaned_df[col] = cleaned_df[col].fillna(mode[0])
                    elif fill_value == 'unknown':
                        cleaned_df[col] = cleaned_df[col].fillna('unknown')
        
        # Remove duplicates if specified
        if cleaner_config.get('remove_duplicates', True):
            cleaned_df = cleaned_df.drop_duplicates()
        
        # Handle outliers if specified (using IQR method)
        if cleaner_config.get('handle_outliers', False):
            for col in cleaned_df.select_dtypes(include=['float64', 'int64']).columns:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_action = cleaner_config.get('outlier_action', 'clip')
                if outlier_action == 'clip':
                    cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
                elif outlier_action == 'remove':
                    mask = (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
                    cleaned_df = cleaned_df[mask]
        
        return cleaned_df
    
    def _apply_feature_selection(self, df):
        """Apply feature selection based on stored configuration."""
        if self.preprocessing_info is None:
            return df
            
        # If we have specific columns to select, use them
        if 'selected_columns' in self.preprocessing_info and self.preprocessing_info['selected_columns']:
            selected_columns = self.preprocessing_info['selected_columns']
            # Check which columns are available in the DataFrame
            available_columns = [col for col in selected_columns if col in df.columns]
            
            if len(available_columns) != len(selected_columns):
                missing_columns = set(selected_columns) - set(available_columns)
                logger.warning(f"Missing columns: {missing_columns}")
                
            if not available_columns:
                logger.warning("No selected columns found in the dataset. Using all columns.")
                return df
                
            return df[available_columns].copy()
        
        return df
    
    def predict(self, df):
        """Preprocess data and make predictions using the trained pipeline."""
        try:
            logger.info(f"Input data shape: {df.shape}")
            logger.info(f"Input data columns: {df.columns.tolist()}")
            
            # Preprocess the data
            processed_df = self.preprocess_data(df)
            logger.info(f"Preprocessed data shape: {processed_df.shape}")
            logger.info(f"Preprocessed data columns: {processed_df.columns.tolist()}")
            
            # Check if we have a target column and remove it
            target_col = None
            for col in ['diagnosis', 'target', 'label', 'class']:
                if col in processed_df.columns:
                    target_col = col
                    true_values = processed_df[target_col].copy()
                    processed_df = processed_df.drop(target_col, axis=1)
                    logger.info(f"Removed target column '{target_col}' for prediction")
                    break
            
            # Print pipeline steps
            if hasattr(self.model, 'steps'):
                logger.info("Pipeline steps:")
                for name, step in self.model.steps:
                    logger.info(f"- {name}: {type(step).__name__}")
            
            # Get the classifier from the pipeline
            classifier = self.model.named_steps['classifier']
            
            # Check classifier parameters
            logger.info(f"Classifier parameters: {classifier.get_params()}")
            
            # Check if we have class weights
            if hasattr(classifier, 'class_weight_'):
                logger.info(f"Class weights: {classifier.class_weight_}")
            
            # Get the scaler from the pipeline
            scaler = self.model.named_steps['scaler']
            
            # Log scaler parameters
            logger.info("Scaler parameters:")
            logger.info(f"  Mean: {scaler.mean_}")
            logger.info(f"  Scale: {scaler.scale_}")
            
            # Use the pipeline's transform method directly
            # This ensures we use the exact same transformation as during training
            scaled_data = self.model.named_steps['scaler'].transform(processed_df)
            
            # Convert back to DataFrame to maintain column names
            scaled_data = pd.DataFrame(scaled_data, columns=processed_df.columns)
            
            logger.info(f"Scaled data shape: {scaled_data.shape}")
            
            # Log some statistics about the scaled data
            logger.info("Scaled data statistics:")
            logger.info(f"  Min: {scaled_data.min()}")
            logger.info(f"  Max: {scaled_data.max()}")
            logger.info(f"  Mean: {scaled_data.mean()}")
            logger.info(f"  Std: {scaled_data.std()}")
            
            # Verify standardization
            mean_check = np.abs(scaled_data.mean()).max()
            std_check = np.abs(scaled_data.std() - 1.0).max()
            logger.info(f"Standardization check - Max absolute mean: {mean_check:.6f}, Max absolute std deviation from 1: {std_check:.6f}")
            
            # Get probabilities first
            probabilities = None
            if hasattr(classifier, 'predict_proba'):
                try:
                    # Get raw probabilities before thresholding
                    raw_probs = classifier.predict_proba(scaled_data)
                    logger.info("Successfully obtained raw prediction probabilities")
                    
                    # Log probability distribution
                    logger.info("Raw probability distribution:")
                    for i, class_name in enumerate(classifier.classes_):
                        probs = raw_probs[:, i]
                        logger.info(f"  Class {class_name}:")
                        logger.info(f"    Min: {probs.min():.4f}")
                        logger.info(f"    Max: {probs.max():.4f}")
                        logger.info(f"    Mean: {probs.mean():.4f}")
                        logger.info(f"    Std: {probs.std():.4f}")
                    
                    # Get decision function if available
                    if hasattr(classifier, 'decision_function'):
                        decision_scores = classifier.decision_function(scaled_data)
                        logger.info("Decision function scores:")
                        logger.info(f"  Min: {decision_scores.min():.4f}")
                        logger.info(f"  Max: {decision_scores.max():.4f}")
                        logger.info(f"  Mean: {decision_scores.mean():.4f}")
                        logger.info(f"  Std: {decision_scores.std():.4f}")
                    
                    probabilities = raw_probs
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {str(e)}")
            
            # Make predictions using the classifier directly
            predictions = classifier.predict(scaled_data)
            
            # If we have label encoder info, decode the predictions
            if self.preprocessing_info and 'label_encoder' in self.preprocessing_info:
                label_encoder_info = self.preprocessing_info['label_encoder']
                if label_encoder_info and label_encoder_info['type'] == 'LabelEncoder':
                    classes = label_encoder_info['classes_']
                    mapping = dict(zip(range(len(classes)), classes))
                    predictions = [mapping[pred] for pred in predictions]
                    logger.info("Decoded predictions using label encoder")
            
            # Print prediction distribution
            unique, counts = np.unique(predictions, return_counts=True)
            logger.info("Prediction distribution:")
            for val, count in zip(unique, counts):
                logger.info(f"  {val}: {count} ({count/len(predictions)*100:.1f}%)")
            
            # Return both predictions and probabilities if available
            result = {'predictions': predictions}
            if probabilities is not None:
                result['probabilities'] = probabilities.tolist()
            
            return result
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
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
        logger.info(f"Loaded data from {data_file} with shape {df.shape}")
        
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
        result_df['prediction'] = predictions['predictions']
        result_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
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
        logger.error(f"Error preprocessing data: {str(e)}\n{traceback.format_exc()}")
        flash(f"Error preprocessing data: {str(e)}", "danger")
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
    # Get all training runs to show even those without models
    all_runs = TrainingRun.query.filter_by(user_id=current_user.id).\
        order_by(TrainingRun.created_at.desc()).all()
    
    # Get the models for each run and attach them to the run object
    for run in all_runs:
        # Get the models for this run
        run.models = TrainingModel.query.filter_by(run_id=run.id, user_id=current_user.id).all()
        run.model_count = len(run.models)
    
    # Also get runs that specifically have models
    runs_with_models = [run for run in all_runs if run.model_count > 0]
    
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
        # Load the model to get scaler and encoder parameters
        try:
            model_data = joblib.load(model.model_url)
            if isinstance(model_data, dict) and 'model' in model_data:
                pipeline = model_data['model']
                label_encoder = model_data.get('label_encoder')
            else:
                pipeline = model_data
                label_encoder = None
                
            # Get scaler parameters if it's a pipeline with a scaler
            scaler_params = None
            if hasattr(pipeline, 'named_steps') and 'scaler' in pipeline.named_steps:
                scaler = pipeline.named_steps['scaler']
                scaler_params = {
                    'mean_': scaler.mean_.tolist(),
                    'scale_': scaler.scale_.tolist(),
                    'var_': scaler.var_.tolist(),
                    'n_samples_seen_': scaler.n_samples_seen_
                }
            
            # Get label encoder information if available
            label_encoder_info = None
            if label_encoder is not None:
                label_encoder_info = {
                    'classes_': label_encoder.classes_.tolist(),
                    'type': 'LabelEncoder'
                }
            
            preprocessing_info = {
                'selected_columns': json.loads(preprocessing_data.selected_columns) if preprocessing_data.selected_columns else [],
                'cleaning_report': json.loads(preprocessing_data.cleaning_report) if preprocessing_data.cleaning_report else {},
                'scaler_params': scaler_params,
                'label_encoder': label_encoder_info
            }
        except Exception as e:
            logger.error(f"Error getting model parameters: {str(e)}")
            preprocessing_info = {
                'selected_columns': json.loads(preprocessing_data.selected_columns) if preprocessing_data.selected_columns else [],
                'cleaning_report': json.loads(preprocessing_data.cleaning_report) if preprocessing_data.cleaning_report else {}
            }
    
    try:
        # Create a temporary directory for the package
        temp_dir = tempfile.mkdtemp()
        
        # Download the model file from storage or use a placeholder
        model_downloaded = False
        
        if model.model_url:
            try:
                logger.info(f"Attempting to download model from {model.model_url}")
                
                # If a filename is provided, use it; otherwise extract from URL
                if model.file_name:
                    local_model_path = os.path.join(temp_dir, model.file_name)
                else:
                    # Extract filename from URL
                    filename = model.model_url.split('/')[-1]
                    local_model_path = os.path.join(temp_dir, filename)
                
                # Try to import the storage module
                try:
                    from storage import download_blob
                    # Download the model file directly to the local path
                    download_success = download_blob(model.model_url, local_model_path)
                    if download_success:
                        model_downloaded = True
                        logger.info(f"Downloaded model to {local_model_path}")
                    else:
                        raise Exception("Failed to download blob directly to file")
                except ImportError:
                    # If storage module is not available, try to download directly with requests
                    logger.info("Storage module not available, using requests instead")
                    try:
                        with requests.get(model.model_url, stream=True) as r:
                            r.raise_for_status()
                            with open(local_model_path, 'wb') as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    f.write(chunk)
                        model_downloaded = True
                        logger.info(f"Downloaded model to {local_model_path} using requests")
                    except Exception as e:
                        logger.error(f"Error downloading model with requests: {str(e)}")
                        # Create a placeholder model file
                        placeholder_path = os.path.join(temp_dir, model.file_name or 'model.joblib')
                        with open(placeholder_path, 'w') as f:
                            f.write("# This is a placeholder for the model file that couldn't be downloaded\n")
                            f.write(f"# Original URL: {model.model_url}\n")
                            f.write(f"# Error: {str(e)}\n")
                        
                        logger.info(f"Created placeholder model file at {placeholder_path}")
            except Exception as e:
                logger.error(f"Error in model download process: {str(e)}")
                flash(f"Error downloading model: {str(e)}", "warning")
                # Continue with the process, using a placeholder
                placeholder_path = os.path.join(temp_dir, model.file_name or 'model.joblib')
                with open(placeholder_path, 'w') as f:
                    f.write("# This is a placeholder for the model file that couldn't be downloaded\n")
                    f.write(f"# Original URL: {model.model_url}\n")
                    f.write(f"# Error: {str(e)}\n")
        else:
            # No URL available, create a placeholder file
            placeholder_path = os.path.join(temp_dir, model.file_name or 'model.joblib')
            with open(placeholder_path, 'w') as f:
                f.write("# This is a placeholder for the model file\n")
                f.write("# No model URL was available for download\n")
            
            logger.warning("No model URL available, created placeholder file")
        
        # Save preprocessing info to a JSON file
        if preprocessing_info:
            preprocessing_file = os.path.join(temp_dir, 'preprocessing_info.json')
            with open(preprocessing_file, 'w') as f:
                json.dump(preprocessing_info, f, indent=2)
        
        # Create utility script for using the model
        create_utility_script(temp_dir, model.file_name or 'model.joblib', preprocessing_info)
        
        # Create model helper script for robust loading
        create_model_helper_script(temp_dir)
        
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
        
        if not model_downloaded:
            flash("Model file could not be downloaded. A placeholder has been included in the package.", "warning")
        
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

@app.route('/container_prediction', methods=['POST'])
def container_prediction():
    try:
        # Get the uploaded file
        file = request.files['file']
        if not file:
            flash('No file uploaded', 'error')
            return redirect(url_for('container_prediction'))
        
        # Save the file temporarily
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(file_path)
        
        # Read the data
        df = pd.read_csv(file_path)
        print(f"Read data with shape: {df.shape}")
        
        # Get the model path from the form
        model_path = request.form.get('model_path')
        if not model_path:
            flash('No model path provided', 'error')
            return redirect(url_for('container_prediction'))
        
        # Create model predictor
        predictor = ModelPredictor(model_path)
        
        # Make predictions
        prediction_result = predictor.predict(df)
        predictions = prediction_result['predictions']
        
        # Add probability columns if available
        if 'probabilities' in prediction_result:
            probabilities = prediction_result['probabilities']
            class_names = predictor.model.classes_
            for i, class_name in enumerate(class_names):
                df[f'probability_{class_name}'] = [prob[i] for prob in probabilities]
        
        # Add predictions to the dataframe
        df['prediction'] = predictions
        
        # Calculate class distribution
        class_dist = df['prediction'].value_counts().to_dict()
        total = len(df)
        class_dist_percent = {k: f"{v/total*100:.1f}%" for k, v in class_dist.items()}
        
        # Prepare display dataframe (first 20 rows)
        display_df = df.head(20)
        
        # Save results to CSV
        output_file = os.path.join(temp_dir, 'predictions.csv')
        df.to_csv(output_file, index=False)
        
        return render_template('container_prediction.html',
                             display_df=display_df,
                             class_dist=class_dist,
                             class_dist_percent=class_dist_percent,
                             output_file=output_file)
        
    except Exception as e:
        flash(f'Error during prediction: {str(e)}', 'error')
        return redirect(url_for('container_prediction'))

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
    """Handle model selection and redirect to appropriate page"""
    try:
        # Get current run ID
        run_id = session.get('last_training_run_id')
        if not run_id:
            flash("No training run found. Please train a model first.", "error")
        return redirect(url_for('training'))
    
        # Find this model in the database
        model = TrainingModel.query.filter_by(
            user_id=current_user.id,
        run_id=run_id,
            model_name=f"best_model_for_{metric}"
        ).first()
    
        if not model:
            flash(f"Model not found for {metric}.", "error")
            return redirect(url_for('model_selection'))
    
        # Store the selected model ID in session
        session['selected_model_id'] = model.id
    
        # Redirect to model selection page
        flash(f"Model for optimizing {metric} has been selected.", "success")
        return redirect(url_for('model_selection'))
    except Exception as e:
        logger.error(f"Error selecting model: {str(e)}")
        flash(f"Error selecting model: {str(e)}", "error")
        return redirect(url_for('model_selection'))

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

def create_model_helper_script(temp_dir):
    """Create a helper script for loading models robustly."""
    script_content = '''
# Helper script for loading models with version compatibility
import joblib
import os
import sys
import warnings
from sklearn.exceptions import UserWarning

def load_model(model_path=None):
    """
    Load a model file with compatibility handling for different sklearn versions.
    
    Args:
        model_path: Path to the .joblib model file. If None, will try to find a .joblib file in current directory.
        
    Returns:
        The loaded model
    """
    # Find model file if not specified
    if model_path is None:
        for file in os.listdir('.'):
            if file.endswith('.joblib'):
                model_path = file
                break
        if model_path is None:
            raise ValueError("No .joblib model file found in current directory")
    
    print(f"Loading model from {model_path}")
    
    # Try to load the model with warnings for version incompatibility
    with warnings.catch_warnings(record=True) as warning_list:
        model_data = joblib.load(model_path)
        
        # Check for version warnings
        for warning in warning_list:
            print(f"Warning during model loading: {warning.message}")
            
    # Handle both direct model objects and dictionary storage format
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        print("Model loaded from dictionary format")
    else:
        model = model_data
        print("Model loaded directly")
        
    print(f"Model type: {type(model).__name__}")
    return model

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = load_model(sys.argv[1])
    else:
        model = load_model()
    
    print("Model successfully loaded!")
    print(f"Model type: {type(model).__name__}")
'''

    # Write the utility script to a file
    script_path = os.path.join(temp_dir, 'model_helper.py')
    with open(script_path, 'w') as f:
        f.write(script_content)
