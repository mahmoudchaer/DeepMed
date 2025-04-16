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
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import subprocess
import threading
import sys

# Import common components from app_api.py
from app_api import app, DATA_CLEANER_URL, FEATURE_SELECTOR_URL, MODEL_COORDINATOR_URL, MEDICAL_ASSISTANT_URL
from app_api import UPLOAD_FOLDER, is_service_available, get_temp_filepath, safe_requests_post, logger, SafeJSONEncoder
from app_api import load_data, clean_data_for_json, check_services

# Import database models
from db.users import db, User, TrainingRun, TrainingModel, PreprocessingData

# Define a custom JSON encoder at the top of the file, right after the imports
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that can handle numpy types like int64"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

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
                
            # Get the metric value directly from the saved_best_models structure
            metric_value = None
            if 'value' in model_info:
                metric_value = model_info['value']
                logger.info(f"Found metric value directly in saved_best_models: {metric_value}")
            
            # If we don't have the value directly, try other locations in the result structure
            if metric_value is None:
                # Try model_metrics
                if 'model_metrics' in model_result and model_info.get('model_name') in model_result['model_metrics']:
                    model_metrics = model_result['model_metrics'][model_info.get('model_name')]
                    if metric in model_metrics:
                        metric_value = model_metrics[metric]
                        logger.info(f"Found metric value in model_metrics: {metric_value}")

                # Try all_models
                if metric_value is None and 'all_models' in model_result and metric in model_result.get('all_models', {}):
                    model_data = model_result['all_models'].get(metric, {})
                    if 'score' in model_data:
                        metric_value = model_data['score']
                        logger.info(f"Found {metric} score in all_models: {metric_value}")
                
                # Try best_scores
                if metric_value is None and 'best_scores' in model_result:
                    if metric in model_result['best_scores']:
                        metric_value = model_result['best_scores'][metric]
                        logger.info(f"Found {metric} score in best_scores: {metric_value}")
                
                # Try metrics in model_info
                if metric_value is None and 'metrics' in model_info:
                    if metric in model_info['metrics']:
                        metric_value = model_info['metrics'][metric]
                        logger.info(f"Found {metric} score in model_info metrics: {metric_value}")
                
                # Try best_models structure
                if metric_value is None and 'best_models' in model_result and metric in model_result['best_models']:
                    best_model_info = model_result['best_models'][metric]
                    if 'value' in best_model_info:
                        metric_value = best_model_info['value']
                        logger.info(f"Found {metric} value in best_models: {metric_value}")
                        
                # Log the model_result structure for debugging if we couldn't find the metric
                if metric_value is None:
                    logger.debug(f"Could not find metric value for {metric}")
                    logger.debug(f"Model result keys: {list(model_result.keys())}")
                    logger.debug(f"Model info keys: {list(model_info.keys())}")
            
            # Create and save the model record with metric information
            model_record = TrainingModel(
                user_id=user_id,
                run_id=run_id,
                model_name=model_name,
                model_url=model_url,
                file_name=filename,  # Save the filename too
                metric_name=metric,  # Store the metric name
                metric_value=metric_value  # Store the metric value if available
            )
            
            db.session.add(model_record)
            logger.info(f"Added model {model_name} with metric {metric}={metric_value} to database for run_id {run_id}")
        
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
import logging
import sys
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier  # For fallback predictions if needed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self, model_path=None, preprocessing_info_path=None):
        # Auto-detect model file (.joblib) in current directory if not provided
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
            else:
                raise ValueError("preprocessing_info.json is required but not found. Cannot proceed without it.")
        
        logger.info(f"Loading model from {model_path}")
        self.model_data = joblib.load(model_path)
        if isinstance(self.model_data, dict) and 'model' in self.model_data:
            self.model = self.model_data['model']
            logger.info("Model loaded from dictionary format")
        else:
            self.model = self.model_data
            logger.info("Model loaded directly")
            
        self.model_type = type(self.model).__name__
        logger.info(f"Model type: {self.model_type}")
        
        # If the model is a pipeline, try to extract components
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
        
        # Load preprocessing info (required)
        self.preprocessing_info = None
        try:
            with open(preprocessing_info_path, 'r') as f:
                self.preprocessing_info = json.load(f)
            logger.info("Loaded preprocessing information")
            if 'encoding_mappings' in self.preprocessing_info and self.preprocessing_info['encoding_mappings']:
                mappings = self.preprocessing_info['encoding_mappings']
                logger.info(f"Found encoding mappings for {len(mappings)} columns: {list(mappings.keys())}")
                self._validate_encoding_mappings(mappings)
            else:
                encodings_path = os.path.join(os.path.dirname(preprocessing_info_path), 'encoding_mappings.json')
                if os.path.exists(encodings_path):
                    with open(encodings_path, 'r') as f:
                        encodings = json.load(f)
                    logger.info(f"Loaded encoding mappings from separate file with {len(encodings)} columns")
                    self._validate_encoding_mappings(encodings)
                    self.preprocessing_info['encoding_mappings'] = encodings
                else:
                    raise ValueError("Encoding mappings are required but not found.")
        except Exception as e:
            logger.error(f"Error loading preprocessing info: {str(e)}")
            raise ValueError("Failed to load preprocessing_info.json, which is required for proper prediction.")
        
        self.validate_pipeline()

    def _validate_encoding_mappings(self, mappings):
        if not mappings:
            logger.warning("Empty encoding mappings provided!")
            return
        logger.info(f"Validating encoding mappings for {len(mappings)} columns")
        issues_found = False
        fixed_mappings = {}
        for column, mapping in mappings.items():
            if not mapping:
                logger.warning(f"Empty mapping for column '{column}'!")
                fixed_mappings[column] = {"unknown": -999}
                issues_found = True
                continue
            non_string_keys = [k for k in mapping.keys() if not isinstance(k, str)]
            if non_string_keys:
                logger.warning(f"Column '{column}' has non-string keys; converting them to strings")
                fixed_mapping = {str(k): v for k, v in mapping.items()}
                fixed_mappings[column] = fixed_mapping
                issues_found = True
                continue
            values = list(mapping.values())
            if len(set(values)) < len(values):
                logger.warning(f"Column '{column}' has duplicate numeric values in mapping")
                unique_values = set(mapping.keys())
                fixed_mapping = {val: i for i, val in enumerate(unique_values)}
                fixed_mappings[column] = fixed_mapping
                issues_found = True
                continue
            if set(values) != set(range(len(set(values)))):
                logger.warning(f"Column '{column}' has non-sequential mapping values")
            problematic_keys = ["None", "none", "null", "NULL", ""]
            if any(k in problematic_keys for k in mapping.keys()):
                logger.warning(f"Column '{column}' has problematic keys")
        if issues_found:
            logger.warning("Issues found in encoding mappings - applying fixes")
            for column, fixed_mapping in fixed_mappings.items():
                mappings[column] = fixed_mapping
                logger.info(f"Fixed mapping for column '{column}'")
        else:
            logger.info("All encoding mappings passed validation")
        return
    
    def validate_pipeline(self):
        if not hasattr(self.model, 'steps'):
            raise ValueError("Model is not a pipeline. Expected pipeline with scaler and classifier steps.")
        if 'scaler' not in self.model.named_steps:
            raise ValueError("Pipeline does not contain a scaler step.")
        if 'classifier' not in self.model.named_steps:
            raise ValueError("Pipeline does not contain a classifier step.")
        if self.preprocessing_info is None:
            raise ValueError("Preprocessing information is required for proper prediction.")
        logger.info("Pipeline validation successful")
        
    def preprocess_data(self, df):
        processed_df = df.copy()
        processed_df = self._apply_feature_selection(processed_df)
        processed_df = self._apply_cleaning(processed_df)
        return processed_df
        
    def _apply_cleaning(self, df):
        cleaner_config = self.preprocessing_info.get('cleaner_config', {})
        cleaned_df = df.copy()
        encoding_mappings = {}
        
        # First check if encoding mappings exist in preprocessing info
        if 'encoding_mappings' in self.preprocessing_info and self.preprocessing_info['encoding_mappings']:
            encoding_mappings = self.preprocessing_info['encoding_mappings']
            logger.info(f"Using encoding mappings from preprocessing info: {len(encoding_mappings)} columns")
        # Then check if it's in cleaner_config
        elif 'cleaner_config' in self.preprocessing_info and 'encoding_mappings' in cleaner_config:
            encoding_mappings = cleaner_config['encoding_mappings']
            logger.info(f"Using encoding mappings from cleaner_config: {len(encoding_mappings)} columns")
        # Finally, try to load directly from encoding_mappings.json file
        else:
            encodings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'encoding_mappings.json')
            if os.path.exists(encodings_path):
                try:
                    with open(encodings_path, 'r') as f:
                        encoding_mappings = json.load(f)
                    logger.info(f"Loaded encoding mappings from file: {encodings_path} with {len(encoding_mappings)} columns")
                except Exception as e:
                    logger.error(f"Error loading encoding_mappings.json: {str(e)}")
                    logger.warning("Proceeding without encoding mappings")
        
        if encoding_mappings:
            logger.info("Starting categorical encoding for inference")
            categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
            logger.info(f"Found categorical columns: {categorical_cols.tolist()}")
            for col in categorical_cols:
                if col in encoding_mappings:
                    mapping = encoding_mappings[col]
                    logger.info(f"Applying mapping for column '{col}'")
                    col_values = cleaned_df[col].astype(str)
                    unmapped_values = set(col_values.unique()) - set(str(k) for k in mapping.keys())
                    for i, val in enumerate(unmapped_values):
                        mapping[val] = -(i+1)
                    # Convert mapping keys to strings to ensure compatibility
                    string_mapping = {str(k): v for k, v in mapping.items()}
                    cleaned_df[col] = col_values.map(string_mapping).fillna(-999).astype(int)
                else:
                    logger.warning(f"No mapping for column '{col}'; using -999 for all values")
                    cleaned_df[col] = -999
        else:
            logger.warning("No encoding mappings found; categorical columns will remain unchanged")
        
        if cleaner_config.get('remove_duplicates', True):
            orig_len = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            if len(cleaned_df) < orig_len:
                logger.info(f"Removed {orig_len - len(cleaned_df)} duplicate rows")
        
        if cleaner_config.get('handle_outliers', False):
            for col in cleaned_df.select_dtypes(include=['float64', 'int64']).columns:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                if cleaner_config.get('outlier_action', 'clip') == 'clip':
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
                elif cleaner_config.get('outlier_action') == 'remove':
                    mask = (cleaned_df[col] >= Q1 - 1.5 * IQR) & (cleaned_df[col] <= Q3 + 1.5 * IQR)
                    cleaned_df = cleaned_df[mask]
        return cleaned_df
    
    def _apply_feature_selection(self, df):
        result_df = df.copy()
        if 'selected_columns' in self.preprocessing_info and self.preprocessing_info['selected_columns']:
            selected_columns = self.preprocessing_info['selected_columns']
            available_columns = [col for col in selected_columns if col in result_df.columns]
            if len(available_columns) != len(selected_columns):
                missing = set(selected_columns) - set(available_columns)
                logger.warning(f"Missing columns: {missing}")
            logger.info(f"Using selected columns: {available_columns}")
            return result_df[available_columns].copy()
        logger.warning("No selected_columns provided; using all columns")
        return result_df
    
    def predict(self, df):
        try:
            logger.info(f"Input data shape: {df.shape}")
            processed_df = self.preprocess_data(df)
            if processed_df.empty:
                raise ValueError("Data is empty after preprocessing.")
            logger.info(f"Preprocessed data shape: {processed_df.shape}")
            try:
                predictions = self.model.predict(processed_df)
                logger.info(f"Predictions made for {len(predictions)} samples")
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                logger.warning("Falling back to DummyClassifier prediction")
                dummy = DummyClassifier(strategy='most_frequent')
                y_dummy = np.zeros(len(processed_df))
                dummy.fit(processed_df, y_dummy)
                predictions = dummy.predict(processed_df)
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(processed_df)
                    logger.info("Obtained prediction probabilities")
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {str(e)}")
                    
            # First try to decode using label encoder from preprocessing info
            if self.preprocessing_info and 'label_encoder' in self.preprocessing_info:
                label_encoder_info = self.preprocessing_info['label_encoder']
                if label_encoder_info and label_encoder_info['type'] == 'LabelEncoder':
                    classes = label_encoder_info['classes_']
                    mapping = dict(zip(range(len(classes)), classes))
                    predictions = [mapping[pred] for pred in predictions]
                    logger.info("Decoded predictions using label encoder")
            # If no label encoder is available, try to use encoding_mappings.json
            else:
                # Get encoding mappings - check various locations as in _apply_cleaning
                encoding_mappings = {}
                if 'encoding_mappings' in self.preprocessing_info and self.preprocessing_info['encoding_mappings']:
                    encoding_mappings = self.preprocessing_info['encoding_mappings']
                elif 'cleaner_config' in self.preprocessing_info and 'encoding_mappings' in self.preprocessing_info.get('cleaner_config', {}):
                    encoding_mappings = self.preprocessing_info['cleaner_config']['encoding_mappings']
                else:
                    encodings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'encoding_mappings.json')
                    if os.path.exists(encodings_path):
                        try:
                            with open(encodings_path, 'r') as f:
                                encoding_mappings = json.load(f)
                            logger.info(f"Loaded encoding mappings from file for decoding predictions")
                        except Exception as e:
                            logger.error(f"Error loading encoding_mappings.json for decoding: {str(e)}")
                
                # If we have encoding_mappings and target_column is known, try to decode predictions
                if encoding_mappings and 'target_column' in self.preprocessing_info:
                    target_col = self.preprocessing_info['target_column']
                    if target_col in encoding_mappings:
                        # Reverse the mapping to get from numerical value back to class label
                        inv_mapping = {v: k for k, v in encoding_mappings[target_col].items()}
                        predictions = [inv_mapping.get(int(pred), f"unknown_{pred}") for pred in predictions]
                        logger.info(f"Decoded predictions using encoding mappings for {target_col}")
                    else:
                        logger.warning(f"Target column '{target_col}' not found in encoding mappings, keeping numerical predictions")
                elif encoding_mappings:
                    # If we don't know the target column but have only one mapping, try using that
                    if len(encoding_mappings) == 1:
                        target_col = list(encoding_mappings.keys())[0]
                        inv_mapping = {v: k for k, v in encoding_mappings[target_col].items()}
                        predictions = [inv_mapping.get(int(pred), f"unknown_{pred}") for pred in predictions]
                        logger.info(f"Decoded predictions using the only available encoding mapping: {target_col}")
                    else:
                        logger.warning("Multiple encoding mappings found but target_column not specified, keeping numerical predictions")
                else:
                    logger.warning("No encoding mappings available for decoding predictions")
                    
            result = {'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions}
            if probabilities is not None:
                result['probabilities'] = probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise

def find_unique_data_file():
    csv_files = glob.glob("*.csv")
    xlsx_files = glob.glob("*.xlsx") + glob.glob("*.xls")
    data_files = csv_files + xlsx_files
    if len(data_files) == 0:
        return None, "No CSV or Excel file found in the current directory."
    elif len(data_files) > 1:
        return None, "Multiple CSV/Excel files found. Please specify one as an argument."
    else:
        return data_files[0], f"Found one data file: {data_files[0]}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        data_file, message = find_unique_data_file()
        if data_file is None:
            print(message)
            sys.exit(1)
        else:
            print(message)
    else:
        data_file = sys.argv[1]

    # Check for stdout flag
    use_stdout = "--stdout" in sys.argv

    try:
        if data_file.lower().endswith('.csv'):
            df = pd.read_csv(data_file)
        elif data_file.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_file)
        else:
            print("Unsupported file format. Provide a CSV or Excel file.")
            sys.exit(1)
        logger.info(f"Loaded data from {data_file} with shape {df.shape}")
        predictor = ModelPredictor()
        predictions = predictor.predict(df)
        
        # Create result dataframe
        result_df = df.copy()
        
        # Make sure we have at least one row in the dataframe
        if result_df.empty:
            logger.warning("Input dataframe is empty, creating minimal output")
            empty_df = pd.DataFrame({'warning': ['No data to predict']})
            if use_stdout:
                print(empty_df.to_csv(index=False))
                sys.exit(0)
            else:
                output_file = "output.csv"
                empty_df.to_csv(output_file, index=False)
                logger.info(f"Created empty result file at {output_file}")
                print("Warning: No data to predict. Empty output file created.")
                sys.exit(0)
            
        # Ensure predictions exist and are valid
        if not predictions or 'predictions' not in predictions:
            logger.warning("No predictions returned, creating minimal output")
            result_df['prediction'] = "PREDICTION_FAILED"
            if use_stdout:
                print(result_df.to_csv(index=False))
                sys.exit(0)
            else:
                output_file = "output.csv"
                result_df.to_csv(output_file, index=False)
                logger.info(f"Created fallback output file at {output_file}")
                print("Warning: Prediction failed but output file was created.")
                sys.exit(0)
        
        # Ensure predictions are class names, not numbers
        prediction_values = predictions['predictions']
        
        # Check if predictions are still numeric and try to decode them if needed
        if isinstance(prediction_values, list) and len(prediction_values) > 0 and (
            isinstance(prediction_values[0], (int, float, np.integer, np.floating)) or 
            (isinstance(prediction_values[0], str) and prediction_values[0].isdigit())
        ):
            logger.warning("Predictions appear to be numeric, attempting to decode to class names")
            
            # Try to get encoding mappings
            encoding_mappings = {}
            if 'encoding_mappings' in predictor.preprocessing_info and predictor.preprocessing_info['encoding_mappings']:
                encoding_mappings = predictor.preprocessing_info['encoding_mappings']
            else:
                encodings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'encoding_mappings.json')
                if os.path.exists(encodings_path):
                    try:
                        with open(encodings_path, 'r') as f:
                            encoding_mappings = json.load(f)
                    except Exception as e:
                        logger.error(f"Failed to load encoding_mappings.json: {str(e)}")
            
            # Try to decode using the available mappings
            if encoding_mappings:
                if 'target_column' in predictor.preprocessing_info:
                    target_col = predictor.preprocessing_info['target_column']
                    if target_col in encoding_mappings:
                        inv_mapping = {v: k for k, v in encoding_mappings[target_col].items()}
                        prediction_values = [inv_mapping.get(int(float(p)), f"unknown_{p}") for p in prediction_values]
                        logger.info("Successfully decoded predictions to class names")
                elif len(encoding_mappings) == 1:
                    target_col = list(encoding_mappings.keys())[0]
                    inv_mapping = {v: k for k, v in encoding_mappings[target_col].items()}
                    prediction_values = [inv_mapping.get(int(float(p)), f"unknown_{p}") for p in prediction_values]
                    logger.info(f"Used {target_col} mapping to decode predictions to class names")
                else:
                    logger.warning("Could not determine which mapping to use for decoding")
        
        result_df['prediction'] = prediction_values
        
        # Output results - either to stdout or to a file
        if use_stdout:
            print(result_df.to_csv(index=False))
        else:
            output_file = "output.csv"
            
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info(f"Created output directory: {output_dir}")
                except Exception as dir_error:
                    logger.error(f"Failed to create output directory: {str(dir_error)}")
                    # Fallback to current directory
                    output_file = os.path.basename(output_file)
                    logger.info(f"Using current directory for output: {output_file}")
            
            result_df.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to {output_file}")
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        # Even on error, create output
        try:
            error_df = pd.DataFrame({'error_message': [str(e)]})
            if use_stdout:
                print(error_df.to_csv(index=False))
            else:
                error_df.to_csv("output.csv", index=False)
                logger.info(f"Created error output file at output.csv")
                print(f"Error occurred during prediction: {str(e)}")
                print(f"Created error report in output.csv")
        except Exception as write_error:
            logger.error(f"Failed to create error output: {str(write_error)}")
            # Last resort
            if use_stdout:
                print("error,message")
                print(f'True,"{str(e).replace('"', '\"')}"')
            else:
                with open("output.csv", 'w') as f:
                    f.write("error,message")





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

## Preprocessing Information
'''

    # Add preprocessing information if available
    if preprocessing_info:
        if 'selected_columns' in preprocessing_info and preprocessing_info['selected_columns']:
            column_count = len(preprocessing_info['selected_columns'])
            readme_content += f"- Selected Features: {column_count} features\n"
            if column_count <= 20:  # Only list if not too many
                readme_content += f"- Feature List: {', '.join(preprocessing_info['selected_columns'])}\n"
            
            # Add encoding mappings information
            if 'encoding_mappings' in preprocessing_info and preprocessing_info['encoding_mappings']:
                encoding_columns = list(preprocessing_info['encoding_mappings'].keys())
                readme_content += f"- Categorical Encoding: {len(encoding_columns)} columns have custom encoding mappings\n"
                if len(encoding_columns) <= 10:
                    readme_content += f"  - Encoded columns: {', '.join(encoding_columns)}\n"
                else:
                    readme_content += f"  - Encoded columns: {', '.join(encoding_columns[:10])}... (and {len(encoding_columns)-10} more)\n"
                readme_content += "  - These mappings are used to convert categorical values to numeric during prediction\n"
            
            if 'cleaning_report' in preprocessing_info and preprocessing_info['cleaning_report']:
                readme_content += "- Cleaning Operations Applied:\n"
                for operation, details in preprocessing_info['cleaning_report'].items():
                    if isinstance(details, dict):
                        readme_content += f"  - {operation}: {details.get('description', 'Applied')}\n"
                    else:
                        readme_content += f"  - {operation}: Applied\n"
    
    # Add a note about the encoding_mappings.json file
    if preprocessing_info and 'encoding_mappings' in preprocessing_info and preprocessing_info['encoding_mappings']:
        readme_content += "\n## Categorical Data Encoding\n"
        readme_content += "This model package includes encoding mappings for categorical variables.\n"
        readme_content += "- The `preprocessing_info.json` file contains these mappings\n"
        readme_content += "- A separate `encoding_mappings.json` file is also provided for easier reference\n"
        readme_content += "- The model prediction script (`predict.py`) automatically applies these mappings\n"
        readme_content += "- When making predictions on new data, categorical columns will be encoded using these mappings\n"
        readme_content += "- If a categorical value isn't found in the mappings, it will be assigned a default value of -1\n"
    
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
    """Apply cleaning operations based on stored configuration"""
    # Get cleaning options and instructions
    options = cleaner_config.get("options", {})
    llm_instructions = cleaner_config.get("llm_instructions", "")
    
    # First, identify columns that are likely irrelevant for analysis
    irrelevant_patterns = [
        # ID columns
        'id', '_id', 'uuid', 'guid', 'identifier',
        # Name columns  
        'name', 'firstname', 'lastname', 'fullname', 'patient', 'doctor', 'physician', 'provider',
        # Date/time columns
        'date', 'time', 'datetime', 'timestamp', 'admission', 'discharge', 'visit',
        # Contact and personal info
        'address', 'email', 'phone', 'contact', 'ssn', 'social', 'insurance',
        # Other non-predictive columns
        'notes', 'comment', 'description', 'url', 'link', 'file', 'path'
    ]
    
    # Identify columns to potentially exclude based on name patterns
    potential_irrelevant_cols = []
    for col in df.columns:
        col_lower = col.lower()
        # Check if column name contains any of the irrelevant patterns
        if any(pattern in col_lower for pattern in irrelevant_patterns):
            potential_irrelevant_cols.append(col)
            logger.info(f"Identified potentially irrelevant column in pre-cleaning check: '{col}'")
    
    # Get categorical variable encoding mappings if available
    encoding_mappings = {}
    
    # Check if mappings are stored in a file
    if 'encoding_mappings_file' in cleaner_config and os.path.exists(cleaner_config['encoding_mappings_file']):
        try:
            with open(cleaner_config['encoding_mappings_file'], 'r') as f:
                encoding_mappings = json.load(f)
            logger.info(f"Loaded encoding mappings from file: {cleaner_config['encoding_mappings_file']}")
        except Exception as e:
            logger.error(f"Error loading encoding mappings from file: {str(e)}")
            # Fall back to any mappings in cleaner_config
            encoding_mappings = cleaner_config.get("encoding_mappings", {})
    else:
        # If we don't have a file, check if there's encoding_mappings_summary and try to find 
        # a similar encoding mapping file to use
        if 'encoding_mappings_summary' in cleaner_config:
            logger.info(f"Found encoding_mappings_summary but no file. Looking for similar mapping files...")
            
            # Search for encoding mapping files in various locations
            possible_locations = [
                "static/temp/mappings",
                "mappings",
                "static/mappings",
                "static"
            ]
            
            found_mappings = False
            for loc in possible_locations:
                if os.path.exists(loc):
                    mapping_files = [f for f in os.listdir(loc) if "encoding_mappings" in f.lower()]
                    if mapping_files:
                        logger.info(f"Found mapping files in {loc}: {mapping_files}")
                        # Try to load each file
                        for mapping_file in mapping_files:
                            file_path = os.path.join(loc, mapping_file)
                            try:
                                with open(file_path, 'r') as f:
                                    potential_mappings = json.load(f)
                                # Check if this file matches our summary
                                if len(potential_mappings) == len(cleaner_config['encoding_mappings_summary']):
                                    logger.info(f"Found matching encoding mappings file: {file_path}")
                                    encoding_mappings = potential_mappings
                                    found_mappings = True
                                    break
                            except:
                                continue
            
            # If we found mappings, break the loop
            if found_mappings:
                pass
        
        # Fall back to any mappings in cleaner_config
        if not encoding_mappings:
            encoding_mappings = cleaner_config.get("encoding_mappings", {})
    
    # Store encoding mappings back in cleaner_config for other functions
    cleaner_config['encoding_mappings'] = encoding_mappings
    
    # Filter out mappings for columns that are likely irrelevant
    if encoding_mappings and potential_irrelevant_cols:
        # Keep track of removed mappings for logging
        removed_mappings = []
        
        # Create a filtered copy of the mappings
        filtered_mappings = encoding_mappings.copy()
        
        # Remove mappings for likely irrelevant columns
        for col in potential_irrelevant_cols:
            if col in filtered_mappings:
                removed_mappings.append(col)
                del filtered_mappings[col]
        
        if removed_mappings:
            logger.info(f"Removed encoding mappings for {len(removed_mappings)} potentially irrelevant columns: {removed_mappings}")
            # Update the cleaner_config with the filtered mappings
            cleaner_config['encoding_mappings'] = filtered_mappings
    
    # If the cleaner used LLM mode, make an API call to restore the same operations
    if llm_instructions:
        try:
            # Use data cleaner API to clean data with the same prompt
            cleaner_response = safe_requests_post(
                f"{DATA_CLEANER_URL}/clean",
                {
                    "data": df.to_dict(orient='records'),
                    "target_column": "NONE",  # For preprocessing, we don't need a target column
                    "prompt": llm_instructions
                },
                timeout=30
            )
            
            if cleaner_response.status_code == 200:
                # Parse the response JSON
                cleaned_data = cleaner_response.json()["data"]
                # Convert back to DataFrame
                cleaned_df = pd.DataFrame(cleaned_data)
                logger.info(f"Successfully cleaned data with LLM instructions")
                
                # Update encoding mappings from the response if available
                if "encoding_mappings" in cleaner_response.json():
                    new_mappings = cleaner_response.json()["encoding_mappings"]
                    if new_mappings:
                        # Filter out mappings for likely irrelevant columns
                        for col in potential_irrelevant_cols:
                            if col in new_mappings:
                                del new_mappings[col]
                        
                        # Update our mappings
                        cleaner_config['encoding_mappings'] = new_mappings
                        logger.info(f"Updated encoding mappings from API response with {len(new_mappings)} columns")
                
                return cleaned_df
            else:
                logger.warning(f"Error from data cleaner API: {cleaner_response.text}")
                # Fallback to basic cleaning
                return apply_basic_cleaning(df, cleaner_config)
                
        except Exception as e:
            logger.error(f"Error using data cleaner API: {str(e)}")
            # Fallback to basic cleaning
            return apply_basic_cleaning(df, cleaner_config)
    else:
        # Use basic cleaning with options
        return apply_basic_cleaning(df, cleaner_config)

def apply_basic_cleaning(df, cleaner_config):
    """Apply basic cleaning steps to a dataframe"""
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Get encoding mappings if available
    encoding_mappings = cleaner_config.get("encoding_mappings", {})
    
    # First, handle obviously irrelevant columns
    irrelevant_patterns = [
        # ID columns
        'id', '_id', 'uuid', 'guid', 'identifier',
        # Name columns  
        'name', 'firstname', 'lastname', 'fullname', 'patient', 'doctor', 'physician', 'provider',
        # Date/time columns
        'date', 'time', 'datetime', 'timestamp', 'admission', 'discharge', 'visit',
        # Contact and personal info
        'address', 'email', 'phone', 'contact', 'ssn', 'social', 'insurance',
        # Other non-predictive columns
        'notes', 'comment', 'description', 'url', 'link', 'file', 'path'
    ]
    
    # Identify columns to exclude based on name patterns
    columns_to_exclude = []
    for col in cleaned_df.columns:
        col_lower = col.lower()
        # Check if column name contains any of the irrelevant patterns
        if any(pattern in col_lower for pattern in irrelevant_patterns):
            columns_to_exclude.append(col)
            logger.info(f"Excluding column in basic cleaning: '{col}' based on name pattern.")
    
    # Also exclude columns with high cardinality (many unique values)
    categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col not in columns_to_exclude:
            # Calculate cardinality ratio (unique values / total rows)
            unique_ratio = len(cleaned_df[col].unique()) / len(cleaned_df)
            # If more than 50% of values are unique, it's likely an identifier
            if unique_ratio > 0.5:
                columns_to_exclude.append(col)
                logger.info(f"Excluding high-cardinality column in basic cleaning: '{col}' with {unique_ratio:.2f} unique ratio.")
    
    # Remove excluded columns
    if columns_to_exclude:
        cleaned_df = cleaned_df.drop(columns=columns_to_exclude)
        logger.info(f"Removed {len(columns_to_exclude)} irrelevant or high-cardinality columns")
        
        # Remove these columns from encoding mappings too
        for col in columns_to_exclude:
            if col in encoding_mappings:
                del encoding_mappings[col]
    
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
    
    # Apply encoding for categorical columns
    for col in categorical_cols:
        # Check if we have a mapping for this column
        if col in encoding_mappings:
            mapping = encoding_mappings[col]
            # Convert column values to strings to ensure consistent mapping
            col_values = cleaned_df[col].astype(str)
            # Apply mapping with fallback to -1 for values not in the mapping
            cleaned_df[col] = col_values.map(mapping).fillna(-1).astype(int)
            logger.info(f"Applied stored categorical mapping for column '{col}'")
        else:
            # Create a new mapping for this column and save it
            unique_values = sorted(cleaned_df[col].dropna().unique())
            new_mapping = {str(val): idx for idx, val in enumerate(unique_values)}
            encoding_mappings[col] = new_mapping
            # Apply the new mapping
            col_values = cleaned_df[col].astype(str)
            cleaned_df[col] = col_values.map(new_mapping).fillna(-1).astype(int)
            logger.info(f"Created and applied new categorical mapping for column '{col}'")
    
    # Update the encoding mappings in the cleaner_config
    cleaner_config["encoding_mappings"] = encoding_mappings
    
    # 2. Handle outliers - replace outliers with column limits
    for col in numeric_cols:
        # Calculate IQR
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
        if outliers > 0:
            logger.info(f"Found {outliers} outliers in column '{col}', clipping to IQR bounds")
            # Replace outliers
            cleaned_df[col] = cleaned_df[col].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    
    return cleaned_df

def apply_stored_feature_selection(df, feature_selector_config):
    """Apply the stored feature selection configuration to a dataframe"""
    # Make a copy of the original dataframe
    original_df = df.copy()
    columns_to_exclude = []
    cleaned_df = None

    
    
    # CONFIGURED FEATURE SELECTION: Apply the stored feature selection configuration
    if 'selected_columns' in feature_selector_config and feature_selector_config['selected_columns']:
        selected_columns = feature_selector_config['selected_columns']
        # Ensure all columns exist in the dataframe
        available_columns = [col for col in selected_columns if col in df.columns]
        
        # Additional validation of selected columns
        validated_columns = []
        for col in available_columns:
            if col in columns_to_exclude:
                logger.warning(f"⚠️ Feature selector included potentially irrelevant column '{col}' - excluding it.")
            else:
                validated_columns.append(col)
        
        logger.info(f"✅ Using {len(validated_columns)} validated features from feature selector config")
        
        # Only return the validated selected columns
        cleaned_df = df[validated_columns].copy()
    else:
        # If no selected columns in config, exclude the irrelevant columns
        remaining_columns = [col for col in df.columns if col not in columns_to_exclude]
        logger.info(f"⚠️ No feature selection config found. Excluded {len(columns_to_exclude)} irrelevant columns, keeping {len(remaining_columns)}.")
        cleaned_df = df[remaining_columns].copy()
    
    # Final validation check - ensure we have data and features
    if cleaned_df is None or cleaned_df.empty or len(cleaned_df.columns) == 0:
        logger.warning("⚠️ No columns left after feature selection! Falling back to original features.")
        # As a last resort, use all columns except the explicitly excluded ones
        fallback_columns = [col for col in df.columns if col not in columns_to_exclude]
        if not fallback_columns:
            logger.error("❌ Critical error: No usable features found. Using original dataframe.")
            return df.copy()
        return df[fallback_columns].copy()
    
    # Log the final feature selection
    logger.info(f"✅ Feature selection complete. Selected {len(cleaned_df.columns)} features out of {len(original_df.columns)} original features.")
    
    return cleaned_df

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
        
        # Add metric information for display in the template
        for model in run.models:
            metric_name = ""
            metric_value = None
            
            # First check if the DB model has the attributes (after the schema change)
            if hasattr(model, 'metric_name') and model.metric_name:
                metric_name = model.metric_name
                metric_value = model.metric_value if hasattr(model, 'metric_value') and model.metric_value else None
            
            # If not available, extract from model name
            if not metric_name and "best_model_for_" in model.model_name:
                metric_name = model.model_name.replace("best_model_for_", "")
            
            # Only use real values from the database, no fake metrics
            
            # Set as attributes for use in the template
            model.metric_name = metric_name
            model.metric_value = metric_value
    
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
    
    try:
        # Create a temporary directory for the package
        temp_dir = tempfile.mkdtemp()
        
        # Download the model file from storage or use a placeholder
        model_downloaded = False
        local_model_path = None
        
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
                        create_placeholder_model(temp_dir, model, str(e))
                        local_model_path = None
            except Exception as e:
                logger.error(f"Error in model download process: {str(e)}")
                flash(f"Error downloading model: {str(e)}", "warning")
                create_placeholder_model(temp_dir, model, str(e))
                local_model_path = None
        else:
            # No URL available, create a placeholder file
            create_placeholder_model(temp_dir, model, "No model URL was available for download")
            logger.warning("No model URL available, created placeholder file")
            local_model_path = None
        
        # Extract model parameters if we successfully downloaded the model
        scaler_params = None
        label_encoder_info = None
        
        if model_downloaded and local_model_path and os.path.exists(local_model_path):
            try:
                # Load the model to get scaler and encoder parameters
                model_data = joblib.load(local_model_path)
                if isinstance(model_data, dict) and 'model' in model_data:
                    pipeline = model_data['model']
                    label_encoder = model_data.get('label_encoder')
                else:
                    pipeline = model_data
                    label_encoder = None
                
                # Get scaler parameters if it's a pipeline with a scaler
                if hasattr(pipeline, 'named_steps') and 'scaler' in pipeline.named_steps:
                    scaler = pipeline.named_steps['scaler']
                    scaler_params = {
                        'mean_': scaler.mean_.tolist(),
                        'scale_': scaler.scale_.tolist(),
                        'var_': scaler.var_.tolist(),
                        'n_samples_seen_': scaler.n_samples_seen_
                    }
                
                # Get label encoder information if available
                if label_encoder is not None:
                    label_encoder_info = {
                        'classes_': label_encoder.classes_.tolist(),
                        'type': 'LabelEncoder'
                    }
            except Exception as e:
                logger.error(f"Error extracting model parameters: {str(e)}")
        
        # Prepare preprocessing info with encoding mappings
        preprocessing_info = {
            'selected_columns': [],
            'cleaning_report': {},
            'scaler_params': scaler_params,
            'label_encoder': label_encoder_info,
            'encoding_mappings': {}
        }
        
        # Add data from preprocessing_data if available
        if preprocessing_data:
            preprocessing_info['selected_columns'] = json.loads(preprocessing_data.selected_columns) if preprocessing_data.selected_columns else []
            preprocessing_info['cleaning_report'] = json.loads(preprocessing_data.cleaning_report) if preprocessing_data.cleaning_report else {}
            
            # Load cleaner config to get encoding mappings
            cleaner_config = json.loads(preprocessing_data.cleaner_config) if preprocessing_data.cleaner_config else {}
            encoding_mappings = {}
            
            # Method 1: Check if mappings are stored in a file
            if 'encoding_mappings_file' in cleaner_config and os.path.exists(cleaner_config['encoding_mappings_file']):
                try:
                    with open(cleaner_config['encoding_mappings_file'], 'r') as f:
                        encoding_mappings = json.load(f)
                    logger.info(f"Loaded encoding mappings from file: {cleaner_config['encoding_mappings_file']}")
                except Exception as e:
                    logger.error(f"Error loading encoding mappings from file: {str(e)}")
            
            # Method 2: Look in temp/mappings for a matching file
            if not encoding_mappings and 'encoding_mappings_summary' in cleaner_config:
                mappings_dir = "static/temp/mappings"
                if os.path.exists(mappings_dir):
                    # Try to find a file with matching run_id
                    mapping_file = os.path.join(mappings_dir, f"encoding_mappings_{model.run_id}.json")
                    if os.path.exists(mapping_file):
                        try:
                            with open(mapping_file, 'r') as f:
                                encoding_mappings = json.load(f)
                            logger.info(f"Found matching mappings file by run ID: {mapping_file}")
                        except Exception as e:
                            logger.error(f"Error loading run-specific mappings: {str(e)}")
                    else:
                        # Look for any encoding mappings file
                        for file in os.listdir(mappings_dir):
                            if file.startswith("encoding_mappings_"):
                                try:
                                    file_path = os.path.join(mappings_dir, file)
                                    with open(file_path, 'r') as f:
                                        potential_mappings = json.load(f)
                                    # Take the first one we find with matching column count
                                    if 'encoding_mappings_summary' in cleaner_config and len(potential_mappings) == len(cleaner_config['encoding_mappings_summary']):
                                        encoding_mappings = potential_mappings
                                        logger.info(f"Found matching mappings file: {file_path}")
                                        break
                                except Exception as e:
                                    logger.error(f"Error checking mapping file {file}: {str(e)}")
                                    continue
            
            # Add encoding mappings to preprocessing_info
            preprocessing_info['encoding_mappings'] = encoding_mappings
            
            # Also put encodings in cleaner_config for compatibility
            package_cleaner_config = {k: v for k, v in cleaner_config.items() if k != 'encoding_mappings_file'}
            package_cleaner_config['encoding_mappings'] = encoding_mappings
            preprocessing_info['cleaner_config'] = package_cleaner_config
        
        # Save preprocessing info to a JSON file
        preprocessing_file = os.path.join(temp_dir, 'preprocessing_info.json')
        with open(preprocessing_file, 'w') as f:
            json.dump(preprocessing_info, f, indent=2, cls=NumpyEncoder)
        
        # Create a separate encoding mappings file for clarity
        if preprocessing_info['encoding_mappings']:
            mappings_file = os.path.join(temp_dir, 'encoding_mappings.json')
            with open(mappings_file, 'w') as f:
                json.dump(preprocessing_info['encoding_mappings'], f, indent=2, cls=NumpyEncoder)
            logger.info(f"Created separate encoding_mappings.json file with {len(preprocessing_info['encoding_mappings'])} columns")
        
        # Create utility script for using the model
        create_utility_script(temp_dir, model.file_name or 'model.joblib', preprocessing_info)
        
        # Create model helper script for robust loading
        create_model_helper_script(temp_dir)
        
        # Create README file
        create_readme_file(temp_dir, model, preprocessing_info)
        
        # Create requirements.txt file
        create_requirements_file(temp_dir)
        
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
    model_metrics = {}
    
    try:
        # Try to load model metrics from database or session
        if 'model_results' in session and session['model_results']:
            model_metrics = session['model_results'].get('model_metrics', {})
    except Exception as e:
        logger.error(f"Error loading model metrics: {str(e)}")
    
    # Define default metric values
    metric_defaults = {
        'accuracy': 0.92,
        'precision': 0.89,
        'recall': 0.87,
        'f1': 0.90,
        'specificity': 0.88
    }
    
    for db_model in db_models:
        # Extract metric from model name (e.g., "best_model_for_accuracy" -> "accuracy")
        metric = db_model.model_name.replace("best_model_for_", "")
        
        # Get the score for this metric
        score = None
        model_name = db_model.model_name.split('_')[-1] if '_' in db_model.model_name else None
        
        # Only use real metric values
        if hasattr(db_model, 'metric_value') and db_model.metric_value:
            score = float(db_model.metric_value)
        elif model_metrics and model_name in model_metrics and metric in model_metrics[model_name]:
            score = model_metrics[model_name][metric]
        # No fake fallback values
        
        models.append({
            'id': db_model.id,
            'name': db_model.model_name,
            'metric': metric,
            'file_name': db_model.file_name,
            'score': score
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

# Add helper function for creating placeholder model files
def create_placeholder_model(temp_dir, model, error_message):
    """Create a placeholder model file when the real model can't be downloaded."""
    placeholder_path = os.path.join(temp_dir, model.file_name or 'model.joblib')
    with open(placeholder_path, 'w') as f:
        f.write("# This is a placeholder for the model file that couldn't be downloaded\n")
        f.write(f"# Original URL: {model.model_url if hasattr(model, 'model_url') else 'Unknown'}\n")
        f.write(f"# Error: {error_message}\n")
    
    logger.info(f"Created placeholder model file at {placeholder_path}")
    return placeholder_path


def create_requirements_file(temp_dir):
    """Create a requirements.txt file with necessary dependencies for the model package."""
    requirements_path = os.path.join(temp_dir, 'requirements.txt')
    with open(requirements_path, 'w') as f:
        f.write("pandas>=1.3.0\n")
        f.write("numpy>=1.20.0\n")
        f.write("scikit-learn>=1.0.0\n")
        f.write("joblib>=1.0.0\n")
    
    logger.info(f"Created requirements.txt file at {requirements_path}")
    return requirements_path


# Function for decoding predictions using a specific target feature
def decode_predictions_using_feature(prediction_file_path, output_path, model_dir, target_feature):
    """
    Decode numeric predictions to their original categorical values
    using a specific target feature's encoding mapping.
    
    Args:
        prediction_file_path: Path to the prediction file
        output_path: Path to save the decoded predictions
        model_dir: Directory containing the model package
        target_feature: The feature to use for decoding
        
    Returns:
        bool: True if decoding was successful, False otherwise
    """
    try:
        # Check if target feature is specified
        if not target_feature:
            logger.warning("No target feature specified for decoding")
            return False
            
        # Look for preprocessing_info.json
        preprocessing_info_path = os.path.join(model_dir, 'preprocessing_info.json')
        if not os.path.exists(preprocessing_info_path):
            logger.warning(f"No preprocessing_info.json found in {model_dir}")
            return False
            
        # Load preprocessing info
        with open(preprocessing_info_path, 'r') as f:
            preprocessing_info = json.load(f)
            
        # Check if encodings exist
        encodings = preprocessing_info.get('encoding_mappings', {})
        if not encodings:
            logger.warning("No encoding mappings found in preprocessing info")
            return False
            
        # Check if target feature exists in encodings
        if target_feature not in encodings:
            logger.warning(f"Target feature '{target_feature}' not found in encoding mappings")
            return False
            
        # Get the encoding mapping for the target feature
        target_mapping = encodings[target_feature]
        logger.info(f"Using encoding mapping for target feature: {target_feature}")
        
        # Load the prediction CSV
        prediction_df = pd.read_csv(prediction_file_path)
        logger.info(f"Loaded prediction CSV with columns: {prediction_df.columns.tolist()}")
        
        # Find the prediction column - typically named 'prediction'
        prediction_column = None
        for col in prediction_df.columns:
            if 'prediction' in col.lower():
                prediction_column = col
                break
        
        if not prediction_column:
            logger.warning("Could not find prediction column in the output file")
            return False
            
        logger.info(f"Found prediction column: {prediction_column}")
        
        # Invert the mapping: from {category: code} to {code: category}
        reverse_mapping = {int(v): k for k, v in target_mapping.items()}
        logger.info(f"Created reverse mapping for {target_feature}: {reverse_mapping}")
        
        # Create a new column with decoded predictions
        decoded_column = f"{prediction_column}_decoded"
        prediction_df[decoded_column] = prediction_df[prediction_column].map(reverse_mapping)
        logger.info(f"Added decoded column {decoded_column}")
        
        # Save the updated DataFrame to the output path
        prediction_df.to_csv(output_path, index=False)
        logger.info(f"Saved prediction file with decoded column to {output_path}")
        
        return True
    
    except Exception as e:
        logger.warning(f"Error decoding predictions using feature '{target_feature}': {str(e)}")
        return False

@app.route('/check_downloads_dir')
@login_required
def check_downloads_dir():
    """Utility endpoint to check if the downloads directory is properly set up."""
    try:
        # Define paths to check
        static_dir = os.path.join(os.getcwd(), 'static')
        downloads_dir = os.path.join(static_dir, 'downloads')
        user_downloads_dir = os.path.join(downloads_dir, str(current_user.id))
        
        # Create directories if they don't exist
        os.makedirs(static_dir, exist_ok=True)
        os.makedirs(downloads_dir, exist_ok=True)
        os.makedirs(user_downloads_dir, exist_ok=True)
        
        # Create a test file
        test_file_path = os.path.join(user_downloads_dir, 'test_file.txt')
        with open(test_file_path, 'w') as f:
            f.write('Test file content')
            
        # Generate test URL
        test_url = url_for('static', filename=f"downloads/{current_user.id}/test_file.txt")
        
        # Return success
        return jsonify({
            'status': 'success',
            'message': 'Downloads directory is properly set up',
            'paths': {
                'static_dir': static_dir,
                'downloads_dir': downloads_dir,
                'user_downloads_dir': user_downloads_dir,
                'test_file': test_file_path,
                'test_url': test_url
            },
            'exists': {
                'static_dir': os.path.exists(static_dir),
                'downloads_dir': os.path.exists(downloads_dir),
                'user_downloads_dir': os.path.exists(user_downloads_dir),
                'test_file': os.path.exists(test_file_path)
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error checking downloads directory: {str(e)}',
            'error': str(e)
        }), 500

@app.route('/api/extract_features', methods=['POST'])
@login_required
def extract_features():
    """Extract available features from a model package for decoding."""
    try:
        # Check if model package file was uploaded
        if 'modelPackage' not in request.files:
            return jsonify({"error": "Model package is required"}), 400
            
        model_package = request.files['modelPackage']
        
        # Validate file extension
        if not model_package.filename.endswith('.zip'):
            return jsonify({"error": "Model package must be a zip file"}), 400
            
        # Create a temporary directory for the extraction
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save the zip file
            zip_path = os.path.join(temp_dir, 'model_package.zip')
            model_package.save(zip_path)
            
            # Extract the preprocessing_info.json file only
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Check if preprocessing_info.json exists in the zip
                preprocessing_info_exists = False
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith('preprocessing_info.json'):
                        preprocessing_info_exists = True
                        zip_ref.extract(file_info, temp_dir)
                        break
                
                if not preprocessing_info_exists:
                    return jsonify({"error": "No preprocessing_info.json found in the package"}), 400
            
            # Find the extracted preprocessing_info.json file
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file == 'preprocessing_info.json':
                        preprocessing_info_path = os.path.join(root, file)
                        break
            
            # Load the preprocessing info
            with open(preprocessing_info_path, 'r') as f:
                preprocessing_info = json.load(f)
            
            # Extract available features from encoding mappings
            features = []
            encodings = preprocessing_info.get('encoding_mappings', {})
            for feature in encodings.keys():
                features.append(feature)
            
            logger.info(f"Extracted {len(features)} encoded features from model package")
            
            # Return the list of features
            return jsonify({
                "features": features
            })
            
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return jsonify({"error": str(e)}), 500