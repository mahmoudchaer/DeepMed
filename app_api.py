from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify, flash
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import io
import time
from datetime import datetime
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NaN, inf, -inf
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        return super().default(obj)

# Safe JSON dump function
def safe_json_dumps(obj):
    return json.dumps(obj, cls=SafeJSONEncoder)

# Modified requests post function that handles problematic float values
# Modified requests post function that handles problematic float values
def safe_requests_post(url, json_data, **kwargs):
    safe_json = clean_data_for_json(json_data)
    return requests.post(url, json=safe_json, **kwargs)


# Load environment variables from .env file
load_dotenv()

# Define service URLs
DATA_CLEANER_URL = "http://localhost:5001"
FEATURE_SELECTOR_URL = "http://localhost:5002"
ANOMALY_DETECTOR_URL = "http://localhost:5003"
MODEL_TRAINER_URL = "http://localhost:5004"
MEDICAL_ASSISTANT_URL = "http://localhost:5005"

# Setup Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Check service health
def check_services():
    services = {
        "Data Cleaner": DATA_CLEANER_URL,
        "Feature Selector": FEATURE_SELECTOR_URL,
        "Anomaly Detector": ANOMALY_DETECTOR_URL,
        "Model Trainer": MODEL_TRAINER_URL,
        "Medical Assistant": MEDICAL_ASSISTANT_URL
    }
    
    status = {}
    for name, url in services.items():
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                status[name] = "healthy"
            else:
                status[name] = f"unhealthy - {response.status_code}"
        except Exception as e:
            status[name] = f"unreachable - {str(e)[:50]}"  # Truncate long error messages
    
    return status

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

def load_data(file_path):
    """Load data from file with error handling and feedback"""
    try:
        if file_path.endswith('.csv'):
            # Attempt to read with UTF-8 encoding first
            try:
                data = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                # If UTF-8 fails, try ISO-8859-1 (Latin-1) encoding
                data = pd.read_csv(file_path, encoding='ISO-8859-1')
        elif file_path.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file_path)
        else:
            return None, "Unsupported file format. Please upload a CSV or Excel file."
        
        # Generate file statistics
        file_stats = {
            'rows': data.shape[0],
            'columns': data.shape[1],
            'memory_usage': f"{data.memory_usage().sum() / 1024:.2f} KB",
            'upload_time': datetime.now().strftime('%H:%M:%S')
        }
        
        return data, file_stats
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        return None, f"Error loading file: {str(e)}"

def is_service_available(service_url):
    """Check if a service is available"""
    try:
        response = requests.get(f"{service_url}/health", timeout=1)
        return response.status_code == 200
    except:
        return False

def clean_data_for_json(data):
    """Clean DataFrame to make it JSON serializable by replacing non-compliant values"""
    if isinstance(data, pd.DataFrame):
        # Create a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Replace inf/-inf with None (which will become null in JSON)
        data_copy = data_copy.replace([np.inf, -np.inf], None)
        
        # Replace NaN with None
        data_copy = data_copy.where(pd.notnull(data_copy), None)
        
        # Handle any remaining problematic float values
        for col in data_copy.select_dtypes(include=['float']).columns:
            data_copy[col] = data_copy[col].apply(
                lambda x: None if x is not None and (np.isnan(x) or np.isinf(x)) else x
            )
            
        return data_copy.to_dict(orient='records')
    elif isinstance(data, pd.Series):
        # For Series objects (like target variables)
        # Create a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Handle problematic values for Series
        if data_copy.dtype.kind == 'f':  # If float type
            data_copy = data_copy.apply(
                lambda x: None if np.isnan(x) or np.isinf(x) else x
            )
        
        return data_copy.replace([np.inf, -np.inf, np.nan], None).tolist()
    elif isinstance(data, list):
        # For lists, recursively clean each item
        return [clean_data_for_json(item) if isinstance(item, (pd.DataFrame, pd.Series, list, dict)) else 
                (None if isinstance(item, float) and (np.isnan(item) or np.isinf(item)) else item) 
                for item in data]
    elif isinstance(data, dict):
        # For dictionaries, recursively clean each value
        return {k: clean_data_for_json(v) if isinstance(v, (pd.DataFrame, pd.Series, list, dict)) else
                (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v)
                for k, v in data.items()}
    
    # Handle single float value
    if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
        return None
    
    return data

@app.route('/')
def index():
    # Reset session data for new session
    for key in list(session.keys()):
        if key != '_flashes':
            session.pop(key)
    
    # Check services health for status display
    services_status = check_services()
    return render_template('index.html', services_status=services_status)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load the data to validate it
        data, result = load_data(filepath)
        if data is None:
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
def training():
    filepath = session.get('uploaded_file')
    
    if not filepath:
        flash('Please upload a file first', 'error')
        return redirect(url_for('index'))
    
    data, _ = load_data(filepath)
    
    if request.method == 'POST':
        # Get target column from form
        target_column = request.form.get('target_column')
        session['target_column'] = target_column
        
        # Get additional parameters (optional)
        session['test_size'] = float(request.form.get('test_size', 0.2))
        
        try:
            # Check if required services are available
            required_services = {
                "Data Cleaner": DATA_CLEANER_URL,
                "Feature Selector": FEATURE_SELECTOR_URL,
                "Anomaly Detector": ANOMALY_DETECTOR_URL,
                "Model Trainer": MODEL_TRAINER_URL
            }
            
            for service_name, service_url in required_services.items():
                if not is_service_available(service_url):
                    flash(f"The {service_name} service is not available. Cannot proceed with training.", 'error')
                    return redirect(url_for('training'))
            
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
            cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_data.csv')
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
            
            # Store feature importances for visualization
            feature_importance = []
            for feature, importance in feature_result["feature_importances"].items():
                feature_importance.append({'Feature': feature, 'Importance': importance})
            
            selected_features_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'selected_features.csv')
            X_selected.to_csv(selected_features_filepath, index=False)
            session['selected_features_file'] = selected_features_filepath
            session['feature_importance'] = feature_importance
            
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
            
            # 4. MODEL TRAINING (via Model Trainer API)
            logger.info(f"Sending data to Model Trainer API")
            
            # Convert all data to simple Python structures
            X_selected_records = X_selected.replace([np.inf, -np.inf], np.nan).where(pd.notnull(X_selected), None).to_dict(orient='records')
            y_list = y.replace([np.inf, -np.inf], np.nan).where(pd.notnull(y), None).tolist()

            # Use our safe request method
            response = safe_requests_post(
                f"{MODEL_TRAINER_URL}/train",
                {
                    "data": X_selected_records,
                    "target": y_list,
                    "test_size": float(session['test_size'])
                },
                timeout=180  # Model training can take time
            )
            
            if response.status_code != 200:
                raise Exception(f"Model Trainer API error: {response.json().get('error', 'Unknown error')}")
            
            model_result = response.json()
            # Store all models in session
            session['trained_models'] = model_result["models"]
            session['saved_models'] = model_result["saved_models"]
            
            # Add logging to verify data being sent to APIs
            logger.info(f"Data being sent to Model Trainer API: X_selected: {X_selected_records[:5]}, y: {y_list[:5]}")
            logger.info(f"Models received: {len(model_result['models'])} models")
            
            return redirect(url_for('model_selection'))
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}", exc_info=True)
            flash(f"Error processing data: {str(e)}", 'error')
            return redirect(url_for('training'))
    
    # Get AI recommendations for the dataset (via Medical Assistant API) - OPTIONAL
    ai_recommendations = None
    if 'ai_recommendations' not in session and is_service_available(MEDICAL_ASSISTANT_URL):
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
                session['ai_recommendations'] = ai_recommendations
                # Add logging to verify data being sent to APIs
                logger.info(f"Data being sent to Medical Assistant API: {data_records[:5]}")
            else:
                logger.warning(f"Medical Assistant API returned an error: {response.text}")
        except Exception as e:
            logger.error(f"Error getting AI recommendations: {str(e)}", exc_info=True)
            # Don't flash this error to avoid confusing the user
            logger.info("Continuing without AI recommendations")
    else:
        ai_recommendations = session.get('ai_recommendations')
    
    return render_template('training.html', 
                          data=data.head().to_html(classes='table table-striped'),
                          columns=data.columns.tolist(),
                          file_stats=session.get('file_stats'),
                          ai_recommendations=ai_recommendations)

@app.route('/download_cleaned')
def download_cleaned():
    cleaned_filepath = session.get('cleaned_file')
    if not cleaned_filepath:
        flash('No cleaned file available', 'error')
        return redirect(url_for('training'))
    return send_file(cleaned_filepath, as_attachment=True, download_name='cleaned_data.csv')

@app.route('/feature_importance')
def feature_importance():
    importance_data = session.get('feature_importance')
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

@app.route('/model_selection')
def model_selection():
    trained_models = session.get('trained_models')
    if not trained_models:
        flash('No trained models available. Please complete the training process first.', 'error')
        return redirect(url_for('training'))
    
    task = session.get('task')
    selected_features_file = session.get('selected_features_file')
    
    # Load selected features data to get column count
    selected_features_data, _ = load_data(selected_features_file)
    feature_count = len(selected_features_data.columns) if selected_features_data is not None else 0
    
    overall_metrics = {
        'models_trained': len(trained_models),
        'features_used': feature_count,
        'test_size': session.get('test_size', 0.2)
    }
    
    return render_template('model_selection.html', 
                          models=trained_models,
                          task=task,
                          overall_metrics=overall_metrics)

@app.route('/select_model/<int:model_id>')
def select_model(model_id):
    trained_models = session.get('trained_models')
    if not trained_models or model_id >= len(trained_models):
        flash('Invalid model selection', 'error')
        return redirect(url_for('model_selection'))
    
    session['selected_model'] = model_id
    return redirect(url_for('prediction'))

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    selected_model_id = session.get('selected_model')
    if selected_model_id is None:
        flash('No model selected', 'error')
        return redirect(url_for('model_selection'))
    
    trained_models = session.get('trained_models')
    if not trained_models or selected_model_id >= len(trained_models):
        flash('Invalid model selection', 'error')
        return redirect(url_for('model_selection'))
        
    selected_model_info = trained_models[selected_model_id]
    
    # Check if required services are available
    required_prediction_services = {
        "Data Cleaner": DATA_CLEANER_URL,
        "Feature Selector": FEATURE_SELECTOR_URL,
        "Model Trainer": MODEL_TRAINER_URL
    }
    
    for service_name, service_url in required_prediction_services.items():
        if not is_service_available(service_url):
            flash(f"The {service_name} service is not available. Cannot make predictions.", 'error')
            return render_template('prediction.html', model=selected_model_info)
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('prediction'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for('prediction'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Load prediction data
                pred_data, _ = load_data(filepath)
                target_column = session.get('target_column')
                
                # 1. Clean data using Data Cleaner API
                logger.info(f"Sending prediction data to Data Cleaner API")
                pred_data_records = pred_data.replace([np.inf, -np.inf], np.nan).where(pd.notnull(pred_data), None).to_dict(orient='records')
                
                response = safe_requests_post(
                    f"{DATA_CLEANER_URL}/clean",
                    {
                        "data": pred_data_records,
                        "target_column": target_column
                    },
                    timeout=60
                )
                
                if response.status_code != 200:
                    raise Exception(f"Data Cleaner API error: {response.json().get('error', 'Unknown error')}")
                
                cleaned_data = pd.DataFrame.from_dict(response.json()["data"])
                
                # Handle case where target column might be in prediction data
                if target_column in cleaned_data.columns:
                    cleaned_data = cleaned_data.drop(columns=[target_column])
                
                # 2. Transform features using Feature Selector API
                logger.info(f"Sending prediction data to Feature Selector API for transformation")
                
                # Load the selected features to get the correct columns
                selected_features_file = session.get('selected_features_file')
                selected_features_data, _ = load_data(selected_features_file)
                
                cleaned_data_records = cleaned_data.replace([np.inf, -np.inf], np.nan).where(pd.notnull(cleaned_data), None).to_dict(orient='records')
                
                response = safe_requests_post(
                    f"{FEATURE_SELECTOR_URL}/transform",
                    {
                        "data": cleaned_data_records
                    },
                    timeout=60
                )
                
                if response.status_code != 200:
                    raise Exception(f"Feature Selector API error: {response.json().get('error', 'Unknown error')}")
                
                transformed_data = pd.DataFrame.from_dict(response.json()["transformed_data"])
                
                # 3. Make predictions using Model Trainer API
                logger.info(f"Sending data to Model Trainer API for prediction")
                transformed_data_records = transformed_data.replace([np.inf, -np.inf], np.nan).where(pd.notnull(transformed_data), None).to_dict(orient='records')
                
                response = safe_requests_post(
                    f"{MODEL_TRAINER_URL}/predict/{selected_model_id}",
                    {
                        "data": transformed_data_records
                    },
                    timeout=60
                )
                
                if response.status_code != 200:
                    raise Exception(f"Model Trainer API error: {response.json().get('error', 'Unknown error')}")
                
                predictions = response.json()["predictions"]
                
                # Create results DataFrame
                results_df = pd.DataFrame({'Prediction': predictions})
                
                # Save results
                results_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
                results_df.to_csv(results_filepath, index=False)
                session['predictions_file'] = results_filepath
                
                # Create distribution data for display
                value_counts = results_df['Prediction'].value_counts()
                distribution = []
                for val, count in value_counts.items():
                    percentage = (count / len(predictions) * 100).round(2)
                    distribution.append({
                        'class': str(val),
                        'count': int(count),
                        'percentage': float(percentage)
                    })
                
                session['prediction_distribution'] = distribution
                
                # Add logging to verify data being sent to APIs
                logger.info(f"Data being sent to Data Cleaner API for prediction: {pred_data_records[:5]}")
                logger.info(f"Data being sent to Feature Selector API for prediction: {cleaned_data_records[:5]}")
                logger.info(f"Data being sent to Model Trainer API for prediction: {transformed_data_records[:5]}")
                
                return redirect(url_for('prediction_results'))
                
            except Exception as e:
                logger.error(f"Error making predictions: {str(e)}", exc_info=True)
                flash(f"Error making predictions: {str(e)}", 'error')
                return redirect(url_for('prediction'))
    
    return render_template('prediction.html', model=selected_model_info)

@app.route('/prediction_results')
def prediction_results():
    predictions_file = session.get('predictions_file')
    if not predictions_file:
        flash('No predictions available', 'error')
        return redirect(url_for('prediction'))
    
    # Load predictions
    predictions_data, _ = load_data(predictions_file)
    distribution = session.get('prediction_distribution')
    
    return render_template('prediction_results.html', 
                          predictions=predictions_data.head(20).to_html(classes='table table-striped'),
                          distribution=distribution)

@app.route('/download_predictions')
def download_predictions():
    predictions_file = session.get('predictions_file')
    if not predictions_file:
        flash('No predictions available', 'error')
        return redirect(url_for('prediction'))
    
    return send_file(predictions_file, as_attachment=True, download_name='predictions.csv')

@app.route('/chat', methods=['GET', 'POST'])
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

@app.route('/service_status')
def service_status():
    """Check the status of all services"""
    services_status = check_services()
    return jsonify(services_status)

if __name__ == '__main__':
    print("Starting MedicAI with API services integration")
    print("Service status:")
    status = check_services()
    for service, health in status.items():
        print(f"- {service}: {health}")
    app.run(debug=True, port=5000) 