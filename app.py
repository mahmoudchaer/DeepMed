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
import joblib
import logging

# Import from models directly since they're now in the root directory
from models.data_cleaner import DataCleaner
from models.feature_selector import FeatureSelector
from models.anomaly_detector import AnomalyDetector
from models.model_trainer import ModelTrainer
from models.medical_assistant import MedicalAssistant

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get the absolute path to the current directory (now the root)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, 
    static_folder=STATIC_DIR,
    template_folder=TEMPLATE_DIR
)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models', 'saved')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize the medical assistant with API key
medical_assistant = None
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    try:
        medical_assistant = MedicalAssistant(api_key)
        logger.info("Medical AI Assistant initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing Medical Assistant: {str(e)}")
        # We'll handle this gracefully - the app will work without AI features
else:
    logger.warning("OPENAI_API_KEY not found in environment variables")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

def load_data(file_path):
    """Load data from file with error handling and feedback"""
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
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

@app.route('/')
@app.route('/index')
def index():
    # Check if we need to reset the session
    if request.args.get('reset'):
        session.clear()
        flash('Session cleared. You can now start a new training.', 'info')
    # Reset session data for new session
    for key in list(session.keys()):
        if key != '_flashes':
            session.pop(key)
    return render_template('index.html')

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
            # Initialize components
            cleaner = DataCleaner()
            feature_selector = FeatureSelector()
            anomaly_detector = AnomalyDetector()
            model_trainer = ModelTrainer()
            
            # Clean data
            cleaned_data = cleaner.clean_data(data, target_column)
            cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_data.csv')
            cleaned_data.to_csv(cleaned_filepath, index=False)
            session['cleaned_file'] = cleaned_filepath
            
            # Feature selection
            X = cleaned_data.drop(columns=[target_column])
            y = cleaned_data[target_column]
            X_selected = feature_selector.fit_transform(X, y)
            
            # Store feature importances for visualization
            importances = feature_selector.feature_importances_
            feature_importance = []
            for feature, importance in importances.items():
                feature_importance.append({'Feature': feature, 'Importance': importance})
            
            selected_features_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'selected_features.csv')
            X_selected.to_csv(selected_features_filepath, index=False)
            session['selected_features_file'] = selected_features_filepath
            session['feature_importance'] = feature_importance
            
            # Anomaly detection
            anomaly_results = anomaly_detector.detect(X_selected)
            session['anomaly_results'] = {
                'is_data_valid': anomaly_results['is_data_valid'],
                'anomaly_percentage': anomaly_results['anomaly_report']['anomaly_percentage']
            }
            
            # Train models
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=session['test_size'], random_state=42
            )
            
            # Detect task type
            unique_values = len(np.unique(y))
            if unique_values < 10 or str(y.dtype) in ['bool', 'object']:
                task = 'classification'
            else:
                task = 'regression'
            session['task'] = task
            
            model_trainer = ModelTrainer(task=task, test_size=session['test_size'])
            best_models = model_trainer.train_models(X_train, X_test, y_train, y_test)
            
            # Save trained models
            models_info = []
            for i, model_info in enumerate(best_models):
                model_path = os.path.join(app.config['MODEL_FOLDER'], f'model_{i}.joblib')
                joblib.dump(model_info, model_path)
                
                # Store simplified model info in session
                models_info.append({
                    'id': i,
                    'model_name': model_info['model_name'],
                    'metric_name': model_info['metric_name'],
                    'metric_value': model_info['metrics'][model_info['display_metric']],
                    'cv_score_mean': model_info['metrics']['cv_score_mean'],
                    'cv_score_std': model_info['metrics']['cv_score_std']
                })
            
            session['trained_models'] = models_info
            
            return redirect(url_for('model_selection'))
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}", exc_info=True)
            flash(f"Error processing data: {str(e)}", 'error')
            return redirect(url_for('training'))
    
    # Get AI recommendations for the dataset if medical assistant is available
    ai_recommendations = None
    if 'ai_recommendations' not in session and medical_assistant:
        try:
            ai_recommendations = medical_assistant.analyze_data(data)
            session['ai_recommendations'] = ai_recommendations
        except Exception as e:
            logger.error(f"Error getting AI recommendations: {str(e)}", exc_info=True)
            flash(f"Error getting AI recommendations: {str(e)}", 'warning')
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
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
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
    selected_model_info = trained_models[selected_model_id]
    
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
                
                # Load the model from disk
                model_path = os.path.join(app.config['MODEL_FOLDER'], f'model_{selected_model_id}.joblib')
                model_full_info = joblib.load(model_path)
                
                # Prepare prediction data
                target_column = session.get('target_column')
                
                # Initialize components
                cleaner = DataCleaner()
                feature_selector = FeatureSelector()
                
                # Clean and transform prediction data
                pred_data = cleaner.clean_data(pred_data, target_column)
                
                # Handle case where target column might be in prediction data
                if target_column in pred_data.columns:
                    pred_data = pred_data.drop(columns=[target_column])
                
                # Load the feature selector state
                selected_features_file = session.get('selected_features_file')
                selected_features_data, _ = load_data(selected_features_file)
                # We assume that feature_selector stored the feature list in the same order
                pred_data = pred_data[selected_features_data.columns]
                
                # Get the model and make predictions
                model = model_full_info['model']
                predictions = model.predict(pred_data)
                
                # Create results DataFrame
                if session.get('task') == 'classification':
                    # Get the label encoder from the model info
                    label_encoder = model_full_info.get('label_encoder')
                    
                    if label_encoder is not None:
                        # Create a mapping dictionary from encoded to decoded values
                        mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
                        # Map predictions to their decoded values
                        decoded_predictions = [mapping[pred] for pred in predictions]
                        results_df = pd.DataFrame({'Prediction': decoded_predictions})
                    else:
                        results_df = pd.DataFrame({'Prediction': predictions})
                else:  # regression
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
    if not medical_assistant:
        flash('OpenAI API key not found or Medical Assistant not initialized! Please add your API key to the .env file.', 'error')
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
            
            # Get AI response
            try:
                response = medical_assistant.chat(prompt)
                
                # Add assistant response to chat history
                session['messages'].append({
                    'role': 'assistant',
                    'content': response
                })
                
            except Exception as e:
                logger.error(f"Error communicating with AI assistant: {str(e)}", exc_info=True)
                flash(f"Error communicating with AI assistant: {str(e)}", 'error')
    
    return render_template('chat.html', messages=session.get('messages', []))

@app.route('/clear_chat')
def clear_chat():
    if 'messages' in session:
        session.pop('messages')
    return redirect(url_for('chat'))

if __name__ == '__main__':
    app.run(debug=True) 