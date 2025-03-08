from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import logging
import sys
import os
import tempfile
import json
import io
import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
os.makedirs(MODELS_DIR, exist_ok=True)

class ModelTrainer:
    def __init__(self, task='auto', test_size=0.2):
        self.models = {}
        self.best_models = []
        self.task = task
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.classification_models = {
            'accuracy_focused': {
                'model': LogisticRegression(
                    multi_class='ovr',
                    tol=1e-4,
                    solver='saga'
                ),
                'params': {
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10],
                    'classifier__class_weight': ['balanced'],
                    'classifier__max_iter': [1000, 2500, 5000, 7500, 10000],
                },
                'scoring': 'accuracy',
                'display_metric': 'accuracy',
                'metric_name': 'Accuracy'
            },
            'sensitivity_focused': {
                'model': LogisticRegression(
                    multi_class='ovr',
                    tol=1e-4,
                    solver='saga'
                ),
                'params': {
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10],
                    'classifier__class_weight': ['balanced'],
                    'classifier__max_iter': [1000, 2500, 5000, 7500, 10000],
                },
                'scoring': 'recall_macro',
                'display_metric': 'true_positive_rate',
                'metric_name': 'Sensitivity'
            },
            'specificity_focused': {  
                'model': LogisticRegression(
                    multi_class='ovr',
                    tol=1e-4,
                    solver='saga'
                ),
                'params': {
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10],
                    'classifier__class_weight': ['balanced'],
                    'classifier__max_iter': [1000, 2500, 5000, 7500, 10000],
                },
                'scoring': 'precision_macro',
                'display_metric': 'true_negative_rate',
                'metric_name': 'Specificity'
            }
        }
    
    def _detect_task(self, y):
        unique_values = len(np.unique(y))
        if unique_values < 2:
            raise ValueError("Classification requires at least 2 different classes in the target variable.")
        return 'classification'
    
    def _handle_imbalance(self, X, y):
        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            logging.warning(f"SMOTE failed: {e}. Using original data.")
            return X, y
    
    def train_models(self, X_train, X_test, y_train, y_test):
        if self.task == 'auto':
            self.task = self._detect_task(y_train)
        
        self.original_classes = np.unique(y_train)
        self.class_mapping = dict(zip(range(len(self.original_classes)), self.original_classes))
        
        # Encode target variable for classification
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        return self._train_classification_models(X_train, X_test, y_train_encoded, y_test_encoded)
    
    def _train_classification_models(self, X_train, X_test, y_train, y_test):
        best_models = []
        
        for model_name, model_info in self.classification_models.items():
            try:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model_info['model'])
                ])
                
                grid_search = GridSearchCV(
                    pipeline,
                    model_info['params'],
                    cv=5,
                    scoring=model_info['scoring'],
                    refit=True,
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                
                cv_scores = cross_val_score(
                    grid_search.best_estimator_,
                    X_train, y_train,
                    cv=5, 
                    scoring=model_info['scoring']
                )
                
                y_pred = grid_search.predict(X_test)
                
                cm = confusion_matrix(y_test, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
                    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
                else:
                    specificity = np.diag(cm) / np.sum(cm, axis=1)
                    true_negative_rate = np.mean(specificity)
                    sensitivity = np.diag(cm) / np.sum(cm, axis=0)
                    true_positive_rate = np.mean(sensitivity)
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'true_negative_rate': true_negative_rate,
                    'true_positive_rate': true_positive_rate,
                    'cv_score_mean': cv_scores.mean(),
                    'cv_score_std': cv_scores.std()
                }
                
                model_info = {
                    'model': grid_search.best_estimator_,
                    'model_name': model_name,
                    'params': grid_search.best_params_,
                    'metrics': metrics,
                    'label_encoder': self.label_encoder,
                    'original_classes': self.original_classes,
                    'class_mapping': self.class_mapping,
                    'display_metric': model_info['display_metric'],
                    'metric_name': model_info['metric_name']
                }
                
                best_models.append(model_info)
                
            except Exception as e:
                logging.error(f"Error training {model_name}: {str(e)}")
                continue
        
        if not best_models:
            raise ValueError("Unable to train models with the provided data.")
        
        self.best_models = best_models
        return self.best_models
    
    def save_models(self, output_dir=MODELS_DIR):
        """Save trained models to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        model_paths = []
        for i, model_info in enumerate(self.best_models):
            model_path = output_path / f'model_{i}.joblib'
            model_info['label_encoder'] = self.label_encoder
            model_info['class_mapping'] = self.class_mapping
            joblib.dump(model_info, model_path)
            model_paths.append(str(model_path))
        
        return model_paths
    
    def load_model(self, model_path):
        """Load a trained model from disk"""
        model_info = joblib.load(model_path)
        if 'label_encoder' in model_info:
            self.label_encoder = model_info['label_encoder']
        if 'class_mapping' in model_info:
            self.class_mapping = model_info['class_mapping']
        return model_info

# Create a global ModelTrainer instance
model_trainer = ModelTrainer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "model_trainer_api"})

@app.route('/train', methods=['POST'])
def train():
    """
    Train models endpoint
    
    Expected JSON input:
    {
        "X_train": {...},  # Training features in JSON format
        "X_test": {...},   # Test features in JSON format 
        "y_train": [...],  # Training target values as list
        "y_test": [...],   # Test target values as list
        "task": "auto",    # Optional task type: "auto", "classification", "regression" (default: "auto")
        "test_size": 0.2   # Optional test size (default: 0.2)
    }
    
    Returns:
    {
        "models": [        # List of trained model information
            {
                "model_name": "...",
                "metric_name": "...",
                "metric_value": 0.XX,
                "cv_score_mean": 0.XX,
                "cv_score_std": 0.XX,
                "model_id": 0
            },
            ...
        ],
        "saved_models": [...],  # List of paths where models are saved
        "message": "Models trained successfully"
    }
    """
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or 'X_train' not in request_data or 'y_train' not in request_data:
            return jsonify({"error": "Invalid request. Missing required training data."}), 400
        
        # Convert JSON to DataFrames/Series
        try:
            X_train = pd.DataFrame.from_dict(request_data['X_train'])
            y_train = pd.Series(request_data['y_train'])
            
            # Handle test data - if not provided, create train/test split
            if 'X_test' in request_data and 'y_test' in request_data:
                X_test = pd.DataFrame.from_dict(request_data['X_test'])
                y_test = pd.Series(request_data['y_test'])
            else:
                # Create train/test split
                test_size = request_data.get('test_size', 0.2)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_train, y_train, test_size=test_size, random_state=42
                )
                
            # Set task if provided
            if 'task' in request_data and request_data['task']:
                model_trainer.task = request_data['task']
                
            # Set test_size if provided
            if 'test_size' in request_data and request_data['test_size']:
                model_trainer.test_size = float(request_data['test_size'])
                
        except Exception as e:
            return jsonify({"error": f"Failed to process input data: {str(e)}"}), 400
        
        # Train models
        best_models = model_trainer.train_models(X_train, X_test, y_train, y_test)
        
        # Save models to disk
        saved_models = model_trainer.save_models()
        
        # Create response with simplified model info
        models_info = []
        for i, model_info in enumerate(best_models):
            models_info.append({
                'model_name': model_info['model_name'],
                'metric_name': model_info['metric_name'],
                'metric_value': model_info['metrics'][model_info['display_metric']],
                'cv_score_mean': model_info['metrics']['cv_score_mean'],
                'cv_score_std': model_info['metrics']['cv_score_std'],
                'model_id': i
            })
        
        return jsonify({
            "models": models_info,
            "saved_models": saved_models,
            "message": "Models trained successfully"
        })
    
    except Exception as e:
        logging.error(f"Error in train endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/predict/<int:model_id>', methods=['POST'])
def predict(model_id):
    """
    Make predictions using a trained model
    
    Expected JSON input:
    {
        "data": {...}  # Features in JSON format
    }
    
    Returns:
    {
        "predictions": [...],  # List of predictions
        "message": "Predictions made successfully"
    }
    """
    try:
        # Validate model_id
        model_path = os.path.join(MODELS_DIR, f'model_{model_id}.joblib')
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model with ID {model_id} not found. Train models first."}), 404
        
        # Get request data
        request_data = request.json
        
        if not request_data or 'data' not in request_data:
            return jsonify({"error": "Invalid request. Missing 'data'."}), 400
        
        # Convert JSON to DataFrame
        try:
            data = pd.DataFrame.from_dict(request_data['data'])
        except Exception as e:
            return jsonify({"error": f"Failed to convert JSON to DataFrame: {str(e)}"}), 400
        
        # Load model and make predictions
        model_info = model_trainer.load_model(model_path)
        model = model_info['model']
        predictions = model.predict(data)
        
        # If classification, decode predictions
        if model_trainer.task == 'classification' and 'label_encoder' in model_info:
            label_encoder = model_info['label_encoder']
            # Create a mapping dictionary from encoded to decoded values
            mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
            # Map predictions to their decoded values
            decoded_predictions = [mapping[pred] for pred in predictions]
            predictions = decoded_predictions
        
        return jsonify({
            "predictions": list(predictions),
            "message": "Predictions made successfully"
        })
    
    except Exception as e:
        logging.error(f"Error in predict endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/download_model/<int:model_id>', methods=['GET'])
def download_model(model_id):
    """Download a trained model file"""
    try:
        model_path = os.path.join(MODELS_DIR, f'model_{model_id}.joblib')
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model with ID {model_id} not found"}), 404
        
        return send_file(model_path, as_attachment=True, download_name=f'model_{model_id}.joblib')
    
    except Exception as e:
        logging.error(f"Error in download_model endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/model_info/<int:model_id>', methods=['GET'])
def model_info(model_id):
    """Get detailed information about a trained model"""
    try:
        model_path = os.path.join(MODELS_DIR, f'model_{model_id}.joblib')
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model with ID {model_id} not found"}), 404
        
        # Load model
        model_info = model_trainer.load_model(model_path)
        
        # Create a clean response (without the actual model object)
        response = {
            'model_name': model_info['model_name'],
            'model_id': model_id,
            'metric_name': model_info['metric_name'],
            'metrics': model_info['metrics'],
            'params': str(model_info['params']),
            'task': model_trainer.task
        }
        
        return jsonify(response)
    
    except Exception as e:
        logging.error(f"Error in model_info endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the app on port 5004
    port = int(os.environ.get('PORT', 5004))
    app.run(host='0.0.0.0', port=port) 