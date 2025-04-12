from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
import logging
import sys
import os
import tempfile
import json
import io
import base64
from waitress import serve
from sklearn.compose import ColumnTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure MLflow
MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'medical_diagnosis')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
os.makedirs(MODELS_DIR, exist_ok=True)

class ModelTrainer:
    def __init__(self, task='auto', test_size=0.2):
        self.models = {}
        self.best_models = []
        self.task = task
        self.test_size = test_size
        # Keep LabelEncoder for target variables
        self.label_encoder = LabelEncoder()  # For target variables
        
        # Define classification models with their hyperparameters
        self.classification_models = {
            'logistic_regression': {
                'model': LogisticRegression(multi_class='ovr'),
                'params': {
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__solver': ['lbfgs', 'saga'],
                    'classifier__max_iter': [1000, 5000, 10000],
                    'classifier__class_weight': ['balanced']
                },
                'scoring': ['accuracy', 'precision_weighted', 'recall_weighted'],
                'refit': 'accuracy'
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(),
                'params': {
                    'classifier__max_depth': [3, 5, 7, 10, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__class_weight': ['balanced']
                },
                'scoring': ['accuracy', 'precision_weighted', 'recall_weighted'],
                'refit': 'accuracy'
            },
            'random_forest': {
                'model': RandomForestClassifier(),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [5, 10, None],
                    'classifier__min_samples_split': [2, 5],
                    'classifier__min_samples_leaf': [1, 2],
                    'classifier__class_weight': ['balanced']
                },
                'scoring': ['accuracy', 'precision_weighted', 'recall_weighted'],
                'refit': 'accuracy'
            },
            'svm': {
                'model': SVC(probability=True),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__kernel': ['rbf', 'linear'],
                    'classifier__gamma': ['scale', 'auto'],
                    'classifier__class_weight': ['balanced']
                },
                'scoring': ['accuracy', 'precision_weighted', 'recall_weighted'],
                'refit': 'accuracy'
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'classifier__n_neighbors': [3, 5, 7, 11],
                    'classifier__weights': ['uniform', 'distance'],
                    'classifier__metric': ['euclidean', 'manhattan']
                },
                'scoring': ['accuracy', 'precision_weighted', 'recall_weighted'],
                'refit': 'accuracy'
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                },
                'scoring': ['accuracy', 'precision_weighted', 'recall_weighted'],
                'refit': 'accuracy'
            }
        }
    
    def _detect_task(self, y):
        unique_values = len(np.unique(y))
        if unique_values < 10 or str(y.dtype) in ['bool', 'object']:
            return 'classification'
        return 'regression'
    
    def _handle_imbalance(self, X, y):
        """Handle imbalanced datasets using SMOTE"""
        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info("Applied SMOTE to handle class imbalance")
            return X_resampled, y_resampled
        except Exception as e:
            logger.warning(f"Could not apply SMOTE: {str(e)}. Using original data.")
            return X, y
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and track with MLflow"""
        if self.task == 'auto':
            self.task = self._detect_task(y_train)
        
        if self.task == 'classification':
            return self._train_classification_models(X_train, X_test, y_train, y_test)
        else:
            raise ValueError("Only classification tasks are currently supported")
    
    def _train_classification_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate classification models"""
        logger.info("Starting classification model training")
        logger.info(f"Input training data shape: {X_train.shape}")
        logger.info(f"Input test data shape: {X_test.shape}")
        
        # Handle imbalanced datasets
        X_train_balanced, y_train_balanced = self._handle_imbalance(X_train, y_train)
        logger.info(f"After SMOTE balancing - Training data shape: {X_train_balanced.shape}")
        
        # Encode labels if they're not numeric
        if not np.issubdtype(y_train.dtype, np.number):
            logger.info("Target variable is categorical, applying LabelEncoder")
            original_classes = np.unique(y_train)
            y_train_balanced = self.label_encoder.fit_transform(y_train_balanced)
            y_test = self.label_encoder.transform(y_test)
            logger.info(f"Classes encoded: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        else:
            logger.info("Target variable is numeric, no encoding needed")
        
        # Save the cleaned and balanced dataset to a CSV file for visualization
        cleaned_data_path = os.path.join(MODELS_DIR, 'cleaned_data.csv')
        X_train_balanced.to_csv(cleaned_data_path, index=False)
        logger.info(f"Cleaned dataset saved to {cleaned_data_path}")
        
        # Log data statistics
        logger.info("Dataset statistics:")
        for col in X_train_balanced.columns:
            logger.info(f"  Column: {col}")
            logger.info(f"    Type: {X_train_balanced[col].dtype}")
            logger.info(f"    Min: {X_train_balanced[col].min()}")
            logger.info(f"    Max: {X_train_balanced[col].max()}")
            logger.info(f"    Mean: {X_train_balanced[col].mean()}")
            logger.info(f"    Std: {X_train_balanced[col].std()}")
        
        best_models = []
        model_metrics = {}
        
        with mlflow.start_run(run_name="model_comparison") as run:
            for model_name, model_info in self.classification_models.items():
                logger.info(f"Training {model_name}")
                
                try:
                    with mlflow.start_run(run_name=model_name, nested=True):
                        # Find categorical columns for one-hot encoding
                        categorical_cols = X_train_balanced.select_dtypes(include=['object', 'category']).columns.tolist()
                        if categorical_cols:
                            logger.info(f"Categorical columns for one-hot encoding: {categorical_cols}")
                            
                            # Create preprocessing steps
                            preprocessor = ColumnTransformer(
                                transformers=[
                                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                                ],
                                remainder='passthrough'
                            )
                            
                            # Create pipeline with preprocessing and classifier
                            pipeline = Pipeline([
                                ('preprocessor', preprocessor),
                                ('classifier', model_info['model'])
                            ])
                            
                            logger.info("Created pipeline with OneHotEncoder for categorical features")
                        else:
                            logger.info("No categorical columns found, using classifier directly")
                            # Create pipeline without scaling
                            pipeline = Pipeline([
                                # ('scaler', StandardScaler()),  # Removed to prevent double scaling
                                ('classifier', model_info['model'])
                            ])
                        
                        # Perform grid search with cross-validation
                        # Adjust parameter names to match the pipeline structure
                        adjusted_params = {}
                        for param_name, param_values in model_info['params'].items():
                            # If we have a preprocessor, we need to adjust classifier parameter names
                            if 'preprocessor' in pipeline.named_steps and param_name.startswith('classifier__'):
                                adjusted_params[param_name] = param_values
                            elif 'preprocessor' not in pipeline.named_steps:
                                adjusted_params[param_name] = param_values
                            
                        grid_search = GridSearchCV(
                            pipeline,
                            adjusted_params,
                            cv=5,
                            scoring=model_info['scoring'],
                            refit=model_info['refit'],
                            n_jobs=-1
                        )
                        
                        # Train model
                        grid_search.fit(X_train_balanced, y_train_balanced)
                        
                        # Get best model
                        best_model = grid_search.best_estimator_
                        
                        # Make predictions
                        y_pred = best_model.predict(X_test)
                        
                        # Calculate metrics
                        metrics = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, average='weighted'),
                            'recall': recall_score(y_test, y_pred, average='weighted'),
                            'f1': f1_score(y_test, y_pred, average='weighted'),
                            'cv_score_mean': grid_search.best_score_,
                            'cv_score_std': grid_search.cv_results_['std_test_accuracy'][grid_search.best_index_]
                        }
                        
                        # Log parameters and metrics
                        mlflow.log_params(grid_search.best_params_)
                        mlflow.log_metrics(metrics)
                        
                        # Log model
                        mlflow.sklearn.log_model(best_model, model_name)
                        
                        # Store model info
                        model_metrics[model_name] = {
                            'model': best_model,
                            'metrics': metrics,
                            'model_name': model_name,
                            'params': grid_search.best_params_
                        }
                        
                        logger.info(f"Completed training {model_name}")
                        
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    continue
        
        # Select top models based on different metrics
        best_accuracy = max(model_metrics.items(), key=lambda x: x[1]['metrics']['accuracy'])
        
        # Prepare best model info - only using the model with best accuracy
        model_name, model_info = best_accuracy
        best_models.append({
            'model': model_info['model'],
            'model_name': model_name,
            'metric_name': 'Accuracy',
            'metrics': model_info['metrics'],
            'display_metric': 'accuracy',
            'label_encoder': self.label_encoder if not np.issubdtype(y_train.dtype, np.number) else None
        })
        
        self.best_models = best_models
        return best_models
    
    def save_models(self, output_dir='models'):
        """Save trained models to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, model_info in enumerate(self.best_models):
            model_path = output_path / f'model_{i}.joblib'
            joblib.dump(model_info, model_path)
        
        return [str(output_path / f'model_{i}.joblib') for i in range(len(self.best_models))]
    
    def load_model(self, model_path):
        """Load a trained model from disk"""
        return joblib.load(model_path)

# Initialize model trainer
model_trainer = ModelTrainer()

@app.route('/health')
def health():
    return jsonify({
        "service": "model_trainer_api",
        "status": "healthy"
    })

@app.route('/train', methods=['POST'])
def train():
    """
    Train models on the provided data
    
    Expected JSON input:
    {
        "data": {...},  # Features in JSON format
        "target": [...],  # Target variable
        "test_size": 0.2  # Optional
    }
    
    Returns:
    {
        "models": [...],  # List of trained model info
        "saved_models": [...],  # List of saved model paths
        "message": "Models trained successfully"
    }
    """
    try:
        request_data = request.json
        
        if not request_data or 'data' not in request_data or 'target' not in request_data:
            return jsonify({"error": "Invalid request. Missing 'data' or 'target'."}), 400
        
        # Convert JSON to DataFrame/array
        try:
            X = pd.DataFrame.from_dict(request_data['data'])
            y = np.array(request_data['target'])
        except Exception as e:
            return jsonify({"error": f"Failed to convert JSON to DataFrame: {str(e)}"}), 400
        
        # Set test size if provided
        if 'test_size' in request_data:
            model_trainer.test_size = float(request_data['test_size'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=model_trainer.test_size, random_state=42
        )
        
        # Train models
        best_models = model_trainer.train_models(X_train, X_test, y_train, y_test)
        
        # Save models
        saved_models = model_trainer.save_models(MODELS_DIR)
        
        # Prepare response
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
        logger.error(f"Error in train endpoint: {str(e)}", exc_info=True)
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
        
        # Convert NumPy values to native Python types
        converted_predictions = []
        for p in predictions:
            if isinstance(p, (np.integer, np.floating)):
                # Convert NumPy numeric types to Python float first
                converted_value = float(p)
                # Check if it's close to an integer before rounding
                if abs(converted_value - int(converted_value)) < 1e-10:
                    converted_value = int(converted_value)
            else:
                # For non-numeric types (e.g., strings), convert to string
                converted_value = str(p)
            converted_predictions.append(converted_value)
        
        return jsonify({
            "predictions": converted_predictions,
            "message": "Predictions made successfully"
        })
    
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}", exc_info=True)
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
        logger.error(f"Error in download_model endpoint: {str(e)}", exc_info=True)
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
        logger.error(f"Error in model_info endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/download_preprocessed_data', methods=['GET'])
def download_preprocessed_data():
    """Download the preprocessed dataset used for training"""
    try:
        # Path to the cleaned data CSV file
        cleaned_data_path = os.path.join(MODELS_DIR, 'cleaned_data.csv')
        
        # Check if the file exists
        if not os.path.exists(cleaned_data_path):
            return jsonify({"error": "Preprocessed data file not found. Train models first."}), 404
        
        # Return the file for download
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        return send_file(
            cleaned_data_path, 
            as_attachment=True, 
            download_name=f'preprocessed_data_{timestamp}.csv',
            mimetype='text/csv'
        )
    
    except Exception as e:
        logger.error(f"Error in download_preprocessed_data endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5004))
    serve(app, host='0.0.0.0', port=port) 