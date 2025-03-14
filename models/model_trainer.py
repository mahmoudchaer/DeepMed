import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
import logging
import os
import time

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, task='auto', test_size=0.2):
        self.models = {}
        self.best_models = []
        self.task = task
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Set up MLflow to store everything locally
        os.environ['MLFLOW_TRACKING_URI'] = 'file:./mlruns'
        mlflow.set_tracking_uri('file:./mlruns')
        mlflow.set_experiment('model_training')
        
        # Define classification models with their hyperparameters
        self.classification_models = {
            'logistic_regression': {
                'model': LogisticRegression(multi_class='ovr'),
                'params': {
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__solver': ['lbfgs', 'saga'],
                    'classifier__max_iter': [1000, 5000, 10000],
                    'classifier__class_weight': ['balanced']
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(),
                'params': {
                    'classifier__max_depth': [3, 5, 7, 10, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__class_weight': ['balanced']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [5, 10, None],
                    'classifier__min_samples_split': [2, 5],
                    'classifier__min_samples_leaf': [1, 2],
                    'classifier__class_weight': ['balanced']
                }
            },
            'svm': {
                'model': SVC(probability=True),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__kernel': ['rbf', 'linear'],
                    'classifier__gamma': ['scale', 'auto'],
                    'classifier__class_weight': ['balanced']
                }
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'classifier__n_neighbors': [3, 5, 7, 11],
                    'classifier__weights': ['uniform', 'distance'],
                    'classifier__metric': ['euclidean', 'manhattan']
                }
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                }
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
        """Train and evaluate all models to find the best performers"""
        if self.task == 'auto':
            self.task = self._detect_task(y_train)
        
        if self.task == 'classification':
            return self._train_classification_models(X_train, X_test, y_train, y_test)
        else:
            raise ValueError("Only classification tasks are currently supported")
    
    def _train_classification_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate classification models"""
        model_metrics = {}
        best_models = []
        
        # Convert to numpy arrays for compatibility
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        # Convert categorical target to numeric
        if not np.issubdtype(y_train.dtype, np.number):
            self.label_encoder.fit(y_train)
            y_train = self.label_encoder.transform(y_train)
            y_test = self.label_encoder.transform(y_test)
            logger.info(f"Transformed labels: {self.label_encoder.classes_}")
        
        # Check for class imbalance
        if len(np.unique(y_train)) > 1:
            value_counts = np.bincount(y_train)
            majority_class_count = np.max(value_counts)
            minority_class_count = np.min(value_counts)
            
            # If imbalanced (majority class is more than 3x the minority class)
            if majority_class_count / minority_class_count > 3:
                logger.info("Class imbalance detected, applying SMOTE")
                X_train, y_train = self._handle_imbalance(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train each model
        for model_name, model_config in self.classification_models.items():
            logger.info(f"Training {model_name}...")
            try:
                # Create a pipeline with the model
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),  # Add scaler in pipeline for consistency
                    ('classifier', model_config['model'])
                ])
                
                # Set up cross-validation with stratification
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scoring = 'accuracy'
                
                # Set up grid search with our pipeline
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid=model_config['params'],
                    cv=cv,
                    scoring=scoring,
                    verbose=1,
                    n_jobs=-1
                )
                
                # Fit the grid search
                with mlflow.start_run(nested=True):
                    mlflow.log_param("model_name", model_name)
                    
                    # Log start time
                    start_time = time.time()
                    
                    # Fit the model
                    grid_search.fit(X_train, y_train)
                    
                    # Log training time
                    training_time = time.time() - start_time
                    mlflow.log_metric("training_time", training_time)
                    
                    # Get best model
                    best_model = grid_search.best_estimator_
                    
                    # Make predictions
                    y_pred = best_model.predict(X_test)
                    
                    # Calculate metrics
                    metrics = {
                        'accuracy': float(accuracy_score(y_test, y_pred)),
                        'precision': float(precision_score(y_test, y_pred, average='weighted')),
                        'recall': float(recall_score(y_test, y_pred, average='weighted')),
                        'f1': float(f1_score(y_test, y_pred, average='weighted')),
                        'cv_score_mean': float(grid_search.best_score_),
                        'cv_score_std': float(grid_search.cv_results_['std_test_accuracy'][grid_search.best_index_])
                    }
                    
                    # Log to MLflow
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(best_model, model_name)
                    
                    # Store model info - ensure all values are properly converted to floating point
                    model_metrics[model_name] = {
                        'model': best_model,
                        'metrics': metrics,
                        'model_name': model_name,
                        'params': grid_search.best_params_
                    }
                    
                    logger.info(f"Completed training {model_name}")
                    logger.info(f"Model {model_name} metrics: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}")
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Select top models based on different metrics
        if model_metrics:
            best_accuracy = max(model_metrics.items(), key=lambda x: x[1]['metrics']['accuracy'])
            
            # Prepare best model info - only using the model with best accuracy
            model_name, model_info = best_accuracy
            best_models.append({
                'model': model_info['model'],
                'model_name': model_name,
                'metric_name': 'accuracy',
                'metric_value': float(model_info['metrics']['accuracy']),  # Ensure float conversion
                'display_metric': 'accuracy',
                'cv_score_mean': float(model_info['metrics']['cv_score_mean']),
                'cv_score_std': float(model_info['metrics']['cv_score_std']),
                'label_encoder': self.label_encoder if not np.issubdtype(y_test.dtype, np.number) else None
            })
            
            # Add all metrics for the best model
            for model in best_models:
                model_name = model['model_name']
                model['metrics'] = model_metrics[model_name]['metrics']
            
            self.best_models = best_models
            return best_models
        else:
            logger.warning("No models were successfully trained")
            return []
    
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
