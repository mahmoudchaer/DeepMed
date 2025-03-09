import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
import logging
import os

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
        logger.info("Starting classification model training")
        
        # Handle imbalanced datasets
        X_train_balanced, y_train_balanced = self._handle_imbalance(X_train, y_train)
        
        # Encode labels if they're not numeric
        if not np.issubdtype(y_train.dtype, np.number):
            y_train_balanced = self.label_encoder.fit_transform(y_train_balanced)
            y_test = self.label_encoder.transform(y_test)
        
        best_models = []
        model_metrics = {}
        
        # Train all models with extensive hyperparameter search
        with mlflow.start_run(run_name="model_comparison") as run:
            mlflow.log_param("dataset_size", len(X_train))
            mlflow.log_param("features_count", X_train.shape[1])
            
            for model_name, model_info in self.classification_models.items():
                logger.info(f"Training {model_name}")
                
                try:
                    with mlflow.start_run(run_name=model_name, nested=True):
                        # Create pipeline
                        pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('classifier', model_info['model'])
                        ])
                        
                        # Perform grid search with cross-validation
                        grid_search = GridSearchCV(
                            pipeline,
                            model_info['params'],
                            cv=5,
                            scoring=['accuracy', 'precision_weighted', 'recall_weighted'],
                            refit='accuracy',
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
                        
                        # Log to MLflow
                        mlflow.log_params(grid_search.best_params_)
                        mlflow.log_metrics(metrics)
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
        best_precision = max(model_metrics.items(), key=lambda x: x[1]['metrics']['precision'])
        best_recall = max(model_metrics.items(), key=lambda x: x[1]['metrics']['recall'])
        
        # Prepare best models info
        for metric_name, (model_name, model_info) in [
            ('Accuracy', best_accuracy),
            ('Precision', best_precision),
            ('Recall', best_recall)
        ]:
            if model_name not in [m['model_name'] for m in best_models]:  # Avoid duplicates
                best_models.append({
                    'model': model_info['model'],
                    'model_name': model_name,
                    'metric_name': metric_name,
                    'metrics': model_info['metrics'],
                    'display_metric': metric_name.lower(),
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
