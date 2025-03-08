import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
import streamlit as st

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
            print(f"SMOTE failed: {e}. Using original data.")
            return X, y
    
    def train_models(self, X_train, X_test, y_train, y_test):
        if self.task == 'auto':
            self.task = self._detect_task(y_train)
        
        self.original_classes = np.unique(y_train)
        self.class_mapping = dict(zip(range(len(self.original_classes)), self.original_classes))
        
        mlflow.log_param("test_size", self.test_size)
        
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
                else:
                    specificity = np.diag(cm) / np.sum(cm, axis=1)
                    true_negative_rate = np.mean(specificity)
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'true_negative_rate': true_negative_rate,
                    'true_positive_rate': tp / (tp + fn) if cm.shape == (2, 2) else np.mean(np.diag(cm) / np.sum(cm, axis=0)),
                    'cv_score_mean': cv_scores.mean(),
                    'cv_score_std': cv_scores.std()
                }
                
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{model_name}_{metric_name}", value)
                
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
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        if not best_models:
            raise ValueError("Unable to train logistic regression models with the provided data.")
        
        self.best_models = best_models
        return self.best_models
    
    def save_models(self, output_dir='models'):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for i, model_info in enumerate(self.best_models):
            model_path = output_path / f'model_{i}.joblib'
            model_info['label_encoder'] = self.label_encoder
            model_info['class_mapping'] = self.class_mapping
            joblib.dump(model_info, model_path)
    
    def load_model(self, model_path):
        model_info = joblib.load(model_path)
        if 'label_encoder' in model_info:
            self.label_encoder = model_info['label_encoder']
        if 'class_mapping' in model_info:
            self.class_mapping = model_info['class_mapping']
        return model_info
