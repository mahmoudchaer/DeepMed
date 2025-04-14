from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
import logging
import sys
import os
from waitress import serve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
app = Flask(__name__)

MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'model_trainer')
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
        self.label_encoder = LabelEncoder()

        self.classification_models = {
            'logistic_regression': {
                'model': LogisticRegression(multi_class='ovr'),
                'params': {
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__solver': ['lbfgs', 'saga'],
                    'classifier__max_iter': [1000],
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
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [5, 10, None],
                    'classifier__min_samples_split': [2, 5],
                    'classifier__min_samples_leaf': [1, 2],
                    'classifier__class_weight': ['balanced']
                },
                'scoring': ['accuracy', 'precision_weighted', 'recall_weighted'],
                'refit': 'accuracy'
            }
        }

    def _detect_task(self, y):
        return 'classification' if len(np.unique(y)) < 10 else 'regression'

    def _handle_imbalance(self, X, y):
        try:
            smote = SMOTE(random_state=42)
            return smote.fit_resample(X, y)
        except Exception as e:
            logger.warning(f"SMOTE failed: {str(e)}")
            return X, y

    def train_models(self, X_train, X_test, y_train, y_test):
        if self.task == 'auto':
            self.task = self._detect_task(y_train)

        if self.task != 'classification':
            raise ValueError("Only classification is supported")

        X_train, y_train = self._handle_imbalance(X_train, y_train)

        if not np.issubdtype(y_train.dtype, np.number):
            y_train = self.label_encoder.fit_transform(y_train)
            y_test = self.label_encoder.transform(y_test)

        best_models = []

        with mlflow.start_run(run_name="model_comparison") as run:
            for name, config in self.classification_models.items():
                try:
                    with mlflow.start_run(run_name=name, nested=True):
                        pipeline = Pipeline([
                            ('classifier', config['model'])
                        ])

                        grid = GridSearchCV(
                            pipeline,
                            config['params'],
                            cv=5,
                            scoring=config['scoring'],
                            refit=config['refit'],
                            n_jobs=-1
                        )

                        grid.fit(X_train, y_train)
                        best_model = grid.best_estimator_
                        y_pred = best_model.predict(X_test)

                        metrics = {
                            'accuracy': float(accuracy_score(y_test, y_pred)),
                            'precision': float(precision_score(y_test, y_pred, average='weighted')),
                            'recall': float(recall_score(y_test, y_pred, average='weighted')),
                            'f1': float(f1_score(y_test, y_pred, average='weighted')),
                            'cv_score_mean': float(grid.best_score_),
                            'cv_score_std': float(grid.cv_results_['std_test_accuracy'][grid.best_index_])
                        }

                        mlflow.log_params(grid.best_params_)
                        mlflow.log_metrics(metrics)
                        mlflow.sklearn.log_model(best_model, name)

                        best_models.append({
                            'model': best_model,
                            'model_name': name,
                            'metrics': metrics,
                            'params': grid.best_params_,
                            'display_metric': 'accuracy',
                            'label_encoder': self.label_encoder
                        })
                except Exception as e:
                    logger.error(f"Error training {name}: {str(e)}")

        self.best_models = best_models
        return best_models

    def save_models(self, output_dir=MODELS_DIR):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, model in enumerate(self.best_models):
            path = output_path / f'model_{i}.joblib'
            joblib.dump(model, path)
            saved_paths.append(str(path))
        return saved_paths

    def load_model(self, path):
        return joblib.load(path)

model_trainer = ModelTrainer()

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "model_trainer"})

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.json
        X = pd.DataFrame.from_dict(data['data'])
        y = np.array(data['target'])

        if 'test_size' in data:
            model_trainer.test_size = float(data['test_size'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model_trainer.test_size, random_state=42)
        best_models = model_trainer.train_models(X_train, X_test, y_train, y_test)
        saved_models = model_trainer.save_models()

        response = []
        for i, m in enumerate(best_models):
            response.append({
                'model_name': m['model_name'],
                'metric_value': m['metrics'][m['display_metric']],
                'cv_score_mean': m['metrics']['cv_score_mean'],
                'cv_score_std': m['metrics']['cv_score_std'],
                'model_id': i
            })

        return jsonify({
            'models': response,
            'saved_models': saved_models,
            'message': 'Models trained successfully'
        })

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5004))
    serve(app, host='0.0.0.0', port=port)