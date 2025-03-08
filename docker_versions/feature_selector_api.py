from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import json
import openai
import os
import logging
import sys
from io import StringIO

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler('logs/feature_selector.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class FeatureSelector:
    def __init__(self, method='fallback', min_features_to_keep=5, performance_threshold=0.001):
        """Initialize the feature selector with LLM capabilities."""
        self.method = method
        self.min_features_to_keep = min_features_to_keep
        self.performance_threshold = performance_threshold
        self.selected_features = None
        self.feature_importances_ = {}
        self.label_encoders = {}
        self.id_columns = []
        
        # Initialize OpenAI client if available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("🔴 OPENAI_API_KEY environment variable not set. Using fallback method.")
            self.method = 'fallback'
            self.client = None
        else:
            try:
                openai.api_key = api_key
                self.client = openai
                self.method = 'llm'
                logger.info("🟢 OpenAI client initialized successfully - LLM feature selection enabled")
            except Exception as e:
                logger.error(f"🔴 Error initializing OpenAI client: {e}")
                logger.warning("🔄 Falling back to traditional feature selection method")
                self.method = 'fallback'
                self.client = None
        
        logger.info(f"Feature selector initialized with method: {self.method}")
        
        self.system_prompt = """You are an expert medical data scientist specializing in feature selection.
        Your task is to analyze medical datasets and recommend which features to keep or remove based on:
        1. Medical relevance and importance
        2. Statistical properties
        3. Potential redundancy
        4. Relationship to the target variable
        Be conservative in removing features - only suggest removal if there's a clear reason."""
    
    def _get_llm_recommendations(self, X, y=None, target_name=None):
        """Get feature selection recommendations from LLM."""
        # If we don't have LLM access, use fallback method
        if self.method == 'fallback' or self.client is None:
            logger.info("🔄 Using fallback method for feature selection (LLM not available)")
            return {
                "features_to_keep": list(X.columns),
                "features_to_remove": [],
                "reasoning": {}
            }
            
        logger.info("🤖 Attempting to get feature recommendations from LLM")
        # Prepare data summary for LLM
        data_summary = {
            "features": list(X.columns),
            "target_variable": target_name,
            "feature_stats": {
                col: {
                    "dtype": str(X[col].dtype),
                    "unique_values": int(X[col].nunique()),
                    "missing_values": int(X[col].isnull().sum()),
                    "sample_values": list(X[col].dropna().head().astype(str)),
                    "correlation_with_target": float(X[col].corr(y)) if y is not None and pd.api.types.is_numeric_dtype(X[col]) else None
                } for col in X.columns
            }
        }
        
        logger.debug(f"Prepared data summary for LLM with {len(X.columns)} features")
        
        prompt = f"""Analyze this medical dataset and recommend which features to keep or remove.

        Dataset Summary: {json.dumps(data_summary, indent=2)}

        Provide your recommendations in JSON format with:
        {{
            "features_to_keep": ["feature1", "feature2", ...],
            "features_to_remove": ["feature3", "feature4", ...],
            "reasoning": {{
                "feature3": "reason for removal",
                "feature4": "reason for removal"
            }}
        }}

        Consider:
        1. Medical relevance of each feature
        2. Statistical properties (missing values, unique values)
        3. Correlation with target (if available)
        4. Potential redundancy between features
        
        Be conservative - only recommend removing features if there's a clear reason to do so."""

        try:
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            recommendations = json.loads(response.choices[0].message.content)
            logger.info(f"🟢 Successfully received LLM recommendations")
            logger.info(f"Features to keep: {len(recommendations['features_to_keep'])}")
            logger.info(f"Features to remove: {len(recommendations['features_to_remove'])}")
            return recommendations
        except Exception as e:
            logger.error(f"🔴 Error getting LLM recommendations: {str(e)}")
            logger.warning("🔄 Falling back to keeping all features")
            return {
                "features_to_keep": list(X.columns),
                "features_to_remove": [],
                "reasoning": {}
            }

    def fit_transform(self, X, y=None):
        """Fit the feature selector and transform the data."""
        logger.info(f"Starting feature selection using method: {self.method}")
        logger.info(f"Input data shape: {X.shape}")
        
        X = X.copy()
        
        # Prepare features (handle categorical variables)
        X = self._prepare_features(X)
        
        if self.method == 'llm':
            logger.info("🤖 Using LLM-based feature selection")
            # Get LLM recommendations
            recommendations = self._get_llm_recommendations(X, y, target_name=y.name if hasattr(y, 'name') else None)
            
            # Store selected features
            self.selected_features = recommendations['features_to_keep']
            
            # Calculate and store feature importances based on LLM decisions
            for feature in X.columns:
                if feature in recommendations['features_to_keep']:
                    self.feature_importances_[feature] = 1.0
                else:
                    self.feature_importances_[feature] = 0.0
                    reason = recommendations['reasoning'].get(feature, "Removed based on analysis")
                    logger.info(f"🔍 Removing feature '{feature}': {reason}")
            
            # Ensure we keep minimum number of features
            if len(self.selected_features) < self.min_features_to_keep:
                logger.warning(f"⚠️ Too few features recommended ({len(self.selected_features)}). Keeping top {self.min_features_to_keep} features based on correlation with target.")
                correlations = X.corrwith(y) if y is not None else pd.Series(1.0, index=X.columns)
                self.selected_features = correlations.abs().nlargest(self.min_features_to_keep).index.tolist()
        else:
            logger.info("📊 Using traditional feature selection method")
            # Fallback to original maximize method
            self.selected_features = self._maximize_features(X, y)
        
        logger.info(f"✅ Feature selection complete. Selected {len(self.selected_features)} features")
        return X[self.selected_features]

    def transform(self, X):
        """Transform new data using the selected features."""
        if self.selected_features is None:
            raise ValueError("Fit the feature selector first using fit_transform()")
        
        X = X.copy()
        X = self._prepare_features(X)
        
        # Ensure all selected features are present
        missing_features = set(self.selected_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        return X[self.selected_features]

    def _prepare_features(self, X):
        """Prepare features by encoding categorical variables."""
        X = X.copy()
        
        # Handle categorical columns
        for column in X.select_dtypes(include=['object']).columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                X[column] = self.label_encoders[column].fit_transform(X[column].astype(str))
            else:
                # Handle unseen categories in test data
                unique_values = X[column].unique()
                for val in unique_values:
                    if val not in self.label_encoders[column].classes_:
                        # We need to handle unseen values by using a default
                        X[column] = X[column].map(lambda x: x if x in self.label_encoders[column].classes_ else self.label_encoders[column].classes_[0])
                        break
                X[column] = self.label_encoders[column].transform(X[column].astype(str))
        
        return X

    def _identify_id_columns(self, X):
        """Identify and remove ID-like columns"""
        id_patterns = ['id', 'customer_id', 'cust_', 'number']
        id_cols = []
        
        for col in X.columns:
            # Check if column name contains ID-like patterns
            if any(pattern in col.lower() for pattern in id_patterns):
                id_cols.append(col)
            # Check if column has unique values for more than 90% of rows
            elif len(X[col].unique()) / len(X) > 0.9:
                id_cols.append(col)
                
        return id_cols
    
    def _evaluate_feature_set(self, X, y, features):
        """Evaluate a set of features using cross-validation"""
        # For very small datasets, use leave-one-out cross validation
        if len(X) < 5:
            cv = len(X)  # Leave-one-out
        else:
            cv = min(5, len(X))  # Use k-fold CV with k = min(5, n_samples)
            
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        model = LogisticRegression(max_iter=1000)
        scores = []
        
        # Convert y to pandas Series if it's not already
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        for train_idx, val_idx in cv_splitter.split(X, y):
            X_train, X_val = X.iloc[train_idx][features], X.iloc[val_idx][features]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            scores.append(accuracy_score(y_val, model.predict(X_val)))
            
        return np.mean(scores)
    
    def _maximize_features(self, X, y):
        """Maximize feature retention while ensuring performance"""
        logger.info("Using maximize_features method for feature selection")
        # Get initial feature importance using Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        
        # Get baseline performance with all features
        baseline_performance = self._evaluate_feature_set(X, y, X.columns)
        best_performance = baseline_performance
        
        # Initialize with all features
        best_features = list(X.columns)
        current_features = list(X.columns)
        
        # Sort features by importance
        sorted_features = importances.sort_values(ascending=True)
        
        # Try removing features one by one, starting from least important
        removed_features = []
        
        for feature in sorted_features.index:
            temp_features = [f for f in current_features if f != feature]
            temp_performance = self._evaluate_feature_set(X, y, temp_features)
            
            # If removing the feature improves or maintains performance (within threshold)
            if temp_performance >= (best_performance - self.performance_threshold):
                current_features = temp_features
                removed_features.append((feature, temp_performance - best_performance))
                if temp_performance > best_performance:
                    best_performance = temp_performance
                    best_features = temp_features
            else:
                # Try combinations with previously removed features
                for removed_feature, perf_impact in removed_features[-3:]:  # Look at last 3 removed features
                    test_features = current_features + [removed_feature]
                    test_performance = self._evaluate_feature_set(X, y, test_features)
                    
                    if test_performance > best_performance:
                        best_performance = test_performance
                        best_features = test_features
                        current_features = test_features
        
        # Store feature importances for selected features
        self.feature_importances_ = {
            feature: float(importances[feature])
            for feature in best_features
        }
        
        # Print feature selection summary
        logger.info(f"Feature Selection Summary:")
        logger.info(f"Baseline performance: {baseline_performance:.4f}")
        logger.info(f"Final performance: {best_performance:.4f}")
        logger.info(f"Features retained: {len(best_features)}/{len(X.columns)}")
        
        return best_features

# Create a global FeatureSelector instance
feature_selector = FeatureSelector()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "feature_selector_api"})

def preprocess_data(data):
    """Preprocess input data into pandas DataFrame regardless of format."""
    try:
        # If data is already in DataFrame format
        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            return pd.DataFrame(data)
        
        # If data is a list of dictionaries
        if isinstance(data, list) and all(isinstance(d, dict) for d in data):
            return pd.DataFrame(data)
        
        # If data is a list of lists with header
        if isinstance(data, dict) and 'headers' in data and 'values' in data:
            return pd.DataFrame(data['values'], columns=data['headers'])
        
        # If data is a CSV string
        if isinstance(data, str):
            try:
                return pd.read_json(data)
            except:
                return pd.read_csv(StringIO(data))
        
        raise ValueError("Unsupported data format")
    except Exception as e:
        raise ValueError(f"Data preprocessing failed: {str(e)}")

def preprocess_target(target):
    """Preprocess target variable into proper format."""
    try:
        # Convert to pandas Series if it's not already
        if not isinstance(target, pd.Series):
            y = pd.Series(target)
        else:
            y = target
        
        # If target is string/categorical, encode it
        if y.dtype == object:
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        
        # Ensure there are at least 2 classes
        unique_values = y.unique()
        if len(unique_values) < 2:
            raise ValueError("Target must have at least 2 classes")
        
        return y
    except Exception as e:
        raise ValueError(f"Target preprocessing failed: {str(e)}")

@app.route('/select_features', methods=['POST'])
def select_features():
    """Feature selection endpoint with flexible data format handling."""
    try:
        logger.info("📥 Received feature selection request")
        request_data = request.json
        
        if not request_data or 'data' not in request_data or 'target' not in request_data:
            logger.error("🔴 Invalid request - missing data or target")
            return jsonify({"error": "Invalid request. Missing 'data' or 'target'"}), 400
        
        try:
            # Preprocess input data
            logger.info("🔄 Preprocessing input data")
            X = preprocess_data(request_data['data'])
            y = preprocess_target(request_data['target'])
            
            logger.info(f"📊 Input data shape: {X.shape}")
            
            # Set parameters if provided
            if 'target_name' in request_data:
                y.name = request_data['target_name']
                logger.info(f"Target name set to: {y.name}")
            if 'min_features' in request_data:
                feature_selector.min_features_to_keep = int(request_data['min_features'])
                logger.info(f"Minimum features to keep set to: {feature_selector.min_features_to_keep}")
            
        except ValueError as e:
            logger.error(f"🔴 Preprocessing error: {str(e)}")
            return jsonify({"error": str(e)}), 400
        
        # Select features
        logger.info("🔄 Starting feature selection process")
        X_selected = feature_selector.fit_transform(X, y)
        logger.info(f"✅ Feature selection complete. Selected {len(feature_selector.selected_features)} features")
        
        return jsonify({
            "selected_features": feature_selector.selected_features,
            "feature_importances": feature_selector.feature_importances_,
            "transformed_data": X_selected.to_dict(orient='records'),
            "message": "Features selected successfully",
            "method_used": feature_selector.method,  # Added to show which method was used
            "data_info": {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_selected_features": len(feature_selector.selected_features),
                "class_distribution": {
                    "class_0": int(np.sum(y == 0)),
                    "class_1": int(np.sum(y == 1))
                }
            }
        })
    
    except Exception as e:
        logger.error(f"🔴 Error in feature selection: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "suggestion": "Check your data format and make sure it's properly structured"
        }), 500

@app.route('/transform', methods=['POST'])
def transform():
    """
    Transform data using previously selected features
    
    Expected JSON input:
    {
        "data": {...}  # Data in JSON format that can be loaded into a pandas DataFrame
    }
    
    Returns:
    {
        "transformed_data": {...},  # Data with only selected features 
        "message": "Data transformed successfully"
    }
    """
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or 'data' not in request_data:
            return jsonify({"error": "Invalid request. Missing 'data'"}), 400
        
        # Validate if feature selection has been done
        if not feature_selector.selected_features:
            return jsonify({"error": "No features have been selected. Call /select_features first."}), 400
        
        # Convert JSON to DataFrame
        try:
            X = preprocess_data(request_data['data'])
        except Exception as e:
            return jsonify({"error": f"Failed to convert JSON to DataFrame: {str(e)}"}), 400
        
        # Transform data
        X_transformed = feature_selector.transform(X)
        
        # Return transformed data
        return jsonify({
            "transformed_data": X_transformed.to_dict(orient='records'),
            "message": "Data transformed successfully"
        })
    
    except Exception as e:
        logging.error(f"Error in transform endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the app on port 5002
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port) 