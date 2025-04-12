from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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
    def __init__(self, method='fallback', min_features_to_keep=5, performance_threshold=0.001, 
                 apply_scaling=False, scaling_method='standard', 
                 apply_discretization=False, n_bins=5, 
                 apply_feature_crossing=False, max_crossing_degree=2):
        """
        Initialize the feature selector.
        
        Parameters:
            method (str): Feature selection method to use - 'llm', 'fallback', or 'auto'
            min_features_to_keep (int): Minimum number of features to select
            performance_threshold (float): Performance difference threshold for adding features
            apply_scaling (bool): DISABLED - No longer used - should be False
            scaling_method (str): DISABLED - No longer used
            apply_discretization (bool): DISABLED - No longer used - should be False
            n_bins (int): DISABLED - No longer used
            apply_feature_crossing (bool): DISABLED - No longer used - should be False
            max_crossing_degree (int): DISABLED - No longer used
        """
        self.method = method
        self.min_features_to_keep = min_features_to_keep
        self.performance_threshold = performance_threshold
        
        # Ensure these are all disabled to prevent data transformation
        self.apply_scaling = False  # Force to False
        self.scaling_method = scaling_method  # Not used
        self.apply_discretization = False  # Force to False
        self.n_bins = n_bins  # Not used
        self.apply_feature_crossing = False  # Force to False
        self.max_crossing_degree = max_crossing_degree  # Not used
        
        # For handling categorical features during selection (no transformation)
        self.encoders = {}
        
        # Store feature importances and selected features
        self.feature_importances_ = {}
        self.selected_features = None
        
        # Initialize empty containers for feature engineering
        self.scalers = {}  # Not used
        self.discretizers = {}  # Not used
        self.crossed_features = []  # Not used
        
        logger.info("⚙️ Initialized FeatureSelector with method: " + method)
        logger.info("✅ Data transformation disabled - Feature selector will ONLY select features, not modify values")
        
        # Initialize OpenAI client if available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("🔴 OPENAI_API_KEY environment variable not set. Using fallback method.")
            self.method = 'fallback'
            self.client = None
        else:
            try:
                # Set the API key directly on the openai module
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
        
        Be conservative - only recommend removing features if there's a clear reason to do so.
        IMPORTANT: Respond ONLY with the JSON object, no additional text."""

        try:
            logger.info("Calling OpenAI API...")
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            # Get the raw response content
            content = response['choices'][0]['message']['content'].strip()
            logger.info("Received response from OpenAI")
            
            # Remove any markdown formatting if present
            if content.startswith("```json"):
                content = content.split("```json")[1]
            if content.startswith("```"):
                content = content.split("```")[1]
            content = content.strip()
            
            try:
                recommendations = json.loads(content)
                # Validate the response format
                required_keys = ["features_to_keep", "features_to_remove", "reasoning"]
                if not all(key in recommendations for key in required_keys):
                    raise ValueError("Response missing required keys")
                
                logger.info(f"🟢 Successfully parsed LLM recommendations")
                logger.info(f"Features to keep: {len(recommendations['features_to_keep'])}")
                logger.info(f"Features to remove: {len(recommendations['features_to_remove'])}")
                return recommendations
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"🔴 Error parsing LLM response: {str(e)}")
                logger.error(f"Raw response content: {content}")
                raise
                
        except Exception as e:
            logger.error(f"🔴 Error getting LLM recommendations: {str(e)}")
            logger.warning("🔄 Falling back to keeping all features")
            return {
                "features_to_keep": list(X.columns),
                "features_to_remove": [],
                "reasoning": {}
            }

    def _get_target_recommendations(self, df):
        """Get target column recommendations from LLM."""
        if self.method == 'fallback' or self.client is None:
            logger.info("🔄 LLM not available for target recommendations")
            return None
            
        logger.info("🤖 Getting target column recommendations from LLM")
        
        # Prepare data summary for LLM
        data_summary = {
            "columns": {
                col: {
                    "dtype": str(df[col].dtype),
                    "unique_values": int(df[col].nunique()),
                    "missing_values": int(df[col].isnull().sum()),
                    "sample_values": list(df[col].dropna().head().astype(str)),
                    "is_numeric": pd.api.types.is_numeric_dtype(df[col])
                } for col in df.columns
            }
        }
        
        prompt = f"""Analyze this medical dataset and recommend which column should be the target (dependent) variable.

        Dataset Summary: {json.dumps(data_summary, indent=2)}

        Consider:
        1. Medical significance (e.g., disease status, patient outcome, diagnosis)
        2. Data properties (categorical/numeric, number of unique values)
        3. Typical medical ML prediction tasks

        Respond with a JSON object:
        {{
            "recommended_target": "column_name",
            "reason": "explanation for why this column is suitable as target",
            "alternative_targets": ["other_possible_column1", "other_possible_column2"],
            "task_type": "classification or regression",
            "target_description": "brief description of what this target represents"
        }}

        IMPORTANT: Respond ONLY with the JSON object, no additional text."""

        try:
            logger.info("Calling OpenAI API for target recommendations...")
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert medical data scientist specializing in identifying appropriate target variables for medical machine learning tasks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            content = response['choices'][0]['message']['content'].strip()
            
            # Remove any markdown formatting if present
            if content.startswith("```json"):
                content = content.split("```json")[1]
            if content.startswith("```"):
                content = content.split("```")[1]
            content = content.strip()
            
            recommendations = json.loads(content)
            logger.info(f"🟢 Successfully received target recommendations")
            logger.info(f"Recommended target: {recommendations['recommended_target']}")
            return recommendations
            
        except Exception as e:
            logger.error(f"🔴 Error getting target recommendations: {str(e)}")
            return None

    def _apply_scaling(self, X, is_fitting=True):
        """Apply feature scaling to numerical columns."""
        X = X.copy()
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        
        if not numerical_cols:
            logger.info("No numerical columns found for scaling")
            return X
            
        logger.info(f"🔄 Applying {self.scaling_method} scaling to {len(numerical_cols)} numerical features")
        
        for col in numerical_cols:
            # Skip if all values are the same (standard deviation is zero)
            if X[col].std() == 0:
                continue
                
            if is_fitting:
                if self.scaling_method == 'standard':
                    self.scalers[col] = StandardScaler()
                elif self.scaling_method == 'minmax':
                    self.scalers[col] = MinMaxScaler()
                
                # Fit and transform
                X[col] = self.scalers[col].fit_transform(X[col].values.reshape(-1, 1)).flatten()
            else:
                # Transform only if scaler exists
                if col in self.scalers:
                    X[col] = self.scalers[col].transform(X[col].values.reshape(-1, 1)).flatten()
        
        return X
    
    def _apply_discretization(self, X, is_fitting=True):
        """Apply discretization to suitable numerical columns."""
        X = X.copy()
        
        # Only apply to numerical columns with sufficient unique values
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        suitable_cols = [col for col in numerical_cols if X[col].nunique() > self.n_bins * 2]
        
        if not suitable_cols:
            logger.info("No suitable columns found for discretization")
            return X
            
        logger.info(f"🔄 Applying discretization to {len(suitable_cols)} suitable numerical features")
        
        for col in suitable_cols:
            # Skip if standard deviation is zero
            if X[col].std() == 0:
                continue
                
            if is_fitting:
                # Use KBinsDiscretizer with quantile strategy for more balanced bins
                self.discretizers[col] = KBinsDiscretizer(
                    n_bins=min(self.n_bins, X[col].nunique() - 1),
                    encode='ordinal',  # Use ordinal to maintain numeric representation
                    strategy='quantile'  # Use quantile for more balanced bin sizes
                )
                
                # Fit and transform, then replace the original column values
                X[col] = self.discretizers[col].fit_transform(
                    X[col].values.reshape(-1, 1)
                ).flatten()
                logger.info(f"✅ Discretized column: {col}")
            else:
                # Transform only if discretizer exists
                if col in self.discretizers:
                    X[col] = self.discretizers[col].transform(
                        X[col].values.reshape(-1, 1)
                    ).flatten()
                    logger.info(f"✅ Applied existing discretization to column: {col}")
        
        return X
    
    def _apply_feature_crossing(self, X, is_fitting=True):
        """Create crossed features from numerical columns."""
        X = X.copy()
        
        # Only consider numerical features for crossing to avoid dimensionality explosion
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        
        # Limit to top 10 most correlated features if there are too many
        if len(numerical_cols) > 10:
            logger.info("Too many numerical features, limiting to 10 most important for crossing")
            numerical_cols = numerical_cols[:10]
        
        if not numerical_cols or len(numerical_cols) < 2:
            logger.info("Not enough suitable columns for feature crossing")
            return X
        
        logger.info(f"🔄 Applying feature crossing to {len(numerical_cols)} numerical features")
        
        # Disable feature crossing when not creating new columns to maintain original columns only
        logger.info("Feature crossing disabled to maintain only original columns")
        return X

    def _prepare_features(self, X):
        """Prepare features for selection while preserving original values."""
        # Return a copy with no modifications
        return X.copy()

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
        # Increase max_iter and use lbfgs solver for better convergence
        model = LogisticRegression(max_iter=10000, solver='lbfgs', multi_class='ovr')
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

    def fit_transform(self, X, y=None):
        """Fit the feature selector and transform the data."""
        logger.info(f"Starting feature selection using method: {self.method}")
        logger.info(f"Input data shape: {X.shape}")
        
        X = X.copy()
        
        # Continue with feature selection
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
        logger.info(f"Selected features: {self.selected_features}")
        return X[self.selected_features]

    def transform(self, X):
        """Transform new data using the selected features."""
        if self.selected_features is None:
            raise ValueError("Fit the feature selector first using fit_transform()")
        
        X = X.copy()
        
        # Filter to only include selected columns available in the dataframe
        available_cols = [col for col in self.selected_features if col in X.columns]
        
        if len(available_cols) < len(self.selected_features):
            missing = set(self.selected_features) - set(available_cols)
            logger.warning(f"Missing columns in input data: {missing}")
        
        return X[available_cols]

# Create a global FeatureSelector instance with default settings
feature_selector = FeatureSelector(
    apply_scaling=False,          # Disable scaling
    scaling_method='standard',   # Not used
    apply_discretization=False,   # Disable discretization
    n_bins=5,                    # Not used
    apply_feature_crossing=False, # Disable feature crossing
    max_crossing_degree=2        # Not used
)

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
        
        if not request_data or 'data' not in request_data:
            logger.error("🔴 Invalid request - missing data")
            return jsonify({"error": "Invalid request. Missing 'data'"}), 400
        
        try:
            # Preprocess input data
            logger.info("🔄 Preprocessing input data")
            df = preprocess_data(request_data['data'])
            logger.info(f"📊 Input data shape: {df.shape}")
            
            # Save original column names before any transformations
            original_columns = list(df.columns)
            
            # Get target recommendations if no target is specified
            target_recommendations = None
            if 'target' not in request_data:
                target_recommendations = feature_selector._get_target_recommendations(df)
                if target_recommendations:
                    recommended_target = target_recommendations['recommended_target']
                    logger.info(f"🎯 Recommended target column: {recommended_target}")
                    y = df[recommended_target]
                else:
                    logger.warning("⚠️ No target recommendations available")
                    return jsonify({"error": "No target column specified or recommended"}), 400
            else:
                y = preprocess_target(request_data['target'])
            
            # Set parameters if provided
            if 'target_name' in request_data:
                y.name = request_data['target_name']
                logger.info(f"Target name set to: {y.name}")
            elif target_recommendations:
                y.name = recommended_target
            
            if 'min_features' in request_data:
                feature_selector.min_features_to_keep = int(request_data['min_features'])
                logger.info(f"Minimum features to keep set to: {feature_selector.min_features_to_keep}")
            
            # Configure feature engineering options if provided
            if 'apply_scaling' in request_data:
                feature_selector.apply_scaling = bool(request_data['apply_scaling'])
                logger.info(f"Feature scaling set to: {feature_selector.apply_scaling}")
                
            if 'scaling_method' in request_data:
                if request_data['scaling_method'] in ['standard', 'minmax']:
                    feature_selector.scaling_method = request_data['scaling_method']
                    logger.info(f"Scaling method set to: {feature_selector.scaling_method}")
            
            if 'apply_discretization' in request_data:
                feature_selector.apply_discretization = bool(request_data['apply_discretization'])
                logger.info(f"Feature discretization set to: {feature_selector.apply_discretization}")
                
            if 'n_bins' in request_data:
                feature_selector.n_bins = int(request_data['n_bins'])
                logger.info(f"Discretization bins set to: {feature_selector.n_bins}")
            
            if 'apply_feature_crossing' in request_data:
                feature_selector.apply_feature_crossing = bool(request_data['apply_feature_crossing'])
                logger.info(f"Feature crossing set to: {feature_selector.apply_feature_crossing}")
                
            if 'max_crossing_degree' in request_data:
                feature_selector.max_crossing_degree = int(request_data['max_crossing_degree'])
                logger.info(f"Max crossing degree set to: {feature_selector.max_crossing_degree}")
            
            # Select features
            logger.info("🔄 Starting feature selection process")
            X_selected = feature_selector.fit_transform(df, y)
            logger.info(f"✅ Feature selection complete. Selected {len(feature_selector.selected_features)} features")
            
            # Make sure we're only returning original column names (no _disc suffix)
            selected_features = [feat for feat in feature_selector.selected_features if feat in original_columns]
            logger.info(f"✅ Filtered to {len(selected_features)} original features")
            
            # Add back any missing original features that were discretized
            for col in original_columns:
                if col not in selected_features and col in X_selected.columns:
                    selected_features.append(col)
                    logger.info(f"✅ Added back original feature: {col}")
            
            response_data = {
                "selected_features": selected_features,
                "feature_importances": {
                    k: v for k, v in feature_selector.feature_importances_.items() 
                    if k in selected_features
                },
                "transformed_data": X_selected.to_dict(orient='records'),
                "message": "Features selected successfully",
                "method_used": feature_selector.method,
                "feature_engineering": {
                    "scaling": feature_selector.apply_scaling,
                    "discretization": feature_selector.apply_discretization,
                    "feature_crossing": feature_selector.apply_feature_crossing,
                    "crossed_features": [name for name, _ in feature_selector.crossed_features] if feature_selector.apply_feature_crossing else []
                }
            }
            
            # Include target recommendations if available
            if target_recommendations:
                response_data["target_recommendations"] = target_recommendations
            
            return jsonify(response_data)
        
        except ValueError as e:
            logger.error(f"🔴 Preprocessing error: {str(e)}")
            return jsonify({"error": str(e)}), 400
        
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