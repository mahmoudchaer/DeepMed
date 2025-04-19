from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import logging
import sys
import os
import traceback
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
import openai
from waitress import serve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Set up OpenAI API if environment variable is provided
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI API key configured")
else:
    logger.warning("OpenAI API key not found in environment variables")

# Custom JSON Encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)

# Configure Flask app to use the custom encoder
app.json_encoder = NumpyEncoder

class RegressionFeatureSelector:
    def __init__(self):
        """Initialize the feature selector with regression-focused methods"""
        self.methods = ['mutual_info', 'f_regression', 'lasso', 'random_forest', 'rfe']
        
    def _mutual_info_selection(self, X, y, k=None):
        """Select features using mutual information regression"""
        if k is None:
            k = min(10, X.shape[1])
            
        # Make sure y is numeric
        y = pd.to_numeric(y, errors='coerce')
        
        selector = SelectKBest(mutual_info_regression, k=k)
        selector.fit(X, y)
        
        # Get selected feature indices
        feature_idx = selector.get_support(indices=True)
        
        # Get feature scores
        scores = selector.scores_
        
        # Create importance dictionary
        importance = {X.columns[i]: float(scores[i]) for i in range(len(X.columns))}
        
        # Return selected features and all feature importances
        return X.iloc[:, feature_idx], importance
        
    def _f_regression_selection(self, X, y, k=None):
        """Select features using F-statistic regression (better for linear relationships)"""
        if k is None:
            k = min(10, X.shape[1])
            
        # Make sure y is numeric
        y = pd.to_numeric(y, errors='coerce')
        
        # Apply feature selection without scaling (scaling should be done by the data cleaner)
        selector = SelectKBest(f_regression, k=k)
        selector.fit(X, y)
        
        # Get selected feature indices
        feature_idx = selector.get_support(indices=True)
        
        # Get feature scores
        scores = selector.scores_
        
        # Create importance dictionary
        importance = {X.columns[i]: float(scores[i]) for i in range(len(X.columns))}
        
        # Return selected features and all feature importances
        return X.iloc[:, feature_idx], importance
        
    def _lasso_selection(self, X, y, alpha=0.05):
        """Select features using Lasso regression (L1 regularization)"""
        # Make sure y is numeric
        y = pd.to_numeric(y, errors='coerce')
        
        # Apply Lasso without scaling (scaling should be done by the data cleaner)
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X, y)
        
        # Get feature coefficients
        coefficients = lasso.coef_
        
        # Create importance dictionary
        importance = {X.columns[i]: abs(float(coefficients[i])) for i in range(len(X.columns))}
        
        # Select features with non-zero coefficients
        selected_features = [X.columns[i] for i in range(len(X.columns)) if coefficients[i] != 0]
        
        # Check if we have too few features
        if len(selected_features) < 3 and X.shape[1] > 3:
            # If Lasso selected too few features, use top 3 by importance
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
            selected_features = [f[0] for f in top_features]
            
        # Return selected features
        return X[selected_features], importance
        
    def _random_forest_selection(self, X, y, threshold=0.01):
        """Select features using Random Forest feature importance"""
        # Make sure y is numeric
        y = pd.to_numeric(y, errors='coerce')
        
        # Fit Random Forest model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Create importance dictionary
        importance = {X.columns[i]: float(importances[i]) for i in range(len(X.columns))}
        
        # Select features above threshold
        selected_features = [X.columns[i] for i in range(len(X.columns)) if importances[i] > threshold]
        
        # If no features meet the threshold, take the top 5
        if not selected_features:
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            selected_features = [f[0] for f in top_features]
            
        # Return selected features
        return X[selected_features], importance
        
    def _rfe_selection(self, X, y, n_features=None):
        """Select features using Recursive Feature Elimination with Ridge regression"""
        if n_features is None:
            n_features = max(3, X.shape[1] // 2)  # Select half the features or at least 3
            
        # Make sure y is numeric
        y = pd.to_numeric(y, errors='coerce')
        
        # Use Ridge as estimator for RFE (better for regression than Logistic)
        estimator = Ridge(alpha=1.0)
        
        # Create RFE
        rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)
        
        # Get selected feature indices
        feature_idx = rfe.get_support(indices=True)
        
        # Get feature ranking (lower is better)
        ranking = rfe.ranking_
        
        # Create importance dictionary - convert rankings to importance scores
        # (lower ranking means higher importance)
        max_rank = max(ranking)
        importance = {X.columns[i]: float(max_rank - ranking[i] + 1) for i in range(len(X.columns))}
        
        # Return selected features
        return X.iloc[:, feature_idx], importance
        
    def _aggregate_feature_importances(self, importances_list):
        """Aggregate feature importances from multiple methods"""
        # Initialize aggregate importance
        aggregate_importance = {}
        
        # For each feature, compute the average of its normalized importance across methods
        all_features = set()
        for importances in importances_list:
            all_features.update(importances.keys())
            
        for feature in all_features:
            # Collect importances for this feature from each method
            feature_importances = []
            for importances in importances_list:
                if feature in importances:
                    feature_importances.append(importances[feature])
            
            # Compute average importance if the feature was selected by any method
            if feature_importances:
                aggregate_importance[feature] = sum(feature_importances) / len(feature_importances)
            else:
                aggregate_importance[feature] = 0.0
                
        return aggregate_importance
        
    def _llm_feature_analysis(self, X, y, target_name, all_importances, selected_features):
        """Use LLM to analyze feature selection and suggest improvements for regression"""
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not available, skipping LLM feature analysis")
            return None
            
        try:
            # Sample data for LLM
            sample_rows = min(5, len(X))
            sample_data = X.head(sample_rows)
            if isinstance(y, pd.Series):
                sample_data = pd.concat([sample_data, y.head(sample_rows)], axis=1)
            
            # Create prompt
            prompt = f"""
            You are a feature selection specialist focusing on regression problems in healthcare.
            
            I'm working on a regression model with target variable: {target_name}
            
            Here's a sample of the data (first {sample_rows} rows):
            {sample_data.to_string()}
            
            Based on several feature selection methods, I've calculated these feature importances:
            {json.dumps(all_importances, indent=2)}
            
            And selected these features:
            {', '.join(selected_features)}
            
            Please provide:
            1. An analysis of the selected features and their relevance for predicting {target_name}
            2. Suggestions for any additional features that might improve regression performance
            3. Potential feature interactions or polynomial terms that could be valuable
            4. Comments on multicollinearity concerns among the selected features
            5. Recommendations for further feature engineering specific to this regression task
            
            Focus on creating a powerful and accurate regression model.
            """
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4-0125-preview",
                messages=[{"role": "system", "content": "You are a feature selection expert for regression problems."},
                          {"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.3
            )
            
            # Extract suggestions
            llm_analysis = response.choices[0].message.content
            logger.info(f"Received LLM feature analysis: {llm_analysis[:100]}...")
            
            return llm_analysis
            
        except Exception as e:
            logger.error(f"Error getting LLM feature analysis: {str(e)}")
            return f"Error getting LLM feature analysis: {str(e)}"
    
    def select_features(self, data, target, target_name=None, options=None):
        """
        Select features for regression.

        Args:
            data: Input data as DataFrame or convertible to DataFrame
            target: Target variable as Series or array-like
            target_name: Name of the target variable
            options: Dictionary of options for feature selection

        Returns:
            X_transformed: DataFrame of selected features
            selected_features: List of selected feature names
            report: Dictionary with summary of feature selection process
            feature_importances: Dictionary of feature importances by method
            llm_analysis: Analysis from LLM if available
        """
        # Set default options
        default_options = {
            'select_features': True,
            'methods': ['statistical', 'tree_based', 'lasso'],
            'num_features': 10,
            'use_llm': False,
            'threshold': 0.1,
            'vif_threshold': 5.0,
            'remove_multicollinearity': True,
            'auto_select': True
        }
        
        # Merge with user options
        if options:
            for key, value in options.items():
                default_options[key] = value
        
        options = default_options
                
        # Ensure data is DataFrame
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
                logger.info(f"Converted data to DataFrame with shape {data.shape}")
            except Exception as e:
                logger.error(f"Could not convert data to DataFrame: {str(e)}")
                raise ValueError("Input data must be a DataFrame or convertible to DataFrame")
        
        # Ensure target is Series
        if not isinstance(target, pd.Series):
            try:
                target = pd.Series(target, name=target_name if target_name else "target")
                logger.info(f"Converted target to Series with length {len(target)}")
            except Exception as e:
                logger.error(f"Could not convert target to Series: {str(e)}")
                raise ValueError("Target must be a Series or convertible to Series")
        
        logger.info(f"Starting feature selection process with {data.shape[1]} features")
        
        # Capture column names
        feature_names = data.columns.tolist()
        target_name = target.name if target.name else target_name if target_name else "target"
        
        # Initialize report
        report = {
            "initial_features": feature_names,
            "steps": [],
            "selected_features": [],
            "method_scores": {}
        }
        
        # Skip feature selection if requested or if there's only one feature
        if not options['select_features'] or len(feature_names) <= 1:
            logger.info(f"Skipping feature selection, returning all {len(feature_names)} features")
            report["steps"].append({
                "step": "skip_feature_selection",
                "message": "Skipping feature selection as requested or only one feature available"
            })
            report["selected_features"] = feature_names
            return data, feature_names, report, {}, None
        
        # Store original feature count
        original_feature_count = data.shape[1]
        
        # Run feature selection methods
        selected_features_dict = {}
        importances_dict = {}
        
        # Get methods to use
        methods = options.get('methods', self.methods)
        
        for method in methods:
            try:
                if method == 'mutual_info':
                    selected_X, importance = self._mutual_info_selection(data, target)
                elif method == 'f_regression':
                    selected_X, importance = self._f_regression_selection(data, target)
                elif method == 'lasso':
                    selected_X, importance = self._lasso_selection(data, target)
                elif method == 'random_forest':
                    selected_X, importance = self._random_forest_selection(data, target)
                elif method == 'rfe':
                    selected_X, importance = self._rfe_selection(data, target)
                else:
                    logger.warning(f"Unknown method: {method}, skipping")
                    continue
                    
                selected_features_dict[method] = selected_X.columns.tolist()
                importances_dict[method] = importance
                
                logger.info(f"Method {method} selected {len(selected_features_dict[method])} features")
                
            except Exception as e:
                logger.error(f"Error running method {method}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        # Combine results from all methods to get final selected features
        # Count how many methods selected each feature
        feature_counts = {}
        for method, features in selected_features_dict.items():
            for feature in features:
                if feature in feature_counts:
                    feature_counts[feature] += 1
                else:
                    feature_counts[feature] = 1
        
        # Calculate the threshold for selection (how many methods need to select a feature)
        # More aggressive = higher threshold
        if options.get('aggressive', True):
            methods_threshold = max(2, len(selected_features_dict) // 2)
        else:
            methods_threshold = max(1, len(selected_features_dict) // 3)
            
        # Select features that meet the threshold
        final_selected_features = [feature for feature, count in feature_counts.items() 
                                  if count >= methods_threshold]
        
        # If too few features are selected, use the top features by aggregate importance
        if len(final_selected_features) < 3 and original_feature_count > 3:
            logger.info("Too few features selected, using aggregate importance ranking")
            
            # Aggregate importances across methods
            aggregate_importance = self._aggregate_feature_importances(importances_dict.values())
            
            # Sort features by importance
            sorted_features = sorted(aggregate_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 5 features or half of original features, whichever is smaller
            top_count = min(5, original_feature_count // 2)
            final_selected_features = [feature for feature, _ in sorted_features[:top_count]]
        
        # Get LLM analysis if requested
        llm_analysis = None
        if options.get('use_llm', True) and OPENAI_API_KEY:
            aggregate_importance = self._aggregate_feature_importances(importances_dict.values())
            
            # Keep only final selected features in X
            X_selected = data[final_selected_features]
            
            llm_analysis = self._llm_feature_analysis(
                X_selected, target, target_name, aggregate_importance, final_selected_features
            )
        
        # Create selection report
        selection_report = {
            'original_features': original_feature_count,
            'selected_features': len(final_selected_features),
            'methods_used': list(selected_features_dict.keys()),
            'method_results': {method: len(features) for method, features in selected_features_dict.items()}
        }
        
        # Filter data to only include selected features
        X_transformed = data[final_selected_features]
        
        # Return the selected feature data and metadata
        return X_transformed, final_selected_features, selection_report, importances_dict, llm_analysis


# Create a feature selector instance
feature_selector = RegressionFeatureSelector()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "regression_feature_selector"
    })

@app.route('/select_features', methods=['POST'])
def select_features():
    """
    API endpoint to select features for regression model
    
    Required JSON format:
    {
        "data": [...], # list of records or dict
        "target": [...], # list of target values
        "target_name": "target_column_name", # optional
        "options": {...} # optional configuration
    }
    """
    try:
        # Get JSON data
        request_data = request.get_json(force=True)
        
        # Log request info
        logger.info(f"Received feature selection request with options: {request_data.get('options', {})}")
        
        # Check if required fields are present
        if 'data' not in request_data or 'target' not in request_data:
            return jsonify({"error": "Missing required fields: data and target"}), 400
            
        # Extract fields
        data = request_data['data']
        target = request_data['target']
        target_name = request_data.get('target_name')
        options = request_data.get('options', {})
        
        # Check data format and size
        logger.info(f"Data type: {type(data).__name__}")
        if isinstance(data, list):
            logger.info(f"Data contains {len(data)} records with {len(data[0]) if data and data[0] else 0} features")
        elif isinstance(data, dict):
            logger.info(f"Data contains {len(next(iter(data.values()))) if data else 0} records")
        
        # Convert data to DataFrame if not already
        try:
            if not isinstance(data, pd.DataFrame):
                data_df = pd.DataFrame(data)
                logger.info(f"Converted data to DataFrame with shape {data_df.shape}")
            else:
                data_df = data
        except Exception as e:
            logger.error(f"Error converting data to DataFrame: {str(e)}")
            return jsonify({"error": f"Could not process data: {str(e)}"}), 400
        
        # Convert target to Series if not already
        try:
            if not isinstance(target, pd.Series):
                target_series = pd.Series(target, name=target_name if target_name else "target")
                logger.info(f"Converted target to Series with length {len(target_series)}")
            else:
                target_series = target
        except Exception as e:
            logger.error(f"Error converting target to Series: {str(e)}")
            return jsonify({"error": f"Could not process target: {str(e)}"}), 400
        
        # Run feature selection
        X_transformed, selected_features, report, feature_importances, llm_analysis = feature_selector.select_features(
            data_df, target_series, target_name, options
        )
        
        # Convert DataFrame to dict for JSON serialization
        transformed_data = X_transformed.to_dict(orient='records')
        
        # Ensure feature importances are JSON serializable
        for method, importances in feature_importances.items():
            if isinstance(importances, pd.Series):
                feature_importances[method] = importances.to_dict()
            elif isinstance(importances, np.ndarray):
                feature_importances[method] = importances.tolist()
            
        # Prepare response
        response = {
            "transformed_data": transformed_data,
            "selected_features": selected_features,
            "report": report,
            "feature_importances": feature_importances
        }
        
        if llm_analysis:
            response["llm_analysis"] = llm_analysis
            
        return json.dumps(response, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}
        
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5032))
    logger.info(f"Starting Regression Feature Selector Service on port {port}")
    serve(app, host='0.0.0.0', port=port) 