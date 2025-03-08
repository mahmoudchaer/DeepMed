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
from openai import OpenAI
import os
import logging

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
            logging.warning("OPENAI_API_KEY environment variable not set. Using fallback method.")
            self.method = 'fallback'
            self.client = None
        else:
            try:
                # Initialize with minimal configuration to avoid proxy issues
                self.client = OpenAI(api_key=api_key)
                self.method = 'llm'
                logging.info("OpenAI client initialized successfully for feature selection")
            except Exception as e:
                logging.error(f"Error initializing OpenAI client for feature selection: {e}")
                self.method = 'fallback'
                self.client = None
        
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
            logging.info("Using fallback method for feature selection")
            return {
                "features_to_keep": list(X.columns),
                "features_to_remove": [],
                "reasoning": {}
            }
            
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
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            recommendations = json.loads(response.choices[0].message.content)
            return recommendations
        except Exception as e:
            logging.error(f"Error getting LLM recommendations: {str(e)}")
            return {
                "features_to_keep": list(X.columns),
                "features_to_remove": [],
                "reasoning": {}
            }

    def fit_transform(self, X, y=None):
        """Fit the feature selector and transform the data."""
        X = X.copy()
        
        # Prepare features (handle categorical variables)
        X = self._prepare_features(X)
        
        if self.method == 'llm':
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
                    logging.info(f"Removing feature '{feature}': {reason}")
            
            # Ensure we keep minimum number of features
            if len(self.selected_features) < self.min_features_to_keep:
                logging.warning(f"Too few features recommended. Keeping top {self.min_features_to_keep} features based on correlation with target.")
                correlations = X.corrwith(y) if y is not None else pd.Series(1.0, index=X.columns)
                self.selected_features = correlations.abs().nlargest(self.min_features_to_keep).index.tolist()
        else:
            # Fallback to original maximize method
            self.selected_features = self._maximize_features(X, y)
        
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
                if not all(val in self.label_encoders[column].classes_ for val in unique_values):
                    X[column] = X[column].map(lambda x: x if x in self.label_encoders[column].classes_ else self.label_encoders[column].classes_[0])
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
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = LogisticRegression(max_iter=1000)
        scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx][features], X.iloc[val_idx][features]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            scores.append(accuracy_score(y_val, model.predict(X_val)))
            
        return np.mean(scores)
    
    def _maximize_features(self, X, y):
        """Maximize feature retention while ensuring performance"""
        logging.info("Using maximize_features method for feature selection")
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
        logging.info(f"Feature Selection Summary:")
        logging.info(f"Baseline performance: {baseline_performance:.4f}")
        logging.info(f"Final performance: {best_performance:.4f}")
        logging.info(f"Features retained: {len(best_features)}/{len(X.columns)}")
        
        return best_features 