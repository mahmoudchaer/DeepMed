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
import re

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
        """Initialize the feature selector."""
        self.method = method
        self.min_features_to_keep = min_features_to_keep
        self.performance_threshold = performance_threshold
        self.apply_scaling = apply_scaling
        self.scaling_method = scaling_method
        self.apply_discretization = apply_discretization
        self.n_bins = n_bins
        self.apply_feature_crossing = apply_feature_crossing
        self.max_crossing_degree = max_crossing_degree
        
        # Store selected features
        self.selected_features = None
        
        # Store feature importances
        self.feature_importances_ = {}
        
        # Store scalers for numeric columns
        self.scalers = {}
        
        # Track all identified problematic columns for comprehensive filtering
        self.problematic_columns = []
        
        # Store encoders for categorical columns
        self.encoders = {}
        
        # Store encoding mappings for categorical columns
        self.encoding_mappings = {}
        
        # Track column metadata for better filtering decisions
        self.column_metadata = {}
        
        # OpenAI client if available
        self.client = None
        if method == 'llm':
            try:
                import openai
                self.client = openai
                self._test_openai_connection()
                
                # Set a more specific system prompt for feature selection
                self.system_prompt = """
You are an advanced feature selection specialist for machine learning. Your expertise is in identifying which features (columns) in a dataset are relevant for prediction and which should be removed.

ALWAYS follow these principles:
1. Be EXTREMELY EXPLICIT about identifying unnecessary features that should be removed.
2. Eliminate irrelevant features that don't contribute to the prediction task.
3. Remove high-risk features that could lead to data leakage or overfitting.
4. Be especially vigilant about removing:
   - Identifiers (names, IDs, patient numbers, unique identifiers, rooms, doctors, hospitals)
   - Administrative metadata (record numbers, file identifiers)
   - Temporal information not available at prediction time (future dates, timestamps created after the target event)
   - Free-text fields with high cardinality (comments, notes, addresses)
   - Redundant or highly correlated features that don't add new information

Your recommendations must be SPECIFIC to the dataset you're analyzing and NOT based on fixed rules. Examine the actual data characteristics when making decisions.

You MUST explicitly list ALL features that should be removed with clear reasoning for each. These will be DIRECTLY removed from the dataset without further validation.
"""
                logger.info("🟢 OpenAI client initialized successfully")
            except ImportError:
                logger.warning("⚠️ OpenAI package not installed, falling back to traditional method")
                self.method = 'fallback'
            except Exception as e:
                logger.error(f"🔴 Error initializing OpenAI client: {str(e)}")
                self.method = 'fallback'

    def _test_openai_connection(self):
        """Test the OpenAI connection to make sure it's working properly."""
        try:
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "Please respond with 'OK' if you can read this message."}
                ],
                max_tokens=5
            )
            
            logger.info("🟢 Successfully tested OpenAI connection")
            return True
        except Exception as e:
            logger.error(f"🔴 Error testing OpenAI connection: {str(e)}")
            return False

    def _get_llm_recommendations(self, X, y=None, target_name=None):
        """Get feature recommendations from LLM."""
        if self.method == 'fallback' or self.client is None:
            logger.info("🔄 LLM not available for feature recommendations")
            return {
                "features_to_keep": list(X.columns),
                "features_to_remove": [],
                "reasoning": {}
            }
            
        logger.info("🤖 Getting feature recommendations from LLM")
        
        # Prepare data summary for LLM
        data_summary = {
            "target_column": target_name,
            "sample_size": len(X),
            "columns": {
                col: {
                    "dtype": str(X[col].dtype),
                    "unique_values": int(X[col].nunique()),
                    "unique_ratio": float(X[col].nunique() / len(X)) if len(X) > 0 else 0,
                    "missing_values": int(X[col].isnull().sum()),
                    "missing_ratio": float(X[col].isnull().sum() / len(X)) if len(X) > 0 else 0,
                    "sample_values": list(X[col].dropna().sample(min(5, max(1, len(X[col].dropna())))).astype(str)),
                    "correlation_with_target": float(X[col].corr(y)) if y is not None and pd.api.types.is_numeric_dtype(X[col]) else None
                } for col in X.columns
            }
        }
        
        # Add column type hints to help the LLM
        for col in X.columns:
            col_data = data_summary["columns"][col]
            col_values = X[col].dropna().astype(str).tolist()[:20]  # Sample for analysis
            
            # Check for potential datetime patterns
            date_patterns = 0
            for val in col_values:
                if isinstance(val, str) and re.search(r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}', val):
                    date_patterns += 1
            
            if date_patterns > min(3, len(col_values) * 0.3):
                col_data["likely_date"] = True
            
            # Check for high uniqueness (potential identifiers)
            if col_data["unique_ratio"] > 0.8 and col_data["unique_values"] > 10:
                col_data["potential_identifier"] = True
                
            # Check for potential name patterns
            name_like = False
            name_patterns = ['name', 'patient', 'doctor', 'provider', 'physician', 'person']
            if any(pattern in col.lower() for pattern in name_patterns):
                name_like = True
                col_data["name_like"] = True
        
        logger.debug(f"Prepared data summary for LLM with {len(X.columns)} features")
        
        prompt = f"""Analyze this dataset and make data-driven decisions about which features to keep or remove.

Dataset Summary: {json.dumps(data_summary, indent=2)}

YOUR TASK:
Analyze each column and determine if it should be kept for model training or removed.

CRITICAL GUIDELINES:
1. You MUST EXPLICITLY identify ALL features to remove and provide clear reasoning.
2. REMOVE any columns that could lead to data leakage or don't contribute to prediction:
   - Unique identifiers (IDs, record numbers, primary keys)
   - Names of people or entities (patient names, doctor names, hospital names)
   - Dates that would not be available at prediction time
   - Administrative metadata unrelated to the prediction task
   - Free-text fields with extremely high cardinality
   - Columns with too many missing values to be useful

3. KEEP columns that contain predictive information:
   - Clinical/domain measurements related to the prediction task
   - Properly encoded categorical variables with reasonable cardinality
   - Features with meaningful correlation to the target (if available)
   - Transformed datetime features that represent cyclical patterns

Your analysis must be specific to THIS dataset's characteristics, not based on generic rules.
All features you identify for removal will be DIRECTLY removed without further validation.

Provide your recommendations in this exact JSON format:
{{
    "features_to_keep": ["feature1", "feature2", ...],
    "features_to_remove": ["feature3", "feature4", ...],
    "reasoning": {{
        "feature3": "Specific reason for removal",
        "feature4": "Specific reason for removal",
        ...
    }}
}}

Respond ONLY with valid JSON, no additional text or explanations outside the JSON structure."""

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
                
                # Directly use LLM recommendations without extensive validation
                # This ensures explicit removal of features identified by the LLM
                logger.info(f"🟢 Successfully parsed LLM recommendations")
                logger.info(f"Features to keep: {len(recommendations['features_to_keep'])}")
                logger.info(f"Features to remove: {len(recommendations['features_to_remove'])}")
                
                # Log each feature the LLM has decided to remove with its reasoning
                logger.info("🔍 LLM REMOVAL DECISIONS:")
                for feature in recommendations['features_to_remove']:
                    reason = recommendations['reasoning'].get(feature, "No specific reason provided")
                    logger.info(f"  • LLM marked '{feature}' for removal: {reason}")
                
                # Add any columns not explicitly classified to features_to_remove
                all_features = set(X.columns)
                classified_features = set(recommendations['features_to_keep'] + recommendations['features_to_remove'])
                unclassified_features = all_features - classified_features
                
                if unclassified_features:
                    logger.warning(f"⚠️ {len(unclassified_features)} features not classified by LLM. Adding to removal list.")
                    logger.info("🔍 UNCLASSIFIED FEATURES BEING REMOVED:")
                    for feature in unclassified_features:
                        recommendations['features_to_remove'].append(feature)
                        recommendations["reasoning"][feature] = "Not explicitly classified by LLM, assuming irrelevant"
                        logger.info(f"  • Adding unclassified feature '{feature}' to removal list")
                
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

    def _identify_problematic_columns(self, X):
        """Comprehensive identification of problematic columns that should be excluded from modeling.
        
        This function identifies columns that are likely to be irrelevant or cause issues:
        1. ID columns and unique identifiers
        2. Name columns (patient, doctor, etc.)
        3. Date/time columns (unless properly transformed)
        4. High cardinality string columns
        5. Low information columns
        """
        logger.info("🔍 Identifying problematic columns for exclusion")
        problematic_cols = []
        column_metadata = {}
        
        total_rows = len(X)
        if total_rows == 0:
            logger.warning("Empty dataframe provided, no problematic columns to identify")
            return [], {}
        
        # Patterns for various problematic column types
        id_patterns = [
            'id', '_id', 'uuid', 'guid', 'identifier', 'record', 'key', 'number', 'nr', 'code',
            'patient_id', 'user_id', 'customer_id', 'client_id', 'employee_id', 'account', 'reference'
        ]
        
        name_patterns = [
            'name', 'firstname', 'lastname', 'fullname', 'patient_name', 'doctor', 'physician', 
            'provider', 'customer', 'client', 'person', 'user', 'employee', 'nurse', 'staff'
        ]
        
        date_patterns = [
            'date', 'time', 'datetime', 'timestamp', 'day', 'month', 'year', 'admission',
            'discharge', 'visit', 'birthday', 'dob', 'created', 'modified', 'updated'
        ]
        
        contact_patterns = [
            'address', 'email', 'phone', 'contact', 'ssn', 'social', 'insurance',
            'zip', 'postal', 'city', 'state', 'country', 'street', 'fax'
        ]
        
        note_patterns = [
            'notes', 'comment', 'description', 'remark', 'observation', 'text',
            'detail', 'explanation', 'reason', 'summary', 'history'
        ]
        
        url_patterns = [
            'url', 'link', 'website', 'web', 'http', 'https', 'www', 'uri', 'path', 'file'
        ]
        
        # Combine all patterns
        all_patterns = {
            'id': id_patterns,
            'name': name_patterns,
            'date': date_patterns,
            'contact': contact_patterns,
            'note': note_patterns,
            'url': url_patterns
        }
        
        for col in X.columns:
            is_problematic = False
            reasons = []
            metadata = {
                "name": col,
                "dtype": str(X[col].dtype),
                "is_numeric": pd.api.types.is_numeric_dtype(X[col]),
                "is_categorical": (not pd.api.types.is_numeric_dtype(X[col])),
                "unique_count": int(X[col].nunique()),
                "unique_ratio": float(X[col].nunique() / total_rows) if total_rows > 0 else 0,
                "missing_count": int(X[col].isnull().sum()),
                "missing_ratio": float(X[col].isnull().sum() / total_rows) if total_rows > 0 else 0,
                "problematic": False,
                "reasons": []
            }
            
            # 1. Check name patterns
            col_lower = col.lower()
            for pattern_type, patterns in all_patterns.items():
                if any(pattern in col_lower for pattern in patterns):
                    reason = f"Column name contains {pattern_type}-like pattern"
                    reasons.append(reason)
                    metadata["pattern_match"] = pattern_type
                    
                    # For certain patterns, we need additional validation from the data
                    if pattern_type == 'id' or pattern_type == 'name':
                        # IDs and names typically have high cardinality
                        if metadata["unique_ratio"] > 0.2:
                            is_problematic = True
                            reasons.append(f"High cardinality ({metadata['unique_ratio']:.2f}) consistent with {pattern_type}")
                    elif pattern_type == 'date':
                        # Check if data looks like dates
                        sample_vals = X[col].dropna().astype(str).sample(min(5, len(X[col].dropna()))).tolist()
                        date_like = any(re.search(r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}', str(val)) for val in sample_vals)
                        if date_like:
                            is_problematic = True
                            reasons.append("Contains date patterns which may require special handling")
                    else:
                        # For other patterns, flag as potentially problematic
                        is_problematic = True
            
            # 2. Check cardinality for non-numeric columns (potential identifiers)
            if not pd.api.types.is_numeric_dtype(X[col]):
                # Very high cardinality in categorical column = likely ID or free text
                if metadata["unique_ratio"] > 0.5 and metadata["unique_count"] > 10:
                    is_problematic = True
                    reasons.append(f"High cardinality categorical column ({metadata['unique_ratio']:.2f})")
                
                # Special case: if every value is unique, it's almost certainly an ID
                if metadata["unique_ratio"] > 0.95:
                    is_problematic = True
                    reasons.append("Nearly unique values, likely an identifier")
            
            # 3. Check for low information columns
            if metadata["unique_count"] == 1:
                is_problematic = True
                reasons.append("Constant column (only one unique value)")
            
            # 4. Check for mostly missing data
            if metadata["missing_ratio"] > 0.8:
                is_problematic = True
                reasons.append(f"Mostly missing data ({metadata['missing_ratio']:.2f})")
            
            # Record results
            if is_problematic:
                problematic_cols.append(col)
                metadata["problematic"] = True
                metadata["reasons"] = reasons
                logger.info(f"🚫 Identified problematic column: '{col}' - {', '.join(reasons)}")
            
            # Store metadata for all columns
            column_metadata[col] = metadata
        
        logger.info(f"✅ Identified {len(problematic_cols)}/{len(X.columns)} problematic columns")
        return problematic_cols, column_metadata

    def _validate_categorical_columns(self, X):
        """Validate categorical columns and ensure they have proper encoding mappings.
        
        This function:
        1. Identifies categorical columns
        2. Checks if they need encoding
        3. Creates encoding mappings if needed
        4. Flags columns that are problematic for encoding
        """
        logger.info("🔍 Validating categorical columns for encoding")
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        logger.info(f"Found {len(categorical_cols)} categorical columns")
        
        # Track columns that need encoding and those that are problematic
        needs_encoding = []
        encoding_issues = []
        
        # Generate encoding mappings
        for col in categorical_cols:
            # Skip if column is already identified as problematic
            if col in self.problematic_columns:
                continue
                
            # Check cardinality - if too many unique values, it's problematic
            unique_values = X[col].dropna().unique()
            unique_ratio = len(unique_values) / len(X) if len(X) > 0 else 0
            
            # Very high cardinality in categorical column = likely ID or free text
            if unique_ratio > 0.5 and len(unique_values) > 10:
                encoding_issues.append({
                    "column": col,
                    "reason": f"High cardinality ({unique_ratio:.2f}, {len(unique_values)} unique values)",
                    "unique_ratio": unique_ratio,
                    "unique_count": len(unique_values)
                })
                self.problematic_columns.append(col)
                continue
            
            # Otherwise, column needs encoding
            needs_encoding.append(col)
            
            # Create encoding mapping if it doesn't exist
            if col not in self.encoding_mappings:
                # Sort values to ensure consistent encoding
                sorted_values = sorted([str(x) for x in unique_values if pd.notna(x)])
                mapping = {val: idx for idx, val in enumerate(sorted_values)}
                self.encoding_mappings[col] = mapping
                logger.info(f"Created encoding mapping for '{col}' with {len(mapping)} values")
        
        if encoding_issues:
            logger.warning(f"⚠️ Found {len(encoding_issues)} categorical columns with encoding issues")
            for issue in encoding_issues:
                logger.warning(f"  - {issue['column']}: {issue['reason']}")
        
        logger.info(f"✅ {len(needs_encoding)} categorical columns validated for encoding")
        return needs_encoding, encoding_issues

    def fit_transform(self, X, y=None):
        """Fit the feature selector and transform the data with improved validation."""
        logger.info(f"Starting feature selection using method: {self.method}")
        logger.info(f"Input data shape: {X.shape}")
        
        X = X.copy()
        original_columns = list(X.columns)
        
        # STAGE 1: Pre-selection validation - identify problematic columns
        # =================================================================
        self.problematic_columns, self.column_metadata = self._identify_problematic_columns(X)
        
        # STAGE 2: Validate categorical columns for encoding
        # =================================================
        valid_categorical_cols, encoding_issues = self._validate_categorical_columns(X)
        
        # STAGE 3: Feature selection based on method
        # =========================================
        if self.method == 'llm':
            logger.info("🤖 Using LLM-based feature selection")
            
            # Get LLM recommendations (features are DIRECTLY removed as specified by the LLM)
            recommendations = self._get_llm_recommendations(X, y, target_name=y.name if hasattr(y, 'name') else None)
            
            # Extract the final lists
            features_to_keep = recommendations['features_to_keep']
            features_to_remove = recommendations['features_to_remove']
            
            # Log detailed information about features being removed
            logger.info("=" * 80)
            logger.info("🚫 FEATURES MARKED FOR REMOVAL BY LLM:")
            logger.info("=" * 80)
            
            if len(features_to_remove) > 0:
                for feature in features_to_remove:
                    reason = recommendations['reasoning'].get(feature, "No specific reason provided")
                    logger.info(f"REMOVING: '{feature}' - Reason: {reason}")
            else:
                logger.info("No features were explicitly marked for removal by the LLM")
            
            # Log unclassified features if any
            all_features = set(X.columns)
            classified_features = set(features_to_keep + features_to_remove)
            unclassified_features = all_features - classified_features
            
            if unclassified_features:
                logger.info("-" * 80)
                logger.info(f"⚠️ UNCLASSIFIED FEATURES (BEING REMOVED):")
                logger.info("-" * 80)
                for feature in unclassified_features:
                    logger.info(f"REMOVING: '{feature}' - Reason: Not explicitly classified by LLM, assuming irrelevant")
            
            logger.info("=" * 80)
            logger.info(f"✅ SUMMARY: Removing {len(features_to_remove) + len(unclassified_features)} features, keeping {len(features_to_keep)} features")
            logger.info("=" * 80)
            
            # Store selected features
            self.selected_features = features_to_keep
            
            # Calculate and store feature importances based on LLM decisions
            for feature in original_columns:
                if feature in features_to_keep:
                    self.feature_importances_[feature] = 1.0
                else:
                    self.feature_importances_[feature] = 0.0
                    if feature in features_to_remove:
                        reason = recommendations['reasoning'].get(feature, "Removed based on analysis")
                        logger.info(f"🔍 Removing feature '{feature}': {reason}")
                    else:
                        logger.info(f"🔍 Feature '{feature}' not classified, assuming irrelevant")
        else:
            logger.info("📊 Using traditional feature selection method")
            
            # First exclude problematic columns before statistical feature selection
            X_filtered = X.drop(columns=self.problematic_columns, errors='ignore')
            logger.info(f"Removed {len(self.problematic_columns)} problematic columns before statistical selection")
            
            if len(X_filtered.columns) == 0:
                logger.warning("⚠️ No columns left after removing problematic ones. Using original data.")
                X_filtered = X
            
            # Apply traditional feature selection
            selected_features = self._maximize_features(X_filtered, y)
            self.selected_features = selected_features
        
        # STAGE 4: Ensure minimum features and handle encoding
        # ==================================================
        # Ensure we keep minimum number of features
        if len(self.selected_features) < self.min_features_to_keep:
            logger.warning(f"⚠️ Too few features recommended ({len(self.selected_features)}). Keeping top {self.min_features_to_keep} features based on correlation with target.")
            
            # If target is available, use correlation
            if y is not None:
                correlations = {}
                for col in original_columns:
                    # Skip explicitly identified problematic columns
                    if col in self.problematic_columns:
                        continue
                        
                    try:
                        if pd.api.types.is_numeric_dtype(X[col]):
                            correlations[col] = abs(X[col].corr(y))
                        else:
                            # Use chi-squared for categorical features
                            from scipy.stats import chi2_contingency
                            contingency = pd.crosstab(X[col], y)
                            chi2, p, dof, expected = chi2_contingency(contingency)
                            correlations[col] = 1 - p  # Higher = more significant relationship
                    except:
                        correlations[col] = 0
                
                # Select top features by correlation
                self.selected_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:self.min_features_to_keep]
                self.selected_features = [col for col, corr in self.selected_features]
            else:
                # If no target, keep features with lowest missing values
                missing_counts = {col: X[col].isna().sum() for col in original_columns if col not in self.problematic_columns}
                self.selected_features = sorted(missing_counts.items(), key=lambda x: x[1])[:self.min_features_to_keep]
                self.selected_features = [col for col, miss in self.selected_features]
        
        # STAGE 5: Final validation of selected features
        # =============================================
        # Validate that categorical features in final selection have encoding mappings
        missing_encodings = []
        for col in self.selected_features:
            if col in X.select_dtypes(include=['object', 'category']).columns and col not in self.encoding_mappings:
                # Create encoding on the fly
                unique_values = sorted([str(x) for x in X[col].dropna().unique() if pd.notna(x)])
                self.encoding_mappings[col] = {val: idx for idx, val in enumerate(unique_values)}
                logger.info(f"Created missing encoding mapping for selected column '{col}'")
        
        logger.info(f"✅ Feature selection complete. Selected {len(self.selected_features)} features")
        logger.info(f"Selected features: {self.selected_features}")
        
        # Generate feature metadata for downstream components
        feature_metadata = self._generate_feature_metadata(X, self.selected_features)
        self.feature_metadata = feature_metadata
        
        # Add encoding mappings to metadata
        feature_metadata["encoding_mappings"] = self.encoding_mappings
        feature_metadata["column_metadata"] = self.column_metadata
        
        # Return the transformed dataset with only selected features
        return X[self.selected_features]

    def _generate_feature_metadata(self, X, selected_features):
        """Generate metadata about selected features for downstream components."""
        metadata = {
            "categorical_features": [],
            "numeric_features": [],
            "date_features": [],
            "feature_details": {}
        }
        
        for col in selected_features:
            # Skip if column doesn't exist
            if col not in X.columns:
                continue
                
            # Determine feature type
            if pd.api.types.is_numeric_dtype(X[col]):
                metadata["numeric_features"].append(col)
                feature_type = "numeric"
            else:
                # Check if it might be a date
                sample_vals = X[col].dropna().astype(str).sample(min(5, len(X[col].dropna()))).tolist()
                date_like = any(re.search(r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}', str(val)) for val in sample_vals)
                
                if date_like and any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'month', 'year']):
                    metadata["date_features"].append(col)
                    feature_type = "date"
                else:
                    metadata["categorical_features"].append(col)
                    feature_type = "categorical"
            
            # Store detailed feature info
            metadata["feature_details"][col] = {
                "type": feature_type,
                "dtype": str(X[col].dtype),
                "unique_count": int(X[col].nunique()),
                "missing_count": int(X[col].isnull().sum()),
                "requires_encoding": feature_type == "categorical"
            }
        
        # Log summary of feature types
        logger.info(f"Feature type summary:")
        logger.info(f"  - Numeric features: {len(metadata['numeric_features'])}")
        logger.info(f"  - Categorical features: {len(metadata['categorical_features'])}")
        logger.info(f"  - Date features: {len(metadata['date_features'])}")
        
        return metadata

    def transform(self, X):
        """Transform new data using previously selected features and encoding.
        
        This applies:
        1. Filtering to only selected features
        2. Ensures consistent encoding of categorical features
        3. Applies any preprocessing transformations
        """
        logger.info(f"Transforming new data with shape {X.shape}")
        
        # Make a copy to avoid modifying the original
        X_transformed = X.copy()
        
        # Check if features have been selected
        if not self.selected_features:
            logger.warning("No features have been selected yet. Returning original data.")
            return X
        
        # Identify which selected features are present in the data
        available_features = [col for col in self.selected_features if col in X.columns]
        
        if len(available_features) < len(self.selected_features):
            missing_features = set(self.selected_features) - set(available_features)
            logger.warning(f"Missing {len(missing_features)} expected features: {missing_features}")
        
        if not available_features:
            logger.error("None of the selected features are present in the input data")
            return pd.DataFrame()
        
        # Filter to only include selected features
        X_filtered = X_transformed[available_features].copy()
        
        # Apply encoding to categorical columns
        if hasattr(self, 'encoding_mappings') and self.encoding_mappings:
            categorical_cols = X_filtered.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols:
                if col in self.encoding_mappings:
                    # Get the encoding mapping
                    mapping = self.encoding_mappings[col]
                    
                    # Convert column values to strings to ensure consistent mapping
                    col_values = X_filtered[col].astype(str)
                    
                    # Apply mapping with fallback to -1 for values not in the mapping
                    # This handles new values not seen during training
                    X_filtered[col] = col_values.map(mapping).fillna(-1).astype(int)
                    logger.info(f"Applied encoding mapping to column '{col}'")
                else:
                    logger.warning(f"No encoding mapping found for categorical column '{col}'")
                    # Create a basic label encoding as fallback
                    X_filtered[col] = X_filtered[col].astype('category').cat.codes
                    logger.info(f"Applied fallback encoding to column '{col}'")
        
        # Apply other preprocessing steps if configured
        if self.apply_scaling:
            X_filtered = self._apply_scaling(X_filtered, is_fitting=False)
        
        if self.apply_discretization:
            X_filtered = self._apply_discretization(X_filtered, is_fitting=False)
        
        if self.apply_feature_crossing:
            X_filtered = self._apply_feature_crossing(X_filtered, is_fitting=False)
        
        logger.info(f"Successfully transformed data to shape {X_filtered.shape}")
        return X_filtered

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
            
            # Select features with comprehensive logging
            logger.info("🔄 Starting feature selection process")
            X_selected = feature_selector.fit_transform(df, y)
            
            # Get feature metadata generated during selection
            feature_metadata = getattr(feature_selector, 'feature_metadata', {})
            
            # Get feature importances
            feature_importances = {feature: float(importance) 
                                  for feature, importance in feature_selector.feature_importances_.items()
                                  if feature in feature_selector.selected_features}
            
            # Make sure we're only returning original column names (no _disc suffix)
            selected_features = [feat for feat in feature_selector.selected_features if feat in original_columns]
            logger.info(f"✅ Filtered to {len(selected_features)} original features")
            
            # Log the final column selection
            ignored_features = [col for col in original_columns if col not in selected_features]
            logger.info(f"📊 Selected features summary:")
            logger.info(f"  - Selected: {len(selected_features)} features")
            logger.info(f"  - Ignored: {len(ignored_features)} features")
            
            # Log detailed removal decisions for better transparency
            if feature_selector.method == 'llm' and hasattr(feature_selector, 'column_metadata'):
                logger.info("✂️ FEATURE REMOVAL DECISION DETAILS:")
                logger.info("=" * 80)
                for col in ignored_features:
                    if col in feature_selector.column_metadata and feature_selector.column_metadata[col].get("problematic", False):
                        reasons = feature_selector.column_metadata[col].get("reasons", ["No specific reason recorded"])
                        logger.info(f"REMOVED '{col}': {'; '.join(reasons)}")
                    else:
                        logger.info(f"REMOVED '{col}': Identified as unnecessary by LLM analysis")
                logger.info("=" * 80)
            
            # Options for the response
            options = {
                'method': feature_selector.method,
                'params': {
                    'min_features_to_keep': feature_selector.min_features_to_keep
                }
            }
            
            # Create the transformed dataframe with only selected columns
            X_selected = df[selected_features]
            
            # Compile information about problematic columns
            problematic_columns = []
            for col in feature_selector.problematic_columns:
                if col in feature_selector.column_metadata:
                    problematic_columns.append({
                        "name": col,
                        "reasons": feature_selector.column_metadata[col].get("reasons", ["Identified as problematic"])
                    })
            
            # Extract encoding mappings
            encoding_mappings = getattr(feature_selector, 'encoding_mappings', {})
            # Filter to only include mappings for selected columns to reduce payload size
            selected_encodings = {col: mapping for col, mapping in encoding_mappings.items() 
                                if col in selected_features}
            
            # Return the results
            response = {
                "transformed_data": X_selected.to_dict(orient='records'),
                "feature_importances": feature_importances,
                "selected_features": selected_features,
                "ignored_features": ignored_features,
                "feature_metadata": feature_metadata,
                "options": options,
                "encoding_mappings": selected_encodings,
                "problematic_columns": problematic_columns
            }
            
            # Add target recommendations if available
            if target_recommendations:
                response["target_recommendations"] = target_recommendations
            
            logger.info("✅ Returning feature selection results")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"🔴 Error in feature selection process: {str(e)}", exc_info=True)
            return jsonify({"error": f"Feature selection failed: {str(e)}"}), 500
    
    except Exception as e:
        logger.error(f"🔴 General error in select_features endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

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
        "transformed_data": {...},  # Data with only selected features and proper encoding
        "encoding_mappings": {...},  # Mappings used for categorical encoding
        "selected_features": [...],  # List of features used
        "message": "Data transformed successfully"
    }
    """
    try:
        logger.info("📥 Received transform request")
        request_data = request.json
        
        if not request_data or 'data' not in request_data:
            logger.error("🔴 Invalid request - missing data")
            return jsonify({"error": "Invalid request. Missing 'data'"}), 400
        
        # Validate if feature selection has been done
        if not feature_selector.selected_features:
            logger.error("🔴 No features have been selected")
            return jsonify({"error": "No features have been selected. Call /select_features first."}), 400
        
        # Convert JSON to DataFrame
        try:
            logger.info("🔄 Preprocessing input data")
            X = preprocess_data(request_data['data'])
            logger.info(f"📊 Input data shape: {X.shape}")
        except Exception as e:
            logger.error(f"🔴 Error preprocessing data: {str(e)}")
            return jsonify({"error": f"Failed to convert JSON to DataFrame: {str(e)}"}), 400
        
        # Transform data using the improved transform method
        logger.info("🔄 Transforming data")
        X_transformed = feature_selector.transform(X)
        
        # Extract encoding mappings used (filtered to only selected columns to reduce payload size)
        selected_encodings = {}
        if hasattr(feature_selector, 'encoding_mappings') and feature_selector.encoding_mappings:
            selected_encodings = {
                col: mapping for col, mapping in feature_selector.encoding_mappings.items()
                if col in feature_selector.selected_features and col in X.columns
            }
            logger.info(f"Included {len(selected_encodings)} encoding mappings in response")
        
        # Return transformed data and metadata
        logger.info("✅ Returning transformed data")
        return jsonify({
            "transformed_data": X_transformed.to_dict(orient='records'),
            "encoding_mappings": selected_encodings,
            "selected_features": feature_selector.selected_features,
            "message": "Data transformed successfully"
        })
    
    except Exception as e:
        logger.error(f"🔴 Error in transform endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": f"Transform failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the app on port 5002
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port) 