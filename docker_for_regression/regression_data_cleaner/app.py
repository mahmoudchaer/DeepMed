from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import logging
import sys
import os
import traceback
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
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

app = Flask(__name__)

class RegressionDataCleaner:
    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.encoders = {}
        self.encoding_maps = {}
        
    def _identify_column_types(self, df):
        """Identify numeric and categorical columns with more regression-specific logic"""
        column_types = {}
        
        for col in df.columns:
            # Check if column is numeric
            if np.issubdtype(df[col].dtype, np.number):
                column_types[col] = 'numeric'
            else:
                # Check if it's a categorical column
                nunique = df[col].nunique()
                if nunique < 20 or (nunique / len(df) < 0.05):  # Fewer than 20 unique values or less than 5% of total rows
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'text'  # Likely not useful for regression, will be flagged
        
        return column_types
    
    def _handle_missing_values(self, df, column_types):
        """Handle missing values based on column type"""
        # For numeric columns, use median imputation (better for regression)
        numeric_cols = [col for col, type_ in column_types.items() if type_ == 'numeric']
        if numeric_cols:
            df[numeric_cols] = self.numeric_imputer.fit_transform(df[numeric_cols])
        
        # For categorical columns, use most frequent imputation
        categorical_cols = [col for col, type_ in column_types.items() if type_ == 'categorical']
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
        
        return df
    
    def _handle_outliers(self, df, column_types, target_column=None):
        """Handle outliers with special care for regression tasks"""
        numeric_cols = [col for col, type_ in column_types.items() if type_ == 'numeric']
        
        outlier_report = {}
        
        for col in numeric_cols:
            # Skip the target column - we don't want to modify the values we're trying to predict
            if col == target_column:
                continue
                
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers_mask = z_scores > 3
            
            if outliers_mask.any():
                num_outliers = outliers_mask.sum()
                outlier_report[col] = num_outliers
                
                # For regression, we use winsorization instead of removing outliers
                # This caps extreme values instead of removing them
                lower_bound = df[col].quantile(0.01)
                upper_bound = df[col].quantile(0.99)
                
                # Cap values outside the bounds
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                logger.info(f"Winsorized {num_outliers} outliers in column {col}")
        
        return df, outlier_report
    
    def _encode_categorical_features(self, df, column_types):
        """Encode categorical features with label encoding and one-hot encoding as appropriate"""
        categorical_cols = [col for col, type_ in column_types.items() if type_ == 'categorical']
        
        for col in categorical_cols:
            if df[col].nunique() <= 2:  # Binary categorical - use label encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                
                # Store mapping for later reference
                self.encoders[col] = le
                self.encoding_maps[f"{col}_encoding"] = dict(zip(le.classes_, le.transform(le.classes_)))
                
                logger.info(f"Applied Label Encoding to binary column: {col}")
            else:
                # For non-binary categorical, use one-hot encoding
                # First create a mapping dictionary
                value_map = {val: idx for idx, val in enumerate(df[col].unique())}
                self.encoding_maps[f"{col}_map"] = value_map
                
                # Then do one-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                
                # Add the new columns to the dataframe
                df = pd.concat([df, dummies], axis=1)
                
                # Drop the original column
                df = df.drop(columns=[col])
                
                logger.info(f"Applied One-Hot Encoding to column: {col}")
        
        return df
    
    def _transform_numeric_features(self, df, column_types, target_column=None):
        """Apply transformations to improve regression model performance"""
        numeric_cols = [col for col, type_ in column_types.items() if type_ == 'numeric']
        transformations = {}
        
        for col in numeric_cols:
            # Skip target column
            if col == target_column:
                continue
                
            # Check for skewness
            if len(df[col].dropna()) > 0:  # Only if we have data
                skewness = df[col].skew()
                
                # Apply log transformation for highly skewed data
                if abs(skewness) > 1:
                    # Make data positive if needed
                    if df[col].min() <= 0:
                        shift = abs(df[col].min()) + 1
                        df[col] = df[col] + shift
                        transformations[col] = f"log(x + {shift})"
                    else:
                        transformations[col] = "log(x)"
                    
                    # Apply log transform
                    df[col] = np.log(df[col])
                    logger.info(f"Applied log transformation to {col} (skewness: {skewness:.2f})")
        
        return df, transformations
    
    def _clean_data_with_llm(self, df, target_column=None):
        """Use LLM to suggest cleaning steps optimized for regression tasks"""
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not available, skipping LLM cleaning suggestions")
            return df, "No LLM suggestions (API key not configured)"
            
        try:
            # Generate sample of dataframe for LLM
            sample_rows = min(5, len(df))
            df_sample = df.head(sample_rows)
            
            # Convert sample to string representation
            df_str = df_sample.to_string()
            
            # Get column information
            column_info = {}
            for col in df.columns:
                column_info[col] = {
                    'dtype': str(df[col].dtype),
                    'missing': df[col].isnull().sum(),
                    'unique_values': df[col].nunique(),
                    'min': df[col].min() if np.issubdtype(df[col].dtype, np.number) else None,
                    'max': df[col].max() if np.issubdtype(df[col].dtype, np.number) else None,
                    'mean': df[col].mean() if np.issubdtype(df[col].dtype, np.number) else None,
                    'median': df[col].median() if np.issubdtype(df[col].dtype, np.number) else None,
                    'skewness': df[col].skew() if np.issubdtype(df[col].dtype, np.number) else None
                }
            
            # Create prompt for the LLM
            prompt = f"""
            You are a data cleaning expert specializing in regression problems in healthcare. 
            I have a dataset for a regression task. 
            
            Here's a sample of the data:
            {df_str}
            
            Column information:
            {json.dumps(column_info, indent=2)}
            
            Target column for regression: {target_column if target_column else 'Not specified'}
            
            Based on this information, please provide:
            1. Suggestions for handling the data specifically for a regression task
            2. Any columns that should be transformed (e.g., log transform for skewed distributions)
            3. Potential issues with multicollinearity or heteroscedasticity
            4. Any columns that should be scaled or normalized
            5. Feature engineering suggestions that might improve regression performance
            
            Focus on maximizing the predictive power for the regression task.
            """
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4-0125-preview",
                messages=[{"role": "system", "content": "You are a data cleaning expert for regression problems."},
                          {"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.3
            )
            
            # Extract suggestions
            llm_suggestions = response.choices[0].message.content
            logger.info(f"Received LLM cleaning suggestions: {llm_suggestions[:100]}...")
            
            return df, llm_suggestions
            
        except Exception as e:
            logger.error(f"Error getting LLM suggestions: {str(e)}")
            return df, f"Error getting LLM suggestions: {str(e)}"
    
    def clean(self, data, target_column=None, options=None):
        """Main method to clean data for regression"""
        logger.info(f"Starting regression data cleaning process for {len(data)} rows")
        
        # Convert to DataFrame if it's a list
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        # Set default options if not provided
        if options is None:
            options = {
                'handle_missing': True,
                'handle_outliers': True,
                'encode_categorical': True,
                'transform_numeric': True,
                'use_llm': True
            }
        
        # Store original column names
        original_columns = df.columns.tolist()
        
        # Identify column types
        column_types = self._identify_column_types(df)
        logger.info(f"Column types identified: {column_types}")
        
        # Clean data
        cleaning_report = {
            'rows_before': len(df),
            'columns_before': len(original_columns),
            'missing_values_before': df.isnull().sum().sum(),
            'column_types': column_types
        }
        
        # Handle missing values if requested
        if options.get('handle_missing', True):
            df = self._handle_missing_values(df, column_types)
            cleaning_report['missing_values_after'] = df.isnull().sum().sum()
        
        # Handle outliers if requested
        if options.get('handle_outliers', True):
            df, outlier_report = self._handle_outliers(df, column_types, target_column)
            cleaning_report['outliers'] = outlier_report
        
        # Transform numeric features if requested (important for regression)
        transformations = {}
        if options.get('transform_numeric', True):
            df, transformations = self._transform_numeric_features(df, column_types, target_column)
            cleaning_report['transformations'] = transformations
        
        # Encode categorical features if requested
        if options.get('encode_categorical', True):
            df = self._encode_categorical_features(df, column_types)
        
        # Use LLM for additional suggestions if requested
        llm_suggestions = None
        if options.get('use_llm', True) and OPENAI_API_KEY:
            _, llm_suggestions = self._clean_data_with_llm(df, target_column)
        
        # Finalize report
        cleaning_report['rows_after'] = len(df)
        cleaning_report['columns_after'] = len(df.columns)
        cleaning_report['columns_added'] = list(set(df.columns) - set(original_columns))
        cleaning_report['columns_removed'] = list(set(original_columns) - set(df.columns))
        
        return df, self.encoding_maps, cleaning_report, llm_suggestions


# Create a cleaner instance
data_cleaner = RegressionDataCleaner()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "regression_data_cleaner"
    })

@app.route('/clean', methods=['POST'])
def clean_data():
    """Clean data endpoint"""
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or 'data' not in request_data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract data and options
        data = request_data['data']
        target_column = request_data.get('target_column')
        options = request_data.get('options')
        previous_prompt = request_data.get('prompt')
        
        # Log basic info
        logger.info(f"Received cleaning request with {len(data)} records")
        logger.info(f"Target column: {target_column}")
        
        # Clean data
        df, encoding_maps, report, llm_suggestions = data_cleaner.clean(data, target_column, options)
        
        # Prepare response
        response = {
            "data": json.loads(df.to_json(orient='records')),
            "encoding_mappings": encoding_maps,
            "report": report,
        }
        
        # Add prompt to response (either new or previous)
        if previous_prompt:
            response["prompt"] = previous_prompt
            logger.info("Using previous prompt")
        elif llm_suggestions:
            response["prompt"] = llm_suggestions
            logger.info("Adding new LLM suggestions")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5031))
    logger.info(f"Starting Regression Data Cleaner Service on port {port}")
    serve(app, host='0.0.0.0', port=port) 