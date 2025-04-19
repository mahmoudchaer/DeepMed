from flask import Flask, request, jsonify
import pandas as pd
import os
import sys
import json
import logging
import re
from sklearn.preprocessing import StandardScaler
import requests
import openai
from datetime import datetime

# Set up logging - using most verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler('logs/data_cleaner.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Print Python version and package versions for debugging
logging.info(f"Python version: {sys.version}")
logging.info(f"OpenAI package version: {openai.__version__ if hasattr(openai, '__version__') else 'unknown'}")

# Print all environment variables at startup
logging.info("Environment variables:")
for key, value in os.environ.items():
    if 'key' in key.lower() or 'secret' in key.lower() or 'token' in key.lower() or 'password' in key.lower():
        masked_value = "***" if value else "None"
        logging.info(f"  {key}={masked_value}")
    else:
        logging.info(f"  {key}={value}")

app = Flask(__name__)

class DataCleaner:
    def __init__(self):
        """
        Initializes the DataCleaner.
        Uses the OpenAI API for LLM-based cleaning.
        Also initializes a StandardScaler for later feature scaling.
        """
        try:
            # For debugging - try with hardcoded key
 
            # First try environment variable (this should be the normal way)
            api_key = os.getenv("OPENAI_API_KEY")
            logging.info(f"API key from environment: {'Found' if api_key else 'Not found'}")
            
            # If not found, use hardcoded key for testing
            if not api_key:
                logging.info("API key not found in environment variables")
            
            # Test if key looks valid
            if api_key and len(api_key) > 20:  # Simple validation that it's not empty or too short
                logging.info(f"API key appears valid (length: {len(api_key)})")
                
                # In version 0.28.1, we set api_key directly on the openai module
                try:
                    openai.api_key = api_key
                    logging.info("Set API key on openai module")
                    # Test if it works
                    self._test_openai_connection()
                    self.has_openai = True  # Flag to check if OpenAI is available
                except Exception as e:
                    logging.error(f"Error with original key: {str(e)}")
                    
                    # If that failed, try removing the prefix if it exists
                    if api_key.startswith("sk-proj-"):
                        try:
                            modified_key = api_key.replace("sk-proj-", "sk-")
                            logging.info("Trying with modified key (removed proj- prefix)...")
                            openai.api_key = modified_key
                            # Test if it works
                            self._test_openai_connection()
                            self.has_openai = True  # Flag to check if OpenAI is available
                        except Exception as e2:
                            logging.error(f"Error with modified key: {str(e2)}")
                            self.has_openai = False
                    else:
                        self.has_openai = False
            else:
                logging.error("API key invalid or too short")
                self.has_openai = False
                
        except Exception as e:
            logging.error(f"Unexpected error in OpenAI initialization: {str(e)}", exc_info=True)
            self.has_openai = False
            
        self.scaler = StandardScaler()
        self.last_cleaning_prompt = None  # Store the last used cleaning instructions
        self.encoding_mappings = {}  # Store encoding mappings for categorical variables
        
    def _test_openai_connection(self):
        """Test if OpenAI client is working correctly"""
        try:
            logging.info("Testing OpenAI connection...")
            # Use a very simple prompt to test - updated for v0.28.1 syntax
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=5
            )
            logging.info(f"OpenAI test successful! Response: {response.choices[0].message.content}")
            return True
        except Exception as e:
            logging.error(f"OpenAI test failed: {str(e)}", exc_info=True)
            return False

    def remove_outliers(self, df, columns, n_std=3):
        """Remove outliers based on standard deviation."""
        for column in columns:
            if df[column].dtype in ['int64', 'float64']:
                mean = df[column].mean()
                std = df[column].std()
                df = df[(df[column] <= mean + (n_std * std)) & 
                        (df[column] >= mean - (n_std * std))]
        return df

    def handle_missing_values(self, df, previous_prompt=None):
        """
        Handle missing values using LLM-based cleaning.
        This method extracts metadata, generates a prompt, calls the LLM,
        parses the cleaning instructions, and applies missing-value handling
        and conversion to numeric values (without normalization).
        
        Args:
            df: DataFrame to clean
            previous_prompt: Optional prompt describing previous cleaning strategy
        
        Returns:
            cleaned_df: Cleaned DataFrame
        """
        return self._llm_clean(df, previous_prompt)

    def scale_features(self, df, columns):
        """Scale numerical features using StandardScaler."""
        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def clean_data(self, df, target_column, previous_prompt=None):
        """
        Main method to clean the data.
        The interface is the same as your old version:
          - Creates a copy of the DataFrame.
          - Handles missing values (using the new LLM logic).
          - Removes outliers from numeric columns (excluding the target).
          - Scales numeric features.
        Returns a fully numeric DataFrame.
        
        Args:
            df: DataFrame to clean
            target_column: Name of the target column
            previous_prompt: Optional prompt describing previous cleaning strategy
        
        Returns:
            cleaned_df: Cleaned DataFrame
        """
        df = df.copy()
        # Determine numeric columns excluding the target.
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_columns = numeric_columns[numeric_columns != target_column]
        
        # Apply LLM-based missing value handling and numeric conversion.
        df = self.handle_missing_values(df, previous_prompt)
        # Remove outliers on the numeric columns.
        df = self.remove_outliers(df, numeric_columns)
        # Scale the numeric features.
        return df

    # ---- Internal methods for LLM-based cleaning ----

    def _extract_metadata(self, df: pd.DataFrame, max_unique: int = 20) -> dict:
        """Extract metadata for each column in the DataFrame."""
        metadata = {}
        for col in df.columns:
            col_data = df[col]
            metadata[col] = {
                "dtype": str(col_data.dtype),
                "num_missing": int(col_data.isna().sum()),
                "num_unique": int(col_data.nunique(dropna=True)),
                "unique_values_sample": col_data.dropna().unique().tolist()[:max_unique]
            }
        return metadata

    def _generate_prompt(self, metadata: dict, previous_prompt: str = None) -> str:
        """
        Build a prompt for the LLM using the extracted metadata.
        If previous_prompt is provided, include it to ensure consistent cleaning.
        """
        base_prompt = (
            "You are an expert data cleaning assistant. Below is metadata for a dataset extracted from a tabular file. "
            "For each column, provide detailed instructions regarding:\n\n"
            "1. Missing values: Specify if rows should be dropped (action: 'drop') or if missing values should be replaced (action: 'replace'). "
            "If replacing, provide a replacement value (e.g., 0, mean, median, mode, etc.).\n\n"
            "2. Conversion to numerical: For non-numerical columns, specify an 'encoding' key with one of these values: "
            "'none' (if already numeric), 'onehot' (for one-hot encoding), or 'label' (for label encoding).\n\n"
            "Return your instructions in JSON format with this structure:\n"
            "{\n"
            '  "column_name": {"missing": {"action": "drop/replace", "value": <value if replace>}, "normalize": false, "encoding": "none/onehot/label"},\n'
            "  ...\n"
            "}\n\n"
        )
        
        if previous_prompt:
            prompt = (
                "You are an expert data cleaning assistant. You previously cleaned a dataset with these instructions:\n\n"
                f"{previous_prompt}\n\n"
                "I need to clean a new dataset using the SAME STRATEGY to ensure consistency. "
                "Apply the same logic, even if it's not optimal for this new data.\n\n"
                "Here is the metadata for the new dataset:\n"
            )
            prompt += json.dumps(metadata, indent=2)
            return prompt
        else:
            prompt = base_prompt + "Here is the metadata:\n"
            prompt += json.dumps(metadata, indent=2)
            return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM via the OpenAI API using the prompt."""
        if not self.has_openai:
            logging.warning("OpenAI not available. Using default cleaning instructions.")
            return '{"error": "No OpenAI available"}'
            
        try:
            logging.info("Calling LLM API...")
            # Updated for v0.28.1 API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Using a more widely available model
                messages=[
                    {"role": "system", "content": "You are an expert data cleaning assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0  # Low temperature for deterministic responses.
            )
            content = response.choices[0].message["content"]
            logging.info("LLM API call successful.")
            return content
        except Exception as e:
            logging.error("Error calling LLM API: %s", e)
            # Return a simple fallback response instead of exiting
            return '{"error": "Failed to get LLM response"}'

    def _parse_instructions(self, llm_response: str) -> dict:
        """
        Remove markdown formatting and extract the JSON response from the LLM.
        This method uses a regex to extract the first JSON object it finds.
        """
        # Remove any markdown code fences.
        if llm_response.strip().startswith("```"):
            llm_response = re.sub(r"^```[^\n]*\n", "", llm_response)
            llm_response = re.sub(r"\n```.*$", "", llm_response)
        # Extract JSON using regex.
        match = re.search(r'({.*})', llm_response, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = llm_response
        try:
            instructions = json.loads(json_str)
            logging.info("Parsed cleaning instructions successfully.")
            return instructions
        except Exception as e:
            logging.error("Error parsing LLM response: %s", e)
            # Return empty dict instead of exiting
            return {}

    def _apply_cleaning(self, df: pd.DataFrame, instructions: dict) -> pd.DataFrame:
        """
        Apply the cleaning, conversion, and missing value instructions to the DataFrame.
        This method ensures all columns are converted to numeric without performing normalization.
        """
        df_clean = df.copy()
        
        # Reset encoding mappings for this cleaning run
        self.encoding_mappings = {}
        
        # If we have no instructions, apply default cleaning
        if not instructions or 'error' in instructions:
            logging.warning("No cleaning instructions available. Applying default cleaning.")
            # Default: fill numeric columns with mean, categorical with mode
            for col in df_clean.columns:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")
                    # Store mapping before encoding
                    unique_vals = sorted(df_clean[col].dropna().unique())
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    self.encoding_mappings[col] = mapping
                    df_clean[col] = df_clean[col].astype("category").cat.codes
            return df_clean
            
        for col in df_clean.columns:
            if col not in instructions:
                logging.warning("Column '%s' not found in LLM instructions. Applying default cleaning.", col)
                # Default cleaning for column
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")
                    # Store mapping before encoding
                    unique_vals = sorted(df_clean[col].dropna().unique())
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    self.encoding_mappings[col] = mapping
                    df_clean[col] = df_clean[col].astype("category").cat.codes
                continue

            instr = instructions[col]
            # Handle missing values
            missing_instr = instr.get("missing", {})
            action = missing_instr.get("action", "").lower()
            if action == "drop":
                df_clean = df_clean.dropna(subset=[col])
                logging.info("Dropped rows with missing values in column '%s'.", col)
            elif action == "replace":
                replacement = missing_instr.get("value", None)
                if replacement is not None:
                    if isinstance(replacement, str):
                        rep_lower = replacement.lower()
                        if rep_lower == "mean":
                            replacement = df_clean[col].mean()
                        elif rep_lower == "median":
                            replacement = df_clean[col].median()
                        elif rep_lower == "mode":
                            mode_series = df_clean[col].mode()
                            replacement = mode_series.iloc[0] if not mode_series.empty else None
                    df_clean[col] = df_clean[col].fillna(replacement)
                    logging.info("Replaced missing values in column '%s' with '%s'.", col, replacement)
            # Convert non-numeric columns to numeric.
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                encoding = instr.get("encoding", "none").lower()
                if encoding in ("onehot", "label"):
                    # Store mapping before encoding
                    unique_vals = sorted(df_clean[col].dropna().unique())
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    self.encoding_mappings[col] = mapping
                    df_clean[col] = df_clean[col].astype("category").cat.codes
                    logging.info("Label-encoded column '%s' (instruction: '%s').", col, encoding)
                else:
                    # For encoding "none": attempt to map two-category columns to 0/1.
                    tmp = df_clean[col].astype(str).str.strip().str.lower()
                    unique_vals = set(tmp.dropna().unique())
                    if len(unique_vals) == 2:
                        sorted_vals = sorted(list(unique_vals))
                        val_to_int = {sorted_vals[0]: 0, sorted_vals[1]: 1}
                        df_clean[col] = tmp.map(val_to_int)
                        self.encoding_mappings[col] = val_to_int
                        logging.info("Mapped two-category column '%s' -> { '%s': 0, '%s': 1 }.", col, sorted_vals[0], sorted_vals[1])
                    else:
                        # Store mapping before encoding
                        unique_vals = sorted(df_clean[col].dropna().unique())
                        mapping = {val: idx for idx, val in enumerate(unique_vals)}
                        self.encoding_mappings[col] = mapping
                        df_clean[col] = df_clean[col].astype("category").cat.codes
                        logging.info("Fallback label-encoded column '%s'.", col)
        # Final step: ensure all columns are numeric.
        df_final = df_clean.apply(pd.to_numeric, errors='coerce')
        return df_final

    def _generate_cleaning_summary(self, instructions: dict) -> str:
        """
        Generate a human-readable summary of the cleaning instructions.
        This will be returned as part of the API response for future reference.
        """
        if not instructions or 'error' in instructions:
            return "Applied default cleaning: filled numeric columns with mean values and categorical columns with mode values, then converted all to numeric."
            
        summary = []
        for col, instr in instructions.items():
            col_summary = []
            
            # Missing values handling
            missing_instr = instr.get("missing", {})
            action = missing_instr.get("action", "").lower()
            if action == "drop":
                col_summary.append(f"dropped rows with missing values")
            elif action == "replace":
                replacement = missing_instr.get("value", None)
                if replacement is not None:
                    col_summary.append(f"replaced missing values with {replacement}")
            
            # Encoding
            encoding = instr.get("encoding", "none").lower()
            if encoding == "onehot":
                col_summary.append("one-hot encoded")
            elif encoding == "label":
                col_summary.append("label encoded")
                
            if col_summary:
                summary.append(f"Column '{col}': {', '.join(col_summary)}")
        
        if not summary:
            return "Applied default cleaning strategy based on column data types."
            
        return "Cleaning strategy:\n" + "\n".join(summary)

    def _llm_clean(self, df: pd.DataFrame, previous_prompt: str = None) -> pd.DataFrame:
        """
        Full LLM-based cleaning pipeline:
          - Extract metadata.
          - Generate prompt.
          - Call LLM.
          - Parse instructions.
          - Apply cleaning (handle missing values and convert to numeric).
        Returns a fully numeric DataFrame.
        
        Args:
            df: DataFrame to clean
            previous_prompt: Optional prompt describing previous cleaning strategy
            
        Returns:
            cleaned_df: Cleaned DataFrame
        """
        try:
            metadata = self._extract_metadata(df)
            logging.info("Extracted metadata for %d columns", len(metadata))
            
            prompt = self._generate_prompt(metadata, previous_prompt)
            logging.info("Generated prompt for LLM.")
            
            llm_response = self._call_llm(prompt)
            
            instructions = self._parse_instructions(llm_response)
            if not instructions:
                logging.warning("Could not parse cleaning instructions. Using default cleaning.")
            
            df_clean = self._apply_cleaning(df, instructions)
            logging.info("Applied cleaning instructions successfully.")
            
            # Save the cleaning instructions for the response
            if previous_prompt:
                self.last_cleaning_prompt = previous_prompt
            else:
                # Generate a human-readable summary of the cleaning performed
                self.last_cleaning_prompt = self._generate_cleaning_summary(instructions)
            
            return df_clean
        except Exception as e:
            logging.error("Error in LLM cleaning pipeline: %s", e)
            # Apply basic cleaning as fallback
            logging.info("Applying fallback cleaning")
            df_clean = df.copy()
            # Simple fallback: fill numeric columns with mean, categorical with mode, and convert to numeric
            for col in df_clean.columns:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                else:
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    # Store mapping before encoding
                    unique_vals = sorted(df_clean[col].dropna().unique())
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    self.encoding_mappings[col] = mapping
                    df_clean[col] = df_clean[col].astype("category").cat.codes
            
            # Set a default cleaning summary
            self.last_cleaning_prompt = "Applied default cleaning: filled numeric columns with mean values and categorical columns with mode values, then converted all to numeric."
            
            return df_clean

# Create a global DataCleaner instance
cleaner = DataCleaner()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/clean', methods=['POST'])
def clean_data():
    try:
        data = request.json
        if not data or 'data' not in data:
            return jsonify({"error": "No data provided"}), 400
        
        # Get the target column (if specified)
        target_column = data.get('target_column', None)
        
        # Get the cleaning prompt (if specified)
        prompt = data.get('prompt', None)
        
        # Convert the data to a pandas DataFrame
        df = pd.DataFrame(data['data'])
        
        # Create a DataCleaner instance
        cleaner = DataCleaner()
        
        # Clean the data
        cleaned_df = cleaner.clean_data(df, target_column, prompt)
        
        # Return the cleaned data and encoding mappings (in separate keys)
        response = {
            "data": json.loads(cleaned_df.to_json(orient='records')),
            "encoding_mappings": cleaner.encoding_mappings,
            "prompt": cleaner.last_cleaning_prompt,
            "options": {},  # Add options if needed
            "report": {
                "rows_before": len(df),
                "rows_after": len(cleaned_df),
                "columns_before": len(df.columns),
                "columns_after": len(cleaned_df.columns),
                "categorical_columns_encoded": len(cleaner.encoding_mappings)
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"Error in clean_data: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/apply_stored_cleaning', methods=['POST'])
def apply_stored_cleaning():
    """Apply cleaning operations based on stored configuration"""
    try:
        data = request.json
        if not data or 'data' not in data or 'cleaner_config' not in data:
            return jsonify({"error": "Missing data or cleaner_config in request"}), 400
        
        # Convert the data to a pandas DataFrame
        df = pd.DataFrame(data['data'])
        cleaner_config = data['cleaner_config']
        
        # Get cleaning options and instructions
        options = cleaner_config.get("options", {})
        llm_instructions = cleaner_config.get("llm_instructions", "")
        
        # First, identify columns that are likely irrelevant for analysis
        irrelevant_patterns = [
            # ID columns
            'id', '_id', 'uuid', 'guid', 'identifier',
            # Name columns  
            'name', 'firstname', 'lastname', 'fullname', 'patient', 'doctor', 'physician', 'provider',
            # Date/time columns
            'date', 'time', 'datetime', 'timestamp', 'admission', 'discharge', 'visit',
            # Contact and personal info
            'address', 'email', 'phone', 'contact', 'ssn', 'social', 'insurance',
            # Other non-predictive columns
            'notes', 'comment', 'description', 'url', 'link', 'file', 'path'
        ]
        
        # Identify columns to potentially exclude based on name patterns
        potential_irrelevant_cols = []
        for col in df.columns:
            col_lower = col.lower()
            # Check if column name contains any of the irrelevant patterns
            if any(pattern in col_lower for pattern in irrelevant_patterns):
                potential_irrelevant_cols.append(col)
                logging.info(f"Identified potentially irrelevant column in pre-cleaning check: '{col}'")
        
        # Get categorical variable encoding mappings if available
        encoding_mappings = cleaner_config.get("encoding_mappings", {})
        
        # Filter out mappings for columns that are likely irrelevant
        if encoding_mappings and potential_irrelevant_cols:
            # Keep track of removed mappings for logging
            removed_mappings = []
            
            # Create a filtered copy of the mappings
            filtered_mappings = encoding_mappings.copy()
            
            # Remove mappings for likely irrelevant columns
            for col in potential_irrelevant_cols:
                if col in filtered_mappings:
                    removed_mappings.append(col)
                    del filtered_mappings[col]
            
            if removed_mappings:
                logging.info(f"Removed encoding mappings for {len(removed_mappings)} potentially irrelevant columns: {removed_mappings}")
                # Update the cleaner_config with the filtered mappings
                encoding_mappings = filtered_mappings
        
        # If the cleaner used LLM mode, make an API call to clean data with the same prompt
        if llm_instructions:
            # Create a DataCleaner instance
            cleaner = DataCleaner()
            
            # Use LLM to clean the data
            cleaned_df = cleaner._llm_clean(df, llm_instructions)
            
            # Update encoding mappings from cleaner
            if cleaner.encoding_mappings:
                # Filter out mappings for likely irrelevant columns
                for col in potential_irrelevant_cols:
                    if col in cleaner.encoding_mappings:
                        del cleaner.encoding_mappings[col]
                
                # Update our mappings
                encoding_mappings = cleaner.encoding_mappings
                logging.info(f"Updated encoding mappings from LLM cleaning with {len(encoding_mappings)} columns")
            
            response = {
                "data": json.loads(cleaned_df.to_json(orient='records')),
                "encoding_mappings": encoding_mappings,
                "report": {
                    "rows_before": len(df),
                    "rows_after": len(cleaned_df),
                    "columns_before": len(df.columns),
                    "columns_after": len(cleaned_df.columns),
                    "categorical_columns_encoded": len(encoding_mappings)
                }
            }
            
            return jsonify(response), 200
        else:
            # Apply basic cleaning
            cleaned_df = apply_basic_cleaning(df, cleaner_config)
            
            response = {
                "data": json.loads(cleaned_df.to_json(orient='records')),
                "encoding_mappings": cleaner_config.get("encoding_mappings", {}),
                "report": {
                    "rows_before": len(df),
                    "rows_after": len(cleaned_df),
                    "columns_before": len(df.columns),
                    "columns_after": len(cleaned_df.columns),
                    "categorical_columns_encoded": len(cleaner_config.get("encoding_mappings", {}))
                }
            }
            
            return jsonify(response), 200
    
    except Exception as e:
        logging.error(f"Error in apply_stored_cleaning: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def apply_basic_cleaning(df, cleaner_config):
    """Apply basic cleaning steps to a dataframe"""
    # Make a copy of the dataframe to avoid modifying the original
    cleaned_df = df.copy()
    encoding_mappings = cleaner_config.get("encoding_mappings", {})
    
    # 1. Handle missing values
    
    # Identify numeric and categorical columns
    numeric_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    # Replace missing values in numeric columns with column median
    for col in numeric_cols:
        if cleaned_df[col].isna().sum() > 0:
            median_value = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_value)
            logging.info(f"Filled {cleaned_df[col].isna().sum()} missing values in numeric column '{col}' with median {median_value}")
    
    # Replace missing values in categorical columns with mode
    for col in categorical_cols:
        if cleaned_df[col].isna().sum() > 0:
            mode_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "MISSING"
            cleaned_df[col] = cleaned_df[col].fillna(mode_value)
            logging.info(f"Filled {cleaned_df[col].isna().sum()} missing values in categorical column '{col}' with mode '{mode_value}'")
    
    # Apply encoding for categorical columns
    for col in categorical_cols:
        # Check if we have a mapping for this column
        if col in encoding_mappings:
            mapping = encoding_mappings[col]
            # Convert column values to strings to ensure consistent mapping
            col_values = cleaned_df[col].astype(str)
            # Apply mapping with fallback to -1 for values not in the mapping
            cleaned_df[col] = col_values.map(mapping).fillna(-1).astype(int)
            logging.info(f"Applied stored categorical mapping for column '{col}'")
        else:
            # Create a new mapping for this column and save it
            unique_values = sorted(cleaned_df[col].dropna().unique())
            new_mapping = {str(val): idx for idx, val in enumerate(unique_values)}
            encoding_mappings[col] = new_mapping
            # Apply the new mapping
            col_values = cleaned_df[col].astype(str)
            cleaned_df[col] = col_values.map(new_mapping).fillna(-1).astype(int)
            logging.info(f"Created and applied new categorical mapping for column '{col}'")
    
    # Update the encoding mappings in the cleaner_config
    cleaner_config["encoding_mappings"] = encoding_mappings
    
    # 2. Handle outliers - replace outliers with column limits
    for col in numeric_cols:
        # Calculate IQR
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
        if outliers > 0:
            logging.info(f"Found {outliers} outliers in column '{col}', clipping to IQR bounds")
            # Replace outliers
            cleaned_df[col] = cleaned_df[col].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    
    return cleaned_df

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True) 