import os
import sys
import json
import logging
import re
import pandas as pd
from openai import OpenAI
from sklearn.preprocessing import StandardScaler

# Set up logging for production-quality output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

class DataCleaner:
    def __init__(self):
        """
        Initializes the DataCleaner.
        Uses the OpenAI API for LLM-based cleaning.
        Also initializes a StandardScaler for later feature scaling.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.warning("OPENAI_API_KEY environment variable not set.")
            logging.info("Proceeding without LLM-based cleaning capabilities.")
            self.client = None
        else:
            try:
                # Initialize with minimal configuration to avoid proxy issues
                self.client = OpenAI(api_key=api_key)
                logging.info("OpenAI client initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing OpenAI client: {e}")
                self.client = None
                
        self.scaler = StandardScaler()

    def remove_outliers(self, df, columns, n_std=3):
        """Remove outliers based on standard deviation."""
        for column in columns:
            if df[column].dtype in ['int64', 'float64']:
                mean = df[column].mean()
                std = df[column].std()
                df = df[(df[column] <= mean + (n_std * std)) & 
                        (df[column] >= mean - (n_std * std))]
        return df

    def handle_missing_values(self, df):
        """
        Handle missing values using LLM-based cleaning.
        This method extracts metadata, generates a prompt, calls the LLM,
        parses the cleaning instructions, and applies missing-value handling
        and conversion to numeric values (without normalization).
        """
        return self._llm_clean(df)

    def scale_features(self, df, columns):
        """Scale numerical features using StandardScaler."""
        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def clean_data(self, df, target_column):
        """
        Main method to clean the data.
        The interface is the same as your old version:
          - Creates a copy of the DataFrame.
          - Handles missing values (using the new LLM logic).
          - Removes outliers from numeric columns (excluding the target).
          - Scales numeric features.
        Returns a fully numeric DataFrame.
        """
        df = df.copy()
        # Determine numeric columns excluding the target.
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_columns = numeric_columns[numeric_columns != target_column]
        
        # Apply LLM-based missing value handling and numeric conversion.
        df = self.handle_missing_values(df)
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

    def _generate_prompt(self, metadata: dict) -> str:
        """
        Build a prompt for the LLM using the extracted metadata.
        """
        prompt = (
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
            "Here is the metadata:\n"
        )
        prompt += json.dumps(metadata, indent=2)
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM via the OpenAI API using the prompt."""
        if self.client is None:
            logging.warning("OpenAI client not available. Using default cleaning instructions.")
            return '{"error": "No LLM client available"}'
            
        try:
            logging.info("Calling LLM API...")
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using a more widely available model
                messages=[
                    {"role": "system", "content": "You are an expert data cleaning assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0  # Low temperature for deterministic responses.
            )
            content = response.choices[0].message.content
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
        
        # If we have no instructions, apply default cleaning
        if not instructions or 'error' in instructions:
            logging.warning("No cleaning instructions available. Applying default cleaning.")
            # Default: fill numeric columns with mean, categorical with mode
            for col in df_clean.columns:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")
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
                        logging.info("Mapped two-category column '%s' -> { '%s': 0, '%s': 1 }.", col, sorted_vals[0], sorted_vals[1])
                    else:
                        df_clean[col] = df_clean[col].astype("category").cat.codes
                        logging.info("Fallback label-encoded column '%s'.", col)
        # Final step: ensure all columns are numeric.
        df_final = df_clean.apply(pd.to_numeric, errors='coerce')
        return df_final

    def _llm_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full LLM-based cleaning pipeline:
          - Extract metadata.
          - Generate prompt.
          - Call LLM.
          - Parse instructions.
          - Apply cleaning (handle missing values and convert to numeric).
        Returns a fully numeric DataFrame.
        """
        try:
            metadata = self._extract_metadata(df)
            logging.info("Extracted metadata for %d columns", len(metadata))
            
            prompt = self._generate_prompt(metadata)
            logging.info("Generated prompt for LLM.")
            
            llm_response = self._call_llm(prompt)
            
            instructions = self._parse_instructions(llm_response)
            if not instructions:
                logging.warning("Could not parse cleaning instructions. Using default cleaning.")
            
            df_clean = self._apply_cleaning(df, instructions)
            logging.info("Applied cleaning instructions successfully.")
            
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
                    df_clean[col] = df_clean[col].astype("category").cat.codes
            
            return df_clean
