import os
import zipfile
import tempfile
import shutil
import subprocess
import pandas as pd
import time
import csv
import io
import json
import sys
import logging
from flask import Flask, request, jsonify
from threading import Thread

# Configure logging to ensure all output appears
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = Flask(__name__)
# Ensure Flask app logs everything to stdout
app.logger.handlers = []
for handler in logging.getLogger().handlers:
    app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "regression_predictor"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Verify that both files are provided.
    if 'model_package' not in request.files or 'input_file' not in request.files:
        return jsonify({"error": "Both 'model_package' and 'input_file' must be provided."}), 400

    model_file = request.files['model_package']
    input_file = request.files['input_file']

    # Validate file types.
    if not model_file.filename.lower().endswith('.zip'):
        return jsonify({"error": "Model package must be a ZIP archive."}), 400
    if not (input_file.filename.lower().endswith('.xlsx') or 
            input_file.filename.lower().endswith('.xls') or 
            input_file.filename.lower().endswith('.csv')):
        return jsonify({"error": "Input file must be an Excel or CSV file."}), 400

    # Create a temporary working directory.
    temp_dir = tempfile.mkdtemp(prefix="session_")
    start_time = time.time()
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        # Save and extract the ZIP package.
        zip_path = os.path.join(temp_dir, "model_package.zip")
        model_file.save(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print(f"Extracted model package to {temp_dir}")

        # Save the input file.
        input_file_path = os.path.join(temp_dir, input_file.filename)
        input_file.save(input_file_path)
        print(f"Saved input file to {input_file_path}")

        # Create a virtual environment in the temporary directory.
        venv_path = os.path.join(temp_dir, "venv")
        print(f"Creating virtual environment at {venv_path}")
        subprocess.run(["python3", "-m", "venv", venv_path], check=True)

        # Determine pip path based on platform
        if os.name == 'nt':  # Windows
            pip_path = os.path.join(venv_path, "Scripts", "pip")
            python_path = os.path.join(venv_path, "Scripts", "python")
        else:  # Linux/Mac
            pip_path = os.path.join(venv_path, "bin", "pip")
            python_path = os.path.join(venv_path, "bin", "python")

        # Install the extracted requirements.
        req_file = os.path.join(temp_dir, "requirements.txt")
        if os.path.exists(req_file):
            print(f"Installing requirements from {req_file}")
            pip_install = subprocess.run([pip_path, "install", "-r", req_file],
                                        capture_output=True, text=True)
            if pip_install.returncode != 0:
                print(f"pip install failed: {pip_install.stderr}")
                return jsonify({"error": f"pip install failed: {pip_install.stderr}"}), 500
        else:
            print("No requirements.txt file found, installing default regression packages")
            subprocess.run([pip_path, "install", "pandas", "numpy", "scikit-learn", "joblib", "matplotlib"], 
                          capture_output=True, text=True)

        # Check if predict.py exists, if not create a simple one for regression
        predict_script = os.path.join(temp_dir, "predict.py")
        if not os.path.exists(predict_script):
            # Try to find the model file
            model_files = []
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.joblib') or file.endswith('.pkl'):
                        model_files.append(os.path.join(root, file))
            
            if not model_files:
                return jsonify({"error": "No model file (.joblib or .pkl) found in package"}), 400
            
            # Create a simple predict.py script
            with open(predict_script, 'w') as f:
                f.write("""import pandas as pd
import numpy as np
import joblib
import sys
import os
import json
import glob
import traceback

def find_file(pattern):
    \"\"\"Find files matching a pattern\"\"\"
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def load_preprocessing_info():
    \"\"\"Load preprocessing information from json files\"\"\"
    preprocessing_info = {}
    
    # Look for encoding/preprocessing files
    encoding_files = []
    for filename in glob.glob("*.json"):
        if any(key in filename.lower() for key in ['encoding', 'preprocess', 'metadata', 'feature']):
            encoding_files.append(filename)
    
    print(f"Found potential preprocessing files: {encoding_files}")
    
    # Load each file and extract preprocessing info
    for file_path in encoding_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract different types of preprocessing information
            if isinstance(data, dict):
                # Look for selected features
                if 'selected_features' in data:
                    preprocessing_info['selected_features'] = data['selected_features']
                    print(f"Found selected features in {file_path}")
                
                # Look for encoding mappings
                if 'encoding_mappings' in data:
                    preprocessing_info['encoding_mappings'] = data['encoding_mappings']
                    print(f"Found encoding mappings in {file_path}")
                
                # The file itself might be encoding mappings
                if 'selected_features' not in data and 'encoding_mappings' not in data:
                    # Check if this looks like an encoding mapping file
                    has_mappings = False
                    for key, value in data.items():
                        if isinstance(value, dict) and len(value) > 0:
                            has_mappings = True
                            break
                    
                    if has_mappings:
                        preprocessing_info['encoding_mappings'] = data
                        print(f"Found encoding mappings in {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
    
    print(f"Extracted preprocessing info: {preprocessing_info.keys()}")
    return preprocessing_info

def preprocess_data(data, preprocessing_info):
    \"\"\"Apply preprocessing to the input data\"\"\"
    if not preprocessing_info:
        return data
    
    # Make a copy to avoid modifying the original
    processed_data = data.copy()
    
    # Apply feature selection if available
    if 'selected_features' in preprocessing_info:
        selected_features = preprocessing_info['selected_features']
        available_features = [f for f in selected_features if f in processed_data.columns]
        missing_features = [f for f in selected_features if f not in processed_data.columns]
        
        if missing_features:
            print(f"Warning: Missing features in input data: {missing_features}")
        
        if available_features:
            processed_data = processed_data[available_features]
            print(f"Selected {len(available_features)} features")
    
    # Apply encoding if available
    if 'encoding_mappings' in preprocessing_info:
        encodings = preprocessing_info['encoding_mappings']
        for column, mapping in encodings.items():
            if column in processed_data.columns:
                # Check if this is a categorical encoding
                if isinstance(mapping, dict):
                    # Create a mapping function that handles missing keys
                    def map_value(val):
                        # Try direct lookup
                        if val in mapping:
                            return mapping[val]
                        # Try string version
                        str_val = str(val)
                        if str_val in mapping:
                            return mapping[str_val]
                        # Return original if not found
                        print(f"Warning: Value '{val}' not found in encoding for column '{column}'")
                        return val
                    
                    # Apply mapping
                    try:
                        processed_data[column] = processed_data[column].map(map_value)
                        print(f"Applied encoding to column '{column}'")
                    except Exception as e:
                        print(f"Error applying encoding to column '{column}': {str(e)}")
    
    return processed_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_file> [--stdout]")
        return 1
    
    print("Starting prediction process...")
    
    input_file = sys.argv[1]
    stdout_mode = "--stdout" in sys.argv
    
    try:
        # Find model file
        model_file = None
        for root, _, files in os.walk('.'):
            for file in files:
                if file.endswith('.joblib') or file.endswith('.pkl'):
                    model_file = os.path.join(root, file)
                    break
            if model_file:
                break
        
        if not model_file:
            print("No model file found")
            return 1
        
        print(f"Using model file: {model_file}")
        
        # Load preprocessing information
        preprocessing_info = load_preprocessing_info()
        
        # Load input data
        try:
            if input_file.endswith('.csv'):
                data = pd.read_csv(input_file)
            elif input_file.endswith('.xlsx') or input_file.endswith('.xls'):
                data = pd.read_excel(input_file)
            else:
                print(f"Unsupported file format: {input_file}")
                return 1
                
            print(f"Loaded input data with shape: {data.shape}")
        except Exception as e:
            print(f"Error loading input file: {str(e)}")
            return 1
        
        # Apply preprocessing
        X = preprocess_data(data, preprocessing_info)
        print(f"Data after preprocessing, shape: {X.shape}")
        
        # Load model
        try:
            model = joblib.load(model_file)
            print(f"Successfully loaded model from {model_file}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(traceback.format_exc())
            return 1
        
        # Make predictions
        try:
            predictions = model.predict(X)
            print(f"Generated {len(predictions)} predictions")
            
            # Add predictions to original data
            data['prediction'] = predictions
            
            # Debug print
            print(f"First few predictions: {predictions[:5]}")
            
            # Output to stdout or file
            if stdout_mode:
                csv_output = data.to_csv(index=False)
                print(csv_output)
            else:
                data.to_csv('output.csv', index=False)
                print(f"Saved predictions to output.csv")
            
            return 0
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            print(traceback.format_exc())
            return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
""")
            print("Created enhanced predict.py script")

            # Also create an empty __init__.py to make imports work better
            with open(os.path.join(temp_dir, "__init__.py"), 'w') as f:
                f.write("# Package initialization file")
                
        print(f"Running prediction with input file: {input_file.filename}")
        
        # Run the prediction with stdout capture mode instead of file output
        result = subprocess.run(
            [python_path, predict_script, input_file_path, "--stdout"],
            cwd=temp_dir,
            capture_output=True, text=True, timeout=300  # 5 minute timeout
        )
        
        print(f"Prediction completed with return code {result.returncode}")
        
        # Check if the process succeeded
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout[:500]}")  # Show beginning of stdout for debugging
            return jsonify({"error": f"Prediction script failed: {result.stderr}"}), 400
            
        # If we have output in stdout, parse it as CSV    
        if result.stdout.strip():
            print("Found prediction output in stdout")
            output_data = result.stdout
            
            # Debug: Print the first few lines of the output
            output_lines = output_data.strip().split('\n')
            print(f"Output first line: {output_lines[0] if output_lines else 'No lines'}")
            print(f"Total output lines: {len(output_lines)}")
            
            # Try to validate it's valid CSV
            try:
                reader = csv.reader(io.StringIO(output_data))
                rows = list(reader)
                if len(rows) > 0:
                    print(f"Valid CSV found with {len(rows)} rows, columns: {rows[0]}")
                    # Check if predictions column exists
                    if 'prediction' in rows[0]:
                        print("Found 'prediction' column in results")
                else:
                    print("Warning: Empty CSV output")
            except Exception as e:
                print(f"Warning: Output is not valid CSV: {str(e)}")
                
            # Return the CSV content
            return jsonify({"output_file": output_data})
            
        # If no stdout, check if output.csv was created
        output_path = os.path.join(temp_dir, "output.csv")
        if os.path.exists(output_path):
            print(f"Reading output from file: {output_path}")
            
            # Read and return the file content
            with open(output_path, 'r') as f:
                output_data = f.read()
                
            # Debug: Print the first few lines of the output
            output_lines = output_data.strip().split('\n')
            print(f"Output file first line: {output_lines[0] if output_lines else 'No lines'}")
            print(f"Total output file lines: {len(output_lines)}")
                
            return jsonify({"output_file": output_data})
        
        # No output found
        return jsonify({"error": "No output generated from prediction script"}), 500
        
    except subprocess.TimeoutExpired:
        print("Prediction script timed out after 5 minutes")
        return jsonify({"error": "Prediction script timed out after 5 minutes"}), 408
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary directory
        try:
            # Delayed cleanup to ensure any file operations are complete
            def delayed_cleanup():
                time.sleep(1)  # Short delay
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory: {temp_dir}")
                    
            Thread(target=delayed_cleanup).start()
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    from waitress import serve
    print(f"Starting Regression Predictor Service on port {port}")
    serve(app, host='0.0.0.0', port=port) 