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
import traceback
from flask import Flask, request, jsonify
import uuid
import datetime
from werkzeug.utils import secure_filename

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

# Set max content length to 100MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Create data directories if they don't exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for service monitoring"""
    request_id = str(uuid.uuid4())[:8]
    print(f"[{request_id}] Health check request received at {datetime.datetime.now().isoformat()}")
    
    # Add system info to health response
    health_info = {
        "status": "healthy", 
        "service": "regression_predictor",
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform
    }
    
    print(f"[{request_id}] Returning health status: {health_info}")
    return jsonify(health_info), 200

@app.route('/extract_encodings', methods=['POST'])
def extract_encodings():
    """Extract encoding maps from a model package"""
    request_id = str(uuid.uuid4())[:8]
    print(f"[{request_id}] ===== ENCODING EXTRACTION REQUEST =====")
    print(f"[{request_id}] Timestamp: {datetime.datetime.now().isoformat()}")
    
    # Check if the post request has the file part
    if 'model_package' not in request.files:
        print(f"[{request_id}] Error: No model_package file in request")
        return jsonify({"error": "No model_package file provided"}), 400
        
    model_file = request.files['model_package']
    
    # If user does not select file, the browser might
    # submit an empty file without a filename
    if model_file.filename == '':
        print(f"[{request_id}] Error: Empty filename")
        return jsonify({"error": "No model file selected"}), 400
        
    # Check if it's a ZIP file
    if not model_file.filename.lower().endswith('.zip'):
        print(f"[{request_id}] Error: File is not a ZIP archive: {model_file.filename}")
        return jsonify({"error": "Model package must be a ZIP file"}), 400
    
    # Create a temporary directory to extract the model package
    temp_dir = tempfile.mkdtemp(prefix="model_extract_")
    print(f"[{request_id}] Created temporary directory: {temp_dir}")
    
    start_time = time.time()
    
    try:
        # Save the model package to the temporary directory
        model_path = os.path.join(temp_dir, secure_filename(model_file.filename))
        model_file.save(model_path)
        print(f"[{request_id}] Saved model package to: {model_path}")
        
        # Extract the package
        import zipfile
        try:
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            print(f"[{request_id}] Extracted model package to: {temp_dir}")
        except Exception as e:
            print(f"[{request_id}] Error extracting ZIP archive: {str(e)}")
            return jsonify({"error": f"Invalid ZIP file: {str(e)}"}), 400
        
        # Process the extracted package contents
        return process_extracted_package(temp_dir, request_id, start_time)
        
    except Exception as e:
        print(f"[{request_id}] Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"[{request_id}] Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"[{request_id}] Error cleaning up: {str(e)}")

def process_extracted_package(temp_dir, request_id, start_time):
    """Process the contents of an extracted model package"""
    print(f"[{request_id}] Processing extracted package contents")
    
    # Look for encoding files - check all JSON files
    encoding_files = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.json'):
                if any(key in file.lower() for key in ['encoding', 'preprocess', 'metadata', 'feature']):
                    encoding_files.append(os.path.join(root, file))
    
    print(f"[{request_id}] Found potential encoding files: {encoding_files}")
    
    # If no encoding files found, return empty result instead of error
    if not encoding_files:
        print(f"[{request_id}] No encoding files found in package")
        return jsonify({
            "encoding_maps": [],
            "message": "No encoding maps found in the package"
        })
    
    # Process encoding files
    encoding_maps = []
    
    for file_path in encoding_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Look for encoding mappings in different formats
            if isinstance(data, dict):
                # Look for direct encoding dictionary {feature_name: {value: encoded}}
                for key, value in data.items():
                    if isinstance(value, dict) and all(isinstance(k, str) for k in value.keys()):
                        # Check if values are numeric (encoding maps typically map to numbers)
                        if all(isinstance(v, (int, float)) for v in value.values()):
                            # This looks like an encoding map!
                            rel_path = os.path.relpath(file_path, temp_dir)
                            encoding_maps.append({
                                "feature_name": key,
                                "source_file": rel_path,
                                "mapping": value,
                                "num_values": len(value)
                            })
                            print(f"[{request_id}] Found encoding map for feature '{key}' with {len(value)} values")
                
                # Look for encoding_mappings section
                if 'encoding_mappings' in data and isinstance(data['encoding_mappings'], dict):
                    for key, value in data['encoding_mappings'].items():
                        if isinstance(value, dict):
                            rel_path = os.path.relpath(file_path, temp_dir)
                            encoding_maps.append({
                                "feature_name": key,
                                "source_file": rel_path,
                                "mapping": value,
                                "num_values": len(value)
                            })
                            print(f"[{request_id}] Found encoding map for feature '{key}' in encoding_mappings")
                
                # Look for column_encodings section
                if 'column_encodings' in data and isinstance(data['column_encodings'], dict):
                    for key, value in data['column_encodings'].items():
                        if isinstance(value, dict):
                            rel_path = os.path.relpath(file_path, temp_dir)
                            encoding_maps.append({
                                "feature_name": key,
                                "source_file": rel_path,
                                "mapping": value,
                                "num_values": len(value)
                            })
                            print(f"[{request_id}] Found encoding map for feature '{key}' in column_encodings")
        
        except Exception as e:
            print(f"[{request_id}] Error processing encoding file {file_path}: {str(e)}")
    
    # Filter out non-feature encoding maps
    filtered_maps = []
    for encoding_map in encoding_maps:
        feature_name = encoding_map["feature_name"]
        # Skip metadata keys that aren't actual features
        if feature_name in ["__model_info__", "__metadata__", "model_type", "version", "timestamp"]:
            continue
        filtered_maps.append(encoding_map)
    
    total_time = time.time() - start_time
    print(f"[{request_id}] ===== ENCODING EXTRACTION COMPLETED in {total_time:.2f}s =====")
    print(f"[{request_id}] Found {len(filtered_maps)} encoding maps")
    
    # Return the list of encoding maps
    return jsonify({
        "encoding_maps": filtered_maps,
        "count": len(filtered_maps)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the provided model and input data"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    print(f"[{request_id}] ===== PREDICTION REQUEST =====")
    print(f"[{request_id}] Timestamp: {datetime.datetime.now().isoformat()}")
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="predict_")
    print(f"[{request_id}] Created temporary directory: {temp_dir}")
    
    try:
        # Check if the post request has the file parts
        if 'input_file' not in request.files:
            print(f"[{request_id}] Error: No input_file in request")
            return jsonify({"error": "No input file provided"}), 400
            
        if 'model_package' not in request.files:
            print(f"[{request_id}] Error: No model_package in request")
            return jsonify({"error": "No model package provided"}), 400
            
        input_file = request.files['input_file']
        model_file = request.files['model_package']
        
        # If user does not select file, browser might submit empty file without filename
        if input_file.filename == '':
            print(f"[{request_id}] Error: Empty input filename")
            return jsonify({"error": "No input file selected"}), 400
            
        if model_file.filename == '':
            print(f"[{request_id}] Error: Empty model filename")
            return jsonify({"error": "No model package selected"}), 400
            
        # Validate file types
        if not (input_file.filename.lower().endswith('.csv') or 
                input_file.filename.lower().endswith('.xlsx') or 
                input_file.filename.lower().endswith('.xls')):
            print(f"[{request_id}] Error: Input file is not CSV or Excel: {input_file.filename}")
            return jsonify({"error": "Input file must be CSV or Excel"}), 400
            
        if not model_file.filename.lower().endswith('.zip'):
            print(f"[{request_id}] Error: Model file is not a ZIP archive: {model_file.filename}")
            return jsonify({"error": "Model package must be a ZIP file"}), 400
        
        # Save the files
        input_file_path = os.path.join(temp_dir, secure_filename(input_file.filename))
        model_path = os.path.join(temp_dir, secure_filename(model_file.filename))
        
        input_file.save(input_file_path)
        model_file.save(model_path)
        
        print(f"[{request_id}] Saved input file to: {input_file_path}")
        print(f"[{request_id}] Saved model package to: {model_path}")
        
        # Extract the model package
        import zipfile
        try:
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            print(f"[{request_id}] Extracted model package to: {temp_dir}")
        except Exception as e:
            print(f"[{request_id}] Error extracting ZIP archive: {str(e)}")
            return jsonify({"error": f"Invalid ZIP file: {str(e)}"}), 400
        
        # Check if we can find preprocessing_info.json
        preprocessing_file = os.path.join(temp_dir, "preprocessing_info.json")
        if not os.path.exists(preprocessing_file):
            # Look for encoding files and create preprocessing info
            encoding_files = []
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.json'):
                        if any(key in file.lower() for key in ['encoding', 'preprocess', 'metadata', 'feature']):
                            encoding_files.append(os.path.join(root, file))
            
            print(f"[{request_id}] Found potential encoding files: {encoding_files}")
            
            # Create preprocessing_info.json if needed
            if encoding_files:
                encoding_mappings = {}
                for file_path in encoding_files:
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        if isinstance(data, dict):
                            # Look for encoding_mappings
                            if 'encoding_mappings' in data and isinstance(data['encoding_mappings'], dict):
                                encoding_mappings.update(data['encoding_mappings'])
                            
                            # Look for column_encodings
                            if 'column_encodings' in data and isinstance(data['column_encodings'], dict):
                                encoding_mappings.update(data['column_encodings'])
                            
                            # Look for direct encoding dictionaries
                            for key, value in data.items():
                                if isinstance(value, dict) and all(isinstance(k, str) for k in value.keys()):
                                    if all(isinstance(v, (int, float)) for v in value.values()):
                                        encoding_mappings[key] = value
                    except Exception as e:
                        print(f"[{request_id}] Error processing encoding file {file_path}: {str(e)}")
                
                if encoding_mappings:
                    with open(preprocessing_file, 'w') as f:
                        json.dump({"encoding_mappings": encoding_mappings}, f)
                    print(f"[{request_id}] Created preprocessing_info.json with {len(encoding_mappings)} encodings")
        
        # Check if we can find requirements.txt and create one if not
        requirements_file = os.path.join(temp_dir, "requirements.txt")
        if not os.path.exists(requirements_file):
            with open(requirements_file, 'w') as f:
                f.write("pandas>=1.0.0\n")
                f.write("scikit-learn>=0.22.0\n")
                f.write("joblib>=0.14.0\n")
                f.write("numpy>=1.18.0\n")
                f.write("openpyxl>=3.0.0\n")  # For Excel support
                f.write("xgboost>=1.0.0\n")   # Common for regression
            print(f"[{request_id}] Created default requirements.txt")
        
        # Create Python virtual environment
        venv_dir = os.path.join(temp_dir, "venv")
        python_path = os.path.join(venv_dir, "bin", "python")
        if sys.platform.startswith('win'):
            python_path = os.path.join(venv_dir, "Scripts", "python.exe")
        
        print(f"[{request_id}] Creating Python virtual environment at {venv_dir}")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
            print(f"[{request_id}] Created virtual environment")
            
            # Install dependencies
            print(f"[{request_id}] Installing dependencies from {requirements_file}")
            pip_path = os.path.join(venv_dir, "bin", "pip")
            if sys.platform.startswith('win'):
                pip_path = os.path.join(venv_dir, "Scripts", "pip.exe")
                
            result = subprocess.run(
                [pip_path, "install", "-r", requirements_file],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"[{request_id}] pip install error: {result.stderr}")
            else:
                print(f"[{request_id}] Successfully installed dependencies")
        except Exception as e:
            print(f"[{request_id}] Error setting up environment: {str(e)}")
            return jsonify({"error": f"Failed to set up environment: {str(e)}"}), 500
        
        # Generate predict.py script
        predict_script = os.path.join(temp_dir, "predict.py")
        with open(predict_script, 'w') as f:
            f.write("""
import os
import sys
import json
import glob
import joblib
import pandas as pd
import traceback

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_file> [--stdout]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)
    
    stdout_mode = "--stdout" in sys.argv
    
    # Find preprocessing info (encodings, selected features, etc.)
    preprocessing_info = {}
    encoding_files = []
    
    for filename in glob.glob("*.json"):
        if any(key in filename.lower() for key in ['encoding', 'preprocess', 'metadata', 'feature']):
            encoding_files.append(filename)
    
    print(f"Found potential preprocessing files: {encoding_files}")
    
    # Load preprocessing information
    for file_path in encoding_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Look for selected features
            if isinstance(data, dict):
                # Add selected features if available
                if 'selected_features' in data and isinstance(data['selected_features'], list):
                    preprocessing_info['selected_features'] = data['selected_features']
                    print(f"Found selected features in {file_path}")
                
                # Add encoding mappings if available
                if 'encoding_mappings' in data and isinstance(data['encoding_mappings'], dict):
                    preprocessing_info['encoding_mappings'] = data['encoding_mappings']
                    print(f"Found encoding mappings in {file_path}")
                
                # For simpler files, the whole file might be encoding mappings
                for key, value in data.items():
                    if isinstance(value, dict) and any(isinstance(v, (int, float)) for v in value.values()):
                        if 'column_encodings' not in preprocessing_info:
                            preprocessing_info['column_encodings'] = {}
                        preprocessing_info['column_encodings'][key] = value
                        print(f"Found encoding for column {key}")
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
    
    # Find model file
    model_file = None
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.joblib') or file.endswith('.pkl'):
                model_file = os.path.join(root, file)
                
                # Check if it's a placeholder
                try:
                    file_size = os.path.getsize(model_file)
                    if file_size < 1000:  # Less than 1KB
                        with open(model_file, 'r') as f:
                            content = f.read()
                            if "placeholder" in content.lower():
                                print(f"Found placeholder instead of real model: {model_file}")
                                print(f"Content: {content}")
                                print("Error: Model file is just a placeholder.")
                                sys.exit(1)
                except Exception as e:
                    # If we can't read it as text, it's probably a real model file
                    pass
                
                break
        if model_file:
            break
    
    if not model_file:
        print("No model file found")
        sys.exit(1)
    
    print(f"Using model file: {model_file}")
    
    # Load model
    try:
        model = joblib.load(model_file)
        print(f"Successfully loaded model from {model_file}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Load input data
    try:
        if input_file.endswith('.csv'):
            data = pd.read_csv(input_file)
        elif input_file.endswith('.xlsx') or input_file.endswith('.xls'):
            data = pd.read_excel(input_file)
        else:
            print(f"Unsupported file format: {input_file}")
            sys.exit(1)
        
        print(f"Loaded input data with shape {data.shape}")
    except Exception as e:
        print(f"Error loading input file: {str(e)}")
        sys.exit(1)
    
    # Apply preprocessing if needed
    input_data = data.copy()
    
    # Apply feature selection if needed
    if 'selected_features' in preprocessing_info:
        selected_features = preprocessing_info['selected_features']
        available_features = [f for f in selected_features if f in input_data.columns]
        missing_features = [f for f in selected_features if f not in input_data.columns]
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
        
        if available_features:
            input_data = input_data[available_features]
            print(f"Selected {len(available_features)} features from original {len(data.columns)}")
    
    # Apply encoding if needed
    encodings = {}
    if 'encoding_mappings' in preprocessing_info:
        encodings.update(preprocessing_info['encoding_mappings'])
    if 'column_encodings' in preprocessing_info:
        encodings.update(preprocessing_info['column_encodings'])
    
    if encodings:
        for column, mapping in encodings.items():
            if column in input_data.columns:
                # Convert categorical values according to the mapping
                try:
                    # Create a mapping function that handles missing keys
                    def map_value(val):
                        if val in mapping:
                            return mapping[val]
                        str_val = str(val)
                        if str_val in mapping:
                            return mapping[str_val]
                        print(f"Warning: Value '{val}' not found in mapping for '{column}'")
                        return val
                    
                    # Apply mapping
                    input_data[column] = input_data[column].map(map_value)
                    print(f"Applied encoding to column '{column}'")
                except Exception as e:
                    print(f"Error applying encoding to column '{column}': {str(e)}")
    
    # Make predictions
    try:
        predictions = model.predict(input_data)
        print(f"Generated {len(predictions)} predictions")
        
        # Add predictions to original data
        data['prediction'] = predictions
        
        # Debug print
        print(f"Sample predictions: {predictions[:5] if len(predictions) >= 5 else predictions}")
        
        # Output
        if stdout_mode:
            print(data.to_csv(index=False))
        else:
            output_file = 'output.csv'
            data.to_csv(output_file, index=False)
            print(f"Saved predictions to {output_file}")
        
        return 0
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
""")
            print("Created basic predict.py script for regression")
        
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
            
            # Try to validate it's valid CSV
            try:
                reader = csv.reader(io.StringIO(output_data))
                rows = list(reader)
                if len(rows) > 0:
                    print(f"Valid CSV found with {len(rows)} rows, columns: {rows[0]}")
                    
                    # Check if this is an error CSV (single column named error_message)
                    if len(rows[0]) == 1 and rows[0][0] == 'error_message':
                        error_message = rows[1][0] if len(rows) > 1 else "Unknown error in prediction"
                        print(f"Error message in CSV: {error_message}")
                        return jsonify({"error": error_message}), 400
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
        print(f"[{request_id}] ERROR: Unexpected exception: {str(e)}")
        import traceback
        print(f"[{request_id}] Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary directory
        try:
            # Delayed cleanup to ensure any file operations are complete
            def delayed_cleanup():
                time.sleep(1)  # Short delay
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"[{request_id}] Cleaned up temporary directory: {temp_dir}")
                    
            from threading import Thread
            Thread(target=delayed_cleanup).start()
        except Exception as e:
            print(f"[{request_id}] Warning: Failed to clean up temporary directory: {str(e)}")

        total_time = time.time() - start_time
        print(f"[{request_id}] ===== PREDICTION REQUEST COMPLETED in {total_time:.2f}s =====")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    from waitress import serve
    
    # Print startup banner with service info
    print("="*80)
    print(f"Starting Regression Predictor Service v1.0")
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Listening on port: {port}")
    print("="*80)
    
    serve(app, host='0.0.0.0', port=port) 