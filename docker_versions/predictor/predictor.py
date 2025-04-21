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
import glob
import uuid

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

def cleanup_all_temp_dirs():
    """Perform a thorough cleanup of all temporary directories we might have created"""
    print("Performing aggressive cleanup of all temporary directories")
    patterns = ["encoding_extract_*", "session_*"]
    
    for pattern in patterns:
        try:
            for dir_path in glob.glob(os.path.join(tempfile.gettempdir(), pattern)):
                try:
                    print(f"Cleaning up temporary directory: {dir_path}")
                    shutil.rmtree(dir_path, ignore_errors=True)
                except Exception as e:
                    print(f"Error cleaning up directory {dir_path}: {str(e)}")
        except Exception as e:
            print(f"Error during cleanup of pattern {pattern}: {str(e)}")

# Add a startup cleanup
cleanup_all_temp_dirs()

@app.route('/health', methods=['GET'])
def health():
    # Periodically clean up temporary directories on health checks
    cleanup_all_temp_dirs()
    return jsonify({"status": "ok"}), 200

@app.route('/cleanup', methods=['GET'])
def cleanup():
    """Endpoint to manually trigger cleanup of temporary directories"""
    cleanup_all_temp_dirs()
    return jsonify({"status": "cleanup completed"}), 200

@app.route('/extract_encodings', methods=['POST'])
def extract_encodings():
    """Extract encoding maps from a model package"""
    if 'model_package' not in request.files:
        return jsonify({"error": "Model package file is required"}), 400
    
    model_file = request.files['model_package']
    
    # Validate file type
    if not model_file.filename.lower().endswith('.zip'):
        return jsonify({"error": "Model package must be a ZIP archive"}), 400
    
    # Check if any temporary directories with similar prefixes exist and clean them up
    for dir_path in glob.glob(os.path.join(tempfile.gettempdir(), "encoding_extract_*")):
        try:
            # Check if it's older than 1 hour
            if time.time() - os.path.getctime(dir_path) > 3600:
                print(f"Cleaning up stale temporary directory: {dir_path}")
                shutil.rmtree(dir_path, ignore_errors=True)
        except Exception as e:
            print(f"Error cleaning up directory {dir_path}: {str(e)}")
    
    # Create a unique temporary directory with UUID to guarantee uniqueness
    unique_id = str(uuid.uuid4())[:8]
    temp_dir = tempfile.mkdtemp(prefix=f"encoding_extract_{int(time.time())}_{unique_id}_")
    
    print(f"Created new COMPLETELY ISOLATED temporary directory: {temp_dir}")
    
    try:
        # Save and extract the ZIP package
        zip_path = os.path.join(temp_dir, "model_package.zip")
        model_file.save(zip_path)
        extraction_time = time.time()  # Record when extraction happens
        print(f"Saved zip file at {zip_path} with timestamp {extraction_time}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        print(f"Extracted model package to {temp_dir}")
        
        # List all extracted files with timestamps for debugging
        print(f"LISTING ALL FILES IN TEMPORARY DIRECTORY {temp_dir}:")
        extracted_files = []
        suspicious_files = []
        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_time = os.path.getmtime(file_path)
                time_diff = abs(file_time - extraction_time)
                rel_path = os.path.relpath(file_path, temp_dir)
                print(f"  - {rel_path}: mod_time={file_time}, diff={time_diff}s")
                
                # Track files that are part of this extraction
                if time_diff < 5:
                    extracted_files.append(rel_path)
                else:
                    suspicious_files.append(file_path)
                    print(f"    WARNING: File {rel_path} has suspicious timestamp, will be excluded from validation!")
        
        # Actually remove suspicious files to prevent them from being used
        for file_path in suspicious_files:
            try:
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    print(f"REMOVING suspicious file: {file_path}")
                    os.remove(file_path)
            except Exception as e:
                print(f"Error removing suspicious file {file_path}: {str(e)}")
        
        print(f"Validated files in extracted package: {extracted_files}")
        
        # Look for encoding files with broader search
        encoding_files = []
        json_files = []
        
        # First pass: look for standard encoding file names
        for file_path in extracted_files:
            full_path = os.path.join(temp_dir, file_path)
            if file_path.lower().endswith('.json'):
                json_files.append(file_path)
                filename = os.path.basename(file_path)
                if (filename == 'encoding_mappings.json' or 
                    filename == 'encoding_mappings' or
                    filename == 'preprocessing_info.json' or 
                    filename == 'preprocessing_info' or
                    filename.endswith('_encoding.json') or 
                    filename == 'encoding.json' or 
                    filename == 'preprocessing.json' or
                    'encod' in filename.lower() or
                    'target' in filename.lower() or
                    'label' in filename.lower() or
                    'map' in filename.lower()):
                    encoding_files.append(file_path)
        
        print(f"Found {len(json_files)} JSON files: {json_files}")
        print(f"Potential encoding files based on name: {encoding_files}")
        
        # If no encoding files found by name, try all JSON files
        if not encoding_files and json_files:
            print("No encoding files found by name pattern, examining all JSON files")
            encoding_files = json_files
        
        if not encoding_files:
            print("No encoding files found in the model package")
            return jsonify({
                "encoding_maps": {},
                "metadata": {
                    "info": {
                        "value_count": 0,
                        "sample_values": [],
                        "display_name": "No Encodings Found",
                        "message": "This model appears to use only numeric features."
                    }
                },
                "message": "No encoding maps found. Select 'No encoding (numeric target)' option."
            }), 200
        
        # Look for encoding files in the validated model files
        for file_path in extracted_files:
            if file_path.lower().endswith('.json'):
                full_path = os.path.join(temp_dir, file_path)
                filename = os.path.basename(file_path)
                
                if (filename == 'encoding_mappings.json' or 
                    filename == 'encoding_mappings' or
                    filename == 'preprocessing_info.json' or 
                    filename == 'preprocessing_info' or
                    filename.endswith('_encoding.json') or 
                    filename == 'encoding.json' or 
                    filename == 'preprocessing.json' or
                    'encod' in filename.lower() or
                    'target' in filename.lower() or
                    'label' in filename.lower() or
                    'map' in filename.lower()):
                    
                    encoding_files.append(full_path)
                    print(f"Adding validated encoding file: {file_path}")
        
        print(f"Found potential encoding files: {encoding_files}")
        
        # Create preprocessing_info.json if it doesn't exist
        preprocessing_info_path = os.path.join(temp_dir, "preprocessing_info.json")
        if not os.path.exists(preprocessing_info_path) or preprocessing_info_path not in encoding_files:
            print("Creating a minimal preprocessing_info.json with empty encoding_mappings")
            with open(preprocessing_info_path, 'w') as f:
                json.dump({"encoding_mappings": {}}, f)
            
            # Add to encoding files if not already there
            if preprocessing_info_path not in encoding_files:
                encoding_files.append(preprocessing_info_path)
            
            print(f"Created preprocessing_info.json at {preprocessing_info_path}")
        
        # Ensure preprocessing_info.json has encoding_mappings
        try:
            with open(preprocessing_info_path, 'r') as f:
                preprocessing_info = json.load(f)
                
            # Add empty encoding_mappings if it doesn't exist
            if 'encoding_mappings' not in preprocessing_info:
                preprocessing_info['encoding_mappings'] = {}
                with open(preprocessing_info_path, 'w') as f:
                    json.dump(preprocessing_info, f)
                print("Updated preprocessing_info.json with empty encoding_mappings field")
        except Exception as e:
            print(f"Error reading or updating preprocessing_info.json: {str(e)}")
            # Create a new one from scratch if there was an error
            with open(preprocessing_info_path, 'w') as f:
                json.dump({"encoding_mappings": {}}, f)
            print("Created new preprocessing_info.json after error")
        
        # Extract encoding maps from each file
        encoding_maps = {}
        for file in encoding_files:
            file_path = os.path.join(temp_dir, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                print(f"Examining file {file} for encoding maps")
                
                # Handle the case where encoding_mappings is empty but exists
                if 'encoding_mappings' in data and data['encoding_mappings'] is None:
                    data['encoding_mappings'] = {}
                
                # Handle the case where preprocessing_info.json exists but has no encoding_mappings key
                if os.path.basename(file) == 'preprocessing_info.json' and 'encoding_mappings' not in data:
                    data['encoding_mappings'] = {}
                    print("Added empty encoding_mappings to preprocessing_info.json")
                    
                    # Save the updated preprocessing_info.json
                    with open(file_path, 'w') as f:
                        json.dump(data, f)
                    print("Updated preprocessing_info.json with empty encoding_mappings")
                
                # Process different potential formats
                if isinstance(data, dict):
                    # Look for dictionary structures that could be encoding maps
                    for column, mapping in data.items():
                        if isinstance(mapping, dict):
                            # Look for typical encoding patterns (numeric keys or values)
                            is_encoding_map = False
                            
                            # Check if it maps numbers to strings (common encoding pattern)
                            numeric_keys = False
                            string_values = False
                            
                            for k, v in mapping.items():
                                # Try to convert key to number
                                try:
                                    float(k)
                                    numeric_keys = True
                                except (ValueError, TypeError):
                                    pass
                                
                                # Check if value is string
                                if isinstance(v, str):
                                    string_values = True
                            
                            # If keys are numbers and values are strings, likely an encoding map
                            if numeric_keys and string_values:
                                is_encoding_map = True
                            
                            # Also check if it has a reasonable number of entries to be a category mapping
                            if len(mapping) > 0 and len(mapping) < 100:
                                # More likely to be an encoding map than a large data structure
                                is_encoding_map = True
                                
                            if is_encoding_map:
                                print(f"Found encoding map for column '{column}' with {len(mapping)} values")
                                encoding_maps[column] = mapping
                    
                    # Also check if there's a 'target_map' or similar key
                    for key in ['target_map', 'target_encoding', 'label_encoding', 'class_mapping', 'label_map']:
                        if key in data and isinstance(data[key], dict):
                            print(f"Found encoding map with key '{key}' containing {len(data[key])} values")
                            encoding_maps[key] = data[key]
                    
                    # Check for patterns like "column_name_encoding" or "column_name_map"
                    for key in data.keys():
                        if (key.endswith('_encoding') or key.endswith('_map') or '_encoding_' in key) and isinstance(data[key], dict):
                            print(f"Found encoding map with pattern-matched key '{key}' containing {len(data[key])} values")
                            encoding_maps[key] = data[key]
            except Exception as e:
                # Skip files that can't be parsed
                print(f"Error parsing {file}: {str(e)}")
                continue
        
        if not encoding_maps:
            print("No encoding maps found in any of the JSON files")
            return jsonify({
                "encoding_maps": {},
                "metadata": {
                    "info": {
                        "value_count": 0,
                        "sample_values": [],
                        "display_name": "No Encodings Found",
                        "message": "This model appears to use only numeric features."
                    }
                },
                "message": "No encoding maps found. Select 'No encoding (numeric target)' option."
            }), 200
        
        # Filter out non-feature encoding maps
        excluded_maps = ['scaler_params', 'encoding_mappings', 'cleaner_config']
        feature_encoding_maps = {k: v for k, v in encoding_maps.items() if k not in excluded_maps}
        
        if not feature_encoding_maps:
            print("No feature encoding maps found after filtering")
            return jsonify({
                "encoding_maps": {},
                "metadata": {
                    "info": {
                        "value_count": 0,
                        "sample_values": [],
                        "display_name": "No Encodings Found",
                        "message": "This model appears to use only numeric features."
                    }
                },
                "message": "No feature encoding maps found. Select 'No encoding (numeric target)' option."
            }), 200
        
        # Add metadata for each encoding map to help the frontend
        encoding_metadata = {}
        for feature_name, mapping in feature_encoding_maps.items():
            # Count the number of values in each encoding map
            value_count = len(mapping)
            # Get some sample values for display
            sample_values = list(mapping.keys())[:3]  # First 3 keys
            # Create metadata entry
            encoding_metadata[feature_name] = {
                "value_count": value_count,
                "sample_values": sample_values,
                "display_name": feature_name
            }
            
        # Return both the encoding maps and metadata
        print(f"Successfully extracted {len(feature_encoding_maps)} feature encoding maps: {list(feature_encoding_maps.keys())}")
        return jsonify({
            "encoding_maps": feature_encoding_maps,
            "metadata": encoding_metadata
        })
    
    except Exception as e:
        print(f"Error in extract_encodings: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.route('/predict', methods=['POST'])
def predict():
    # Verify that both files are provided.
    if 'model_package' not in request.files or 'input_file' not in request.files:
        return jsonify({"error": "Both 'model_package' and 'input_file' must be provided."}), 400

    model_file = request.files['model_package']
    input_file = request.files['input_file']
    
    # Get selected encoding map if provided
    selected_encoding = request.form.get('encoding_column', None)
    
    # Check if "No encoding" option was selected
    skip_decoding = False
    if selected_encoding == "no_encoding":
        skip_decoding = True
        selected_encoding = None
        print("'No encoding' option selected - will skip decoding step")

    # Validate file types.
    if not model_file.filename.lower().endswith('.zip'):
        return jsonify({"error": "Model package must be a ZIP archive."}), 400
    if not (input_file.filename.lower().endswith('.xlsx') or 
            input_file.filename.lower().endswith('.xls') or 
            input_file.filename.lower().endswith('.csv')):
        return jsonify({"error": "Input file must be an Excel or CSV file."}), 400

    # Check if any temporary directories with similar prefixes exist and clean them up
    for dir_path in glob.glob(os.path.join(tempfile.gettempdir(), "session_*")):
        try:
            # Check if it's older than 1 hour
            if time.time() - os.path.getctime(dir_path) > 3600:
                print(f"Cleaning up stale session directory: {dir_path}")
                shutil.rmtree(dir_path, ignore_errors=True)
        except Exception as e:
            print(f"Error cleaning up directory {dir_path}: {str(e)}")
    
    # Create a temporary working directory with a unique identifier.
    unique_id = str(uuid.uuid4())[:8]
    temp_dir = tempfile.mkdtemp(prefix=f"session_{int(time.time())}_{unique_id}_")
    start_time = time.time()
    print(f"Created new COMPLETELY ISOLATED temporary directory: {temp_dir}")
    
    try:
        # Save and extract the ZIP package.
        zip_path = os.path.join(temp_dir, "model_package.zip")
        model_file.save(zip_path)
        extraction_time = time.time()  # Record exact extraction time
        print(f"Saved zip file at {zip_path} with timestamp {extraction_time}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print(f"Extracted model package to {temp_dir}")

        # List all extracted files with timestamps for debugging
        print(f"LISTING ALL FILES IN TEMPORARY DIRECTORY {temp_dir}:")
        extracted_files = []
        suspicious_files = []
        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_time = os.path.getmtime(file_path)
                time_diff = abs(file_time - extraction_time)
                rel_path = os.path.relpath(file_path, temp_dir)
                print(f"  - {rel_path}: mod_time={file_time}, diff={time_diff}s")
                
                # Track files that are part of this extraction
                if time_diff < 5:
                    extracted_files.append(rel_path)
                else:
                    suspicious_files.append(file_path)
                    print(f"    WARNING: File {rel_path} has suspicious timestamp, will be excluded from validation!")
        
        # Actually remove suspicious files to prevent them from being used
        for file_path in suspicious_files:
            try:
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    print(f"REMOVING suspicious file: {file_path}")
                    os.remove(file_path)
            except Exception as e:
                print(f"Error removing suspicious file {file_path}: {str(e)}")
        
        print(f"Validated files in extracted package: {extracted_files}")

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
            print("No requirements.txt file found, installing default packages")
            subprocess.run([pip_path, "install", "pandas", "numpy", "scikit-learn", "joblib"], 
                          capture_output=True, text=True)

        # Run the predict.py script from the package
        predict_script = os.path.join(temp_dir, "predict.py")
        if not os.path.exists(predict_script):
            return jsonify({"error": "predict.py script not found in model package"}), 400

        # Ensure preprocessing_info.json exists
        preprocessing_info_path = os.path.join(temp_dir, 'preprocessing_info.json')
        if not os.path.exists(preprocessing_info_path):
            print("Creating a minimal preprocessing_info.json with empty encoding_mappings")
            with open(preprocessing_info_path, 'w') as f:
                json.dump({"encoding_mappings": {}}, f)

        print(f"Running prediction with input file: {input_file.filename}")
        
        # Run the prediction with stdout capture mode instead of file output
        result = subprocess.run(
            [python_path, predict_script, input_file_path, "--stdout"],
            cwd=temp_dir,
            capture_output=True, text=True, timeout=300  # 5 minute timeout
        )
        
        print(f"Prediction completed with return code {result.returncode}")
        
        # Record extraction time for validation
        extraction_time = os.path.getmtime(os.path.join(temp_dir, "predict.py"))
        print(f"Extraction timestamp reference: {extraction_time}")
        
        # Check for any suspicious files that shouldn't be there
        print("Checking for suspicious files that may cause encoding issues:")
        suspicious_files = []
        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                if filename.endswith('.json'):
                    file_path = os.path.join(root, filename)
                    file_time = os.path.getmtime(file_path)
                    time_diff = abs(file_time - extraction_time)
                    rel_path = os.path.relpath(file_path, temp_dir)
                    print(f"  - JSON file: {rel_path}: mod_time={file_time}, diff={time_diff}s")
                    
                    if time_diff > 5:
                        suspicious_files.append(file_path)
                        print(f"    WARNING: JSON file {rel_path} has suspicious timestamp, will be removed!")
        
        # Remove any suspicious JSON files
        for file_path in suspicious_files:
            try:
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    print(f"REMOVING suspicious JSON file: {file_path}")
                    os.remove(file_path)
            except Exception as e:
                print(f"Error removing suspicious file {file_path}: {str(e)}")
        
        # Check if the process succeeded
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            
            # Check for specific exit code 2 which indicates missing features
            if result.returncode == 2:
                # Try to parse the error output from stdout
                try:
                    error_df = pd.read_csv(io.StringIO(result.stdout))
                    if 'error_message' in error_df.columns:
                        error_message = error_df['error_message'].iloc[0]
                        print(f"Feature validation error: {error_message}")
                        return jsonify({"error": f"Feature validation error: {error_message}"}), 400
                except Exception:
                    pass
                
                # Fallback error message
                return jsonify({"error": "Missing required features. The input dataset must contain all features used during training."}), 400
            else:
                # Generic error handling
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
                    print(f"Valid CSV found with {len(rows)} rows")
                else:
                    print("Warning: Empty CSV output")
            except Exception as e:
                print(f"Warning: Output is not valid CSV: {str(e)}")
            
            # If skip_decoding is True, just parse and return the CSV as is
            if skip_decoding:
                print("Skipping decoding as requested - returning numeric predictions as is")
                try:
                    df = pd.read_csv(io.StringIO(output_data))
                    output_csv = df.to_csv(index=False)
                    return jsonify({"output_file": output_csv})
                except Exception as e:
                    print(f"Error parsing output CSV: {str(e)}")
                    return jsonify({"output_file": output_data})  # Return as is if parsing fails
            
            # If selected_encoding is provided, try to decode the prediction column
            if selected_encoding:
                try:
                    # Find encoding files
                    encoding_files = []
                    print(f"Looking for encoding files to decode predictions using '{selected_encoding}'")
                    
                    # First get a list of original model files to verify encoding file belongs to this model
                    # Use the extraction time to validate files
                    model_files = []
                    for root, _, files in os.walk(temp_dir):
                        for filename in files:
                            if filename.endswith('.py') or filename.endswith('.json'):
                                file_path = os.path.join(root, filename)
                                file_time = os.path.getmtime(file_path)
                                # Only include files that were part of this extraction (within 10 seconds)
                                if abs(file_time - extraction_time) < 10:
                                    rel_path = os.path.relpath(file_path, temp_dir)
                                    model_files.append(rel_path)
                    
                    print(f"Model files that are part of this package: {model_files}")
                    
                    # Look for encoding files in the validated model files
                    for file_path in model_files:
                        if file_path.lower().endswith('.json'):
                            full_path = os.path.join(temp_dir, file_path)
                            filename = os.path.basename(file_path)
                            
                            if (filename == 'encoding_mappings.json' or 
                                filename == 'encoding_mappings' or
                                filename == 'preprocessing_info.json' or 
                                filename == 'preprocessing_info' or
                                filename.endswith('_encoding.json') or 
                                filename == 'encoding.json' or 
                                filename == 'preprocessing.json' or
                                'encod' in filename.lower() or
                                'target' in filename.lower() or
                                'label' in filename.lower() or
                                'map' in filename.lower()):
                                
                                encoding_files.append(full_path)
                                print(f"Adding validated encoding file: {file_path}")
                    
                    print(f"Found potential encoding files: {encoding_files}")
                    
                    # Create preprocessing_info.json if it doesn't exist
                    preprocessing_info_path = os.path.join(temp_dir, "preprocessing_info.json")
                    if not os.path.exists(preprocessing_info_path):
                        print("Creating a minimal preprocessing_info.json with empty encoding_mappings")
                        with open(preprocessing_info_path, 'w') as f:
                            json.dump({"encoding_mappings": {}}, f)
                        
                        # Add to encoding files
                        encoding_files.append(preprocessing_info_path)
                        print(f"Created preprocessing_info.json at {preprocessing_info_path}")
                    else:
                        # Ensure it has encoding_mappings
                        try:
                            with open(preprocessing_info_path, 'r') as f:
                                preprocessing_info = json.load(f)
                                
                            # Add empty encoding_mappings if it doesn't exist
                            if 'encoding_mappings' not in preprocessing_info:
                                preprocessing_info['encoding_mappings'] = {}
                                with open(preprocessing_info_path, 'w') as f:
                                    json.dump(preprocessing_info, f)
                                print("Updated preprocessing_info.json with empty encoding_mappings field")
                        except Exception as e:
                            print(f"Error reading or updating preprocessing_info.json: {str(e)}")
                            # Create a new one from scratch if there was an error
                            with open(preprocessing_info_path, 'w') as f:
                                json.dump({"encoding_mappings": {}}, f)
                            print("Created new preprocessing_info.json after error")
                    
                    # Extract encoding maps from files
                    encoding_map = None
                    for file_path in encoding_files:
                        try:
                            filename = os.path.basename(file_path)
                            print(f"Examining file {filename} for encoding map")
                            
                            # Verify the timestamp matches expected
                            file_time = os.path.getmtime(file_path)
                            time_diff = abs(file_time - extraction_time)
                            if time_diff > 5:
                                print(f"SUSPICIOUS: File {filename} has inconsistent timestamp (diff: {time_diff:.2f}s), skipping")
                                continue
                                
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            
                            # Log the entire data structure for debugging
                            print(f"JSON data in {filename}: {json.dumps(data)[:500]}... (truncated)")
                            
                            # Check if the selected encoding column exists in this file
                            if selected_encoding in data:
                                encoding_map = data[selected_encoding]
                                print(f"Found encoding map for '{selected_encoding}' in {filename}")
                                print(f"Encoding map contains {len(encoding_map)} items")
                                print(f"Sample items: {dict(list(encoding_map.items())[:3])}")
                                break
                                
                            # Restructure the encoding mapping - we only want the specific encoding
                            # Check if this is an encoding_mappings.json file with multiple columns
                            if os.path.basename(file_path) == "encoding_mappings.json" or "encoding" in os.path.basename(file_path).lower():
                                if selected_encoding in data:
                                    # Extract only the specific column's encoding
                                    app.logger.info(f"Using specific encoding for column '{selected_encoding}' from encoding_mappings.json")
                                    encoding_map = data[selected_encoding]
                                else:
                                    app.logger.info(f"Column '{selected_encoding}' not found in encoding_mappings.json")
                                    app.logger.info(f"Available columns: {list(data.keys())}")
                            
                            # Also check under common keys
                            for key in ['target_map', 'target_encoding', 'label_encoding']:
                                if key in data and selected_encoding == key:
                                    encoding_map = data[key]
                                    print(f"Found encoding map for '{selected_encoding}' under key '{key}' in {os.path.basename(file_path)}")
                                    break
                                
                            # Check if we need to find the encoding for a specific column
                            if selected_encoding in data.keys():
                                for col, mapping in data.items():
                                    if col == selected_encoding and isinstance(mapping, dict):
                                        encoding_map = mapping
                                        print(f"Found encoding map for column '{selected_encoding}' in {os.path.basename(file_path)}")
                                        break
                                        
                            if encoding_map:
                                break
                        except Exception as e:
                            print(f"Error reading {os.path.basename(file_path)}: {str(e)}")
                            continue
                    
                    # If we found an encoding map, decode the prediction column
                    if encoding_map:
                        # Log that we're beginning the decoding process
                        app.logger.info(f"===== BEGINNING DECODING PROCESS FOR '{selected_encoding}' =====")
                        app.logger.info(f"Encoding map contains {len(encoding_map)} mappings")
                        
                        # Validate encoding map format - should be a direct mapping of category -> code
                        if encoding_map and all(isinstance(v, (int, float)) for v in encoding_map.values()):
                            app.logger.info("Encoding map is in the correct format (category -> code)")
                        else:
                            app.logger.warning("Encoding map may be in the wrong format, attempting to fix")
                            try:
                                # Check if it's the full encoding_mappings structure
                                if selected_encoding in encoding_map:
                                    app.logger.info(f"Found full encoding structure, extracting only '{selected_encoding}' mapping")
                                    encoding_map = encoding_map[selected_encoding]
                                
                                # If values are dictionaries instead of codes, it's the wrong format
                                if isinstance(next(iter(encoding_map.values()), None), dict):
                                    app.logger.info("Encoding map contains nested dictionaries, extracting data")
                                    for key, value in list(encoding_map.items()):
                                        if isinstance(value, dict) and selected_encoding in value:
                                            encoding_map = value[selected_encoding]
                                            break
                            except Exception as e:
                                app.logger.error(f"Error fixing encoding map format: {str(e)}")
                            
                        app.logger.info(f"Final encoding map for decoding: {encoding_map}")
                        
                        # *** CREATE INVERSE MAPPING FOR DECODING ***
                        # The encoding map is category -> code, but for decoding we need code -> category
                        inverse_mapping = {}
                        for category, code in encoding_map.items():
                            # Handle both numeric and string codes
                            if isinstance(code, (int, float)):
                                inverse_mapping[code] = category
                                # Also add string version of the code
                                inverse_mapping[str(code)] = category
                            else:
                                inverse_mapping[code] = category
                        
                        app.logger.info(f"Created inverse mapping for decoding: {inverse_mapping}")
                        
                        # Parse the CSV
                        df = pd.read_csv(io.StringIO(output_data))
                        print(f"CSV columns: {df.columns.tolist()}")
                        
                        # Identify the prediction column - usually named 'prediction'
                        pred_col = None
                        potential_pred_cols = ['prediction', 'predicted', 'target', 'label', 'class', 
                                             'result', 'output', 'outcome', selected_encoding]
                        
                        # Try exact matches first
                        for col_name in potential_pred_cols:
                            if col_name in df.columns:
                                pred_col = col_name
                                print(f"Found exact match prediction column: '{pred_col}'")
                                break
                        
                        # If no exact match, try case-insensitive search
                        if not pred_col:
                            for col in df.columns:
                                for pattern in potential_pred_cols:
                                    if pattern.lower() in col.lower():
                                        pred_col = col
                                        print(f"Found prediction column by pattern match: '{pred_col}'")
                                        break
                                if pred_col:
                                    break
                                
                        # If no prediction column found, use the last column
                        if not pred_col and len(df.columns) > 0:
                            pred_col = df.columns[-1]
                            print(f"No prediction column identified, using last column: '{pred_col}'")
                            
                        if pred_col:
                            app.logger.info(f"Using prediction column: '{pred_col}'")
                            app.logger.info(f"Sample predictions: {df[pred_col].head(5).tolist()}")
                            
                            # COMPLETELY SIMPLIFIED APPROACH:
                            # 1. Create the decoded values
                            decoded_values = df[pred_col].apply(lambda x: inverse_mapping.get(x, 
                                                                               inverse_mapping.get(str(x), 
                                                                                   inverse_mapping.get(int(float(x)) if isinstance(x, (int, float, str)) and x != '' else None,
                                                                                       None))))
                            
                            # 2. Remove ALL prediction-related columns
                            prediction_columns = [col for col in df.columns if 'predict' in col.lower() or 'original' in col.lower()]
                            app.logger.info(f"Removing all prediction columns: {prediction_columns}")
                            df.drop(columns=prediction_columns, inplace=True)
                            
                            # 3. Add only the final decoded column
                            df["Prediction (Decoded)"] = decoded_values
                            
                            # 4. Ensure it's at the end
                            all_cols = df.columns.tolist()
                            all_cols.remove("Prediction (Decoded)")
                            all_cols.append("Prediction (Decoded)")
                            df = df[all_cols]
                            
                            app.logger.info(f"Final output has only one prediction column at the end")
                            
                            # Check if decoding worked
                            null_count = df["Prediction (Decoded)"].isna().sum()
                            decoded_count = len(df) - null_count
                            
                            if decoded_count > 0:
                                # At least some values were decoded
                                success_percentage = (decoded_count / len(df)) * 100
                                app.logger.info(f"===== DECODING SUCCEEDED for {decoded_count}/{len(df)} values ({success_percentage:.1f}%) =====")
                                
                                # No message row - just log to console
                            else:
                                app.logger.info("===== DECODING COMPLETELY FAILED - NO VALUES DECODED =====")
                                
                                # If decoding completely failed, we'll remove the decoded column
                                df.drop(columns=["Prediction (Decoded)"], inplace=True)
                except Exception as decode_error:
                    print(f"Error decoding prediction: {str(decode_error)}")
                    # Continue without decoding if there's an error
            
            # Return the CSV content with all columns including decoded values
            output_csv = df.to_csv(index=False)
            app.logger.info(f"Total CSV size before return: {len(output_csv)} bytes")
            return jsonify({"output_file": output_csv})
            
        # If no stdout, check if output.csv was created
        output_path = os.path.join(temp_dir, "output.csv")
        if os.path.exists(output_path):
            print(f"Reading output from file: {output_path}")
            
            # If skip_decoding is True, just read and return the CSV as is
            if skip_decoding:
                print("Skipping decoding as requested - returning numeric predictions as is")
                try:
                    with open(output_path, 'r') as f:
                        output_data = f.read()
                    return jsonify({"output_file": output_data})
                except Exception as e:
                    print(f"Error reading output CSV: {str(e)}")
                    return jsonify({"error": f"Error reading prediction results: {str(e)}"}), 500
            
            # If selected_encoding is provided, try to decode the prediction column
            if selected_encoding:
                try:
                    # Load CSV
                    df = pd.read_csv(output_path)
                    
                    # Find encoding files and decode as above
                    encoding_files = []
                    print(f"Looking for encoding files to decode predictions in output.csv using '{selected_encoding}'")
                    
                    # Use timestamp-based validation to find model files
                    model_files = []
                    for root, _, files in os.walk(temp_dir):
                        for filename in files:
                            if filename.endswith('.py') or filename.endswith('.json'):
                                file_path = os.path.join(root, filename)
                                file_time = os.path.getmtime(file_path)
                                # Only include files from this model package (within 10 seconds of extraction_time)
                                if abs(file_time - extraction_time) < 10:
                                    rel_path = os.path.relpath(file_path, temp_dir)
                                    model_files.append(rel_path)
                                    
                                    # Add encoding files based on filename pattern
                                    if filename.endswith('.json'):
                                        if (filename == 'encoding_mappings.json' or 
                                            filename == 'encoding_mappings' or
                                            filename == 'preprocessing_info.json' or 
                                            filename == 'preprocessing_info' or
                                            filename.endswith('_encoding.json') or 
                                            filename == 'encoding.json' or 
                                            filename == 'preprocessing.json' or
                                            'encod' in filename.lower() or
                                            'target' in filename.lower() or
                                            'label' in filename.lower() or
                                            'map' in filename.lower()):
                                            
                                            full_path = os.path.join(root, filename)
                                            encoding_files.append(full_path)
                                            print(f"Adding validated encoding file: {rel_path} (timestamp diff: {abs(file_time - extraction_time):.2f}s)")
                                else:
                                    rel_path = os.path.relpath(file_path, temp_dir)
                                    print(f"Ignoring file with suspicious timestamp: {rel_path} (diff: {abs(file_time - extraction_time):.2f}s)")
                                    # Optionally remove suspicious files
                                    try:
                                        os.remove(file_path)
                                        print(f"Removed suspicious file: {rel_path}")
                                    except Exception as e:
                                        print(f"Failed to remove suspicious file {rel_path}: {str(e)}")
                    
                    print(f"Model files that are part of this package: {model_files}")
                    print(f"Found potential encoding files: {encoding_files}")
                    
                    # Validate file contents if needed
                    for file_path in model_files:
                        if file_path.lower().endswith('.json'):
                            full_path = os.path.join(temp_dir, file_path)
                            filename = os.path.basename(file_path)
                            
                            if (filename == 'encoding_mappings.json' or 
                                filename == 'encoding_mappings' or
                                filename == 'preprocessing_info.json' or 
                                filename == 'preprocessing_info' or
                                filename.endswith('_encoding.json') or 
                                filename == 'encoding.json' or 
                                filename == 'preprocessing.json' or
                                'encod' in filename.lower() or
                                'target' in filename.lower() or
                                'label' in filename.lower() or
                                'map' in filename.lower()):
                                
                                encoding_files.append(full_path)
                                print(f"Adding validated encoding file: {file_path}")
                    
                    print(f"Found potential encoding files: {encoding_files}")
                    
                    # Create preprocessing_info.json if it doesn't exist
                    preprocessing_info_path = os.path.join(temp_dir, "preprocessing_info.json")
                    if not os.path.exists(preprocessing_info_path):
                        print("Creating a minimal preprocessing_info.json with empty encoding_mappings")
                        with open(preprocessing_info_path, 'w') as f:
                            json.dump({"encoding_mappings": {}}, f)
                        
                        # Add to encoding files
                        encoding_files.append(preprocessing_info_path)
                        print(f"Created preprocessing_info.json at {preprocessing_info_path}")
                    else:
                        # Ensure it has encoding_mappings
                        try:
                            with open(preprocessing_info_path, 'r') as f:
                                preprocessing_info = json.load(f)
                                
                            # Add empty encoding_mappings if it doesn't exist
                            if 'encoding_mappings' not in preprocessing_info:
                                preprocessing_info['encoding_mappings'] = {}
                                with open(preprocessing_info_path, 'w') as f:
                                    json.dump(preprocessing_info, f)
                                print("Updated preprocessing_info.json with empty encoding_mappings field")
                        except Exception as e:
                            print(f"Error reading or updating preprocessing_info.json: {str(e)}")
                            # Create a new one from scratch if there was an error
                            with open(preprocessing_info_path, 'w') as f:
                                json.dump({"encoding_mappings": {}}, f)
                            print("Created new preprocessing_info.json after error")
                    
                    # Extract encoding maps from files
                    encoding_map = None
                    for file_path in encoding_files:
                        try:
                            filename = os.path.basename(file_path)
                            print(f"Examining file {filename} for encoding map")
                            
                            # Verify the timestamp matches expected
                            file_time = os.path.getmtime(file_path)
                            time_diff = abs(file_time - extraction_time)
                            if time_diff > 5:
                                print(f"SUSPICIOUS: File {filename} has inconsistent timestamp (diff: {time_diff:.2f}s), skipping")
                                continue
                                
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            
                            # Log the entire data structure for debugging
                            print(f"JSON data in {filename}: {json.dumps(data)[:500]}... (truncated)")
                            
                            # Check if the selected encoding column exists in this file
                            if selected_encoding in data:
                                encoding_map = data[selected_encoding]
                                print(f"Found encoding map for '{selected_encoding}' in {filename}")
                                print(f"Encoding map contains {len(encoding_map)} items")
                                print(f"Sample items: {dict(list(encoding_map.items())[:3])}")
                                break
                                
                            # Restructure the encoding mapping - we only want the specific encoding
                            # Check if this is an encoding_mappings.json file with multiple columns
                            if os.path.basename(file_path) == "encoding_mappings.json" or "encoding" in os.path.basename(file_path).lower():
                                if selected_encoding in data:
                                    # Extract only the specific column's encoding
                                    app.logger.info(f"Using specific encoding for column '{selected_encoding}' from encoding_mappings.json")
                                    encoding_map = data[selected_encoding]
                                else:
                                    app.logger.info(f"Column '{selected_encoding}' not found in encoding_mappings.json")
                                    app.logger.info(f"Available columns: {list(data.keys())}")
                            
                            # Also check under common keys
                            for key in ['target_map', 'target_encoding', 'label_encoding']:
                                if key in data and selected_encoding == key:
                                    encoding_map = data[key]
                                    print(f"Found encoding map for '{selected_encoding}' under key '{key}' in {os.path.basename(file_path)}")
                                    break
                                
                            # Check if we need to find the encoding for a specific column
                            if selected_encoding in data.keys():
                                for col, mapping in data.items():
                                    if col == selected_encoding and isinstance(mapping, dict):
                                        encoding_map = mapping
                                        print(f"Found encoding map for column '{selected_encoding}' in {os.path.basename(file_path)}")
                                        break
                                        
                            if encoding_map:
                                break
                        except Exception as e:
                            print(f"Error reading {os.path.basename(file_path)}: {str(e)}")
                            continue
                    
                    # If we found an encoding map, decode the prediction column
                    if encoding_map:
                        # Log that we're beginning the decoding process
                        app.logger.info(f"===== BEGINNING DECODING PROCESS FOR '{selected_encoding}' =====")
                        app.logger.info(f"Encoding map contains {len(encoding_map)} mappings")
                        
                        # Validate encoding map format - should be a direct mapping of category -> code
                        if encoding_map and all(isinstance(v, (int, float)) for v in encoding_map.values()):
                            app.logger.info("Encoding map is in the correct format (category -> code)")
                        else:
                            app.logger.warning("Encoding map may be in the wrong format, attempting to fix")
                            try:
                                # Check if it's the full encoding_mappings structure
                                if selected_encoding in encoding_map:
                                    app.logger.info(f"Found full encoding structure, extracting only '{selected_encoding}' mapping")
                                    encoding_map = encoding_map[selected_encoding]
                                
                                # If values are dictionaries instead of codes, it's the wrong format
                                if isinstance(next(iter(encoding_map.values()), None), dict):
                                    app.logger.info("Encoding map contains nested dictionaries, extracting data")
                                    for key, value in list(encoding_map.items()):
                                        if isinstance(value, dict) and selected_encoding in value:
                                            encoding_map = value[selected_encoding]
                                            break
                            except Exception as e:
                                app.logger.error(f"Error fixing encoding map format: {str(e)}")
                            
                        app.logger.info(f"Final encoding map for decoding: {encoding_map}")
                        
                        # *** CREATE INVERSE MAPPING FOR DECODING ***
                        # The encoding map is category -> code, but for decoding we need code -> category
                        inverse_mapping = {}
                        for category, code in encoding_map.items():
                            # Handle both numeric and string codes
                            if isinstance(code, (int, float)):
                                inverse_mapping[code] = category
                                # Also add string version of the code
                                inverse_mapping[str(code)] = category
                            else:
                                inverse_mapping[code] = category
                        
                        app.logger.info(f"Created inverse mapping for decoding: {inverse_mapping}")
                        
                        # Parse the CSV
                        df = pd.read_csv(output_path)
                        print(f"CSV columns: {df.columns.tolist()}")
                        
                        # Identify the prediction column - usually named 'prediction'
                        pred_col = None
                        potential_pred_cols = ['prediction', 'predicted', 'target', 'label', 'class', 
                                             'result', 'output', 'outcome', selected_encoding]
                        
                        # Try exact matches first
                        for col_name in potential_pred_cols:
                            if col_name in df.columns:
                                pred_col = col_name
                                print(f"Found exact match prediction column: '{pred_col}'")
                                break
                        
                        # If no exact match, try case-insensitive search
                        if not pred_col:
                            for col in df.columns:
                                for pattern in potential_pred_cols:
                                    if pattern.lower() in col.lower():
                                        pred_col = col
                                        print(f"Found prediction column by pattern match: '{pred_col}'")
                                        break
                                if pred_col:
                                    break
                                
                        # If no prediction column found, use the last column
                        if not pred_col and len(df.columns) > 0:
                            pred_col = df.columns[-1]
                            print(f"No prediction column identified, using last column: '{pred_col}'")
                            
                        if pred_col:
                            app.logger.info(f"Using prediction column: '{pred_col}'")
                            app.logger.info(f"Sample predictions: {df[pred_col].head(5).tolist()}")
                            
                            # COMPLETELY SIMPLIFIED APPROACH:
                            # 1. Create the decoded values
                            decoded_values = df[pred_col].apply(lambda x: inverse_mapping.get(x, 
                                                                               inverse_mapping.get(str(x), 
                                                                                   inverse_mapping.get(int(float(x)) if isinstance(x, (int, float, str)) and x != '' else None,
                                                                                       None))))
                            
                            # 2. Remove ALL prediction-related columns
                            prediction_columns = [col for col in df.columns if 'predict' in col.lower() or 'original' in col.lower()]
                            app.logger.info(f"Removing all prediction columns: {prediction_columns}")
                            df.drop(columns=prediction_columns, inplace=True)
                            
                            # 3. Add only the final decoded column
                            df["Prediction (Decoded)"] = decoded_values
                            
                            # 4. Ensure it's at the end
                            all_cols = df.columns.tolist()
                            all_cols.remove("Prediction (Decoded)")
                            all_cols.append("Prediction (Decoded)")
                            df = df[all_cols]
                            
                            app.logger.info(f"Final output has only one prediction column at the end")
                            
                            # Check if decoding worked
                            null_count = df["Prediction (Decoded)"].isna().sum()
                            decoded_count = len(df) - null_count
                            
                            if decoded_count > 0:
                                # At least some values were decoded
                                success_percentage = (decoded_count / len(df)) * 100
                                app.logger.info(f"===== DECODING SUCCEEDED for {decoded_count}/{len(df)} values ({success_percentage:.1f}%) =====")
                                
                                # No message row - just log to console
                            else:
                                app.logger.info("===== DECODING COMPLETELY FAILED - NO VALUES DECODED =====")
                                
                                # If decoding completely failed, we'll remove the decoded column
                                df.drop(columns=["Prediction (Decoded)"], inplace=True)
                except Exception as decode_error:
                    print(f"Error decoding prediction from file: {str(decode_error)}")
                    # Continue without decoding if there's an error
            
            with open(output_path, "r") as f:
                output_data = f.read()
            return jsonify({"output_file": output_data})
            
        # No output found in stdout or file, create a fallback response
        print("No output found in stdout or file, creating fallback")
        
        try:
            # Try to read the input file to get its structure
            if input_file_path.lower().endswith('.csv'):
                df = pd.read_csv(input_file_path)
            elif input_file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(input_file_path)
            
            # Add error column
            df['prediction'] = "ERROR"
            df['error_message'] = "Prediction script did not produce any output"
            
            # Convert to CSV string
            output_buffer = io.StringIO()
            df.to_csv(output_buffer, index=False)
            output_data = output_buffer.getvalue()
            print("Created fallback response from input data")
            
            return jsonify({"output_file": output_data})
            
        except Exception as fallback_error:
            print(f"Error creating fallback response: {str(fallback_error)}")
            # Last resort - create minimal error CSV
            error_csv = "error,message\nTrue,\"No output produced by prediction script\"\n"
            return jsonify({"output_file": error_csv})

    except subprocess.TimeoutExpired:
        print("Prediction timed out")
        return jsonify({"error": "Inference process timed out."}), 504
    except Exception as e:
        print(f"Prediction failed with error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(temp_dir)
        elapsed_time = time.time() - start_time
        print(f"Deleted temporary directory: {temp_dir} (process took {elapsed_time:.2f} seconds)")

if __name__ == '__main__':
    # Make sure logging is properly initialized
    logging.info("Starting predictor service")
    app.run(host='0.0.0.0', port=5101)