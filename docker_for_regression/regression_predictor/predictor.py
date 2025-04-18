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
import uuid
import datetime

# Import storage module for blob operations
try:
    from storage import download_blob
    BLOB_STORAGE_AVAILABLE = True
    print("Azure Blob Storage integration available")
except ImportError:
    BLOB_STORAGE_AVAILABLE = False
    print("WARNING: Azure Blob Storage integration not available")

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

def download_model_from_blob(model_url, target_path):
    """Download a model file from Azure Blob Storage.
    
    Args:
        model_url (str): The URL or path to the model file
        target_path (str): The local path to save the downloaded model
        
    Returns:
        bool: True if successful, False otherwise
    """
    request_id = str(uuid.uuid4())[:8]
    print(f"[{request_id}] Attempting to download model from {model_url} to {target_path}")
    
    # Early return if blob storage is not available
    if not BLOB_STORAGE_AVAILABLE:
        print(f"[{request_id}] Cannot download model - Azure Blob Storage integration not available")
        return False
    
    try:
        # Use our storage module to download the blob
        result = download_blob(model_url, target_path)
        if result:
            # Verify the file was downloaded and is a valid model file
            if os.path.exists(target_path):
                file_size = os.path.getsize(target_path)
                print(f"[{request_id}] Model downloaded successfully - size: {file_size} bytes")
                
                # Verify it's not a text file (placeholder)
                if file_size < 1000:  # Less than 1KB
                    try:
                        with open(target_path, 'r') as f:
                            content = f.read(100)  # Read first 100 chars
                            if "placeholder" in content.lower() or "could not be downloaded" in content.lower():
                                print(f"[{request_id}] WARNING: Downloaded file appears to be a placeholder: {content}")
                                return False
                    except UnicodeDecodeError:
                        # If we can't decode as text, it's likely a binary file (good)
                        pass
                
                # Try to load with joblib to verify it's a valid model
                try:
                    import joblib
                    model = joblib.load(target_path)
                    print(f"[{request_id}] Successfully loaded model with joblib")
                    return True
                except Exception as e:
                    print(f"[{request_id}] Error loading model with joblib: {str(e)}")
                    return False
            else:
                print(f"[{request_id}] Target path {target_path} does not exist after download")
                return False
        else:
            print(f"[{request_id}] Failed to download model from {model_url}")
            return False
    except Exception as e:
        print(f"[{request_id}] Exception in download_model_from_blob: {str(e)}")
        return False

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
        "platform": sys.platform,
        "blob_storage_available": BLOB_STORAGE_AVAILABLE
    }
    
    print(f"[{request_id}] Returning health status: {health_info}")
    return jsonify(health_info), 200

@app.route('/extract_encodings', methods=['POST'])
def extract_encodings():
    """Extract encoding maps from a model package"""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]  # Generate a short unique ID for this request
    
    print(f"[{request_id}] ===== EXTRACT ENCODINGS REQUEST STARTED =====")
    print(f"[{request_id}] Timestamp: {datetime.datetime.now().isoformat()}")
    
    # Check if this is a model URL request
    if request.is_json and 'model_url' in request.json:
        # This is a direct model URL extraction request
        model_url = request.json['model_url']
        print(f"[{request_id}] Extract encodings from model URL: {model_url}")
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp(prefix="encoding_extract_")
        print(f"[{request_id}] Created temporary directory: {temp_dir}")
        
        try:
            # Download the model file
            target_filename = model_url.split('/')[-1] if '/' in model_url else 'model.joblib'
            target_path = os.path.join(temp_dir, target_filename)
            
            print(f"[{request_id}] Downloading model to {target_path}")
            download_success = download_model_from_blob(model_url, target_path)
            
            if not download_success:
                print(f"[{request_id}] Failed to download model from {model_url}")
                return jsonify({"error": f"Failed to download model from {model_url}"}), 400
            
            # Verify the file exists and is a valid model
            if not os.path.exists(target_path):
                print(f"[{request_id}] Model file not found at {target_path}")
                return jsonify({"error": f"Model file not found after download"}), 400
            
            # Look for encoded features in the model
            try:
                import joblib
                model_data = joblib.load(target_path)
                print(f"[{request_id}] Loaded model data of type: {type(model_data)}")
                
                # Check if it's a dictionary with preprocessing info
                if isinstance(model_data, dict):
                    # Try various keys where encodings might be stored
                    encoding_maps = {}
                    
                    # Look for encoding_mappings at the top level
                    if 'encoding_mappings' in model_data and isinstance(model_data['encoding_mappings'], dict):
                        encoding_maps.update(model_data['encoding_mappings'])
                        print(f"[{request_id}] Found encoding_mappings at top level with {len(model_data['encoding_mappings'])} items")
                    
                    # Look for preprocessing_info
                    if 'preprocessing_info' in model_data and isinstance(model_data['preprocessing_info'], dict):
                        preproc = model_data['preprocessing_info']
                        if 'encoding_mappings' in preproc and isinstance(preproc['encoding_mappings'], dict):
                            encoding_maps.update(preproc['encoding_mappings'])
                            print(f"[{request_id}] Found encoding_mappings in preprocessing_info with {len(preproc['encoding_mappings'])} items")
                    
                    # Look for column_encodings
                    if 'column_encodings' in model_data and isinstance(model_data['column_encodings'], dict):
                        encoding_maps.update(model_data['column_encodings'])
                        print(f"[{request_id}] Found column_encodings with {len(model_data['column_encodings'])} items")
                    
                    if encoding_maps:
                        # Generate metadata for each encoding map
                        encoding_metadata = {}
                        for feature_name, mapping in encoding_maps.items():
                            if isinstance(mapping, dict):
                                value_count = len(mapping)
                                sample_values = list(mapping.keys())[:3]  # First 3 keys
                                encoding_metadata[feature_name] = {
                                    "value_count": value_count,
                                    "sample_values": sample_values,
                                    "display_name": feature_name
                                }
                        
                        print(f"[{request_id}] Successfully extracted {len(encoding_maps)} encoding maps with metadata")
                        return jsonify({
                            "encoding_maps": encoding_maps,
                            "metadata": encoding_metadata,
                            "source": "model_url"
                        })
                    else:
                        print(f"[{request_id}] No encoding maps found in model data")
                        # Return an empty result rather than an error
                        return jsonify({
                            "encoding_maps": {},
                            "metadata": {},
                            "source": "model_url",
                            "warning": "No encoding maps found in model data"
                        })
                        
                else:
                    # It's a direct model object - cannot extract encodings
                    print(f"[{request_id}] Model data is not a dictionary, cannot extract encodings")
                    return jsonify({
                        "encoding_maps": {},
                        "metadata": {},
                        "source": "model_url",
                        "warning": "Model data is not a dictionary structure with encodings"
                    })
                    
            except Exception as e:
                print(f"[{request_id}] Error analyzing model file: {str(e)}")
                return jsonify({"error": f"Error analyzing model file: {str(e)}"}), 500
                
        except Exception as e:
            print(f"[{request_id}] Error processing model URL: {str(e)}")
            return jsonify({"error": str(e)}), 500
            
        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"[{request_id}] Cleaned up temporary directory: {temp_dir}")
            
            total_time = time.time() - start_time
            print(f"[{request_id}] ===== EXTRACT ENCODINGS FROM URL COMPLETED in {total_time:.2f}s =====")
    
    # Standard model package flow (ZIP file upload)
    if 'model_package' not in request.files:
        print(f"[{request_id}] Error: 'model_package' file not provided")
        return jsonify({"error": "Model package file is required"}), 400
    
    model_file = request.files['model_package']
    print(f"[{request_id}] Received model package: {model_file.filename}")
    
    # Validate file type
    if not model_file.filename.lower().endswith('.zip'):
        print(f"[{request_id}] Error: Invalid model package format: {model_file.filename}")
        return jsonify({"error": "Model package must be a ZIP archive"}), 400
    
    # Create a temporary directory to extract files
    temp_dir = tempfile.mkdtemp(prefix="encoding_extract_")
    print(f"[{request_id}] Created temporary directory: {temp_dir}")
    
    try:
        # Save and extract the ZIP package
        zip_path = os.path.join(temp_dir, "model_package.zip")
        model_file.save(zip_path)
        print(f"[{request_id}] Saved model package to {zip_path}, size: {os.path.getsize(zip_path)} bytes")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            # Log the files in the zip for debugging
            file_list = zip_ref.namelist()
            print(f"[{request_id}] Extracted {len(file_list)} files from model package")
            print(f"[{request_id}] Files in package: {', '.join(file_list[:10])}{'...' if len(file_list) > 10 else ''}")
        
        # Process the extracted files
        return process_extracted_package(temp_dir, request_id, start_time)
        
    except Exception as e:
        print(f"[{request_id}] ERROR: {str(e)}")
        import traceback
        print(f"[{request_id}] Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"[{request_id}] Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"[{request_id}] Warning: Failed to clean up temporary directory: {str(e)}")

def process_extracted_package(temp_dir, request_id, start_time):
    """Process an extracted model package directory to find encoding maps"""
    # List all files for debugging
    all_files = os.listdir(temp_dir)
    print(f"[{request_id}] All files in extracted package root: {all_files}")
    
    # Look for encoding files with broader search
    encoding_files = []
    json_files = []
    
    # First pass: look for standard encoding file names
    for filename in all_files:
        if filename.lower().endswith('.json'):
            json_files.append(filename)
            if (filename == 'encoding_mappings.json' or 
                filename == 'encoding_mappings' or
                filename == 'preprocessing_info.json' or 
                filename == 'preprocessing_info' or
                filename.endswith('_encoding.json') or 
                filename == 'encoding.json' or 
                filename == 'preprocessing.json' or
                'encod' in filename.lower() or
                'feature' in filename.lower() or
                'map' in filename.lower()):
                encoding_files.append(filename)
    
    print(f"[{request_id}] Found {len(json_files)} JSON files: {json_files}")
    print(f"[{request_id}] Potential encoding files based on name: {encoding_files}")
    
    # If no encoding files found by name, try all JSON files
    if not encoding_files and json_files:
        print(f"[{request_id}] No encoding files found by name pattern, examining all JSON files")
        encoding_files = json_files
    
    if not encoding_files:
        # Look deeper in subdirectories
        print(f"[{request_id}] Looking in subdirectories for encoding files")
        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                if filename.lower().endswith('.json'):
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, temp_dir)
                    if (filename == 'encoding_mappings.json' or 
                        filename == 'encoding_mappings' or
                        filename == 'preprocessing_info.json' or 
                        filename == 'preprocessing_info' or
                        filename.endswith('_encoding.json') or 
                        filename == 'encoding.json' or 
                        filename == 'preprocessing.json' or
                        'encod' in filename.lower() or
                        'feature' in filename.lower() or
                        'map' in filename.lower()):
                        encoding_files.append(rel_path)
                    elif rel_path not in json_files:
                        json_files.append(rel_path)
        
        print(f"[{request_id}] After subdirectory search - JSON files: {json_files}")
        print(f"[{request_id}] After subdirectory search - Potential encoding files: {encoding_files}")
        
        # If still no encoding files but found JSON files, try them all
        if not encoding_files and json_files:
            print(f"[{request_id}] No encoding files found in subdirectories by name pattern, examining all JSON files")
            encoding_files = json_files
    
    if not encoding_files:
        print(f"[{request_id}] No JSON files found in the model package")
        return jsonify({"error": "No encoding files found in the model package"}), 404
    
    # Extract encoding maps from each file
    encoding_maps = {}
    for file in encoding_files:
        file_path = os.path.join(temp_dir, file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            print(f"[{request_id}] Examining file {file} for encoding maps")
            
            # Process different potential formats
            if isinstance(data, dict):
                # Look for dictionary structures that could be encoding maps
                for column, mapping in data.items():
                    if isinstance(mapping, dict):
                        # Look for typical encoding patterns (numeric keys or values)
                        is_encoding_map = False
                        
                        # Check if it maps strings to numbers (common encoding pattern)
                        string_keys = False
                        numeric_values = False
                        
                        for k, v in mapping.items():
                            # Check if key is string
                            if isinstance(k, str):
                                string_keys = True
                            
                            # Check if value is numeric
                            if isinstance(v, (int, float)):
                                numeric_values = True
                        
                        # If keys are strings and values are numbers, likely an encoding map
                        if string_keys and numeric_values:
                            is_encoding_map = True
                        
                        # Also check if it has a reasonable number of entries to be a category mapping
                        if len(mapping) > 0 and len(mapping) < 100:
                            # More likely to be an encoding map than a large data structure
                            is_encoding_map = True
                            
                        if is_encoding_map:
                            print(f"[{request_id}] Found encoding map for column '{column}' with {len(mapping)} values")
                            encoding_maps[column] = mapping
                
                # Also check for special information like selected_features
                if 'selected_features' in data and isinstance(data['selected_features'], list):
                    print(f"[{request_id}] Found selected_features list with {len(data['selected_features'])} features")
                    encoding_maps['selected_features'] = data['selected_features']
                
                # Check for patterns like "column_name_encoding" or "column_name_map"
                for key in data.keys():
                    if (key.endswith('_encoding') or key.endswith('_map') or '_encoding_' in key) and isinstance(data[key], dict):
                        print(f"[{request_id}] Found encoding map with pattern-matched key '{key}' containing {len(data[key])} values")
                        encoding_maps[key] = data[key]
                        
                # Check for encoding_mappings key directly (common format)
                if 'encoding_mappings' in data and isinstance(data['encoding_mappings'], dict):
                    print(f"[{request_id}] Found encoding_mappings dictionary with {len(data['encoding_mappings'])} entries")
                    for col, mapping in data['encoding_mappings'].items():
                        encoding_maps[col] = mapping
                
                # Check for column_encodings key directly (alternate format)
                if 'column_encodings' in data and isinstance(data['column_encodings'], dict):
                    print(f"[{request_id}] Found column_encodings dictionary with {len(data['column_encodings'])} entries")
                    for col, mapping in data['column_encodings'].items():
                        encoding_maps[col] = mapping
        except Exception as e:
            # Skip files that can't be parsed
            print(f"[{request_id}] Error parsing {file}: {str(e)}")
            continue
    
    if not encoding_maps:
        print(f"[{request_id}] No encoding maps found in any of the JSON files")
        return jsonify({"error": "No valid encoding maps found in the model package"}), 404
    
    # Filter out non-feature encoding maps
    excluded_maps = ['scaler_params', 'config']
    feature_encoding_maps = {k: v for k, v in encoding_maps.items() if k not in excluded_maps}
    
    if not feature_encoding_maps:
        print(f"[{request_id}] No feature encoding maps found after filtering")
        return jsonify({"error": "No feature encoding maps found in the model package"}), 404
    
    # Add metadata for each encoding map to help the frontend
    encoding_metadata = {}
    for feature_name, mapping in feature_encoding_maps.items():
        if isinstance(mapping, dict): 
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
        elif feature_name == 'selected_features' and isinstance(mapping, list):
            encoding_metadata[feature_name] = {
                "value_count": len(mapping),
                "sample_values": mapping[:3],
                "display_name": "Selected Features"
            }
        
    # Return both the encoding maps and metadata
    print(f"[{request_id}] Successfully extracted {len(feature_encoding_maps)} feature encoding maps: {list(feature_encoding_maps.keys())}")
    
    total_time = time.time() - start_time
    print(f"[{request_id}] ===== EXTRACT ENCODINGS REQUEST COMPLETED in {total_time:.2f}s =====")
    
    return jsonify({
        "encoding_maps": feature_encoding_maps,
        "metadata": encoding_metadata,
        "source": "model_package"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Process a prediction request with a model package and input data.
    Returns CSV with predictions or error message.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]  # Generate a short unique ID for this request
    
    print(f"[{request_id}] ===== PREDICTION REQUEST STARTED =====")
    print(f"[{request_id}] Timestamp: {datetime.datetime.now().isoformat()}")
    
    # Check if we have a direct model URL
    if request.is_json and 'model_url' in request.json and 'input_file' in request.files:
        model_url = request.json['model_url']
        input_file = request.files['input_file']
        
        print(f"[{request_id}] Direct model URL prediction: model={model_url}, input={input_file.filename}")
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp(prefix="session_")
        print(f"[{request_id}] Created temporary directory: {temp_dir}")
        
        try:
            # Save the input file
            input_file_path = os.path.join(temp_dir, input_file.filename)
            input_file.save(input_file_path)
            print(f"[{request_id}] Saved input file to {input_file_path}, size: {os.path.getsize(input_file_path)} bytes")
            
            # Validate input file type
            if not (input_file.filename.lower().endswith('.xlsx') or 
                    input_file.filename.lower().endswith('.xls') or 
                    input_file.filename.lower().endswith('.csv')):
                print(f"[{request_id}] Error: Invalid input file format: {input_file.filename}")
                return jsonify({"error": "Input file must be an Excel or CSV file."}), 400
            
            # Download the model directly
            target_filename = model_url.split('/')[-1] if '/' in model_url else 'model.joblib'
            target_path = os.path.join(temp_dir, target_filename)
            
            print(f"[{request_id}] Downloading model to {target_path}")
            download_success = download_model_from_blob(model_url, target_path)
            
            if not download_success:
                print(f"[{request_id}] Failed to download model from {model_url}")
                return jsonify({
                    "error": f"Failed to download model from {model_url}. Please check if the model URL is correct and the Azure Blob Storage credentials are properly configured."
                }), 400
            
            # Create model package from the downloaded model
            # We'll create the minimum files needed for prediction
            preprocessing_info = {}
            
            # Try to extract preprocessing info directly from the model
            try:
                import joblib
                model_data = joblib.load(target_path)
                print(f"[{request_id}] Loaded model data of type: {type(model_data)}")
                
                # If it's a dictionary with preprocessing info, use it
                if isinstance(model_data, dict):
                    if 'preprocessing_info' in model_data:
                        preprocessing_info = model_data['preprocessing_info']
                        print(f"[{request_id}] Found preprocessing_info in model data")
                    
                    # Check for encoding_mappings
                    if 'encoding_mappings' in model_data:
                        if 'encoding_mappings' not in preprocessing_info:
                            preprocessing_info['encoding_mappings'] = {}
                        preprocessing_info['encoding_mappings'].update(model_data['encoding_mappings'])
                        print(f"[{request_id}] Found encoding_mappings in model data")
                    
                    # Check for selected_features
                    if 'selected_features' in model_data:
                        preprocessing_info['selected_features'] = model_data['selected_features']
                        print(f"[{request_id}] Found selected_features in model data")
                    
                    # Check for column_encodings (alternative format)
                    if 'column_encodings' in model_data:
                        if 'encoding_mappings' not in preprocessing_info:
                            preprocessing_info['encoding_mappings'] = {}
                        preprocessing_info['encoding_mappings'].update(model_data['column_encodings'])
                        print(f"[{request_id}] Found column_encodings in model data")
                
            except Exception as e:
                print(f"[{request_id}] Error extracting preprocessing info from model: {str(e)}")
                # Continue without preprocessing info
            
            # Create a preprocessing_info.json file if we have any info
            if preprocessing_info:
                preprocessing_file = os.path.join(temp_dir, 'preprocessing_info.json')
                with open(preprocessing_file, 'w') as f:
                    json.dump(preprocessing_info, f, indent=2)
                print(f"[{request_id}] Created preprocessing_info.json file")
                
                # Create separate encoding_mappings.json if available
                if 'encoding_mappings' in preprocessing_info and preprocessing_info['encoding_mappings']:
                    mappings_file = os.path.join(temp_dir, 'encoding_mappings.json')
                    with open(mappings_file, 'w') as f:
                        json.dump(preprocessing_info['encoding_mappings'], f, indent=2)
                    print(f"[{request_id}] Created encoding_mappings.json with {len(preprocessing_info['encoding_mappings'])} mappings")
            
            # Create a basic requirements.txt file
            req_file = os.path.join(temp_dir, 'requirements.txt')
            with open(req_file, 'w') as f:
                f.write("numpy\npandas\nscikit-learn\njoblib\npython-dotenv\nopenpyxl\n")
            print(f"[{request_id}] Created basic requirements.txt file")
            
            # Continue with the rest of the predict function using this temp_dir
            # ...
            # Now we can proceed with the standard predict function from here
            
            # The following is almost identical to the standard flow, just without the extraction step
            
            # Create a virtual environment in the temporary directory.
            venv_path = os.path.join(temp_dir, "venv")
            print(f"[{request_id}] Creating virtual environment at {venv_path}")
            subprocess.run(["python3", "-m", "venv", venv_path], check=True)

            # Determine pip path based on platform
            if os.name == 'nt':  # Windows
                pip_path = os.path.join(venv_path, "Scripts", "pip")
                python_path = os.path.join(venv_path, "Scripts", "python")
            else:  # Linux/Mac
                pip_path = os.path.join(venv_path, "bin", "pip")
                python_path = os.path.join(venv_path, "bin", "python")
            
            print(f"[{request_id}] Using Python: {python_path}, pip: {pip_path}")

            # Install the requirements
            print(f"[{request_id}] Installing requirements from {req_file}")
            pip_install = subprocess.run([pip_path, "install", "-r", req_file],
                                        capture_output=True, text=True)
            if pip_install.returncode != 0:
                print(f"[{request_id}] pip install failed: {pip_install.stderr}")
                return jsonify({"error": f"pip install failed: {pip_install.stderr}"}), 500
            print(f"[{request_id}] Successfully installed requirements")
            
            # Continue with prediction script creation and execution
            # ...
            
            # Rest of function remains the same
            
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
    
    # Standard model package upload flow
    # Verify that both files are provided.
    if 'model_package' not in request.files or 'input_file' not in request.files:
        print(f"[{request_id}] Error: Missing required files (model_package or input_file)")
        return jsonify({"error": "Both 'model_package' and 'input_file' must be provided."}), 400

    model_file = request.files['model_package']
    input_file = request.files['input_file']
    
    print(f"[{request_id}] Received files: model={model_file.filename}, input={input_file.filename}")

    # Validate file types.
    if not model_file.filename.lower().endswith('.zip'):
        print(f"[{request_id}] Error: Invalid model package format: {model_file.filename}")
        return jsonify({"error": "Model package must be a ZIP archive."}), 400
    if not (input_file.filename.lower().endswith('.xlsx') or 
            input_file.filename.lower().endswith('.xls') or 
            input_file.filename.lower().endswith('.csv')):
        print(f"[{request_id}] Error: Invalid input file format: {input_file.filename}")
        return jsonify({"error": "Input file must be an Excel or CSV file."}), 400

    # Create a temporary working directory.
    temp_dir = tempfile.mkdtemp(prefix="session_")
    print(f"[{request_id}] Created temporary directory: {temp_dir}")
    
    try:
        # Save and extract the ZIP package.
        zip_path = os.path.join(temp_dir, "model_package.zip")
        model_file.save(zip_path)
        print(f"[{request_id}] Saved model package to {zip_path}, size: {os.path.getsize(zip_path)} bytes")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            # Log the files in the zip for debugging
            file_list = zip_ref.namelist()
            print(f"[{request_id}] Extracted {len(file_list)} files from model package")
            print(f"[{request_id}] Files in package: {', '.join(file_list[:10])}{'...' if len(file_list) > 10 else ''}")

        # Save the input file.
        input_file_path = os.path.join(temp_dir, input_file.filename)
        input_file.save(input_file_path)
        print(f"[{request_id}] Saved input file to {input_file_path}, size: {os.path.getsize(input_file_path)} bytes")

        # Check for placeholder model files
        model_files = []
        placeholder_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                # Check for model files
                if file.endswith('.joblib') or file.endswith('.pkl'):
                    model_path = os.path.join(root, file)
                    model_files.append(model_path)
                    file_size = os.path.getsize(model_path)
                    print(f"[{request_id}] Found model file: {model_path}, size: {file_size} bytes")
                    
                    # Check if it's a placeholder (very small file)
                    if file_size < 1000:  # Less than 1KB
                        try:
                            with open(model_path, 'r') as f:
                                content = f.read()
                                if "placeholder" in content.lower() or "could not be downloaded" in content.lower():
                                    placeholder_files.append(model_path)
                                    print(f"[{request_id}] WARNING: Found placeholder model file: {model_path}")
                                    print(f"[{request_id}] Placeholder content: {content[:100]}...")
                        except UnicodeDecodeError:
                            # Not a text file, probably a real model
                            print(f"[{request_id}] Model file is binary (good)")
                            pass
                
                # Check for specific error files
                if file == "model_error.txt" or file == "download_error.txt":
                    placeholder_files.append(os.path.join(root, file))
                    with open(os.path.join(root, file), 'r') as f:
                        content = f.read()
                    print(f"[{request_id}] WARNING: Found error file: {os.path.join(root, file)}")
                    print(f"[{request_id}] Error content: {content[:100]}...")
        
        print(f"[{request_id}] Model files found: {len(model_files)}, placeholder files: {len(placeholder_files)}")
        
        # If we have placeholder files but no valid model files, return specific error
        if placeholder_files and (len(placeholder_files) == len(model_files) or len(model_files) == 0):
            error_msg = "The model file could not be loaded. This is likely because the model was not properly saved or downloaded from MLflow."
            
            # Try to get more specific error from any error files
            for error_file in placeholder_files:
                try:
                    with open(error_file, 'r') as f:
                        error_content = f.read().strip()
                        if error_content:
                            error_msg = f"{error_msg} Error details: {error_content}"
                            break
                except Exception as e:
                    print(f"[{request_id}] Error reading error file: {str(e)}")
                    pass
                    
            print(f"[{request_id}] ERROR: Returning error due to placeholder model: {error_msg}")
            return jsonify({
                "error": error_msg,
                "suggestion": "Please retrain the model and ensure MLflow tracking server is properly configured."
            }), 400

        # Create a virtual environment in the temporary directory.
        venv_path = os.path.join(temp_dir, "venv")
        print(f"[{request_id}] Creating virtual environment at {venv_path}")
        subprocess.run(["python3", "-m", "venv", venv_path], check=True)

        # Determine pip path based on platform
        if os.name == 'nt':  # Windows
            pip_path = os.path.join(venv_path, "Scripts", "pip")
            python_path = os.path.join(venv_path, "Scripts", "python")
        else:  # Linux/Mac
            pip_path = os.path.join(venv_path, "bin", "pip")
            python_path = os.path.join(venv_path, "bin", "python")
        
        print(f"[{request_id}] Using Python: {python_path}, pip: {pip_path}")

        # Install the extracted requirements.
        req_file = os.path.join(temp_dir, "requirements.txt")
        if os.path.exists(req_file):
            print(f"[{request_id}] Installing requirements from {req_file}")
            with open(req_file, 'r') as f:
                req_content = f.read()
                print(f"[{request_id}] Requirements file content: {req_content}")
                
            pip_install = subprocess.run([pip_path, "install", "-r", req_file],
                                        capture_output=True, text=True)
            if pip_install.returncode != 0:
                print(f"[{request_id}] pip install failed: {pip_install.stderr}")
                return jsonify({"error": f"pip install failed: {pip_install.stderr}"}), 500
            print(f"[{request_id}] Successfully installed requirements")
        else:
            print(f"[{request_id}] No requirements.txt file found, installing default regression packages")
            subprocess.run([pip_path, "install", "pandas", "numpy", "scikit-learn", "joblib", "matplotlib"], 
                          capture_output=True, text=True)
            print(f"[{request_id}] Successfully installed default packages")

        # Check if a predict.py script exists, or create one for regression
        predict_script = os.path.join(temp_dir, "predict.py")
        if not os.path.exists(predict_script):
            # Find model file
            model_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.joblib') or file.endswith('.pkl'):
                        model_files.append(os.path.join(root, file))
            
            if not model_files:
                return jsonify({"error": "No model file (.joblib or .pkl) found in package"}), 400
                
            # Create a basic predict.py script for regression
            with open(predict_script, 'w') as f:
                f.write("""import pandas as pd
import numpy as np
import joblib
import sys
import os
import json
import glob
import traceback

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_file> [--stdout]")
        sys.exit(1)
    
    input_file = sys.argv[1]
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
                            if "placeholder" in content.lower() or "could not be downloaded" in content.lower():
                                print(f"Found placeholder instead of real model: {model_file}")
                                print(f"Content: {content}")
                                print("Error: Model file is just a placeholder. The actual model could not be downloaded.")
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

@app.route('/test_model_download', methods=['POST'])
def test_model_download():
    """Test endpoint to verify model downloads are working correctly"""
    request_id = str(uuid.uuid4())[:8]
    print(f"[{request_id}] ===== MODEL DOWNLOAD TEST REQUEST =====")
    print(f"[{request_id}] Timestamp: {datetime.datetime.now().isoformat()}")
    
    # Parse the request
    data = request.json
    
    if not data or 'model_url' not in data:
        print(f"[{request_id}] Error: Missing required 'model_url' field")
        return jsonify({"error": "Missing required 'model_url' field"}), 400
    
    model_url = data['model_url']
    print(f"[{request_id}] Testing model download from URL: {model_url}")
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp(prefix="model_test_")
    print(f"[{request_id}] Created temporary directory: {temp_dir}")
    
    try:
        # Determine the target filename
        if 'file_name' in data and data['file_name']:
            target_filename = data['file_name']
        else:
            # Extract from URL
            target_filename = model_url.split('/')[-1]
            if not target_filename:
                target_filename = 'model.joblib'
        
        target_path = os.path.join(temp_dir, target_filename)
        print(f"[{request_id}] Target path for download: {target_path}")
        
        # Try direct download first
        direct_success = download_model_from_blob(model_url, target_path)
        
        if direct_success:
            # Try to load the model to verify it's valid
            try:
                import joblib
                model = joblib.load(target_path)
                model_info = {
                    "model_type": type(model).__name__,
                    "has_predict": hasattr(model, 'predict'),
                    "file_size": os.path.getsize(target_path)
                }
                
                print(f"[{request_id}] Model loaded successfully: {model_info}")
                
                return jsonify({
                    "success": True,
                    "message": "Model downloaded and loaded successfully",
                    "model_info": model_info
                })
            except Exception as e:
                print(f"[{request_id}] Error loading model: {str(e)}")
                return jsonify({
                    "success": False, 
                    "error": f"Model file was downloaded but could not be loaded: {str(e)}"
                }), 400
        else:
            # Try fallback to requests
            print(f"[{request_id}] Direct download failed, trying fallback method")
            try:
                import requests
                
                if not model_url.startswith('http'):
                    # Construct full URL from relative path
                    azure_account = os.environ.get('AZURE_STORAGE_ACCOUNT')
                    azure_container = os.environ.get('AZURE_CONTAINER')
                    if azure_account and azure_container:
                        if model_url.startswith('/'):
                            model_url = model_url[1:]
                        model_url = f"https://{azure_account}.blob.core.windows.net/{azure_container}/{model_url}"
                        print(f"[{request_id}] Constructed full URL: {model_url}")
                
                print(f"[{request_id}] Attempting fallback download from: {model_url}")
                response = requests.get(model_url, stream=True, timeout=60)
                response.raise_for_status()
                
                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"[{request_id}] Fallback download completed, file size: {os.path.getsize(target_path)} bytes")
                
                # Verify with joblib
                try:
                    import joblib
                    model = joblib.load(target_path)
                    model_info = {
                        "model_type": type(model).__name__,
                        "has_predict": hasattr(model, 'predict'),
                        "file_size": os.path.getsize(target_path),
                        "method": "fallback"
                    }
                    
                    print(f"[{request_id}] Model loaded successfully via fallback: {model_info}")
                    
                    return jsonify({
                        "success": True,
                        "message": "Model downloaded and loaded successfully via fallback method",
                        "model_info": model_info
                    })
                except Exception as e:
                    print(f"[{request_id}] Error loading model from fallback download: {str(e)}")
                    return jsonify({
                        "success": False, 
                        "error": f"Model file was downloaded via fallback but could not be loaded: {str(e)}"
                    }), 400
                
            except Exception as e:
                print(f"[{request_id}] Error in fallback download: {str(e)}")
                return jsonify({
                    "success": False,
                    "error": f"Failed to download model via any method: {str(e)}"
                }), 400
    
    except Exception as e:
        print(f"[{request_id}] Unexpected error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
        
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"[{request_id}] Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"[{request_id}] Error cleaning up: {str(e)}")
        
        print(f"[{request_id}] ===== MODEL DOWNLOAD TEST COMPLETED =====")
        return jsonify({"success": False, "error": "Failed to download model"}), 400

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