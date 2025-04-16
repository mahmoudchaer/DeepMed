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
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/extract_encodings', methods=['POST'])
def extract_encodings():
    """Extract encoding maps from a model package"""
    if 'model_package' not in request.files:
        return jsonify({"error": "Model package file is required"}), 400
    
    model_file = request.files['model_package']
    
    # Validate file type
    if not model_file.filename.lower().endswith('.zip'):
        return jsonify({"error": "Model package must be a ZIP archive"}), 400
    
    # Create a temporary directory to extract files
    temp_dir = tempfile.mkdtemp(prefix="encoding_extract_")
    
    try:
        # Save and extract the ZIP package
        zip_path = os.path.join(temp_dir, "model_package.zip")
        model_file.save(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        print(f"Extracted model package to {temp_dir}")
        
        # List all files for debugging
        all_files = os.listdir(temp_dir)
        print(f"All files in extracted package: {all_files}")
        
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
                    'target' in filename.lower() or
                    'label' in filename.lower() or
                    'map' in filename.lower()):
                    encoding_files.append(filename)
        
        print(f"Found {len(json_files)} JSON files: {json_files}")
        print(f"Potential encoding files based on name: {encoding_files}")
        
        # If no encoding files found by name, try all JSON files
        if not encoding_files and json_files:
            print("No encoding files found by name pattern, examining all JSON files")
            encoding_files = json_files
        
        if not encoding_files:
            # Look deeper in subdirectories
            print("Looking in subdirectories for encoding files")
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
                            'target' in filename.lower() or
                            'label' in filename.lower() or
                            'map' in filename.lower()):
                            encoding_files.append(rel_path)
                        elif rel_path not in json_files:
                            json_files.append(rel_path)
            
            print(f"After subdirectory search - JSON files: {json_files}")
            print(f"After subdirectory search - Potential encoding files: {encoding_files}")
            
            # If still no encoding files but found JSON files, try them all
            if not encoding_files and json_files:
                print("No encoding files found in subdirectories by name pattern, examining all JSON files")
                encoding_files = json_files
        
        if not encoding_files:
            print("No JSON files found in the model package")
            return jsonify({"error": "No encoding files found in the model package"}), 404
        
        # Extract encoding maps from each file
        encoding_maps = {}
        for file in encoding_files:
            file_path = os.path.join(temp_dir, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                print(f"Examining file {file} for encoding maps")
                
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
            return jsonify({"error": "No valid encoding maps found in the model package"}), 404
        
        print(f"Successfully extracted {len(encoding_maps)} encoding maps: {list(encoding_maps.keys())}")
        return jsonify({"encoding_maps": encoding_maps})
    
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
            print("No requirements.txt file found, installing default packages")
            subprocess.run([pip_path, "install", "pandas", "numpy", "scikit-learn", "joblib"], 
                          capture_output=True, text=True)

        # Run the predict.py script from the package
        predict_script = os.path.join(temp_dir, "predict.py")
        if not os.path.exists(predict_script):
            return jsonify({"error": "predict.py script not found in model package"}), 500
            
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
            
            # If selected_encoding is provided, try to decode the prediction column
            if selected_encoding:
                try:
                    # Find encoding files
                    encoding_files = []
                    print(f"Looking for encoding files to decode predictions using '{selected_encoding}'")
                    
                    for filename in os.listdir(temp_dir):
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
                            encoding_files.append(os.path.join(temp_dir, filename))
                    
                    print(f"Found potential encoding files: {encoding_files}")
                    
                    # Extract encoding maps from files
                    encoding_map = None
                    for file_path in encoding_files:
                        try:
                            print(f"Examining file {os.path.basename(file_path)} for encoding map")
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                
                            # Check if the selected encoding column exists in this file
                            if selected_encoding in data:
                                encoding_map = data[selected_encoding]
                                print(f"Found encoding map for '{selected_encoding}' in {os.path.basename(file_path)}")
                                break
                                
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
                        print(f"Decoding prediction column using {selected_encoding} encoding map")
                        print(f"Encoding map contains {len(encoding_map)} mappings: {encoding_map}")
                        
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
                            
                        # Apply decoding if we found a prediction column
                        if pred_col:
                            print(f"Decoding column: '{pred_col}'")
                            print(f"Column values before decoding: {df[pred_col].value_counts().to_dict()}")
                            
                            # Convert encoding map from string keys to the appropriate type if needed
                            fixed_map = {}
                            for k, v in encoding_map.items():
                                # The predictions are likely to be numbers, so try to convert keys
                                try:
                                    # First try int, then float if that fails
                                    try:
                                        fixed_map[int(k)] = v
                                    except ValueError:
                                        fixed_map[float(k)] = v
                                except ValueError:
                                    # Keep as string if conversion fails
                                    fixed_map[k] = v
                            
                            print(f"Fixed encoding map: {fixed_map}")
                            
                            # Make a copy of the original column
                            original_col = f"{pred_col}_original"
                            df[original_col] = df[pred_col].copy()
                            
                            # Map values using the encoding
                            decoded_col = f"{pred_col}_decoded"
                            df[decoded_col] = df[pred_col].map(fixed_map)
                            
                            # Check if decoding worked
                            null_count = df[decoded_col].isna().sum()
                            if null_count > 0:
                                print(f"Warning: {null_count} values couldn't be decoded")
                                print(f"Unique values in prediction column: {df[pred_col].unique().tolist()}")
                                print(f"Keys in encoding map: {list(fixed_map.keys())}")
                                
                                # Try converting types if needed
                                if df[pred_col].dtype != 'object':
                                    print(f"Converting prediction column from {df[pred_col].dtype} to object type")
                                    df[pred_col] = df[pred_col].astype('object')
                                    # Try mapping again
                                    df[decoded_col] = df[pred_col].map(fixed_map)
                                    null_count = df[decoded_col].isna().sum()
                                    print(f"After type conversion: {null_count} values couldn't be decoded")
                            
                            # If decoding still failed for some values, try matching keys more flexibly
                            if null_count > 0:
                                print("Attempting flexible key matching...")
                                # For each unmatched value, try to find the closest key
                                for idx, val in df[df[decoded_col].isna()][pred_col].items():
                                    # Convert val to string for comparison
                                    str_val = str(val)
                                    # Look for exact string match
                                    if str_val in fixed_map:
                                        df.at[idx, decoded_col] = fixed_map[str_val]
                                    # Try matching numeric values
                                    elif isinstance(val, (int, float)):
                                        for k in fixed_map.keys():
                                            try:
                                                if float(k) == float(val):
                                                    df.at[idx, decoded_col] = fixed_map[k]
                                                    break
                                            except (ValueError, TypeError):
                                                pass
                            
                            print(f"Column values after decoding: {df[decoded_col].value_counts().to_dict()}")
                            
                            # Add a message at the top of the CSV indicating decoding was applied
                            df_with_message = pd.DataFrame([
                                {df.columns[0]: f"DECODED USING '{selected_encoding}' MAP", 
                                 decoded_col: "Original values preserved in column " + original_col}
                            ])
                            df = pd.concat([df_with_message, df], ignore_index=True)
                            
                            # Convert back to CSV
                            output_data = df.to_csv(index=False)
                            print("Successfully decoded prediction column")
                except Exception as decode_error:
                    print(f"Error decoding prediction: {str(decode_error)}")
                    # Continue without decoding if there's an error
            
            return jsonify({"output_file": output_data})
            
        # If no stdout, check if output.csv was created
        output_path = os.path.join(temp_dir, "output.csv")
        if os.path.exists(output_path):
            print(f"Reading output from file: {output_path}")
            
            # If selected_encoding is provided, try to decode the prediction column
            if selected_encoding:
                try:
                    # Load CSV
                    df = pd.read_csv(output_path)
                    
                    # Find encoding files and decode as above
                    encoding_files = []
                    print(f"Looking for encoding files to decode predictions in output.csv using '{selected_encoding}'")
                    
                    for filename in os.listdir(temp_dir):
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
                            encoding_files.append(os.path.join(temp_dir, filename))
                    
                    print(f"Found potential encoding files: {encoding_files}")
                    
                    # Extract encoding maps from files
                    encoding_map = None
                    for file_path in encoding_files:
                        try:
                            print(f"Examining file {os.path.basename(file_path)} for encoding map")
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            
                            # Check if the selected encoding column exists in this file
                            if selected_encoding in data:
                                encoding_map = data[selected_encoding]
                                print(f"Found encoding map for '{selected_encoding}' in {os.path.basename(file_path)}")
                                break
                            
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
                        print(f"Decoding prediction column using {selected_encoding} encoding map")
                        print(f"Encoding map contains {len(encoding_map)} mappings: {encoding_map}")
                        
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
                            
                        # Apply decoding if we found a prediction column
                        if pred_col:
                            print(f"Decoding column: '{pred_col}'")
                            print(f"Column values before decoding: {df[pred_col].value_counts().to_dict()}")
                            
                            # Convert encoding map from string keys to the appropriate type if needed
                            fixed_map = {}
                            for k, v in encoding_map.items():
                                # The predictions are likely to be numbers, so try to convert keys
                                try:
                                    # First try int, then float if that fails
                                    try:
                                        fixed_map[int(k)] = v
                                    except ValueError:
                                        fixed_map[float(k)] = v
                                except ValueError:
                                    # Keep as string if conversion fails
                                    fixed_map[k] = v
                            
                            print(f"Fixed encoding map: {fixed_map}")
                            
                            # Make a copy of the original column
                            original_col = f"{pred_col}_original"
                            df[original_col] = df[pred_col].copy()
                            
                            # Map values using the encoding
                            decoded_col = f"{pred_col}_decoded"
                            df[decoded_col] = df[pred_col].map(fixed_map)
                            
                            # Check if decoding worked
                            null_count = df[decoded_col].isna().sum()
                            if null_count > 0:
                                print(f"Warning: {null_count} values couldn't be decoded")
                                print(f"Unique values in prediction column: {df[pred_col].unique().tolist()}")
                                print(f"Keys in encoding map: {list(fixed_map.keys())}")
                                
                                # Try converting types if needed
                                if df[pred_col].dtype != 'object':
                                    print(f"Converting prediction column from {df[pred_col].dtype} to object type")
                                    df[pred_col] = df[pred_col].astype('object')
                                    # Try mapping again
                                    df[decoded_col] = df[pred_col].map(fixed_map)
                                    null_count = df[decoded_col].isna().sum()
                                    print(f"After type conversion: {null_count} values couldn't be decoded")
                            
                            # If decoding still failed for some values, try matching keys more flexibly
                            if null_count > 0:
                                print("Attempting flexible key matching...")
                                # For each unmatched value, try to find the closest key
                                for idx, val in df[df[decoded_col].isna()][pred_col].items():
                                    # Convert val to string for comparison
                                    str_val = str(val)
                                    # Look for exact string match
                                    if str_val in fixed_map:
                                        df.at[idx, decoded_col] = fixed_map[str_val]
                                    # Try matching numeric values
                                    elif isinstance(val, (int, float)):
                                        for k in fixed_map.keys():
                                            try:
                                                if float(k) == float(val):
                                                    df.at[idx, decoded_col] = fixed_map[k]
                                                    break
                                            except (ValueError, TypeError):
                                                pass
                            
                            print(f"Column values after decoding: {df[decoded_col].value_counts().to_dict()}")
                            
                            # Add a message at the top of the CSV indicating decoding was applied
                            df_with_message = pd.DataFrame([
                                {df.columns[0]: f"DECODED USING '{selected_encoding}' MAP", 
                                 decoded_col: "Original values preserved in column " + original_col}
                            ])
                            df = pd.concat([df_with_message, df], ignore_index=True)
                            
                            # Write back to the file
                            df.to_csv(output_path, index=False)
                          else:
                              print("No prediction column found to decode")
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
    app.run(host='0.0.0.0', port=5101)
