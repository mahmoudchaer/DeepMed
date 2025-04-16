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
        
        # Look for encoding files
        encoding_files = []
        for filename in os.listdir(temp_dir):
            if filename.endswith('_encoding.json') or filename == 'encoding.json' or filename == 'preprocessing.json':
                encoding_files.append(filename)
        
        if not encoding_files:
            return jsonify({"error": "No encoding files found in the model package"}), 404
        
        # Extract encoding maps from each file
        encoding_maps = {}
        for file in encoding_files:
            file_path = os.path.join(temp_dir, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Process different potential formats
                if isinstance(data, dict):
                    # If it's a dict of column name -> encoding map
                    for column, mapping in data.items():
                        if isinstance(mapping, dict):
                            encoding_maps[column] = mapping
                    
                    # Also check if there's a 'target_map' or similar key
                    for key in ['target_map', 'target_encoding', 'label_encoding']:
                        if key in data and isinstance(data[key], dict):
                            encoding_maps[key] = data[key]
            except Exception as e:
                # Skip files that can't be parsed
                print(f"Error parsing {file}: {str(e)}")
                continue
        
        if not encoding_maps:
            return jsonify({"error": "No valid encoding maps found in the model package"}), 404
        
        return jsonify({"encoding_maps": encoding_maps})
    
    except Exception as e:
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
                    for filename in os.listdir(temp_dir):
                        if filename.endswith('_encoding.json') or filename == 'encoding.json' or filename == 'preprocessing.json':
                            encoding_files.append(os.path.join(temp_dir, filename))
                    
                    # Extract encoding maps from files
                    encoding_map = None
                    for file_path in encoding_files:
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                
                            # Check if the selected encoding column exists in this file
                            if selected_encoding in data:
                                encoding_map = data[selected_encoding]
                                break
                                
                            # Also check under common keys
                            for key in ['target_map', 'target_encoding', 'label_encoding']:
                                if key in data and selected_encoding == key:
                                    encoding_map = data[key]
                                    break
                                    
                            if encoding_map:
                                break
                        except:
                            continue
                    
                    # If we found an encoding map, decode the prediction column
                    if encoding_map:
                        print(f"Decoding prediction column using {selected_encoding} encoding map")
                        # Parse the CSV
                        df = pd.read_csv(io.StringIO(output_data))
                        
                        # Identify the prediction column - usually named 'prediction'
                        pred_col = None
                        for col in df.columns:
                            if col.lower() in ['prediction', 'predicted', 'target', 'label']:
                                pred_col = col
                                break
                                
                        # If no prediction column found, use the last column
                        if not pred_col and len(df.columns) > 0:
                            pred_col = df.columns[-1]
                            
                        # Apply decoding if we found a prediction column
                        if pred_col:
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
                            
                            # Map values using the encoding
                            df[f"{pred_col}_decoded"] = df[pred_col].map(fixed_map)
                            
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
                    for filename in os.listdir(temp_dir):
                        if filename.endswith('_encoding.json') or filename == 'encoding.json' or filename == 'preprocessing.json':
                            encoding_files.append(os.path.join(temp_dir, filename))
                    
                    # Extract encoding maps from files
                    encoding_map = None
                    for file_path in encoding_files:
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                
                            # Check if the selected encoding column exists in this file
                            if selected_encoding in data:
                                encoding_map = data[selected_encoding]
                                break
                                
                            # Also check under common keys
                            for key in ['target_map', 'target_encoding', 'label_encoding']:
                                if key in data and selected_encoding == key:
                                    encoding_map = data[key]
                                    break
                                    
                            if encoding_map:
                                break
                        except:
                            continue
                    
                    # If we found an encoding map, decode the prediction column
                    if encoding_map:
                        # Identify the prediction column - usually named 'prediction'
                        pred_col = None
                        for col in df.columns:
                            if col.lower() in ['prediction', 'predicted', 'target', 'label']:
                                pred_col = col
                                break
                                
                        # If no prediction column found, use the last column
                        if not pred_col and len(df.columns) > 0:
                            pred_col = df.columns[-1]
                            
                        # Apply decoding if we found a prediction column
                        if pred_col:
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
                            
                            # Map values using the encoding
                            df[f"{pred_col}_decoded"] = df[pred_col].map(fixed_map)
                            
                            # Write back to the file
                            df.to_csv(output_path, index=False)
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
