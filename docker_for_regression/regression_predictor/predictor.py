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

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_file> [--stdout]")
        return 1
    
    input_file = sys.argv[1]
    stdout_mode = "--stdout" in sys.argv
    
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
    
    # Load model
    model = joblib.load(model_file)
    
    # Load metadata for feature names if available
    feature_names = None
    for path in ['metadata.json', 'model_metadata.json', 'features.json']:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    metadata = json.load(f)
                    if 'selected_features' in metadata:
                        feature_names = metadata['selected_features']
                        print(f"Found feature names: {feature_names}")
                        break
            except:
                pass
    
    # Load input data
    try:
        if input_file.endswith('.csv'):
            data = pd.read_csv(input_file)
        elif input_file.endswith('.xlsx') or input_file.endswith('.xls'):
            data = pd.read_excel(input_file)
        else:
            print(f"Unsupported file format: {input_file}")
            return 1
    except Exception as e:
        print(f"Error loading input file: {str(e)}")
        return 1
    
    # Filter to only include needed features if we know them
    X = data
    if feature_names:
        missing_cols = [col for col in feature_names if col not in data.columns]
        if missing_cols:
            print(f"Warning: Input data missing columns: {missing_cols}")
        
        # Only use available columns from feature_names
        available_features = [col for col in feature_names if col in data.columns]
        if available_features:
            X = data[available_features]
    
    # Make predictions
    try:
        predictions = model.predict(X)
        data['prediction'] = predictions
        
        # Output to stdout or file
        if stdout_mode:
            print(data.to_csv(index=False))
        else:
            data.to_csv('output.csv', index=False)
        
        return 0
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
""")
            print("Created default predict.py script")
            
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
                
            # Return the CSV content
            return jsonify({"output_file": output_data})
            
        # If no stdout, check if output.csv was created
        output_path = os.path.join(temp_dir, "output.csv")
        if os.path.exists(output_path):
            print(f"Reading output from file: {output_path}")
            
            # Read and return the file content
            with open(output_path, 'r') as f:
                output_data = f.read()
                
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
                    
            from threading import Thread
            Thread(target=delayed_cleanup).start()
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    from waitress import serve
    print(f"Starting Regression Predictor Service on port {port}")
    serve(app, host='0.0.0.0', port=port) 