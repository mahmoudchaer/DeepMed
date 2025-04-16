import os
import zipfile
import tempfile
import shutil
import subprocess
import pandas as pd
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

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
            print("No requirements.txt file found, installing default packages")
            subprocess.run([pip_path, "install", "pandas", "numpy", "scikit-learn", "joblib"], 
                          capture_output=True, text=True)

        # Run the predict.py script from the package, passing the input file as an argument
        predict_script = os.path.join(temp_dir, "predict.py")
        if not os.path.exists(predict_script):
            return jsonify({"error": "predict.py script not found in model package"}), 500
            
        print(f"Running prediction with input file: {input_file.filename}")
        result = subprocess.run(
            [python_path, predict_script, input_file_path],
            cwd=temp_dir,
            capture_output=True, text=True, timeout=300  # 5 minute timeout
        )
        
        print(f"Prediction completed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            
        # Read the fixed output file "output.csv"
        output_path = os.path.join(temp_dir, "output.csv")
        
        # If output file doesn't exist, try to create it as a fallback
        if not os.path.exists(output_path):
            print(f"Output file not found at {output_path}")
            
            # Try to find if output was created with a different name
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv') and f != input_file.filename]
            if csv_files:
                print(f"Found alternative CSV files: {csv_files}")
                output_path = os.path.join(temp_dir, csv_files[0])
                print(f"Using alternative output file: {output_path}")
            else:
                # Create a minimal output file with error info
                print("Creating fallback output file")
                try:
                    # Try to read the input file to get its structure
                    if input_file_path.lower().endswith('.csv'):
                        df = pd.read_csv(input_file_path, nrows=5)
                    elif input_file_path.lower().endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(input_file_path, nrows=5)
                    
                    # Add error column
                    df['prediction'] = "ERROR"
                    df['error_message'] = f"Prediction failed: {result.stderr if result.returncode != 0 else 'No output file created'}"
                    
                    # Save as output.csv
                    df.to_csv(output_path, index=False)
                    print(f"Created fallback output file at {output_path}")
                except Exception as fallback_error:
                    # Last resort - create minimal CSV
                    with open(output_path, 'w') as f:
                        f.write("error,message\n")
                        f.write(f"True,\"Prediction failed: {result.stderr if result.returncode != 0 else 'No output file created'}\"\n")
                    print(f"Created minimal fallback file at {output_path}")

        with open(output_path, "r") as f:
            output_data = f.read()

        elapsed_time = time.time() - start_time
        print(f"Prediction completed in {elapsed_time:.2f} seconds")
        return jsonify({"output_file": output_data})
    except subprocess.TimeoutExpired:
        print("Prediction timed out")
        return jsonify({"error": "Inference process timed out."}), 504
    except Exception as e:
        print(f"Prediction failed with error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(temp_dir)
        print(f"Deleted temporary directory: {temp_dir}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5101)
