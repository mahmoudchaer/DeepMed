import os
import zipfile
import tempfile
import shutil
import subprocess
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Check that both files are provided in the request.
    if 'model_package' not in request.files or 'input_file' not in request.files:
        return jsonify({"error": "Both 'model_package' and 'input_file' must be provided."}), 400

    model_file = request.files['model_package']
    input_file = request.files['input_file']
    
    # Optionally validate file extensions: ZIP for model package and Excel for input
    if not model_file.filename.lower().endswith('.zip'):
        return jsonify({"error": "Model package must be a ZIP archive."}), 400

    if not (input_file.filename.lower().endswith('.xlsx') or input_file.filename.lower().endswith('.xls')):
        return jsonify({"error": "Input file must be an Excel file (.xls or .xlsx)."}), 400

    # Create a temporary directory for this session.
    temp_dir = tempfile.mkdtemp(prefix="session_")
    
    try:
        # Save the uploaded ZIP file.
        zip_path = os.path.join(temp_dir, "model_package.zip")
        model_file.save(zip_path)

        # Extract the contents of the ZIP.
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Save the input Excel file into the temp directory.
        excel_file_path = os.path.join(temp_dir, input_file.filename)
        input_file.save(excel_file_path)

        # Create a virtual environment within the temporary directory.
        venv_path = os.path.join(temp_dir, "venv")
        subprocess.run(["python3", "-m", "venv", venv_path], check=True)

        # Install the requirements from the extracted requirements.txt.
        req_file = os.path.join(temp_dir, "requirements.txt")
        pip_path = os.path.join(venv_path, "bin", "pip")
        pip_install = subprocess.run([pip_path, "install", "-r", req_file],
                                     capture_output=True, text=True)
        if pip_install.returncode != 0:
            return jsonify({"error": f"pip install failed: {pip_install.stderr}"}), 500

        # Prepare to run the prediction script.
        # The script is assumed to be named predict.py (extracted from the ZIP).
        python_path = os.path.join(venv_path, "bin", "python")
        predict_script = os.path.join(temp_dir, "predict.py")
        
        # Run the prediction script.
        # If your model's predict.py auto-detects the Excel file in its folder, you can run without arguments.
        # You can pass arguments if needed (e.g., "--input", excel_file_path).
        result = subprocess.run(
            [python_path, predict_script],
            cwd=temp_dir,  # Ensure the script runs in the directory with the Excel file.
            capture_output=True, text=True, timeout=120  # Increase timeout if necessary
        )
        
        # If the prediction script fails, return an error.
        if result.returncode != 0:
            return jsonify({"error": result.stderr}), 400

        # Return the output from the prediction script.
        return jsonify({"output": result.stdout})
    
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Inference process timed out."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Always clean up the temporary directory.
        shutil.rmtree(temp_dir)
        print(f"Directory {temp_dir} deleted!")

if __name__ == '__main__':
    # Run the Flask app on host 0.0.0.0 and port 5101.
    app.run(host='0.0.0.0', port=5101)
