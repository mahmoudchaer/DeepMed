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
    try:
        # Save and extract the ZIP package.
        zip_path = os.path.join(temp_dir, "model_package.zip")
        model_file.save(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Save the input file.
        input_file_path = os.path.join(temp_dir, input_file.filename)
        input_file.save(input_file_path)

        # Create a virtual environment in the temporary directory.
        venv_path = os.path.join(temp_dir, "venv")
        subprocess.run(["python3", "-m", "venv", venv_path], check=True)

        # Install the extracted requirements.
        req_file = os.path.join(temp_dir, "requirements.txt")
        pip_path = os.path.join(venv_path, "bin", "pip")
        pip_install = subprocess.run([pip_path, "install", "-r", req_file],
                                     capture_output=True, text=True)
        if pip_install.returncode != 0:
            return jsonify({"error": f"pip install failed: {pip_install.stderr}"}), 500

        # Run the predict.py script from the package.
        python_path = os.path.join(venv_path, "bin", "python")
        predict_script = os.path.join(temp_dir, "predict.py")
        result = subprocess.run(
            [python_path, predict_script],
            cwd=temp_dir,
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return jsonify({"error": result.stderr}), 400

        # Read the fixed output file "output.csv"
        output_path = os.path.join(temp_dir, "output.csv")
        if not os.path.exists(output_path):
            return jsonify({"error": "Output file not found after prediction."}), 500

        with open(output_path, "r") as f:
            output_data = f.read()

        return jsonify({"output_file": output_data})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Inference process timed out."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(temp_dir)
        print(f"Deleted temporary directory: {temp_dir}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5101)
