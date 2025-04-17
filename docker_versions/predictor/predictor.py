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
    return jsonify({"status": "ok"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'model_package' not in request.files or 'input_file' not in request.files:
        return jsonify({"error": "Both 'model_package' and 'input_file' must be provided."}), 400

    model_file = request.files['model_package']
    input_file = request.files['input_file']

    if not model_file.filename.lower().endswith('.zip'):
        return jsonify({"error": "Model package must be a ZIP archive."}), 400
    if not (input_file.filename.lower().endswith('.xlsx') or 
            input_file.filename.lower().endswith('.xls') or 
            input_file.filename.lower().endswith('.csv')):
        return jsonify({"error": "Input file must be an Excel or CSV file."}), 400

    temp_dir = tempfile.mkdtemp(prefix="session_")
    start_time = time.time()
    print(f"Created temporary directory: {temp_dir}")

    try:
        zip_path = os.path.join(temp_dir, "model_package.zip")
        model_file.save(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print(f"Extracted model package to {temp_dir}")

        input_file_path = os.path.join(temp_dir, input_file.filename)
        input_file.save(input_file_path)
        print(f"Saved input file to {input_file_path}")

        predict_script = os.path.join(temp_dir, "predict.py")
        if not os.path.exists(predict_script):
            return jsonify({"error": "predict.py script not found in model package"}), 500

        python_path = sys.executable  # Use system Python from Docker

        result = subprocess.run(
            [python_path, predict_script, input_file_path, "--stdout"],
            cwd=temp_dir,
            capture_output=True, text=True, timeout=300
        )

        print(f"Prediction completed with return code {result.returncode}")

        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            return jsonify({"error": f"Prediction script failed: {result.stderr}"}), 400

        if result.stdout.strip():
            print("Found prediction output in stdout")
            output_data = result.stdout
            try:
                reader = csv.reader(io.StringIO(output_data))
                rows = list(reader)
                if len(rows) > 0:
                    print(f"Valid CSV found with {len(rows)} rows")
                else:
                    print("Warning: Empty CSV output")
            except Exception as e:
                print(f"Warning: Output is not valid CSV: {str(e)}")

            return jsonify({"output_file": output_data})

        output_path = os.path.join(temp_dir, "output.csv")
        if os.path.exists(output_path):
            print(f"Reading output from file: {output_path}")
            with open(output_path, "r") as f:
                output_data = f.read()
            return jsonify({"output_file": output_data})

        print("No output found in stdout or file, creating fallback")
        try:
            if input_file_path.lower().endswith('.csv'):
                df = pd.read_csv(input_file_path)
            elif input_file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(input_file_path)

            df['prediction'] = "ERROR"
            df['error_message'] = "Prediction script did not produce any output"

            output_buffer = io.StringIO()
            df.to_csv(output_buffer, index=False)
            output_data = output_buffer.getvalue()
            print("Created fallback response from input data")

            return jsonify({"output_file": output_data})

        except Exception as fallback_error:
            print(f"Error creating fallback response: {str(fallback_error)}")
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
    logging.info("Starting predictor service")
    app.run(host='0.0.0.0', port=5101)
