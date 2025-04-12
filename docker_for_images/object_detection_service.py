import os
import io
import json
import zipfile
import tempfile
import logging
import traceback
import shutil
import yaml
from pathlib import Path
import subprocess
from flask import Flask, request, jsonify, Response, send_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define YOLOv5 paths
YOLOV5_DIR = "/app/yolov5"

# Track temp files for cleanup
temp_dirs = []

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "object-detection-service"})

@app.route('/finetune', methods=['POST'])
def finetune_model():
    """
    Endpoint to fine-tune a YOLOv5 model
    
    Expects a multipart/form-data POST with:
    - zipFile: A zip file with dataset in YOLOv5 format
    - level: Level of fine-tuning (1-5)
    
    Returns a zip file of the fine-tuned model
    """
    try:
        if 'zipFile' not in request.files:
            return jsonify({"error": "No ZIP file uploaded"}), 400
        
        # Get the zip file from the request
        zip_file = request.files['zipFile']
        
        # Get level parameter with default
        level = int(request.form.get('level', 3))
        
        # Define preset configurations for each level
        level_configs = {
            1: {
                'model_size': 'nano',
                'epochs': 20,
                'batch_size': 16,
                'img_size': 320
            },
            2: {
                'model_size': 'small',
                'epochs': 30,
                'batch_size': 16,
                'img_size': 416
            },
            3: {
                'model_size': 'small',
                'epochs': 50,
                'batch_size': 16,
                'img_size': 640
            },
            4: {
                'model_size': 'medium',
                'epochs': 80,
                'batch_size': 8,
                'img_size': 640
            },
            5: {
                'model_size': 'large',
                'epochs': 100,
                'batch_size': 8,
                'img_size': 640
            }
        }
        
        # Validate level
        if level < 1 or level > 5:
            return jsonify({"error": "Level must be between 1 and 5"}), 400
        
        # Get configuration for the selected level
        config = level_configs[level]
        model_size = config['model_size']
        epochs = config['epochs']
        batch_size = config['batch_size']
        img_size = config['img_size']
        
        # Log the start of processing
        logger.info(f"Starting YOLOv5 fine-tuning with level {level} (model: {model_size}, epochs: {epochs})")
        
        # Create a temporary working directory
        temp_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_dir)
        
        # Create dataset directory
        dataset_dir = os.path.join(temp_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save the uploaded zip file
        zip_path = os.path.join(temp_dir, "dataset.zip")
        zip_file.save(zip_path)
        
        # Extract the dataset
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        
        # Check for proper dataset structure (data.yaml and folders)
        data_yaml_path = os.path.join(dataset_dir, "data.yaml")
        if not os.path.exists(data_yaml_path):
            # Try to find a data.yaml file somewhere in the extracted directory
            yaml_files = list(Path(dataset_dir).rglob("*.yaml"))
            if yaml_files:
                data_yaml_path = str(yaml_files[0])
            else:
                return jsonify({"error": "Dataset must contain a data.yaml file that defines classes and paths"}), 400
        
        # Read the data.yaml file
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Check if we have required paths
        if 'train' not in data_config or 'val' not in data_config:
            return jsonify({"error": "data.yaml must contain 'train' and 'val' paths"}), 400
        
        # Update paths to be absolute
        train_path = os.path.join(dataset_dir, data_config['train'].lstrip('/')) if not os.path.isabs(data_config['train']) else data_config['train']
        val_path = os.path.join(dataset_dir, data_config['val'].lstrip('/')) if not os.path.isabs(data_config['val']) else data_config['val']
        
        data_config['train'] = train_path
        data_config['val'] = val_path
        
        # Write back the updated data.yaml
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f)
        
        # Create the results directory
        results_dir = os.path.join(temp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Determine model path
        model_name = f"yolov5{model_size[0]}"  # Convert size to letter (n, s, m, l, x)
        
        # Run YOLOv5 training command
        train_cmd = [
            "python", f"{YOLOV5_DIR}/train.py",
            "--img", str(img_size),
            "--batch", str(batch_size),
            "--epochs", str(epochs),
            "--data", data_yaml_path,
            "--weights", f"{model_name}.pt",
            "--project", results_dir,
            "--name", "exp",
            "--cache",
            "--device", "cpu"  # Explicitly use CPU
        ]
        
        logger.info(f"Running training command: {' '.join(train_cmd)}")
        
        # Execute the training command
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Log the training output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        
        # Check training success
        if process.returncode != 0:
            return jsonify({"error": "YOLOv5 training failed"}), 500
        
        # Find the best model
        exp_dir = os.path.join(results_dir, "exp")
        weights_dir = os.path.join(exp_dir, "weights")
        best_model_path = os.path.join(weights_dir, "best.pt")
        
        if not os.path.exists(best_model_path):
            logger.error(f"Best model not found at {best_model_path}")
            return jsonify({"error": "Training completed but best model not found"}), 500
        
        # Create a zip file to return the model and related files
        temp_output_zip = os.path.join(temp_dir, "model_output.zip")
        
        with zipfile.ZipFile(temp_output_zip, 'w') as zipf:
            # Add the best model
            zipf.write(best_model_path, arcname="best.pt")
            
            # Add the last model
            last_model_path = os.path.join(weights_dir, "last.pt")
            if os.path.exists(last_model_path):
                zipf.write(last_model_path, arcname="last.pt")
            
            # Add training results (plots, etc.)
            results_files = os.listdir(exp_dir)
            for file in results_files:
                if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.txt'):
                    file_path = os.path.join(exp_dir, file)
                    zipf.write(file_path, arcname=f"results/{file}")
        
        # Read the zip file into memory
        with open(temp_output_zip, 'rb') as f:
            memory_file = io.BytesIO(f.read())
        
        # Clean up
        cleanup_temp_dirs()
        
        # Return the zip file
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='yolov5_model.zip'
        )
        
    except Exception as e:
        logger.error(f"Error during YOLOv5 fine-tuning: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500
    finally:
        # Make sure to clean up temp files even if there's an error
        cleanup_temp_dirs()

def cleanup_temp_dirs():
    """Clean up temporary directories"""
    global temp_dirs
    for dir_path in temp_dirs:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.info(f"Cleaned up temporary directory: {dir_path}")
        except Exception as e:
            logger.error(f"Error cleaning up {dir_path}: {e}")
    temp_dirs = []

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5027))
    logger.info(f"Starting object detection service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False) 