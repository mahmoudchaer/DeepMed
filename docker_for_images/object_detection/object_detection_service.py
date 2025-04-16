import os
import io
import json
import zipfile
import tempfile
import logging
import traceback
import shutil
import yaml
import uuid
import datetime
from pathlib import Path
import subprocess
from flask import Flask, request, jsonify, Response, send_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define YOLOv5 paths
YOLOV5_DIR = "/app/yolov5"

# Define a cache directory for model outputs
CACHE_DIR = os.path.join(tempfile.gettempdir(), "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Track temp files for cleanup
temp_dirs = []

# Keep a dictionary of cached models with metadata
cached_models = {}

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
    - num_classes: (Optional) Manually specify the number of classes
    
    Returns a zip file of the fine-tuned model
    """
    try:
        if 'zipFile' not in request.files:
            return jsonify({"error": "No ZIP file uploaded"}), 400
        
        # Get the zip file from the request
        zip_file = request.files['zipFile']
        
        # Get level parameter with default
        level = int(request.form.get('level', 3))
        
        # New: Get manual num_classes if provided
        manual_num_classes = request.form.get('num_classes')
        
        # Create a unique ID for this training run
        training_id = str(uuid.uuid4())
        
        # Define preset configurations for each level
        level_configs = {
            1: {
                'model_size': 'nano',
                'epochs': 5,
                'batch_size': 16,
                'img_size': 160
            },
            2: {
                'model_size': 'nano',
                'epochs': 10,
                'batch_size': 16,
                'img_size': 320
            },
            3: {
                'model_size': 'small',
                'epochs': 30,
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
        
        # Look for the dataset root directory
        # We expect a structure where the zip contains a folder, and inside that folder are train/valid folders
        subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        
        # Check if we have exactly one subdirectory (the dataset root)
        dataset_root = dataset_dir
        if len(subdirs) == 1:
            dataset_root = os.path.join(dataset_dir, subdirs[0])
        
        # Check for train and valid directories
        train_dir = os.path.join(dataset_root, "train")
        valid_dir = os.path.join(dataset_root, "valid")
        
        if not (os.path.exists(train_dir) and os.path.exists(valid_dir)):
            return jsonify({"error": "Dataset must contain 'train' and 'valid' directories"}), 400
        
        # Check for images and labels directories
        train_images_dir = os.path.join(train_dir, "images")
        train_labels_dir = os.path.join(train_dir, "labels")
        valid_images_dir = os.path.join(valid_dir, "images")
        valid_labels_dir = os.path.join(valid_dir, "labels")
        
        if not all(os.path.exists(d) for d in [train_images_dir, train_labels_dir, valid_images_dir, valid_labels_dir]):
            return jsonify({"error": "Dataset must contain 'images' and 'labels' subdirectories in both 'train' and 'valid' folders"}), 400
        
        # Get class names by analyzing label files
        class_ids = set()
        # Scan ALL label files in both train and valid directories
        for labels_dir in [train_labels_dir, valid_labels_dir]:
            for label_file in os.listdir(labels_dir):
                if not label_file.endswith('.txt'):
                    continue
                with open(os.path.join(labels_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # Should have class_id + 4 coordinates
                            try:
                                class_id = int(parts[0])
                                class_ids.add(class_id)
                            except ValueError:
                                continue
        
        # Sort the class IDs to ensure consistent order
        class_ids = sorted(list(class_ids))
        
        # If manual_num_classes is provided, use it
        if manual_num_classes:
            try:
                num_classes = int(manual_num_classes)
                logger.info(f"Using user-specified number of classes: {num_classes}")
            except ValueError:
                return jsonify({"error": "num_classes must be a valid integer"}), 400
        else:
            # Otherwise auto-detect from labels
            # Get the number of classes (max class_id + 1)
            if not class_ids:
                return jsonify({"error": "Could not detect any valid class IDs in the label files"}), 400
                
            num_classes = max(class_ids) + 1
            logger.info(f"Auto-detected {num_classes} classes with IDs: {class_ids}")
        
        # Create generic class names
        class_names = [f"class_{i}" for i in range(num_classes)]
        
        if num_classes < 1:
            return jsonify({"error": "Number of classes must be at least 1"}), 400
        
        # Create a new data.yaml file
        data_yaml_path = os.path.join(dataset_root, "data.yaml")
        data_config = {
            'path': dataset_root,
            'train': os.path.join('train', 'images'),
            'val': os.path.join('valid', 'images'),
            'nc': num_classes,
            'names': class_names
        }
        
        # Write the config to data.yaml
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
            "--device", "cpu",  # Explicitly use CPU
            "--workers", "2"    # Use more worker threads for data loading
        ]
        
        # For lower levels, add flags to speed up training
        if level <= 2:
            train_cmd.extend([
                "--noval",        # Skip validation
                "--nosave",       # Don't save each epoch
                "--patience", "0" # Disable early stopping
            ])
        
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
        output_zip_path = os.path.join(temp_dir, "model_output.zip")
        
        with zipfile.ZipFile(output_zip_path, 'w') as zipf:
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
        
        # Save a copy of the zip file to the cache directory
        cache_file_path = os.path.join(CACHE_DIR, f"{training_id}.zip")
        shutil.copy(output_zip_path, cache_file_path)
        
        # Cache the model metadata for 24 hours
        expiration_time = datetime.datetime.now() + datetime.timedelta(hours=24)
        cached_models[training_id] = {
            "path": cache_file_path,
            "expires": expiration_time,
            "level": level,
            "num_classes": num_classes,
            "size": os.path.getsize(cache_file_path)
        }
        
        logger.info(f"Model cached at {cache_file_path} with ID {training_id}")
        
        # Read the zip file into memory
        with open(output_zip_path, 'rb') as f:
            memory_file = io.BytesIO(f.read())
        
        # Clean up
        cleanup_temp_dirs()
        
        # Include the training ID in the response headers
        response = send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='yolov5_model.zip'
        )
        
        response.headers["X-Training-ID"] = training_id
        
        return response
        
    except Exception as e:
        logger.error(f"Error during YOLOv5 fine-tuning: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500
    finally:
        # Make sure to clean up temp files even if there's an error
        cleanup_temp_dirs()

@app.route('/retrieve_model/<training_id>', methods=['GET'])
def retrieve_model(training_id):
    """Retrieve a previously trained model by its ID"""
    try:
        if training_id not in cached_models:
            return jsonify({"error": "Model not found or has expired"}), 404
        
        model_info = cached_models[training_id]
        cache_file_path = model_info["path"]
        
        # Check if the model file still exists
        if not os.path.exists(cache_file_path):
            # Remove from cache if file doesn't exist
            del cached_models[training_id]
            return jsonify({"error": "Model file no longer exists"}), 404
            
        # Check if the model has expired
        if datetime.datetime.now() > model_info["expires"]:
            # Remove expired model
            os.remove(cache_file_path)
            del cached_models[training_id]
            return jsonify({"error": "Model has expired and was removed"}), 404
        
        # Return the model file
        return send_file(
            cache_file_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name='yolov5_model.zip'
        )
        
    except Exception as e:
        logger.error(f"Error retrieving model: {str(e)}")
        return jsonify({"error": str(e)}), 500

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

# Add cleanup of expired cached models
def cleanup_expired_models():
    """Clean up expired cached models"""
    now = datetime.datetime.now()
    to_remove = []
    
    for training_id, model_info in cached_models.items():
        if now > model_info["expires"]:
            # Clean up the file
            try:
                if os.path.exists(model_info["path"]):
                    os.remove(model_info["path"])
                    logger.info(f"Removed expired model file: {model_info['path']}")
            except Exception as e:
                logger.error(f"Error removing expired model file {model_info['path']}: {e}")
            
            # Mark for removal from cache dictionary
            to_remove.append(training_id)
    
    # Remove expired entries from dictionary
    for training_id in to_remove:
        del cached_models[training_id]
        logger.info(f"Removed expired model from cache: {training_id}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5027))
    logger.info(f"Starting object detection service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False) 