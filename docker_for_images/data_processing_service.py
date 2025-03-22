import os
import io
import zipfile
import tempfile
import json
import logging
import shutil
import random

# Add compatibility fix for Werkzeug/Flask version mismatch
import werkzeug
if not hasattr(werkzeug.urls, 'url_quote'):
    werkzeug.urls.url_quote = werkzeug.urls.quote

from flask import Flask, request, jsonify, Response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "data-processing-service"})

@app.route('/process', methods=['POST'])
def process_data():
    """Process data by splitting it into training, validation, and test sets"""
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    
    # Get parameters from the form
    test_size = float(request.form.get('testSize', 0.2))
    val_size = float(request.form.get('valSize', 0.1))
    
    # Validate sizes
    if test_size <= 0 or test_size >= 1:
        return jsonify({"error": "Test size must be between 0 and 1"}), 400
    
    if val_size <= 0 or val_size >= 1:
        return jsonify({"error": "Validation size must be between 0 and 1"}), 400
    
    if test_size + val_size >= 1:
        return jsonify({"error": "Sum of test and validation sizes must be less than 1"}), 400
    
    try:
        # Create temporary directories
        with tempfile.TemporaryDirectory() as extract_dir, tempfile.TemporaryDirectory() as output_dir:
            # Save and extract the ZIP file
            zip_path = os.path.join(extract_dir, "data.zip")
            zip_file.save(zip_path)
            
            # Extract the contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
                
            # Get class folders (assuming each folder is a class)
            class_folders = []
            for item in os.listdir(extract_dir):
                item_path = os.path.join(extract_dir, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    class_folders.append(item)
            
            # Create output directories
            train_dir = os.path.join(output_dir, "train")
            val_dir = os.path.join(output_dir, "val")
            test_dir = os.path.join(output_dir, "test")
            
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            
            # Create class subdirectories in each split
            for class_folder in class_folders:
                os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
                os.makedirs(os.path.join(val_dir, class_folder), exist_ok=True)
                os.makedirs(os.path.join(test_dir, class_folder), exist_ok=True)
            
            # Process each class
            metrics = {
                "classes": len(class_folders),
                "class_distribution": {},
                "total_files": 0,
                "train_files": 0,
                "val_files": 0,
                "test_files": 0
            }
            
            for class_folder in class_folders:
                src_class_dir = os.path.join(extract_dir, class_folder)
                
                # Get all image files
                image_files = []
                for root, _, files in os.walk(src_class_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                            image_files.append(os.path.join(root, file))
                
                # Shuffle the files for random splitting
                random.shuffle(image_files)
                
                # Calculate split sizes
                total_files = len(image_files)
                test_count = int(total_files * test_size)
                val_count = int(total_files * val_size)
                train_count = total_files - test_count - val_count
                
                # Split indices
                train_end = train_count
                val_end = train_count + val_count
                
                # Copy files to respective directories
                for i, src_file in enumerate(image_files):
                    file_name = os.path.basename(src_file)
                    
                    if i < train_end:
                        # Train split
                        dst_file = os.path.join(train_dir, class_folder, file_name)
                        shutil.copy2(src_file, dst_file)
                        metrics["train_files"] += 1
                    elif i < val_end:
                        # Validation split
                        dst_file = os.path.join(val_dir, class_folder, file_name)
                        shutil.copy2(src_file, dst_file)
                        metrics["val_files"] += 1
                    else:
                        # Test split
                        dst_file = os.path.join(test_dir, class_folder, file_name)
                        shutil.copy2(src_file, dst_file)
                        metrics["test_files"] += 1
                
                # Update metrics
                metrics["class_distribution"][class_folder] = {
                    "total": total_files,
                    "train": train_count,
                    "val": val_count,
                    "test": test_count
                }
                metrics["total_files"] += total_files
            
            # Create a new ZIP file containing the processed data
            output_zip_path = os.path.join(output_dir, "processed_data.zip")
            with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add each folder's files to the ZIP
                for folder in ["train", "val", "test"]:
                    for root, _, files in os.walk(os.path.join(output_dir, folder)):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, output_dir)
                            zipf.write(file_path, arcname)
            
            # Create a response with the processed ZIP file
            with open(output_zip_path, "rb") as f:
                processed_zip_bytes = io.BytesIO(f.read())
            
            response = Response(processed_zip_bytes.getvalue())
            response.headers["Content-Type"] = "application/octet-stream"
            response.headers["Content-Disposition"] = "attachment; filename=processed_data.zip"
            response.headers["X-Processing-Metrics"] = json.dumps(metrics)
            
            return response
            
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5011))
    app.run(host='0.0.0.0', port=port, debug=False) 