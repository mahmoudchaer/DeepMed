import os
import io
import zipfile
import tempfile
import json
import random
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "data-processing-service"})

@app.route('/process', methods=['POST'])
def process_data():
    """
    Process the data by organizing and splitting it into training, validation, and test sets.
    Input: ZIP file containing folders (each folder is a class)
    Output: Processed ZIP file with train, val, and test folders
    """
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    
    # Get validation and test split ratios from form (default 0.2 and 0.1)
    val_split = float(request.form.get('validationSplit', 0.2))
    test_split = float(request.form.get('testSplit', 0.1))
    
    try:
        # Create temporary directories for extraction and processing
        with tempfile.TemporaryDirectory() as extract_dir, tempfile.TemporaryDirectory() as process_dir:
            # Extract the ZIP file to the extraction directory
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Create train, validation, and test directories
            train_dir = os.path.join(process_dir, 'train')
            val_dir = os.path.join(process_dir, 'val')
            test_dir = os.path.join(process_dir, 'test')
            
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            
            # Get all class directories (first level subdirectories in the extract directory)
            class_dirs = [d for d in os.listdir(extract_dir) 
                        if os.path.isdir(os.path.join(extract_dir, d))]
            
            # Process each class
            for class_name in class_dirs:
                # Create corresponding class directories in train, val, test
                os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
                os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
                os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
                
                # Get all image files in the class directory
                class_path = os.path.join(extract_dir, class_name)
                image_files = [f for f in os.listdir(class_path) 
                            if os.path.isfile(os.path.join(class_path, f))]
                
                # Shuffle the files for random split
                random.shuffle(image_files)
                
                # Calculate split indices
                total_files = len(image_files)
                test_idx = int(total_files * test_split)
                val_idx = int(total_files * val_split) + test_idx
                
                # Split the files and copy to respective directories
                test_files = image_files[:test_idx]
                val_files = image_files[test_idx:val_idx]
                train_files = image_files[val_idx:]
                
                # Copy files to their respective directories
                for file_list, target_dir in [
                    (train_files, os.path.join(train_dir, class_name)),
                    (val_files, os.path.join(val_dir, class_name)),
                    (test_files, os.path.join(test_dir, class_name))
                ]:
                    for file_name in file_list:
                        src_path = os.path.join(class_path, file_name)
                        dst_path = os.path.join(target_dir, file_name)
                        shutil.copy2(src_path, dst_path)
            
            # Create a new ZIP file with the processed data
            processed_zip_buffer = io.BytesIO()
            with zipfile.ZipFile(processed_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as processed_zip:
                # Add all files from the process directory to the ZIP
                for root, _, files in os.walk(process_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Get the relative path with respect to the process directory
                        rel_path = os.path.relpath(file_path, process_dir)
                        processed_zip.write(file_path, rel_path)
            
            # Prepare the response
            processed_zip_buffer.seek(0)
            response = Response(processed_zip_buffer.getvalue())
            response.headers["Content-Type"] = "application/zip"
            response.headers["Content-Disposition"] = "attachment; filename=processed_data.zip"
            
            # Add metrics as custom header
            metrics = {
                "total_files": total_files,
                "train_files": len(train_files),
                "val_files": len(val_files),
                "test_files": len(test_files),
                "num_classes": len(class_dirs)
            }
            response.headers["X-Processing-Metrics"] = json.dumps(metrics)
            
            return response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5011))
    app.run(host='0.0.0.0', port=port, debug=False) 