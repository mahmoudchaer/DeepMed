import os
import io
import json
import tempfile
from flask import Flask, request, jsonify, send_file
from data_augmentation import ClassificationDatasetAugmentor
import traceback
import atexit
import time
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Track temp files for cleanup
temp_files = []

# Track processing status
processing_jobs = {}

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "augmentation-service"})

@app.route('/augment', methods=['POST'])
def augment_dataset():
    """
    Endpoint to augment a dataset
    
    Expects a multipart/form-data POST with:
    - zipFile: A zip file with folders organized by class
    - level: Augmentation level (1-5) which determines both strength and number of augmentations
    
    Returns a zip file of the augmented dataset
    """
    try:
        if 'zipFile' not in request.files:
            return jsonify({"error": "No ZIP file uploaded"}), 400
        
        # Get the zip file from the request
        zip_file = request.files['zipFile']
        
        # Get the level parameter with default
        level = int(request.form.get('level', 3))
        
        # Validate parameters
        if level < 1 or level > 5:
            return jsonify({"error": "Augmentation level must be between 1 and 5"}), 400
        
        # Log the start of processing
        file_size = 0
        zip_file.seek(0, os.SEEK_END)
        file_size = zip_file.tell()
        zip_file.seek(0)
        logger.info(f"Starting augmentation of file {zip_file.filename} ({file_size/1024/1024:.2f} MB)")
        
        # Save the uploaded zip file temporarily
        temp_input_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        zip_file.save(temp_input_zip.name)
        temp_input_zip.close()
        
        # Add to the list of files to clean up
        temp_files.append(temp_input_zip.name)
        
        # Create a temporary output zip file
        temp_output_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        temp_output_zip.close()
        
        # Add to the list of files to clean up
        temp_files.append(temp_output_zip.name)
        
        # Initialize the augmentor with more worker processes for large datasets
        num_workers = min(os.cpu_count(), 8)  # Use at most 8 workers
        augmentor = ClassificationDatasetAugmentor(
            image_size=(224, 224),
            num_workers=num_workers,
            batch_size=32,  # Larger batch size for efficiency
            use_gpu=False  # Set to True if GPU is available
        )
        
        start_time = time.time()
        
        # Process the dataset - number of augmentations is determined by the level
        augmentor.process_dataset(
            zip_path=temp_input_zip.name,
            output_zip=temp_output_zip.name,
            level=level
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Completed augmentation in {processing_time:.2f} seconds")
        
        # Read the file into memory
        with open(temp_output_zip.name, 'rb') as f:
            file_data = io.BytesIO(f.read())
        
        # Clean up the temporary files immediately
        cleanup_temp_files([temp_input_zip.name, temp_output_zip.name])
        
        # Return the file from memory
        return send_file(
            file_data,
            mimetype='application/zip',
            as_attachment=True,
            download_name='augmented_dataset.zip'
        )
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"Error: {error_msg}")
        return jsonify({"error": str(e), "details": error_msg}), 500

def cleanup_temp_files(file_list=None):
    """Clean up temporary files"""
    global temp_files
    
    if file_list is None:
        file_list = temp_files
    
    for file_path in file_list:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                if file_path in temp_files:
                    temp_files.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up {file_path}: {e}")

# Register cleanup function to run at exit
atexit.register(cleanup_temp_files)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5023))
    logger.info(f"Starting augmentation service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False) 