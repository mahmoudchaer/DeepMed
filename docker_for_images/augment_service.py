import os
import io
import json
import tempfile
from flask import Flask, request, jsonify, send_file
from data_augmentation import ClassificationDatasetAugmentor
import traceback

app = Flask(__name__)

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
    - level: Augmentation level (1-5)
    - numAugmentations: Number of augmentations per image
    
    Returns a zip file of the augmented dataset
    """
    try:
        if 'zipFile' not in request.files:
            return jsonify({"error": "No ZIP file uploaded"}), 400
        
        # Get the zip file from the request
        zip_file = request.files['zipFile']
        
        # Get parameters with defaults
        level = int(request.form.get('level', 3))
        num_augmentations = int(request.form.get('numAugmentations', 2))
        
        # Validate parameters
        if level < 1 or level > 5:
            return jsonify({"error": "Augmentation level must be between 1 and 5"}), 400
        
        if num_augmentations < 1 or num_augmentations > 10:
            return jsonify({"error": "Number of augmentations must be between 1 and 10"}), 400
        
        # Save the uploaded zip file temporarily
        temp_input_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        zip_file.save(temp_input_zip.name)
        temp_input_zip.close()
        
        # Create a temporary output zip file
        temp_output_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        temp_output_zip.close()
        
        # Initialize the augmentor
        augmentor = ClassificationDatasetAugmentor(
            image_size=(224, 224),
            num_workers=4,
            batch_size=16,
            use_gpu=False  # Set to True if GPU is available
        )
        
        # Process the dataset
        augmentor.process_dataset(
            zip_path=temp_input_zip.name,
            output_zip=temp_output_zip.name,
            level=level,
            num_augmentations=num_augmentations
        )
        
        # Return the augmented dataset
        return send_file(
            temp_output_zip.name,
            mimetype='application/zip',
            as_attachment=True,
            download_name='augmented_dataset.zip',
            # Clean up temporary files after sending
            after_request=lambda: cleanup_temp_files([temp_input_zip.name, temp_output_zip.name])
        )
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Error: {error_msg}")
        return jsonify({"error": str(e), "details": error_msg}), 500

def cleanup_temp_files(file_list):
    """Clean up temporary files"""
    for file_path in file_list:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up {file_path}: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5023))
    app.run(host="0.0.0.0", port=port, debug=False) 