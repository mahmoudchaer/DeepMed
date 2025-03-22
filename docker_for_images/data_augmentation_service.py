import os
import io
import zipfile
import tempfile
import json
import logging
import random
import shutil
from pathlib import Path

from flask import Flask, request, jsonify, Response
import torch
from torchvision import transforms, datasets
from PIL import Image
from flask_cors import CORS  # Import CORS for cross-origin support

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants for augmentation
AUGMENTATION_LEVELS = {
    1: 1,  # Level 1: Generate 1 variation per image
    2: 2,  # Level 2: Generate 2 variations per image
    3: 3,  # Level 3: Generate 3 variations per image
    4: 5,  # Level 4: Generate 5 variations per image
    5: 8   # Level 5: Generate 8 variations per image
}

def augment_data(zip_file, augmentation_level=3):
    """
    Augment image data by applying transformations to increase dataset size
    """
    logger.info(f"Starting data augmentation with level {augmentation_level}")
    
    # Validate augmentation level
    if augmentation_level not in AUGMENTATION_LEVELS:
        augmentation_level = 3  # Default to level 3 if invalid
    
    # Number of variations to generate per image
    num_variations = AUGMENTATION_LEVELS[augmentation_level]
    logger.info(f"Will generate {num_variations} variations per image")
    
    # Define augmentation transformations based on level
    # More aggressive transformations for higher levels
    transform_options = [
        # Basic transformations (all levels)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        
        # Medium transformations (level 2+)
        transforms.ColorJitter(brightness=0.2, contrast=0.2) if augmentation_level >= 2 else None,
        
        # More aggressive transformations (level 3+)
        transforms.RandomVerticalFlip(p=0.2) if augmentation_level >= 3 else None,
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)) if augmentation_level >= 3 else None,
        
        # Strong transformations (level 4+)
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3) if augmentation_level >= 4 else None,
        transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.8, 1.2)) if augmentation_level >= 4 else None,
        
        # Extreme transformations (level 5 only)
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5) if augmentation_level >= 5 else None,
        transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.7, 1.3)) if augmentation_level >= 5 else None
    ]
    
    # Filter out None transformations
    transform_options = [t for t in transform_options if t is not None]
    logger.info(f"Using {len(transform_options)} different transformations")
    
    # Create temporary directories for processing
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        # Extract the ZIP file
        zip_path = os.path.join(input_dir, "data.zip")
        zip_file.save(zip_path)
        
        logger.info(f"Saved uploaded zip to {zip_path}")
        
        extract_dir = os.path.join(input_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        
        logger.info(f"Extracted zip to {extract_dir}")
        
        # Check the class directory structure
        class_dirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
        if not class_dirs:
            # If no class directories, assume all images are in the root
            class_dirs = [""]
        
        logger.info(f"Found {len(class_dirs)} class directories")
        
        # Create corresponding output directories
        for class_dir in class_dirs:
            class_path = os.path.join(output_dir, class_dir)
            os.makedirs(class_path, exist_ok=True)
        
        # Track metrics
        original_count = 0
        augmented_count = 0
        
        # Process each class directory
        for class_dir in class_dirs:
            input_class_dir = os.path.join(extract_dir, class_dir)
            output_class_dir = os.path.join(output_dir, class_dir)
            
            logger.info(f"Processing class: {class_dir}")
            
            # List all image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend([f for f in os.listdir(input_class_dir) if f.lower().endswith(ext)])
            
            logger.info(f"Found {len(image_files)} images in class {class_dir}")
            original_count += len(image_files)
            
            # Process each image
            for img_file in image_files:
                img_path = os.path.join(input_class_dir, img_file)
                try:
                    # Open the original image
                    img = Image.open(img_path)
                    
                    # Copy the original image to output
                    img.save(os.path.join(output_class_dir, img_file))
                    
                    # Generate variations
                    for i in range(num_variations):
                        # Create a random combination of transformations for this variation
                        # Choose between 1 and 3 transformations randomly
                        num_transforms = random.randint(1, min(3, len(transform_options)))
                        selected_transforms = random.sample(transform_options, num_transforms)
                        
                        # Apply each transformation
                        augmented_img = img.copy()
                        for transform in selected_transforms:
                            augmented_img = transform(augmented_img)
                        
                        # Save the augmented image with a unique name
                        base_name, ext = os.path.splitext(img_file)
                        aug_filename = f"{base_name}_aug_{i+1}{ext}"
                        augmented_img.save(os.path.join(output_class_dir, aug_filename))
                        augmented_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {str(e)}")
                    continue
        
        logger.info(f"Augmentation complete. Original: {original_count}, Augmented: {augmented_count}, Total: {original_count + augmented_count}")
        
        # Create a ZIP file of the augmented dataset
        output_zip_path = os.path.join(output_dir, "augmented_data.zip")
        with zipfile.ZipFile(output_zip_path, "w") as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    # Skip the output ZIP itself
                    if file == "augmented_data.zip":
                        continue
                    
                    file_path = os.path.join(root, file)
                    # Create archive path relative to output_dir
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
        
        logger.info(f"Created output zip at {output_zip_path}")
        
        # Read the ZIP file into memory for return
        with open(output_zip_path, "rb") as f:
            zip_bytes = io.BytesIO(f.read())
        
        # Return the augmented dataset as a ZIP file
        metrics = {
            "original_images": original_count,
            "augmented_images": augmented_count,
            "total_images": original_count + augmented_count,
            "classes": len(class_dirs),
            "augmentation_level": augmentation_level,
            "variations_per_image": num_variations
        }
        
        return zip_bytes, metrics

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    logger.info("Health check request received")
    return jsonify({"status": "healthy", "service": "data-augmentation-service"})

@app.route('/augment', methods=['POST'])
def augment():
    """
    Process a ZIP file containing image folders and return augmented images
    """
    logger.info("Received /augment request")
    
    if 'zipFile' not in request.files:
        logger.error("No zipFile in request")
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    try:
        zip_file = request.files['zipFile']
        augmentation_level = int(request.form.get('augmentationLevel', 3))
        
        logger.info(f"Processing augmentation request with level {augmentation_level}")
        
        # Augment the data
        augmented_zip, metrics = augment_data(zip_file, augmentation_level=augmentation_level)
        
        # Create response with the augmented data
        response = Response(augmented_zip.getvalue())
        response.headers["Content-Type"] = "application/zip"
        response.headers["Content-Disposition"] = "attachment; filename=augmented_data.zip"
        
        # Add metrics as a custom header
        response.headers["X-Augmentation-Metrics"] = json.dumps(metrics)
        
        logger.info(f"Augmentation complete: {metrics}")
        return response
    
    except Exception as e:
        logger.error(f"Error in augmentation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5111))
    logger.info(f"Starting data augmentation service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 