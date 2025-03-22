import os
import io
import zipfile
import tempfile
import json
import logging
import shutil
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from flask import Flask, request, jsonify, Response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

def augment_image(image, augmentation_level=3):
    """Apply different augmentation techniques based on augmentation level"""
    augmented_images = [image]  # Original image
    
    # Define augmentation techniques for each level
    if augmentation_level >= 1:
        # Level 1: Basic flips and rotations
        augmented_images.append(ImageOps.flip(image))
        augmented_images.append(ImageOps.mirror(image))
        augmented_images.append(image.rotate(90))
    
    if augmentation_level >= 2:
        # Level 2: Add color transformations
        brightness_enhancer = ImageEnhance.Brightness(image)
        augmented_images.append(brightness_enhancer.enhance(0.8))
        augmented_images.append(brightness_enhancer.enhance(1.2))
        
        contrast_enhancer = ImageEnhance.Contrast(image)
        augmented_images.append(contrast_enhancer.enhance(0.8))
        augmented_images.append(contrast_enhancer.enhance(1.2))
    
    if augmentation_level >= 3:
        # Level 3: Add more rotations and transforms
        augmented_images.append(image.rotate(45))
        augmented_images.append(image.rotate(135))
        augmented_images.append(image.rotate(225))
        augmented_images.append(image.rotate(315))
    
    if augmentation_level >= 4:
        # Level 4: Add sharpness and color balance
        sharpness_enhancer = ImageEnhance.Sharpness(image)
        augmented_images.append(sharpness_enhancer.enhance(0.5))
        augmented_images.append(sharpness_enhancer.enhance(1.5))
        
        color_enhancer = ImageEnhance.Color(image)
        augmented_images.append(color_enhancer.enhance(0.7))
        augmented_images.append(color_enhancer.enhance(1.3))
    
    if augmentation_level >= 5:
        # Level 5: More extreme transforms
        # Perspective transform
        width, height = image.size
        new_width, new_height = int(width * 0.8), int(height * 0.8)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        augmented_images.append(image.crop((left, top, right, bottom)).resize((width, height)))
        
        # More extreme color changes
        brightness_enhancer = ImageEnhance.Brightness(image)
        augmented_images.append(brightness_enhancer.enhance(0.6))
        augmented_images.append(brightness_enhancer.enhance(1.4))
        
        contrast_enhancer = ImageEnhance.Contrast(image)
        augmented_images.append(contrast_enhancer.enhance(0.6))
        augmented_images.append(contrast_enhancer.enhance(1.4))
    
    return augmented_images

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "data-augmentation-service"})

@app.route('/augment', methods=['POST'])
def augment_data():
    """Augment data by applying various transformations to images"""
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    
    # Get parameters from the form
    augmentation_level = int(request.form.get('augmentationLevel', 3))
    
    # Validate augmentation level
    if augmentation_level < 1 or augmentation_level > 5:
        return jsonify({"error": "Augmentation level must be between 1 and 5"}), 400
    
    try:
        # Create temporary directories
        with tempfile.TemporaryDirectory() as extract_dir, tempfile.TemporaryDirectory() as output_dir:
            # Save and extract the ZIP file
            zip_path = os.path.join(extract_dir, "data.zip")
            zip_file.save(zip_path)
            
            # Extract the contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
                
            # Get all class folders
            class_folders = []
            for item in os.listdir(extract_dir):
                item_path = os.path.join(extract_dir, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    class_folders.append(item)
            
            # Create output directories for each class
            for class_folder in class_folders:
                os.makedirs(os.path.join(output_dir, class_folder), exist_ok=True)
            
            # Process and augment each image
            metrics = {
                "classes": len(class_folders),
                "original_files": 0,
                "augmented_files": 0,
                "class_distribution": {}
            }
            
            for class_folder in class_folders:
                src_class_dir = os.path.join(extract_dir, class_folder)
                dst_class_dir = os.path.join(output_dir, class_folder)
                
                original_count = 0
                augmented_count = 0
                
                # Process all image files in this class
                for root, _, files in os.walk(src_class_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                            file_path = os.path.join(root, file)
                            file_name = os.path.basename(file_path)
                            file_name_without_ext, file_ext = os.path.splitext(file_name)
                            
                            try:
                                # Open the original image
                                with Image.open(file_path) as img:
                                    img = img.convert('RGB')  # Convert to RGB to ensure consistency
                                    
                                    # Copy original image
                                    original_file_path = os.path.join(dst_class_dir, file_name)
                                    img.save(original_file_path)
                                    original_count += 1
                                    
                                    # Generate augmented versions
                                    augmented_images = augment_image(img, augmentation_level)
                                    
                                    # Save augmented images (skip the first one as it's the original)
                                    for i, aug_img in enumerate(augmented_images[1:], 1):
                                        aug_file_name = f"{file_name_without_ext}_aug{i}{file_ext}"
                                        aug_file_path = os.path.join(dst_class_dir, aug_file_name)
                                        aug_img.save(aug_file_path)
                                        augmented_count += 1
                            except Exception as e:
                                logger.error(f"Error processing image {file_path}: {str(e)}")
                                # Skip this image but continue with others
                                continue
                
                # Update metrics
                metrics["original_files"] += original_count
                metrics["augmented_files"] += augmented_count
                metrics["class_distribution"][class_folder] = {
                    "original": original_count,
                    "augmented": augmented_count,
                    "total": original_count + augmented_count
                }
            
            # Create a ZIP file with the augmented dataset
            output_zip_path = os.path.join(output_dir, "augmented_data.zip")
            with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        if file != "augmented_data.zip":  # Skip the zip file itself
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, output_dir)
                            zipf.write(file_path, arcname)
            
            # Create a response with the augmented ZIP file
            with open(output_zip_path, "rb") as f:
                augmented_zip_bytes = io.BytesIO(f.read())
            
            response = Response(augmented_zip_bytes.getvalue())
            response.headers["Content-Type"] = "application/octet-stream"
            response.headers["Content-Disposition"] = "attachment; filename=augmented_data.zip"
            response.headers["X-Augmentation-Metrics"] = json.dumps(metrics)
            
            return response
            
    except Exception as e:
        logger.error(f"Error augmenting data: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5012))
    app.run(host='0.0.0.0', port=port, debug=False) 