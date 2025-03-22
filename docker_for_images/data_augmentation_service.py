import os
import io
import zipfile
import tempfile
import json
import random
import shutil
from pathlib import Path

from flask import Flask, request, jsonify, Response
import torch
from torchvision import transforms, datasets
from PIL import Image

app = Flask(__name__)

def augment_data(zip_file, augmentation_level=3):
    """
    Augment image data with various transformations based on the augmentation level
    
    Args:
        zip_file: The ZIP file containing image folders
        augmentation_level: Level of augmentation intensity (1-5)
        
    Returns:
        A BytesIO object containing the augmented data as a ZIP file and metrics
    """
    # Map augmentation level to transformation parameters
    augmentation_params = {
        1: {"num_augmentations": 1, "transformations": "basic"},      # Minimal augmentation
        2: {"num_augmentations": 2, "transformations": "basic"},      # Light augmentation
        3: {"num_augmentations": 3, "transformations": "standard"},   # Standard augmentation
        4: {"num_augmentations": 4, "transformations": "advanced"},   # Advanced augmentation
        5: {"num_augmentations": 5, "transformations": "advanced"}    # Extensive augmentation
    }
    
    # Get parameters based on augmentation level
    params = augmentation_params.get(int(augmentation_level), augmentation_params[3])
    num_augmentations = params["num_augmentations"]
    transformation_type = params["transformations"]
    
    # Set up augmentation transformations based on the type
    if transformation_type == "basic":
        # Basic transformations
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10)
        ]
    elif transformation_type == "standard":
        # Standard transformations
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ]
    else:  # advanced
        # Advanced transformations
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ]
    
    # Create temporary directories for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the uploaded ZIP file
        zip_path = os.path.join(tmpdir, "data.zip")
        zip_file.save(zip_path)
        
        # Extract the ZIP file
        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Create directory for augmented data
        augmented_dir = os.path.join(tmpdir, "augmented")
        os.makedirs(augmented_dir, exist_ok=True)
        
        # Find all class folders
        class_folders = [f for f in os.listdir(extract_dir) 
                        if os.path.isdir(os.path.join(extract_dir, f))]
        
        # Initialize counters for metrics
        total_original_images = 0
        total_augmented_images = 0
        classes_found = len(class_folders)
        augmentation_metrics = {}
        
        # Process each class folder
        for class_folder in class_folders:
            # Create corresponding folder in augmented directory
            augmented_class_dir = os.path.join(augmented_dir, class_folder)
            os.makedirs(augmented_class_dir, exist_ok=True)
            
            # Copy original images to the augmented folder
            src_dir = os.path.join(extract_dir, class_folder)
            image_files = [f for f in os.listdir(src_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            # Count original images
            class_original_count = len(image_files)
            total_original_images += class_original_count
            
            # Copy original images
            for img_file in image_files:
                src_path = os.path.join(src_dir, img_file)
                dst_path = os.path.join(augmented_class_dir, img_file)
                shutil.copy2(src_path, dst_path)
            
            # Generate augmented images
            class_augmented_count = 0
            for img_file in image_files:
                try:
                    # Load the image
                    img_path = os.path.join(src_dir, img_file)
                    img = Image.open(img_path).convert('RGB')
                    
                    # Get file base name and extension
                    base_name, ext = os.path.splitext(img_file)
                    
                    # Generate augmented versions
                    for aug_idx in range(num_augmentations):
                        # Create a composition of randomly selected transformations
                        # For variety, use a random subset of the transformations
                        num_transforms = random.randint(1, len(augmentation_transforms))
                        selected_transforms = random.sample(augmentation_transforms, num_transforms)
                        
                        # Apply the transforms
                        transform = transforms.Compose(selected_transforms)
                        augmented_img = transform(img)
                        
                        # Save the augmented image
                        aug_filename = f"{base_name}_aug{aug_idx+1}{ext}"
                        aug_path = os.path.join(augmented_class_dir, aug_filename)
                        augmented_img.save(aug_path)
                        
                        class_augmented_count += 1
                except Exception as e:
                    print(f"Error augmenting image {img_file}: {str(e)}")
                    continue
            
            # Update metrics
            total_augmented_images += class_augmented_count
            augmentation_metrics[class_folder] = {
                "original_images": class_original_count,
                "augmented_images": class_augmented_count,
                "total_images": class_original_count + class_augmented_count
            }
        
        # Create a ZIP file with the augmented data
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_out:
            for root, _, files in os.walk(augmented_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, augmented_dir)
                    zip_out.write(file_path, arcname)
        
        # Reset buffer position
        zip_buffer.seek(0)
        
        # Create metrics summary
        metrics = {
            "original_images": total_original_images,
            "augmented_images": total_augmented_images,
            "total_images": total_original_images + total_augmented_images,
            "classes": classes_found,
            "augmentation_level": augmentation_level,
            "class_metrics": augmentation_metrics
        }
        
        return zip_buffer, metrics

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "data-augmentation-service"})

@app.route('/augment', methods=['POST'])
def api_augment_data():
    """API endpoint for data augmentation"""
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    try:
        # Get augmentation level from the form
        augmentation_level = int(request.form.get('augmentationLevel', 3))
        
        # Validate augmentation level
        if augmentation_level < 1 or augmentation_level > 5:
            return jsonify({"error": "Augmentation level must be between 1 and 5"}), 400
        
        # Augment the data
        augmented_zip, metrics = augment_data(zip_file, augmentation_level=augmentation_level)
        
        # Create a response with the augmented data ZIP and metrics
        response = Response(augmented_zip.getvalue())
        response.headers["Content-Type"] = "application/zip"
        response.headers["Content-Disposition"] = "attachment; filename=augmented_data.zip"
        
        # Add metrics as a JSON string in a custom header
        response.headers["X-Augmentation-Metrics"] = json.dumps(metrics)
        
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5111))
    app.run(host='0.0.0.0', port=port, debug=False) 