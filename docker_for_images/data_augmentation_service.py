import os
import io
import zipfile
import tempfile
import json
from PIL import Image
import random
from flask import Flask, request, jsonify, Response
import torchvision.transforms as transforms

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "data-augmentation-service"})

def apply_augmentation(image_path, output_dir, base_filename, augmentation_level=3):
    """
    Apply augmentation to an image and save multiple augmented versions
    
    Parameters:
    - image_path: Path to the original image
    - output_dir: Directory to save augmented images
    - base_filename: Base name for the augmented files
    - augmentation_level: Level of augmentation (1-5)
    
    Returns:
    - List of paths to the augmented images
    """
    # Define number of augmentations based on level
    num_augmentations = {
        1: 1,  # Minimal augmentation
        2: 2,  # Light augmentation
        3: 3,  # Standard augmentation
        4: 5,  # Extended augmentation
        5: 8   # Heavy augmentation
    }.get(augmentation_level, 3)
    
    # Load the image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return []
    
    # Define various augmentation transforms
    augmentations = [
        # 1. Horizontal Flip
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
        ]),
        
        # 2. Slight Rotation
        transforms.Compose([
            transforms.RandomRotation(15),
        ]),
        
        # 3. Color Jitter
        transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]),
        
        # 4. Crop and Resize
        transforms.Compose([
            transforms.RandomResizedCrop(size=(img.height, img.width), scale=(0.8, 1.0)),
        ]),
        
        # 5. Gaussian Blur
        transforms.Compose([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]),
        
        # 6. Perspective Transform
        transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
        ]),
        
        # 7. Combination: Flip + Rotation
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(10),
        ]),
        
        # 8. Combination: Color Jitter + Crop
        transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(size=(img.height, img.width), scale=(0.85, 1.0)),
        ])
    ]
    
    # Randomly select augmentations based on the desired number
    selected_augmentations = random.sample(augmentations, min(num_augmentations, len(augmentations)))
    
    augmented_images = []
    for i, augmentation in enumerate(selected_augmentations):
        augmented_img = augmentation(img)
        augmented_filename = f"{os.path.splitext(base_filename)[0]}_aug{i+1}{os.path.splitext(base_filename)[1]}"
        augmented_path = os.path.join(output_dir, augmented_filename)
        augmented_img.save(augmented_path)
        augmented_images.append(augmented_path)
    
    return augmented_images

@app.route('/augment', methods=['POST'])
def augment_data():
    """
    Augment images in the training set by applying various transformations.
    Input: ZIP file with train, val, and test folders (from data processing)
    Output: ZIP file with augmented training images
    """
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    
    # Get augmentation level (default is 3)
    augmentation_level = int(request.form.get('augmentationLevel', 3))
    
    try:
        # Create temporary directories for extraction and processing
        with tempfile.TemporaryDirectory() as extract_dir, tempfile.TemporaryDirectory() as augment_dir:
            # Extract the ZIP file to the extraction directory
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Copy the directory structure to the augmentation directory
            for root, dirs, files in os.walk(extract_dir):
                for directory in dirs:
                    dir_path = os.path.join(root, directory)
                    rel_path = os.path.relpath(dir_path, extract_dir)
                    os.makedirs(os.path.join(augment_dir, rel_path), exist_ok=True)
            
            # First, copy all original files to maintain the structure
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, extract_dir)
                    output_path = os.path.join(augment_dir, rel_path)
                    
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Copy the file
                    with open(file_path, 'rb') as src, open(output_path, 'wb') as dst:
                        dst.write(src.read())
            
            # Count files before augmentation
            original_count = sum(1 for _ in os.walk(augment_dir) for _ in _[2])
            
            # Only augment images in the training set
            train_dir = os.path.join(extract_dir, 'train')
            if os.path.exists(train_dir):
                # Process each class in the training set
                for class_name in os.listdir(train_dir):
                    class_dir = os.path.join(train_dir, class_name)
                    if os.path.isdir(class_dir):
                        # Create corresponding directory in augmentation directory
                        augment_class_dir = os.path.join(augment_dir, 'train', class_name)
                        os.makedirs(augment_class_dir, exist_ok=True)
                        
                        # Process each image file
                        for file_name in os.listdir(class_dir):
                            file_path = os.path.join(class_dir, file_name)
                            if os.path.isfile(file_path):
                                # Apply augmentation
                                apply_augmentation(
                                    file_path, 
                                    augment_class_dir, 
                                    file_name, 
                                    augmentation_level
                                )
            
            # Count files after augmentation
            augmented_count = sum(1 for _ in os.walk(augment_dir) for _ in _[2])
            
            # Create a new ZIP file with the augmented data
            augmented_zip_buffer = io.BytesIO()
            with zipfile.ZipFile(augmented_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as augmented_zip:
                # Add all files from the augmentation directory to the ZIP
                for root, _, files in os.walk(augment_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Get the relative path with respect to the augmentation directory
                        rel_path = os.path.relpath(file_path, augment_dir)
                        augmented_zip.write(file_path, rel_path)
            
            # Prepare the response
            augmented_zip_buffer.seek(0)
            response = Response(augmented_zip_buffer.getvalue())
            response.headers["Content-Type"] = "application/zip"
            response.headers["Content-Disposition"] = "attachment; filename=augmented_data.zip"
            
            # Add metrics as custom header
            metrics = {
                "original_files": original_count,
                "augmented_files": augmented_count,
                "new_files_created": augmented_count - original_count,
                "augmentation_level": augmentation_level
            }
            response.headers["X-Augmentation-Metrics"] = json.dumps(metrics)
            
            return response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5012))
    app.run(host='0.0.0.0', port=port, debug=False) 