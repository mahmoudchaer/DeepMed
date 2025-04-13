import os
import io
import json
import zipfile
import tempfile
import logging
import traceback
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
from flask import Flask, request, jsonify, Response, send_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Track temp files for cleanup
temp_dirs = []

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None, class_labels=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.class_labels = class_labels
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Verify matching masks exist
        self.valid_images = []
        for img_name in self.images:
            mask_name = img_name  # Assuming mask filenames match image filenames
            if os.path.exists(os.path.join(mask_dir, mask_name)):
                self.valid_images.append(img_name)
            else:
                logger.warning(f"Mask not found for image {img_name}")
        
        logger.info(f"Found {len(self.valid_images)} valid image-mask pairs")
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        img_name = self.valid_images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask = Image.open(mask_path)
        
        # Convert grayscale mask to single channel
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Default mask transformation: convert to tensor
            mask = transforms.ToTensor()(mask)
            # Ensure mask has integer labels
            mask = mask.long().squeeze(0)
        
        return image, mask

def train_deeplab(dataset, num_classes, epochs, batch_size, device='cpu'):
    """Train a DeepLabV3 model with ResNet-50 backbone"""
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Initialize model
    model = deeplabv3_resnet50(pretrained_backbone=True)
    
    # Modify classifier for our num_classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    metrics_history = {'epoch': [], 'loss': [], 'pixel_acc': [], 'miou': []}
    
    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        running_loss = 0.0
        total_pixels = 0
        correct_pixels = 0
        intersection = torch.zeros(num_classes).to(device)
        union = torch.zeros(num_classes).to(device)
        
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running statistics
            running_loss += loss.item() * images.size(0)
            
            # Calculate metrics
            _, preds = torch.max(outputs, 1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()
            
            # Calculate IoU
            for cls in range(num_classes):
                pred_mask = (preds == cls)
                target_mask = (masks == cls)
                intersection[cls] += (pred_mask & target_mask).sum().item()
                union[cls] += (pred_mask | target_mask).sum().item()
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(dataloader.dataset)
        pixel_accuracy = 100.0 * correct_pixels / total_pixels
        
        # Calculate mean IoU, avoiding division by zero
        class_iou = torch.zeros(num_classes).to(device)
        for cls in range(num_classes):
            if union[cls] > 0:
                class_iou[cls] = intersection[cls] / union[cls]
        miou = 100.0 * torch.mean(class_iou).item()
        
        # Store metrics
        metrics_history['epoch'].append(epoch + 1)
        metrics_history['loss'].append(epoch_loss)
        metrics_history['pixel_acc'].append(pixel_accuracy)
        metrics_history['miou'].append(miou)
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Pixel Acc: {pixel_accuracy:.2f}%, mIoU: {miou:.2f}%")
    
    return model, metrics_history

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "semantic-segmentation-service"})

@app.route('/train', methods=['POST'])
def train_model():
    """
    Endpoint to train a semantic segmentation model using DeepLabV3 with ResNet50 backbone
    
    Expects a multipart/form-data POST with:
    - zipFile: A zip file containing:
      - /images: folder with input images (JPEG/PNG)
      - /masks: folder with segmentation masks (same filenames as images)
      - Optional: labels.txt for class names
    - level: Level of training (1-5) which determines the complexity and training time
    - image_size: (Optional) Size to resize images to (default: 256)
    
    Returns a zip file with the trained model and instructions
    """
    try:
        if 'zipFile' not in request.files:
            return jsonify({"error": "No ZIP file uploaded"}), 400
        
        # Get the zip file from the request
        zip_file = request.files['zipFile']
        
        # Get level parameter with default
        level = int(request.form.get('level', 3))
        
        # Get image size with default
        image_size = int(request.form.get('image_size', 256))
        
        # Define preset configurations for each level
        level_configs = {
            1: {
                'epochs': 2,
                'batch_size': 8,
                'learning_rate': 1e-3
            },
            2: {
                'epochs': 5,
                'batch_size': 8,
                'learning_rate': 1e-3
            },
            3: {
                'epochs': 10,
                'batch_size': 8,
                'learning_rate': 1e-4
            },
            4: {
                'epochs': 20,
                'batch_size': 4,
                'learning_rate': 5e-5
            },
            5: {
                'epochs': 30,
                'batch_size': 4,
                'learning_rate': 1e-5
            }
        }
        
        # Validate level
        if level < 1 or level > 5:
            return jsonify({"error": "Level must be between 1 and 5"}), 400
        
        # Get configuration for the selected level
        config = level_configs[level]
        epochs = config['epochs']
        batch_size = config['batch_size']
        learning_rate = config['learning_rate']
        
        # Log the start of processing
        logger.info(f"Starting Semantic Segmentation training with level {level} (epochs: {epochs})")
        
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
        
        # Find images and masks directories
        images_dir = os.path.join(dataset_dir, "images")
        masks_dir = os.path.join(dataset_dir, "masks")
        
        # Check if the expected directories exist
        if not os.path.isdir(images_dir):
            return jsonify({"error": "Images directory not found in the ZIP file. Please ensure your ZIP contains an 'images' folder."}), 400
        
        if not os.path.isdir(masks_dir):
            return jsonify({"error": "Masks directory not found in the ZIP file. Please ensure your ZIP contains a 'masks' folder."}), 400
        
        # Count the number of valid image files
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            return jsonify({"error": "No image files found in the images directory"}), 400
        
        logger.info(f"Found {len(image_files)} images for training")
        
        # Look for labels.txt file
        labels_file = os.path.join(dataset_dir, "labels.txt")
        num_classes = 2  # Default: binary segmentation (background + 1 class)
        class_labels = ["background", "foreground"]
        
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                class_labels = ["background"] + [line.strip() for line in f.readlines() if line.strip()]
                num_classes = len(class_labels)
                logger.info(f"Found labels.txt with {num_classes-1} classes (total {num_classes} with background)")
        
        # Create results directory
        results_dir = os.path.join(temp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create transformations
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        # Create dataset
        dataset = SegmentationDataset(
            image_dir=images_dir,
            mask_dir=masks_dir,
            transform=transform,
            mask_transform=mask_transform,
            class_labels=class_labels
        )
        
        if len(dataset) == 0:
            return jsonify({"error": "No valid image-mask pairs found for training"}), 400
        
        # Train the model
        device = torch.device('cpu')
        model, metrics = train_deeplab(
            dataset=dataset,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            device=device
        )
        
        # Save the model
        model_path = os.path.join(results_dir, "model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_classes': num_classes,
            'class_labels': class_labels,
            'image_size': image_size,
            'metrics': metrics,
            'backbone': 'resnet50',
            'architecture': 'deeplabv3'
        }, model_path)
        
        # Create README.txt with usage instructions
        readme_path = os.path.join(results_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write("DEEPLABV3 SEMANTIC SEGMENTATION MODEL\n")
            f.write("=====================================\n\n")
            f.write(f"This model was trained for semantic segmentation using DeepLabV3 with a ResNet-50 backbone.\n\n")
            f.write(f"Classes ({num_classes}):\n")
            for i, label in enumerate(class_labels):
                f.write(f"  {i}: {label}\n")
            f.write("\n")
            f.write("Performance Metrics:\n")
            f.write(f"  Mean IoU: {metrics['miou'][-1]:.2f}%\n")
            f.write(f"  Pixel Accuracy: {metrics['pixel_acc'][-1]:.2f}%\n\n")
            f.write("How to load this model in PyTorch:\n")
            f.write("```python\n")
            f.write("import torch\n")
            f.write("from torchvision.models.segmentation import deeplabv3_resnet50\n\n")
            f.write("# Load checkpoint\n")
            f.write("checkpoint = torch.load('model.pth', map_location=torch.device('cpu'))\n\n")
            f.write("# Initialize model\n")
            f.write("num_classes = checkpoint['num_classes']\n")
            f.write("model = deeplabv3_resnet50(pretrained_backbone=False)\n")
            f.write("model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))\n\n")
            f.write("# Load trained weights\n")
            f.write("model.load_state_dict(checkpoint['model_state_dict'])\n")
            f.write("model.eval()  # Set to evaluation mode\n")
            f.write("```\n\n")
            f.write("Example for prediction:\n")
            f.write("```python\n")
            f.write("from PIL import Image\n")
            f.write("import torchvision.transforms as transforms\n")
            f.write("import torch.nn.functional as F\n\n")
            f.write("# Load and preprocess image\n")
            f.write(f"image_size = {image_size}\n")
            f.write("transform = transforms.Compose([\n")
            f.write("    transforms.Resize((image_size, image_size)),\n")
            f.write("    transforms.ToTensor(),\n")
            f.write("    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n")
            f.write("])\n\n")
            f.write("image = Image.open('your_image.jpg').convert('RGB')\n")
            f.write("input_tensor = transform(image).unsqueeze(0)  # Add batch dimension\n\n")
            f.write("# Make prediction\n")
            f.write("with torch.no_grad():\n")
            f.write("    output = model(input_tensor)['out']\n")
            f.write("    prediction = output.argmax(1).squeeze(0).cpu().numpy()\n")
            f.write("```\n")
        
        # Create requirements.txt
        requirements_path = os.path.join(results_dir, "requirements.txt")
        with open(requirements_path, 'w') as f:
            f.write("torch>=1.9.0\n")
            f.write("torchvision>=0.10.0\n")
            f.write("pillow>=8.0.0\n")
            f.write("numpy>=1.19.0\n")
        
        # Create a zip file of the model, README, and requirements
        zip_output_path = os.path.join(temp_dir, "segmentation_model.zip")
        with zipfile.ZipFile(zip_output_path, 'w') as zipf:
            zipf.write(model_path, arcname="model.pth")
            zipf.write(readme_path, arcname="README.txt")
            zipf.write(requirements_path, arcname="requirements.txt")
        
        # Return the zip file
        return send_file(
            zip_output_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name='segmentation_model.zip'
        )
        
    except Exception as e:
        logger.error(f"Error in semantic segmentation training: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def cleanup_temp_dirs():
    """Clean up temporary directories"""
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory {temp_dir}: {str(e)}")

# Clean up temp directories when the app exits
import atexit
atexit.register(cleanup_temp_dirs)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5031))
    logger.info(f"Starting semantic segmentation service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False) 