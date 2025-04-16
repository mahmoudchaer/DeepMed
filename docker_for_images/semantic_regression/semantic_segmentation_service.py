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
import uuid
import psutil
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Track temp files for cleanup
temp_dirs = []

class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation task"""
    
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None, class_labels=None):
        """
        Initialize the dataset
        
        Args:
            image_dir (str): Directory containing the input images
            mask_dir (str): Directory containing the segmentation masks
            transform (callable, optional): Transform to apply to the input images
            mask_transform (callable, optional): Transform to apply to the masks
            class_labels (list, optional): List of class labels
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.class_labels = class_labels
        
        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                 if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Check if masks exist for all images
        valid_pairs = []
        self.mask_type = None  # Will be set when we analyze the first mask
        
        for img_file in self.image_files:
            # Check for corresponding mask with same name
            mask_file = img_file
            
            # Also check for .png version if image is jpg/jpeg
            if not os.path.exists(os.path.join(mask_dir, mask_file)) and mask_file.endswith(('.jpg', '.jpeg')):
                mask_file = os.path.splitext(img_file)[0] + '.png'
            
            if os.path.exists(os.path.join(mask_dir, mask_file)):
                valid_pairs.append((img_file, mask_file))
            
        self.valid_pairs = valid_pairs
        
        # Analyze a sample mask if available
        if len(valid_pairs) > 0:
            self.mask_type = self._analyze_mask_type(os.path.join(mask_dir, valid_pairs[0][1]))
            logger.info(f"Dataset initialized with {len(valid_pairs)} image-mask pairs. Detected mask type: {self.mask_type}")
        else:
            logger.warning("No valid image-mask pairs found")
    
    def _analyze_mask_type(self, mask_path):
        """
        Analyze a sample mask to determine its type
        
        Args:
            mask_path (str): Path to the mask file
            
        Returns:
            str: Type of mask ('grayscale', 'rgb', 'binary', or 'other')
        """
        try:
            with Image.open(mask_path) as mask:
                # Log the basic information
                logger.info(f"Sample mask: mode={mask.mode}, size={mask.size}")
                
                # Convert to numpy array for analysis
                mask_array = np.array(mask)
                
                # Log shape and unique values
                logger.info(f"Mask array shape: {mask_array.shape}")
                unique_values = np.unique(mask_array)
                logger.info(f"Mask unique values: {unique_values}")
                
                # Determine the mask type
                if mask.mode == 'L' or len(mask_array.shape) == 2:
                    # Grayscale mask
                    if len(unique_values) == 2 and ((0 in unique_values and 1 in unique_values) or
                                                   (0 in unique_values and 255 in unique_values)):
                        return 'binary'
                    return 'grayscale'
                elif mask.mode in ('RGB', 'RGBA') or len(mask_array.shape) == 3:
                    if mask_array.shape[2] == 3 or mask_array.shape[2] == 4:
                        # Check if it's actually a color-coded mask or just a binary mask saved as RGB
                        if len(unique_values) <= 2:
                            return 'binary'
                        return 'rgb'
                
                return 'other'
        except Exception as e:
            logger.error(f"Error analyzing mask type: {str(e)}")
            return 'unknown'

    def __len__(self):
        """Get the number of samples in the dataset"""
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, mask) where mask is the segmentation mask
        """
        try:
            # Get the file names
            img_file, mask_file = self.valid_pairs[idx]
            
            # Load the image
            img_path = os.path.join(self.image_dir, img_file)
            image = Image.open(img_path).convert('RGB')
            
            # Load the mask
            mask_path = os.path.join(self.mask_dir, mask_file)
            mask = Image.open(mask_path)
            
            # Process the mask based on its type
            if self.mask_type == 'rgb':
                # Convert RGB mask to grayscale class indices
                mask = mask.convert('L')
            elif self.mask_type == 'binary':
                # Ensure binary masks are properly handled
                mask_array = np.array(mask)
                
                # Check if values are 0/255
                if np.max(mask_array) > 1:
                    mask_array = (mask_array > 0).astype(np.uint8)
                
                mask = Image.fromarray(mask_array)
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                # Default processing if no mask_transform
                mask = torch.from_numpy(np.array(mask)).long()
            
            # Ensure mask has the right format
            if mask.dim() == 3:
                # If mask has 3 dimensions (C,H,W), convert to (H,W) by taking first channel
                mask = mask[0,:,:]
            
            # Sanity check on mask values and dimensions
            if torch.max(mask) > 100:  # If values are very high, likely 0-255 range
                logger.info(f"Converting mask with max value {torch.max(mask).item()} to binary")
                mask = (mask > 0).long()
            
            return image, mask
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            # Create an empty dummy sample
            empty_img = torch.zeros(3, 224, 224)
            empty_mask = torch.zeros(224, 224).long()
            return empty_img, empty_mask

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
    
    # Check if we have a valid dataset
    sample_batch = next(iter(dataloader))
    images, masks = sample_batch
    logger.info(f"Sample batch - Images shape: {images.shape}, Masks shape: {masks.shape}")
    logger.info(f"Mask unique values: {torch.unique(masks).tolist()}")
    
    if torch.max(masks) >= num_classes:
        logger.warning(f"Mask contains values ({torch.max(masks).item()}) >= num_classes ({num_classes})")
        logger.warning("This may cause errors during training. Adjusting num_classes.")
        
        # Adjust num_classes to match the maximum mask value + 1
        adjusted_num_classes = torch.max(masks).item() + 1
        logger.info(f"Adjusting num_classes from {num_classes} to {adjusted_num_classes}")
        num_classes = int(adjusted_num_classes)
        
        # Rebuild the model with the adjusted num_classes
        model = deeplabv3_resnet50(pretrained_backbone=True)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        running_loss = 0.0
        total_pixels = 0
        correct_pixels = 0
        intersection = torch.zeros(num_classes).to(device)
        union = torch.zeros(num_classes).to(device)
        
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Debug info about tensor shapes
            logger.debug(f"Batch - Images shape: {images.shape}, Masks shape: {masks.shape}")
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)['out']
            
            # Debug info
            logger.debug(f"Model output shape: {outputs.shape}")
            
            # Ensure masks are properly formatted for CrossEntropyLoss
            # CrossEntropyLoss expects targets with shape [batch, height, width] (no channels dimension)
            if masks.dim() == 4:
                masks = masks.squeeze(1)  # Remove channel dimension if present
            
            # Check for any NaN values in the masks
            if torch.isnan(masks).any():
                logger.warning("NaN values detected in masks!")
                masks = torch.nan_to_num(masks, nan=0)
            
            # Check for any values outside the valid range
            if torch.max(masks) >= num_classes:
                logger.warning(f"Mask contains values >= num_classes ({num_classes})")
                masks = torch.clamp(masks, 0, num_classes - 1)
            
            # Proceed with loss calculation
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
    
    return model, metrics_history, num_classes

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "semantic-segmentation-service"})

@app.post("/train")
def train_model():
    """Train a semantic segmentation model on the uploaded data"""
    try:
        # Get uploaded file
        uploaded_file = request.files.get('file')
        if not uploaded_file:
            return jsonify({'error': 'No file provided'}), 400
        
        # Define working directory
        work_dir = os.path.join(tempfile.gettempdir(), f'segmentation_training_{uuid.uuid4().hex}')
        os.makedirs(work_dir, exist_ok=True)
        logger.info(f"Working directory: {work_dir}")
        
        # Track memory usage
        def log_memory():
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            logger.info(f"Memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")

        # Set up directories
        dataset_dir = os.path.join(work_dir, 'dataset')
        output_dir = os.path.join(work_dir, 'output')
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save and extract zip file
        zip_path = os.path.join(work_dir, 'data.zip')
        uploaded_file.save(zip_path)
        logger.info(f"Saved uploaded file to {zip_path}")
        
        # Validate zip file before extraction
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Check if it's a valid zip file
                bad_file = zip_ref.testzip()
                if bad_file:
                    return jsonify({'error': f'Corrupt file in archive: {bad_file}'}), 400
                
                # Extract the archive
                zip_ref.extractall(dataset_dir)
        except zipfile.BadZipFile:
            return jsonify({'error': 'Invalid ZIP file'}), 400
        except Exception as e:
            return jsonify({'error': f'Error extracting ZIP file: {str(e)}'}), 500
        
        # Verify dataset structure
        images_dir = None
        masks_dir = None
        
        # Look for common dataset folder structures
        potential_structures = [
            # Direct "images" and "masks" folders
            {'images': 'images', 'masks': 'masks'},
            {'images': 'imgs', 'masks': 'masks'},
            {'images': 'images', 'masks': 'labels'},
            {'images': 'input', 'masks': 'output'},
            # Nested inside a dataset folder
            {'images': 'dataset/images', 'masks': 'dataset/masks'},
            {'images': 'dataset/imgs', 'masks': 'dataset/masks'},
            {'images': 'dataset/images', 'masks': 'dataset/labels'},
            # Train folder
            {'images': 'train/images', 'masks': 'train/masks'},
            {'images': 'train/imgs', 'masks': 'train/masks'},
            {'images': 'train/images', 'masks': 'train/labels'},
        ]
        
        # Try to find a valid dataset structure
        for structure in potential_structures:
            img_dir = os.path.join(dataset_dir, structure['images'])
            msk_dir = os.path.join(dataset_dir, structure['masks'])
            
            if os.path.exists(img_dir) and os.path.exists(msk_dir):
                images_dir = img_dir
                masks_dir = msk_dir
                break
        
        # If we couldn't find a standard structure, look for any folders containing images
        if images_dir is None:
            for root, dirs, files in os.walk(dataset_dir):
                img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(img_files) > 5:  # Arbitrary threshold to identify an image folder
                    potential_img_dir = root
                    # Look for a sibling directory that might contain masks
                    parent_dir = os.path.dirname(potential_img_dir)
                    sibling_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
                    
                    for sibling in sibling_dirs:
                        sibling_path = os.path.join(parent_dir, sibling)
                        if sibling_path != potential_img_dir:
                            # Check if this might be a masks directory
                            mask_files = [f for f in os.listdir(sibling_path) 
                                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                            if len(mask_files) > 5:
                                images_dir = potential_img_dir
                                masks_dir = sibling_path
                                break
                
                if images_dir:
                    break
        
        # If we still can't find the directories, return an error
        if not images_dir or not masks_dir:
            return jsonify({'error': 'Could not find valid images and masks directories in the uploaded data'}), 400
        
        logger.info(f"Found images directory: {images_dir}")
        logger.info(f"Found masks directory: {masks_dir}")
        
        # Check if images and masks directories contain files
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        mask_files = [f for f in os.listdir(masks_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(image_files) == 0:
            return jsonify({'error': f'No image files found in {images_dir}'}), 400
        
        if len(mask_files) == 0:
            return jsonify({'error': f'No mask files found in {masks_dir}'}), 400
        
        logger.info(f"Found {len(image_files)} image files and {len(mask_files)} mask files")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Set up image transformations
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        # Create dataset and dataloader
        try:
            dataset = SegmentationDataset(images_dir, masks_dir, transform=transform, mask_transform=mask_transform)
            
            # Determine number of classes from dataset
            num_classes = 1  # Default for binary segmentation
            
            # Check a sample mask to determine number of classes
            if len(dataset) > 0:
                _, sample_mask = dataset[0]
                unique_values = torch.unique(sample_mask)
                logger.info(f"Unique mask values: {unique_values}")
                
                # Count unique values excluding background (0)
                num_classes = len(unique_values)
                if 0 in unique_values:
                    num_classes -= 1
                
                # Ensure minimum 1 class (binary segmentation)
                num_classes = max(1, num_classes)
                
                logger.info(f"Detected {num_classes} class(es) for segmentation")
            
            # Create dataloader with reasonable batch size based on dataset size
            batch_size = min(8, max(1, len(dataset) // 10))  # Adjust batch size based on dataset size
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            
            logger.info(f"Created dataloader with batch size {batch_size}")
            log_memory()
            
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            return jsonify({'error': f'Failed to create dataset: {str(e)}'}), 500
        
        # Create and train the model
        try:
            # Check if we need a binary or multi-class model
            if num_classes == 1:
                # Binary segmentation (background vs foreground)
                model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
                logger.info("Created U-Net model for binary segmentation")
            else:
                # Multi-class segmentation
                model = UNet(n_channels=3, n_classes=num_classes + 1, bilinear=False).to(device)
                logger.info(f"Created U-Net model for {num_classes+1} classes")
            
            # Configure optimizer and loss function
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Choose loss function based on segmentation type
            if num_classes == 1:
                # Binary segmentation
                criterion = torch.nn.BCEWithLogitsLoss()
            else:
                # Multi-class segmentation
                criterion = torch.nn.CrossEntropyLoss()
            
            logger.info("Starting training...")
            
            # Training parameters
            num_epochs = min(50, max(10, 100 // len(dataloader)))  # Scale epochs based on dataset size
            logger.info(f"Training for {num_epochs} epochs")
            
            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            max_patience = 5
            early_stop = False
            
            for epoch in range(num_epochs):
                if early_stop:
                    break
                    
                model.train()
                epoch_loss = 0
                
                # Monitor and limit memory usage
                log_memory()
                
                # Check memory before training loop - prevent OOM
                process = psutil.Process(os.getpid())
                if process.memory_info().rss > 2 * 1024 * 1024 * 1024:  # 2GB limit
                    logger.warning("Memory usage too high, reducing batch size")
                    # Reduce batch size on the fly if memory usage is high
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5
                
                for batch_idx, (images, masks) in enumerate(dataloader):
                    try:
                        # Clear gradients
                        optimizer.zero_grad()
                        
                        # Move data to device
                        images = images.to(device)
                        masks = masks.to(device)
                        
                        # Forward pass
                        outputs = model(images)
                        
                        # Process outputs based on segmentation type
                        if num_classes == 1:
                            # Binary segmentation
                            masks = masks.float().unsqueeze(1)
                            loss = criterion(outputs, masks)
                        else:
                            # Multi-class segmentation
                            loss = criterion(outputs, masks)
                        
                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        
                        # Free memory
                        del images, masks, outputs, loss
                        torch.cuda.empty_cache()
                        
                        # Log progress
                        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
                            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                            log_memory()
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.error(f"Out of memory error: {str(e)}")
                            # Reduce batch size for next iteration
                            batch_size = max(1, batch_size // 2)
                            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                            logger.info(f"Reduced batch size to {batch_size}")
                            
                            # Clear cache and try to recover
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise
                
                # Average loss for the epoch
                avg_loss = epoch_loss / len(dataloader)
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
                
                # Check for early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    # Save the best model
                    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        logger.info(f"Early stopping after {epoch+1} epochs")
                        early_stop = True
            
            # Evaluate the model
            model.eval()
            
            # Functions for evaluation metrics
            def calculate_iou(pred, target, smooth=1e-6):
                # Convert predictions to binary masks
                if num_classes == 1:
                    pred = (pred > 0).float()
                    intersection = (pred * target).sum()
                    union = pred.sum() + target.sum() - intersection
                else:
                    pred = torch.argmax(pred, dim=1)
                    intersection = (pred == target).float().sum()
                    union = pred.numel()  # Total number of pixels
                
                iou = (intersection + smooth) / (union + smooth)
                return iou.item()
            
            def calculate_pixel_accuracy(pred, target):
                if num_classes == 1:
                    pred = (pred > 0).float()
                    correct = (pred == target).float().sum()
                else:
                    pred = torch.argmax(pred, dim=1)
                    correct = (pred == target).float().sum()
                
                total = target.numel()
                return (correct / total).item()
            
            # Evaluation loop
            val_iou = 0
            val_accuracy = 0
            test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            with torch.no_grad():
                for images, masks in test_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    
                    val_iou += calculate_iou(outputs, masks)
                    val_accuracy += calculate_pixel_accuracy(outputs, masks)
            
            val_iou /= len(test_loader)
            val_accuracy /= len(test_loader)
            
            logger.info(f"Validation IoU: {val_iou:.4f}")
            logger.info(f"Validation Pixel Accuracy: {val_accuracy:.4f}")
            
            # Save the final model
            model_path = os.path.join(output_dir, 'model.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Create a README file with usage instructions
            readme_path = os.path.join(output_dir, 'README.txt')
            with open(readme_path, 'w') as f:
                f.write("# Semantic Segmentation Model\n\n")
                f.write(f"Model type: U-Net\n")
                f.write(f"Number of classes: {num_classes}\n")
                f.write(f"Input size: 256x256\n\n")
                f.write("## Training metrics\n")
                f.write(f"IoU: {val_iou:.4f}\n")
                f.write(f"Pixel Accuracy: {val_accuracy:.4f}\n\n")
                f.write("## Usage\n")
                f.write("To use this model for inference:\n")
                f.write("```python\n")
                f.write("import torch\n")
                f.write("from unet_model import UNet\n\n")
                f.write(f"model = UNet(n_channels=3, n_classes={'1' if num_classes == 1 else str(num_classes+1)})\n")
                f.write("model.load_state_dict(torch.load('model.pth'))\n")
                f.write("model.eval()\n")
                f.write("```\n")
            
            # Create a requirements.txt file
            requirements_path = os.path.join(output_dir, 'requirements.txt')
            with open(requirements_path, 'w') as f:
                f.write("torch>=1.7.0\n")
                f.write("torchvision>=0.8.0\n")
                f.write("numpy>=1.19.0\n")
                f.write("Pillow>=7.0.0\n")
            
            # Save the model architecture
            model_code_path = os.path.join(output_dir, 'unet_model.py')
            with open(__file__, 'r') as source_file:
                source_code = source_file.read()
                
                # Extract UNet model class definition
                unet_class_match = re.search(r'class UNet\(.*?\):.*?(?=class|\Z)', source_code, re.DOTALL)
                double_conv_match = re.search(r'class DoubleConv\(.*?\):.*?(?=class)', source_code, re.DOTALL)
                down_match = re.search(r'class Down\(.*?\):.*?(?=class)', source_code, re.DOTALL)
                up_match = re.search(r'class Up\(.*?\):.*?(?=class)', source_code, re.DOTALL)
                outconv_match = re.search(r'class OutConv\(.*?\):.*?(?=class)', source_code, re.DOTALL)
                
                with open(model_code_path, 'w') as model_file:
                    model_file.write("import torch\n")
                    model_file.write("import torch.nn as nn\n")
                    model_file.write("import torch.nn.functional as F\n\n")
                    
                    if double_conv_match:
                        model_file.write(double_conv_match.group(0))
                    if down_match:
                        model_file.write(down_match.group(0))
                    if up_match:
                        model_file.write(up_match.group(0))
                    if outconv_match:
                        model_file.write(outconv_match.group(0))
                    if unet_class_match:
                        model_file.write(unet_class_match.group(0))
            
            # Create a ZIP file with the trained model and other files
            output_zip_path = os.path.join(work_dir, 'trained_model.zip')
            with zipfile.ZipFile(output_zip_path, 'w') as zip_ref:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zip_ref.write(file_path, os.path.relpath(file_path, output_dir))
            
            logger.info(f"Created output ZIP file: {output_zip_path}")
            
            # Return the trained model as a downloadable file
            return send_file(
                output_zip_path,
                as_attachment=True,
                download_name='trained_segmentation_model.zip',
                mimetype='application/zip'
            )
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Model training failed: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    
    finally:
        # Cleanup
        try:
            if 'work_dir' in locals() and os.path.exists(work_dir):
                logger.info(f"Cleaning up working directory: {work_dir}")
                shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {str(cleanup_error)}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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