import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import zipfile
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
import torch
from typing import Tuple, Optional, Union, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
from functools import partial


class DataAugmentor:
    def __init__(self, image_size: Tuple[int, int] = (224, 224), use_gpu: bool = False):
        self.image_size = image_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if self.use_gpu:
            print(f"Using GPU acceleration: {torch.cuda.get_device_name(0)}")
        else:
            print("Running on CPU")
            
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        self.transforms = {
            1: self._get_level1_transforms(),
            2: self._get_level2_transforms(),
            3: self._get_level3_transforms(),
            4: self._get_level4_transforms(),
            5: self._get_level5_transforms()
        }
    
    def _get_level1_transforms(self) -> A.Compose:
        """Light augmentation - minimal changes"""
        return A.Compose([
            A.Resize(*self.image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2, brightness_limit=0.1, contrast_limit=0.1),
            A.Normalize(),
            ToTensorV2(),
        ])
    
    def _get_level2_transforms(self) -> A.Compose:
        """Moderate augmentation - slight geometric and color changes"""
        return A.Compose([
            A.Resize(*self.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
            A.Rotate(limit=15, p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ])
    
    def _get_level3_transforms(self) -> A.Compose:
        """Medium augmentation - more geometric and color changes"""
        return A.Compose([
            A.Resize(*self.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.4, brightness_limit=0.3, contrast_limit=0.3),
            A.Rotate(limit=30, p=0.4),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ])
    
    def _get_level4_transforms(self) -> A.Compose:
        """Strong augmentation - significant changes"""
        return A.Compose([
            A.Resize(*self.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.4),
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.4, contrast_limit=0.4),
            A.Rotate(limit=45, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=0.4),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])
    
    def _get_level5_transforms(self) -> A.Compose:
        """Very strong augmentation - extreme changes"""
        return A.Compose([
            A.Resize(*self.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.6, brightness_limit=0.5, contrast_limit=0.5),
            A.Rotate(limit=60, p=0.6),
            A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=60, p=0.5),
            A.ElasticTransform(alpha=180, sigma=180 * 0.05, alpha_affine=180 * 0.03, p=0.4),
            A.GaussNoise(p=0.3),
            A.RandomGamma(p=0.3),
            A.RandomFog(p=0.2),
            A.RandomShadow(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])
    
    def _batch_process(self, images: Union[np.ndarray, torch.Tensor], level: int) -> torch.Tensor:
        """Process a batch of images with GPU acceleration"""
        if isinstance(images, np.ndarray):
            # Convert to torch tensor if it's a numpy array
            images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
            images_tensor = images_tensor.to(self.device)
        else:
            images_tensor = images.to(self.device)
            
        # Apply transforms (note: this is simplified, actual implementation would need more work)
        # For a complete implementation, you would need GPU-optimized transforms 
        # This is a placeholder for the concept
        return images_tensor
    
    def augment(self, image: Union[np.ndarray, torch.Tensor], level: int, mask: Optional[np.ndarray] = None) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[np.ndarray]]:
        """
        Apply augmentation to the image with specified level
        
        Args:
            image: Input image as numpy array or torch tensor
            level: Augmentation level (1-5)
            mask: Optional mask for segmentation tasks
            
        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
        if level not in self.transforms:
            raise ValueError(f"Level must be between 1 and 5, got {level}")
        
        # Check if it's a batch
        is_batch = len(image.shape) == 4
        
        # For batch processing with GPU
        if is_batch and self.use_gpu:
            return self._batch_process(image, level), mask
        
        # Standard processing for single image
        transform = self.transforms[level]
        
        # Apply the transform
        if mask is not None:
            transformed = transform(image=image, mask=mask)
            
            # Move to GPU if needed
            if self.use_gpu and isinstance(transformed['image'], torch.Tensor):
                transformed['image'] = transformed['image'].to(self.device)
                
            return transformed['image'], transformed['mask']
        else:
            transformed = transform(image=image)
            
            # Move to GPU if needed
            if self.use_gpu and isinstance(transformed['image'], torch.Tensor):
                transformed['image'] = transformed['image'].to(self.device)
                
            return transformed['image'], None
    
    def to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor back to CPU if it was on GPU"""
        if self.use_gpu and tensor is not None and isinstance(tensor, torch.Tensor):
            return tensor.cpu()
        return tensor


class ClassificationDatasetAugmentor:
    def __init__(self, image_size: tuple = (224, 224), num_workers: int = None, batch_size: int = 16, use_gpu: bool = False):
        """
        Initialize the dataset augmentor
        
        Args:
            image_size: Target size for augmented images
            num_workers: Number of workers for parallel processing (default: CPU count)
            batch_size: Number of images to process in memory at once
            use_gpu: Whether to use GPU acceleration for augmentations
        """
        self.augmentor = DataAugmentor(image_size=image_size, use_gpu=use_gpu)
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()
        self.batch_size = batch_size
        
    def extract_zip(self, zip_path: str) -> Path:
        """
        Extract the zip file to a temporary directory
        
        Args:
            zip_path: Path to the zip file
            
        Returns:
            Path to the extracted directory
        """
        # Create a temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        
        print(f"Extracting dataset from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total size for progress reporting
            total_size = sum(file_info.file_size for file_info in zip_ref.infolist())
            extracted_size = 0
            
            # Create progress bar
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting")
            
            for file_info in zip_ref.infolist():
                zip_ref.extract(file_info, temp_dir)
                extracted_size += file_info.file_size
                pbar.update(file_info.file_size)
            
            pbar.close()
            
        return temp_dir
    
    def create_zip(self, source_dir: Path, output_zip: str):
        """
        Create a zip file from a directory
        
        Args:
            source_dir: Directory to zip
            output_zip: Path to the output zip file
        """
        print(f"\nCreating zip file: {output_zip}")
        
        # Get total files and size for progress reporting
        total_files = 0
        total_size = 0
        for root, _, files in os.walk(source_dir):
            total_files += len(files)
            total_size += sum(os.path.getsize(os.path.join(root, file)) for file in files)
        
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            processed_size = 0
            processed_files = 0
            
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Creating zip")
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    arcname = os.path.relpath(file_path, source_dir)
                    zip_ref.write(file_path, arcname)
                    
                    processed_size += file_size
                    processed_files += 1
                    
                    # Update progress bar
                    pbar.update(file_size)
                    pbar.set_postfix({"files": f"{processed_files}/{total_files}"})
            
            pbar.close()

    def _process_image_batch(self, batch, output_class_folder, level, num_augmentations):
        """
        Process a batch of images
        
        Args:
            batch: List of image paths
            output_class_folder: Path to save augmented images
            level: Augmentation level (1-5)
            num_augmentations: Number of augmented versions to create for each image
            
        Returns:
            Number of successfully processed images
        """
        processed_count = 0
        
        for img_path in batch:
            try:
                # Read image in a memory-efficient way
                with open(img_path, 'rb') as f:
                    buffer = np.frombuffer(f.read(), dtype=np.uint8)
                    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                
                if image is None:
                    continue
                    
                # Convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Copy original image to output folder
                shutil.copy2(img_path, output_class_folder / img_path.name)
                
                # Create augmented versions
                for i in range(num_augmentations):
                    # Apply augmentation
                    augmented_image, _ = self.augmentor.augment(image_rgb, level)
                    
                    # Convert back to numpy array and BGR if it's a tensor
                    if isinstance(augmented_image, torch.Tensor):
                        augmented_image = self.augmentor.to_cpu(augmented_image)
                        augmented_image = augmented_image.permute(1, 2, 0).numpy()
                        augmented_image = (augmented_image * 255).astype(np.uint8)
                    
                    # Save augmented image
                    output_name = f"{img_path.stem}_aug{i+1}{img_path.suffix}"
                    output_path = output_class_folder / output_name
                    
                    # Save using memory-efficient encoding
                    _, buffer = cv2.imencode(img_path.suffix, augmented_image)
                    with open(output_path, 'wb') as f:
                        f.write(buffer)
                
                processed_count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
                
        return processed_count

    def _process_class_worker(self, args):
        """Worker function for parallel processing"""
        class_folder, output_class_folder, level, num_augmentations, batch_size = args
        
        # Create output class folder
        output_class_folder.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in class_folder.glob("*") 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        # Process in batches
        total_processed = 0
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i+batch_size]
            total_processed += self._process_image_batch(batch, output_class_folder, level, num_augmentations)
            
        return class_folder.name, total_processed, len(image_files)
    
    def _find_class_directories(self, base_dir: Path) -> List[Path]:
        """
        Recursively find all directories containing image files
        
        Args:
            base_dir: Base directory to search in
            
        Returns:
            List of directories containing images
        """
        class_dirs = []
        
        # First check if the base_dir itself contains images
        image_files = [f for f in base_dir.glob("*") 
                      if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        if image_files:
            return [base_dir]
        
        # Check immediate subdirectories
        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                # Check if this directory contains images
                image_files = [f for f in subdir.glob("*") 
                              if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                if image_files:
                    class_dirs.append(subdir)
                else:
                    # If not, recursively check its subdirectories
                    class_dirs.extend(self._find_class_directories(subdir))
        
        return class_dirs
    
    def process_dataset(self, zip_path: str, output_zip: str, level: int, num_augmentations: int = 2):
        """
        Process a zipped classification dataset and create a zip file of the augmented dataset
        
        Args:
            zip_path: Path to the zip file containing the dataset
            output_zip: Path to the output zip file
            level: Augmentation level (1-5)
            num_augmentations: Number of augmented versions to create for each image
        """
        # Create a temporary directory for processing
        temp_dir = Path(tempfile.mkdtemp())
        output_dir = temp_dir / "augmented_dataset"
        
        # Extract the input zip file
        input_dir = self.extract_zip(zip_path)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all directories containing images (these are our class folders)
        class_folders = self._find_class_directories(input_dir)
        
        if not class_folders:
            raise ValueError("No class folders with images found. Please ensure the ZIP contains folders with images.")
        
        print(f"Found {len(class_folders)} classes: {[f.relative_to(input_dir) for f in class_folders]}")
        print(f"Starting augmentation with level {level} using {self.num_workers} workers")
        print(f"Number of augmentations per image: {num_augmentations}")
        
        start_time = time.time()
        
        # Prepare arguments for parallel processing
        args_list = []
        for class_folder in class_folders:
            # Maintain the same directory structure in the output
            relative_path = class_folder.relative_to(input_dir)
            output_class_folder = output_dir / relative_path
            
            args_list.append((
                class_folder,
                output_class_folder,
                level,
                num_augmentations,
                self.batch_size
            ))
        
        # Process classes in parallel
        total_images = 0
        processed_images = 0
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            class_results = list(tqdm(
                executor.map(self._process_class_worker, args_list),
                total=len(args_list),
                desc="Processing classes",
                unit="class"
            ))
        
        # Collect and display results
        for class_name, processed, total in class_results:
            print(f"Class {class_name}: Processed {processed}/{total} images")
            total_images += total
            processed_images += processed
            
        elapsed_time = time.time() - start_time
        images_per_second = processed_images / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        print(f"Average processing speed: {images_per_second:.2f} images/second")
        print(f"Total processed: {processed_images}/{total_images} images")
        
        # Create zip file of the augmented dataset
        self.create_zip(output_dir, output_zip)
        
        # Clean up temporary directories
        shutil.rmtree(temp_dir)
        shutil.rmtree(input_dir)
        
        print("\nAugmentation completed!")
        print(f"Augmented dataset saved to: {output_zip}")
        print(f"Original dataset size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")
        print(f"Augmented dataset size: {os.path.getsize(output_zip) / (1024*1024):.2f} MB")


def main():
    # Example usage
    zip_path = "dataset.zip"  # Path to your zipped dataset
    output_zip = "augmented_dataset.zip"  # Path to the output zip file
    
    # Initialize augmentor with 4 workers and GPU if available
    augmentor = ClassificationDatasetAugmentor(
        image_size=(224, 224),
        num_workers=4, 
        batch_size=16,
        use_gpu=torch.cuda.is_available()
    )
    
    # Process the dataset
    augmentor.process_dataset(
        zip_path=zip_path,
        output_zip=output_zip,
        level=2,  # Augmentation level (1-5)
        num_augmentations=2  # Number of augmented versions per image
    )

if __name__ == "__main__":
    main() 