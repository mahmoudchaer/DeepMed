"""
EfficientNet-B0 Model for Image Classification

This module provides functionality to train an EfficientNet-B0 model
on medical images organized in folders by category.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
import time
from datetime import datetime
import json
import h5py
import psutil
import gc

# Configure TensorFlow to use memory growth to avoid allocating all GPU memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")

# Set up logging
logger = logging.getLogger(__name__)

class EfficientNetB0Classifier:
    """
    Class for training and using an EfficientNet-B0 model for image classification.
    """
    
    def __init__(self, dataset_path, output_dir=None, img_size=224, batch_size=32, epochs=20, 
                 learning_rate=0.001, augmentation_intensity='medium'):
        """
        Initialize the EfficientNet-B0 classifier.
        
        Args:
            dataset_path (str): Path to the directory containing category folders with images
            output_dir (str, optional): Directory to save models and logs
            img_size (int): Image size for model input (square)
            batch_size (int): Batch size for training
            epochs (int): Maximum number of epochs for training
            learning_rate (float): Learning rate for the optimizer
            augmentation_intensity (str): Intensity of data augmentation ('low', 'medium', 'high')
        """
        self.dataset_path = dataset_path
        
        # Create output directory if not provided
        if output_dir is None:
            self.output_dir = os.path.join('static', 'trained_models', 
                                         f'efficientnet_b0_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Model hyperparameters
        self.img_size = img_size
        self.batch_size = self._adjust_batch_size(batch_size)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.augmentation_intensity = augmentation_intensity
        self.model = None
        self.history = None
        self.categories = None
        self.num_classes = 0
        
        # Set model paths
        self.checkpoint_path = os.path.join(self.output_dir, 'best_model.h5')
        self.final_model_path = os.path.join(self.output_dir, 'final_model.h5')
        self.model_info_path = os.path.join(self.output_dir, 'model_info.json')
        
        # Create logs directory
        self.logs_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Progress tracking
        self.progress = {
            'status': 'initialized',
            'current_epoch': 0,
            'total_epochs': epochs,
            'accuracy': 0,
            'loss': 0,
            'val_accuracy': 0,
            'val_loss': 0,
            'training_time': 0,
            'message': 'Model initialized'
        }
        
        # Save initial progress
        self._save_progress()
    
    def _adjust_batch_size(self, requested_batch_size):
        """
        Dynamically adjusts batch size based on available memory to prevent OOM errors
        
        Args:
            requested_batch_size (int): The batch size requested by the user
            
        Returns:
            int: Adjusted batch size that should work with available resources
        """
        # Get available system memory in GB
        available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        
        # Check for GPU memory if available
        gpu_memory_gb = 0
        if gpus:
            try:
                # This is a simplification, actual implementation would use tf.config.experimental.get_memory_info
                # or nvidia-smi command to get available GPU memory
                gpu_memory_gb = 4  # Assume 4GB as default if we can't determine
            except:
                pass
        
        # Use the higher of available system or GPU memory
        available_memory_gb = max(available_memory_gb, gpu_memory_gb)
        
        # Adjust batch size based on available memory (simplified heuristic)
        # Assume each image in batch takes about ~15MB of memory during training (including gradients, etc.)
        # This is a rough estimate for 224x224 images with EfficientNet-B0
        memory_per_image_mb = 15
        safe_batch_size = int((available_memory_gb * 1024 * 0.7) / memory_per_image_mb)
        
        # Cap at requested batch size
        adjusted_batch_size = min(safe_batch_size, requested_batch_size)
        
        # Ensure batch size is at least 4 for effective training
        adjusted_batch_size = max(adjusted_batch_size, 4)
        
        if adjusted_batch_size < requested_batch_size:
            logger.warning(f"Batch size adjusted from {requested_batch_size} to {adjusted_batch_size} due to memory constraints")
        
        return adjusted_batch_size
    
    def _save_progress(self):
        """Save the current progress to a JSON file."""
        progress_path = os.path.join(self.output_dir, 'progress.json')
        with open(progress_path, 'w') as f:
            json.dump(self.progress, f)
    
    def _update_progress(self, status, message, **kwargs):
        """Update the progress dictionary and save it."""
        self.progress['status'] = status
        self.progress['message'] = message
        
        # Update any additional fields
        for key, value in kwargs.items():
            if key in self.progress:
                self.progress[key] = value
        
        self._save_progress()
        logger.info(f"Progress: {status} - {message}")
    
    def prepare_data(self):
        """
        Prepare the dataset for training using data augmentation.
        
        Returns:
            tuple: (train_generator, validation_generator, test_generator)
        """
        self._update_progress('preparing', 'Preparing dataset')
        
        # Get the list of categories (subdirectories)
        self.categories = [d for d in os.listdir(self.dataset_path) 
                         if os.path.isdir(os.path.join(self.dataset_path, d))]
        self.num_classes = len(self.categories)
        
        if self.num_classes < 2:
            raise ValueError(f"At least 2 categories needed for classification, found {self.num_classes}")
        
        # Create training, validation, and test dirs
        train_dir = os.path.join(self.output_dir, 'train')
        val_dir = os.path.join(self.output_dir, 'val')
        test_dir = os.path.join(self.output_dir, 'test')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Split data: 70% train, 15% validation, 15% test
        import shutil
        import random
        
        for category in self.categories:
            # Create category directories
            os.makedirs(os.path.join(train_dir, category), exist_ok=True)
            os.makedirs(os.path.join(val_dir, category), exist_ok=True)
            os.makedirs(os.path.join(test_dir, category), exist_ok=True)
            
            # Get image files
            category_path = os.path.join(self.dataset_path, category)
            image_files = [f for f in os.listdir(category_path) 
                          if os.path.isfile(os.path.join(category_path, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff'))]
            
            # Shuffle the list
            random.shuffle(image_files)
            
            # Calculate split points
            train_count = int(0.7 * len(image_files))
            val_count = int(0.15 * len(image_files))
            
            # Split into train, val, test
            train_files = image_files[:train_count]
            val_files = image_files[train_count:train_count + val_count]
            test_files = image_files[train_count + val_count:]
            
            # Copy files to their respective directories
            for file in train_files:
                src = os.path.join(category_path, file)
                dst = os.path.join(train_dir, category, file)
                shutil.copy2(src, dst)
            
            for file in val_files:
                src = os.path.join(category_path, file)
                dst = os.path.join(val_dir, category, file)
                shutil.copy2(src, dst)
                
            for file in test_files:
                src = os.path.join(category_path, file)
                dst = os.path.join(test_dir, category, file)
                shutil.copy2(src, dst)
        
        # Get data augmentation parameters based on intensity
        augmentation_params = self._get_augmentation_params()
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            **augmentation_params
        )
        
        # Only rescaling for validation and testing
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        validation_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Save class indices
        class_indices = train_generator.class_indices
        with open(os.path.join(self.output_dir, 'class_indices.json'), 'w') as f:
            json.dump(class_indices, f)
        
        self._update_progress('data_ready', f'Dataset prepared with {self.num_classes} categories', 
                             num_train=len(train_generator.filenames),
                             num_val=len(validation_generator.filenames),
                             num_test=len(test_generator.filenames))
        
        return train_generator, validation_generator, test_generator
    
    def _get_augmentation_params(self):
        """
        Get data augmentation parameters based on intensity setting.
        
        Returns:
            dict: Augmentation parameters for ImageDataGenerator
        """
        # Low augmentation (minimal transformations)
        if self.augmentation_intensity == 'low':
            return {
                'rotation_range': 10,
                'width_shift_range': 0.1,
                'height_shift_range': 0.1,
                'shear_range': 0.1,
                'zoom_range': 0.1,
                'horizontal_flip': True,
                'fill_mode': 'nearest'
            }
        # High augmentation (aggressive transformations)
        elif self.augmentation_intensity == 'high':
            return {
                'rotation_range': 30,
                'width_shift_range': 0.3,
                'height_shift_range': 0.3,
                'shear_range': 0.3,
                'zoom_range': 0.3,
                'horizontal_flip': True,
                'vertical_flip': True,  # More aggressive
                'fill_mode': 'nearest',
                'brightness_range': [0.7, 1.3]  # Brightness variation
            }
        # Medium augmentation (default)
        else:
            return {
                'rotation_range': 20,
                'width_shift_range': 0.2,
                'height_shift_range': 0.2,
                'shear_range': 0.2,
                'zoom_range': 0.2,
                'horizontal_flip': True,
                'fill_mode': 'nearest'
            }
    
    def build_model(self):
        """
        Build the EfficientNet-B0 model architecture with transfer learning.
        """
        self._update_progress('building', 'Building EfficientNet-B0 model')
        
        # Create the base model from EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze the base model layers for transfer learning
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add a classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save model summary to file
        with open(os.path.join(self.output_dir, 'model_summary.txt'), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        self._update_progress('model_built', 'Model built successfully')
        
        return self.model
    
    def build_feature_model(self, feature_dim):
        """
        Build a model that takes pre-extracted features as input.
        
        Args:
            feature_dim (int): Dimension of input features
            
        Returns:
            Model: Keras model for training on features
        """
        self._update_progress('building', 'Building feature-based model')
        
        # Define input shape based on feature dimension
        inputs = Input(shape=(feature_dim,))
        
        # Add classification layers on top of the features
        x = Dense(512, activation='relu')(inputs)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        self.model = Model(inputs=inputs, outputs=predictions)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save model summary to file
        with open(os.path.join(self.output_dir, 'model_summary.txt'), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        self._update_progress('model_built', 'Feature-based model built successfully')
        
        return self.model
    
    def train(self, train_generator, validation_generator):
        """
        Train the EfficientNet-B0 model.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            
        Returns:
            dict: Training history
        """
        self._update_progress('training', 'Starting model training')
        
        # Define callbacks
        callbacks = [
            # Save model checkpoints
            ModelCheckpoint(
                filepath=self.checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # Early stopping if model stops improving
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when plateau is reached
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            # TensorBoard logs
            TensorBoard(log_dir=self.logs_dir)
        ]
        
        # Custom callback to track progress
        class TrainingProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, classifier):
                self.classifier = classifier
                self.start_time = None
                
            def on_train_begin(self, logs=None):
                self.start_time = time.time()
                
            def on_epoch_begin(self, epoch, logs=None):
                self.classifier._update_progress(
                    'training',
                    f'Training epoch {epoch+1}/{self.classifier.epochs}',
                    current_epoch=epoch+1
                )
                
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    self.classifier._update_progress(
                        'training',
                        f'Completed epoch {epoch+1}/{self.classifier.epochs}',
                        current_epoch=epoch+1,
                        accuracy=float(logs.get('accuracy', 0)),
                        loss=float(logs.get('loss', 0)),
                        val_accuracy=float(logs.get('val_accuracy', 0)),
                        val_loss=float(logs.get('val_loss', 0)),
                        training_time=time.time() - self.start_time
                    )
                    
            def on_train_end(self, logs=None):
                total_time = time.time() - self.start_time
                self.classifier._update_progress(
                    'trained',
                    f'Model training completed in {total_time:.1f} seconds',
                    training_time=total_time
                )
        
        callbacks.append(TrainingProgressCallback(self))
        
        # Calculate steps per epoch based on data size
        steps_per_epoch = len(train_generator)
        validation_steps = len(validation_generator)
        
        # Train the model
        start_time = time.time()
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save(self.final_model_path)
        
        # Save training history
        history_dict = {
            'accuracy': [float(val) for val in self.history.history['accuracy']],
            'loss': [float(val) for val in self.history.history['loss']],
            'val_accuracy': [float(val) for val in self.history.history['val_accuracy']],
            'val_loss': [float(val) for val in self.history.history['val_loss']]
        }
        
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history_dict, f)
        
        return self.history
    
    def train_with_features(self, features_path):
        """
        Train a model using pre-extracted features instead of raw images.
        
        Args:
            features_path (str): Path to the JSON file with features
            
        Returns:
            dict: Training history
        """
        self._update_progress('preparing', 'Loading pre-extracted features')
        
        # Load the features
        with open(features_path, 'r') as f:
            feature_data = json.load(f)
        
        features = feature_data.get('features', {})
        self.categories = feature_data.get('categories', [])
        self.num_classes = len(self.categories)
        
        if self.num_classes < 2:
            raise ValueError(f"At least 2 categories needed for classification, found {self.num_classes}")
        
        # Prepare data arrays
        all_features = []
        all_labels = []
        
        for i, category in enumerate(self.categories):
            category_features = features.get(category, [])
            for item in category_features:
                if 'features' in item:
                    all_features.append(item['features'])
                    # One-hot encode labels
                    label = [0] * self.num_classes
                    label[i] = 1
                    all_labels.append(label)
        
        # Convert to NumPy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Determine feature dimension from the first feature
        if len(X) > 0:
            feature_dim = X.shape[1]
        else:
            raise ValueError("No valid features found in the input data")
        
        # Split data: 70% train, 15% validation, 15% test
        from sklearn.model_selection import train_test_split
        
        # First split off the test set (15% of data)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        # Then split the remaining data into train and validation
        # 15% of total data = 15/(100-15) = 17.65% of the remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
        )
        
        # Save class indices
        class_indices = {cat: i for i, cat in enumerate(self.categories)}
        with open(os.path.join(self.output_dir, 'class_indices.json'), 'w') as f:
            json.dump(class_indices, f)
        
        # Build model for feature data
        self.build_feature_model(feature_dim)
        
        # Define callbacks
        callbacks = [
            # Save model checkpoints
            ModelCheckpoint(
                filepath=self.checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # Early stopping if model stops improving
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when plateau is reached
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            # TensorBoard logs
            TensorBoard(log_dir=self.logs_dir)
        ]
        
        # Custom callback to track progress
        class TrainingProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, classifier):
                self.classifier = classifier
                self.start_time = None
                
            def on_train_begin(self, logs=None):
                self.start_time = time.time()
                
            def on_epoch_begin(self, epoch, logs=None):
                self.classifier._update_progress(
                    'training',
                    f'Training epoch {epoch+1}/{self.classifier.epochs}',
                    current_epoch=epoch+1
                )
                
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    self.classifier._update_progress(
                        'training',
                        f'Completed epoch {epoch+1}/{self.classifier.epochs}',
                        current_epoch=epoch+1,
                        accuracy=float(logs.get('accuracy', 0)),
                        loss=float(logs.get('loss', 0)),
                        val_accuracy=float(logs.get('val_accuracy', 0)),
                        val_loss=float(logs.get('val_loss', 0)),
                        training_time=time.time() - self.start_time
                    )
                    
            def on_train_end(self, logs=None):
                total_time = time.time() - self.start_time
                self.classifier._update_progress(
                    'trained',
                    f'Model training completed in {total_time:.1f} seconds',
                    training_time=total_time
                )
        
        callbacks.append(TrainingProgressCallback(self))
        
        # Train the model
        start_time = time.time()
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save(self.final_model_path)
        
        # Save training history
        history_dict = {
            'accuracy': [float(val) for val in self.history.history['accuracy']],
            'loss': [float(val) for val in self.history.history['loss']],
            'val_accuracy': [float(val) for val in self.history.history['val_accuracy']],
            'val_loss': [float(val) for val in self.history.history['val_loss']]
        }
        
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history_dict, f)
        
        # Evaluate on test set
        self._update_progress('evaluating', 'Evaluating model on test data')
        test_results = self.model.evaluate(X_test, y_test, verbose=1)
        
        # Save evaluation results
        eval_results = {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'num_test_samples': len(X_test)
        }
        
        with open(os.path.join(self.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(eval_results, f)
        
        self._update_progress(
            'evaluated',
            f'Model evaluation completed with accuracy: {eval_results["test_accuracy"]:.4f}',
            test_accuracy=eval_results['test_accuracy'],
            test_loss=eval_results['test_loss']
        )
        
        return self.history

    def evaluate(self, test_generator):
        """
        Evaluate the model on the test dataset.
        
        Args:
            test_generator: Test data generator
            
        Returns:
            dict: Evaluation results
        """
        self._update_progress('evaluating', 'Evaluating model on test data')
        
        # Ensure we're using the best model from training
        if os.path.exists(self.checkpoint_path):
            try:
                self.model.load_weights(self.checkpoint_path)
                print("Loaded best model weights from checkpoint")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        
        # Evaluate the model
        test_results = self.model.evaluate(
            test_generator,
            steps=len(test_generator),
            verbose=1
        )
        
        # Get prediction probabilities
        predictions = self.model.predict(
            test_generator,
            steps=len(test_generator),
            verbose=1
        )
        
        # Get true labels
        y_true = test_generator.classes
        
        # Get predicted labels
        y_pred = np.argmax(predictions, axis=1)
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate classification report
        target_names = list(test_generator.class_indices.keys())
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        
        # Save evaluation results
        eval_results = {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'num_test_samples': len(test_generator.filenames)
        }
        
        with open(os.path.join(self.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(eval_results, f)
        
        self._update_progress(
            'evaluated',
            f'Model evaluation completed with accuracy: {eval_results["test_accuracy"]:.4f}',
            test_accuracy=eval_results['test_accuracy'],
            test_loss=eval_results['test_loss']
        )
        
        return eval_results
    
    def export_model(self, format='keras'):
        """
        Export the model in different formats.
        
        Args:
            format (str): Export format (keras, tflite, or saved_model)
            
        Returns:
            str: Path to the exported model
        """
        self._update_progress('exporting', f'Exporting model in {format} format')
        
        if format == 'keras':
            # Already saved during training
            export_path = self.final_model_path
            
        elif format == 'tflite':
            # Convert to TFLite
            export_path = os.path.join(self.output_dir, 'model.tflite')
            
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            # Save the model
            with open(export_path, 'wb') as f:
                f.write(tflite_model)
        
        elif format == 'saved_model':
            # Export as SavedModel
            export_path = os.path.join(self.output_dir, 'saved_model')
            tf.saved_model.save(self.model, export_path)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        # Update model info
        with open(self.model_info_path, 'w') as f:
            model_info = {
                'model_type': 'EfficientNetB0',
                'num_classes': self.num_classes,
                'img_size': self.img_size,
                'categories': self.categories,
                'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'export_formats': [format]
            }
            json.dump(model_info, f)
        
        self._update_progress('exported', f'Model exported in {format} format')
        
        return export_path
    
    def run_full_training_pipeline(self):
        """
        Execute the complete training pipeline.
        
        Returns:
            dict: Results of model evaluation
        """
        try:
            # Prepare data
            train_generator, validation_generator, test_generator = self.prepare_data()
            
            # Build model
            self.build_model()
            
            # Train model
            self.train(train_generator, validation_generator)
            
            # Evaluate model
            results = self.evaluate(test_generator)
            
            # Export model in different formats
            self.export_model('keras')
            self.export_model('tflite')
            self.export_model('saved_model')
            
            self._update_progress('completed', 'Training pipeline completed successfully')
            
            # Clear memory
            tf.keras.backend.clear_session()
            gc.collect()
            
            return results
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error in training pipeline: {error_message}")
            self._update_progress('failed', f'Training failed: {error_message}')
            raise
    
    def run_feature_training_pipeline(self, features_path):
        """
        Execute the training pipeline using pre-extracted features.
        
        Args:
            features_path (str): Path to the JSON file with features
            
        Returns:
            dict: Results of model evaluation
        """
        try:
            # Train model using features
            self.train_with_features(features_path)
            
            # Export model in different formats
            self.export_model('keras')
            self.export_model('tflite')
            self.export_model('saved_model')
            
            self._update_progress('completed', 'Feature-based training pipeline completed successfully')
            
            # Clear memory
            tf.keras.backend.clear_session()
            gc.collect()
            
            return {
                'status': 'success',
                'message': 'Feature-based training completed successfully'
            }
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error in feature training pipeline: {error_message}")
            self._update_progress('failed', f'Training failed: {error_message}')
            raise 