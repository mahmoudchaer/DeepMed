import os
import io
import zipfile
import tempfile
import json

from flask import Flask, request, jsonify, Response
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split

app = Flask(__name__)

def train_model(zip_file, num_classes=5, training_level=3):
    # Map training level to hyperparameters
    training_params = {
        1: {"epochs": 1, "batch_size": 32, "learning_rate": 0.01},    # Fastest, lowest accuracy
        2: {"epochs": 2, "batch_size": 24, "learning_rate": 0.005},   # Fast, better accuracy
        3: {"epochs": 3, "batch_size": 16, "learning_rate": 0.001},   # Default balanced option
        4: {"epochs": 5, "batch_size": 16, "learning_rate": 0.0005},  # Higher quality, slower
        5: {"epochs": 8, "batch_size": 8, "learning_rate": 0.0001}    # Best quality, slowest
    }
    
    # Get hyperparameters based on training level
    params = training_params.get(int(training_level), training_params[3])  # Default to level 3 if invalid
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    
    # Create a temporary directory to extract the ZIP file
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "data.zip")
        zip_file.save(zip_path)
        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Create a dataset from the extracted images using folder names as labels
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        dataset = datasets.ImageFolder(root=extract_dir, transform=transform)
        
        # Use the user-specified number of classes
        num_classes = int(num_classes)
        
        # Calculate minimum dataset size needed for meaningful splits
        # We want at least 30 samples per class for training and 10 each for val/test
        min_samples_per_class = 50  # 30 for training + 10 for val + 10 for test
        min_dataset_size = num_classes * min_samples_per_class
        
        dataset_size = len(dataset)
        print(f"Dataset size: {dataset_size}, Minimum recommended: {min_dataset_size}")
        
        # Determine whether to split the dataset
        should_split = dataset_size >= min_dataset_size
        
        if should_split:
            # Split dataset into training, validation, and test sets
            train_size = int(0.7 * dataset_size)
            val_size = int(0.15 * dataset_size)
            test_size = dataset_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            print(f"Split dataset: Train={train_size}, Validation={val_size}, Test={test_size}")
        else:
            # Use the entire dataset for training and evaluation
            train_size = dataset_size
            val_size = 0
            test_size = 0
            
            train_dataset = dataset
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Create empty loaders for validation and testing
            # We'll use the training set for evaluation when these are empty
            val_loader = test_loader = None
            
            print(f"Using entire dataset ({dataset_size} images) for both training and evaluation")
        
        # Load EfficientNet-B0 with pretrained weights
        model = models.efficientnet_b0(pretrained=True)
        # Replace the classifier for our dataset's number of classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Track metrics during training
        epoch_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        final_metrics = {}
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss/len(train_loader)
            train_acc = 100 * correct / total
            epoch_losses.append(train_loss)
            
            # Validation phase
            if should_split and val_loader:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        # Calculate validation accuracy
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_loss = val_loss / len(val_loader)
                val_acc = 100 * val_correct / val_total
            else:
                # Use training metrics as validation metrics when no separate validation set
                val_loss = train_loss
                val_acc = train_acc
                
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%" + 
                 (f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%" if should_split else ""))
            
            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
        
        # Load the best model for final testing
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Test phase
        if should_split and test_loader:
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    
                    # Calculate test accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            test_loss = test_loss / len(test_loader)
            test_acc = 100 * test_correct / test_total
        else:
            # Use training metrics as test metrics when no separate test set
            test_loss = train_loss
            test_acc = train_acc
        
        # Calculate final metrics
        final_metrics = {
            "final_train_loss": epoch_losses[-1],
            "final_val_loss": val_losses[-1],
            "test_loss": test_loss,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "test_accuracy": test_acc,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": str(device),
            "num_classes": num_classes,
            "num_total_images": dataset_size,
            "num_train_images": train_size,
            "num_val_images": val_size,
            "num_test_images": test_size,
            "training_level": training_level,
            "data_was_split": should_split,
            "class_to_idx": dataset.class_to_idx,
            "classes": dataset.classes
        }
        
        # Save the trained model to an in-memory bytes object
        model_bytes = io.BytesIO()
        torch.save(model.state_dict(), model_bytes)
        model_bytes.seek(0)
        
        # Return both the model bytes and metrics
        return model_bytes, final_metrics

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "model-training-service"})

@app.route('/train', methods=['POST'])
def api_train_model():
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    try:
        # Get parameters from the form
        num_classes = int(request.form.get('numClasses', 5))
        training_level = int(request.form.get('trainingLevel', 3))
        
        # Train the model on the provided data
        model_bytes, metrics = train_model(zip_file, num_classes=num_classes, training_level=training_level)
        
        # Log metrics for debugging
        print(f"Training metrics: {metrics}")
        
        # Create a response with both the model file and metrics
        response = Response(model_bytes.getvalue())
        response.headers["Content-Type"] = "application/octet-stream"
        response.headers["Content-Disposition"] = "attachment; filename=trained_model.pt"
        
        # Add metrics as a JSON string in a custom header
        response.headers["X-Training-Metrics"] = json.dumps(metrics)
        
        # Add CORS headers to ensure all custom headers are exposed
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Expose-Headers"] = "X-Training-Metrics"
        
        # Log header information
        print("Response headers set:")
        for header, value in response.headers.items():
            print(f"  {header}: {value[:100]}{'...' if len(str(value)) > 100 else ''}")
        
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5021))
    app.run(host='0.0.0.0', port=port, debug=False) 