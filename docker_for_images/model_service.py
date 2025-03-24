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
from torch.utils.data import DataLoader

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
        ])
        dataset = datasets.ImageFolder(root=extract_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Use the user-specified number of classes
        num_classes = int(num_classes)
        
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
        final_metrics = {}
        
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss/len(dataloader)
            epoch_acc = 100 * correct / total
            epoch_losses.append(epoch_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        # Calculate final metrics
        final_metrics = {
            "final_loss": epoch_losses[-1],
            "training_accuracy": epoch_acc,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": str(device),
            "num_classes": num_classes,
            "num_images": len(dataset),
            "training_level": training_level
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
        
        # Create a response with both the model file and metrics
        response = Response(model_bytes.getvalue())
        response.headers["Content-Type"] = "application/octet-stream"
        response.headers["Content-Disposition"] = "attachment; filename=trained_model.pt"
        
        # Add metrics as a JSON string in a custom header
        response.headers["X-Training-Metrics"] = json.dumps(metrics)
        
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5021))
    app.run(host='0.0.0.0', port=port, debug=False) 