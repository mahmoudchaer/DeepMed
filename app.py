import os
import io
import zipfile
import tempfile

from flask import Flask, request, jsonify, send_file, render_template
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

app = Flask(__name__)

def train_model(zip_file, epochs=1, batch_size=32, learning_rate=0.001):
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
        
        num_classes = 5
        
        # Load EfficientNet-B0 with pretrained weights
        model = models.efficientnet_b0(pretrained=True)
        # Replace the classifier for our dataset's number of classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")
        
        # Save the trained model to an in-memory bytes object
        model_bytes = io.BytesIO()
        torch.save(model.state_dict(), model_bytes)
        model_bytes.seek(0)
        return model_bytes

@app.route('/')
def index():
    # Render the training page template
    return render_template('train_model.html')

@app.route('/api/train_model', methods=['POST'])
def api_train_model():
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    zip_file = request.files['zipFile']
    try:
        # Get hyperparameters from the form (default values provided)
        epochs = int(request.form.get('epochs', 1))
        batch_size = int(request.form.get('batchSize', 32))
        learning_rate = float(request.form.get('learningRate', 0.001))
        
        # Train the model on the provided data
        model_bytes = train_model(zip_file, epochs, batch_size, learning_rate)
        # Return the model file as an attachment for download
        return send_file(model_bytes,
                         mimetype='application/octet-stream',
                         as_attachment=True,
                         attachment_filename='trained_model.pt')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
