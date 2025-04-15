import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
import os

# Define the Autoencoder architecture (must match the one used for training)
class Autoencoder(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def detect_anomaly(image_path, model_path='autoencoder.pt', metadata_path='metadata.json'):
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    threshold = metadata.get('threshold')
    image_size = metadata.get('image_size', 256)
    
    # Load model
    model = Autoencoder(input_channels=3)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Detect anomaly
    with torch.no_grad():
        output = model(image_tensor)
        mse = ((image_tensor - output) ** 2).mean().item()
    
    is_anomaly = mse > threshold
    
    return {
        'reconstruction_error': mse,
        'threshold': threshold,
        'is_anomaly': is_anomaly
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python detect_anomaly.py <image_path> [model_path] [metadata_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'autoencoder.pt'
    metadata_path = sys.argv[3] if len(sys.argv) > 3 else 'metadata.json'
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        sys.exit(1)
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file {metadata_path} not found")
        sys.exit(1)
    
    result = detect_anomaly(image_path, model_path, metadata_path)
    print(f"Anomaly detection result:")
    print(f"  Reconstruction error: {result['reconstruction_error']:.6f}")
    print(f"  Threshold: {result['threshold']:.6f}")
    print(f"  Is anomaly: {result['is_anomaly']}") 