import requests
import json
import time
import pandas as pd
import numpy as np

# List of model services to test
MODEL_SERVICES = {
    "logistic_regression": "http://localhost:5010",
    "decision_tree": "http://localhost:5011", 
    "random_forest": "http://localhost:5012",
    "svm": "http://localhost:5013",
    "knn": "http://localhost:5014",
    "naive_bayes": "http://localhost:5015"
}

# Create some simple test data - just 20 samples with 3 features
np.random.seed(42)
num_samples = 20
data = {
    "feature1": np.random.normal(0, 1, num_samples).tolist(),
    "feature2": np.random.normal(0, 1, num_samples).tolist(),
    "feature3": np.random.normal(0, 1, num_samples).tolist()
}

# Create simple binary target variable
target = np.random.choice([0, 1], size=num_samples).tolist()

# Create the payload
payload = {
    "data": data,
    "target": target,
    "test_size": 0.2
}

print(f"Testing with payload: {json.dumps(payload)[:200]}...")
print(f"Data has {len(data)} features and {len(target)} samples")

results = {}

# Test each model service
for model_name, base_url in MODEL_SERVICES.items():
    print(f"\nTesting {model_name} at {base_url}...")
    
    # First, check if the service is running
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code == 200:
            print(f"  ✓ Health check passed: {health_response.json()}")
        else:
            print(f"  ✗ Health check failed: {health_response.status_code} - {health_response.text}")
            continue
    except Exception as e:
        print(f"  ✗ Health check error: {str(e)}")
        continue
    
    # Then, try to train a model
    try:
        train_start = time.time()
        train_response = requests.post(f"{base_url}/train", json=payload, timeout=30)
        train_time = time.time() - train_start
        
        if train_response.status_code == 200:
            response_data = train_response.json()
            print(f"  ✓ Training successful ({train_time:.2f}s)")
            print(f"  - Metrics: {response_data.get('metrics', {})}")
            results[model_name] = response_data
        else:
            print(f"  ✗ Training failed: {train_response.status_code} - {train_response.text}")
    except Exception as e:
        print(f"  ✗ Training error: {str(e)}")

# Print summary
print("\n=== SUMMARY ===")
successful = [name for name, _ in results.items()]
print(f"Successfully trained {len(successful)}/{len(MODEL_SERVICES)} models")
print(f"Successful models: {', '.join(successful) if successful else 'None'}")

if results:
    print("\nDetailed Results:")
    for model_name, result in results.items():
        metrics = result.get('metrics', {})
        print(f"{model_name}: Accuracy={metrics.get('accuracy', 'N/A')}")

# Try the coordinator as well
print("\nTesting the model coordinator...")
coordinator_url = "http://localhost:5020"

try:
    health_response = requests.get(f"{coordinator_url}/health", timeout=5)
    if health_response.status_code == 200:
        print(f"  ✓ Coordinator health check passed: {health_response.json()}")
    else:
        print(f"  ✗ Coordinator health check failed: {health_response.status_code} - {health_response.text}")
except Exception as e:
    print(f"  ✗ Coordinator health check error: {str(e)}")
    
try:
    train_start = time.time()
    train_response = requests.post(f"{coordinator_url}/train", json=payload, timeout=60)
    train_time = time.time() - train_start
    
    if train_response.status_code == 200:
        response_data = train_response.json()
        print(f"  ✓ Coordinator training successful ({train_time:.2f}s)")
        print(f"  - Trained {len(response_data.get('models', []))} models")
    else:
        print(f"  ✗ Coordinator training failed: {train_response.status_code} - {train_response.text}")
except Exception as e:
    print(f"  ✗ Coordinator training error: {str(e)}") 