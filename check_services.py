from app_api import DATA_CLEANER_URL, FEATURE_SELECTOR_URL, ANOMALY_DETECTOR_URL, MODEL_COORDINATOR_URL
import requests

# Print service URLs
print("DATA_CLEANER_URL:", DATA_CLEANER_URL)
print("FEATURE_SELECTOR_URL:", FEATURE_SELECTOR_URL)
print("ANOMALY_DETECTOR_URL:", ANOMALY_DETECTOR_URL)
print("MODEL_COORDINATOR_URL:", MODEL_COORDINATOR_URL)

# Function to check service availability
def check_service(name, url):
    try:
        print(f"Checking {name} at {url}...")
        response = requests.get(f"{url}/health", timeout=2)
        if response.status_code == 200:
            print(f"✅ {name} is healthy")
            return True
        else:
            print(f"❌ {name} returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name} is not available: {str(e)}")
        return False

# Check each service
check_service("Data Cleaner", DATA_CLEANER_URL)
check_service("Feature Selector", FEATURE_SELECTOR_URL)
check_service("Anomaly Detector", ANOMALY_DETECTOR_URL)
check_service("Model Coordinator", MODEL_COORDINATOR_URL) 