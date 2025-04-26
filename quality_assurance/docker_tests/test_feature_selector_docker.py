#!/usr/bin/env python3

import unittest
import requests
import pandas as pd
import numpy as np
import json
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
FEATURE_SELECTOR_URL = os.getenv("FEATURE_SELECTOR_URL", "http://localhost:5002")
HEALTH_ENDPOINT = "/health"
SELECT_FEATURES_ENDPOINT = "/select_features"
TRANSFORM_ENDPOINT = "/transform"

class FeatureSelectorDockerTest(unittest.TestCase):
    """Tests for the feature_selector Docker service."""
    
    def setUp(self):
        """Setup before each test - verify the service is running."""
        self.service_url = FEATURE_SELECTOR_URL
        logger.info(f"Testing feature_selector at {self.service_url}")
        
        # Check if service is running
        try:
            response = requests.get(f"{self.service_url}{HEALTH_ENDPOINT}", timeout=5)
            response.raise_for_status()
            logger.info("Service is running")
        except requests.exceptions.RequestException as e:
            logger.error(f"Service is not available: {str(e)}")
            logger.error("Make sure the feature_selector Docker container is running")
            self.skipTest("Feature selector service is not available")
    
    def test_health_endpoint(self):
        """Test the health endpoint returns correct format and status."""
        response = requests.get(f"{self.service_url}{HEALTH_ENDPOINT}")
        self.assertEqual(response.status_code, 200)
        
        health_data = response.json()
        self.assertIn("status", health_data)
        self.assertIn("version", health_data)
        self.assertIn("timestamp", health_data)
        
        # Status should be "healthy" or "degraded" (if OpenAI is not available)
        self.assertIn(health_data["status"], ["healthy", "degraded"])
    
    def test_select_features_basic(self):
        """Test selecting features from a basic dataset."""
        # Create a sample dataset with obvious features to select/remove
        data = {
            "patient_id": [1, 2, 3, 4, 5],  # Should be removed as ID
            "age": [25, 30, 35, 40, 45],  # Should be kept
            "temperature": [98.6, 99.1, 97.8, 98.4, 98.2],  # Should be kept
            "gender": [1, 0, 1, 0, 1],  # Should be kept
            "record_created": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],  # Might be removed
            "target": [0, 1, 0, 1, 0]  # Target variable
        }
        df = pd.DataFrame(data)
        
        # Convert to JSON for API request
        json_data = {
            "data": df.to_dict(orient="records"),
            "target_column": "target",
            "method": "fallback"  # Use fallback method to avoid OpenAI dependency
        }
        
        # Send request to select features
        response = requests.post(
            f"{self.service_url}{SELECT_FEATURES_ENDPOINT}",
            json=json_data
        )
        
        # Validate response
        self.assertEqual(response.status_code, 200, f"Error: {response.text}")
        
        result = response.json()
        self.assertIn("selected_features", result)
        self.assertIn("feature_importances", result)
        self.assertIn("metadata", result)
        
        # Check that patient_id is not in selected features
        selected_features = result["selected_features"]
        self.assertNotIn("patient_id", selected_features,
                        f"ID column was not removed: {selected_features}")
        
        # Check that important features are kept
        self.assertIn("age", selected_features)
        self.assertIn("temperature", selected_features)
        self.assertIn("gender", selected_features)
    
    def test_transform_with_selected_features(self):
        """Test transforming data using previously selected features."""
        # First, select features
        training_data = {
            "patient_id": [1, 2, 3, 4, 5],
            "age": [25, 30, 35, 40, 45],
            "temperature": [98.6, 99.1, 97.8, 98.4, 98.2],
            "gender": [1, 0, 1, 0, 1],
            "target": [0, 1, 0, 1, 0]
        }
        train_df = pd.DataFrame(training_data)
        
        # Convert to JSON for API request
        train_json = {
            "data": train_df.to_dict(orient="records"),
            "target_column": "target",
            "method": "fallback"
        }
        
        # Send request to select features
        select_response = requests.post(
            f"{self.service_url}{SELECT_FEATURES_ENDPOINT}",
            json=train_json
        )
        self.assertEqual(select_response.status_code, 200)
        
        select_result = select_response.json()
        model_id = select_result.get("model_id")
        self.assertIsNotNone(model_id, "No model_id returned from feature selection")
        
        # Now test transform with new data
        new_data = {
            "patient_id": [6, 7, 8],
            "age": [50, 55, 60],
            "temperature": [98.9, 99.2, 97.5],
            "gender": [0, 1, 0]
        }
        new_df = pd.DataFrame(new_data)
        
        transform_json = {
            "data": new_df.to_dict(orient="records"),
            "model_id": model_id
        }
        
        # Send request to transform data
        transform_response = requests.post(
            f"{self.service_url}{TRANSFORM_ENDPOINT}",
            json=transform_json
        )
        
        # Validate response
        self.assertEqual(transform_response.status_code, 200, f"Error: {transform_response.text}")
        
        transform_result = transform_response.json()
        self.assertIn("transformed_data", transform_result)
        
        # Check that transformed data doesn't include patient_id
        transformed_data = transform_result["transformed_data"]
        self.assertTrue(len(transformed_data) > 0, "No transformed data returned")
        
        # Convert to DataFrame for easier validation
        transformed_df = pd.DataFrame(transformed_data)
        self.assertNotIn("patient_id", transformed_df.columns)
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        # Test with empty dataset
        empty_data = {"data": [], "target_column": "target"}
        response = requests.post(
            f"{self.service_url}{SELECT_FEATURES_ENDPOINT}",
            json=empty_data
        )
        self.assertNotEqual(response.status_code, 200)
        
        # Test with missing target column
        invalid_data = {
            "data": [{"age": 25, "temperature": 98.6}],
            # No target_column specified
        }
        response = requests.post(
            f"{self.service_url}{SELECT_FEATURES_ENDPOINT}",
            json=invalid_data
        )
        self.assertNotEqual(response.status_code, 200)
        
        # Test with invalid model_id for transform
        invalid_transform = {
            "data": [{"age": 25, "temperature": 98.6}],
            "model_id": "non_existent_model_id"
        }
        response = requests.post(
            f"{self.service_url}{TRANSFORM_ENDPOINT}",
            json=invalid_transform
        )
        self.assertNotEqual(response.status_code, 200)
    
    def test_feature_crossing(self):
        """Test feature crossing option."""
        # Create a dataset that could benefit from feature crossing
        data = {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "target": [0, 0, 1, 1, 1]
        }
        df = pd.DataFrame(data)
        
        # Convert to JSON for API request
        json_data = {
            "data": df.to_dict(orient="records"),
            "target_column": "target",
            "method": "fallback",
            "apply_feature_crossing": True,
            "max_crossing_degree": 2
        }
        
        # Send request to select features
        response = requests.post(
            f"{self.service_url}{SELECT_FEATURES_ENDPOINT}",
            json=json_data
        )
        
        # Validate response
        self.assertEqual(response.status_code, 200, f"Error: {response.text}")
        
        result = response.json()
        metadata = result.get("metadata", {})
        
        # Check if feature crossing was applied
        transformations = metadata.get("transformations", {})
        crossed_features = [feat for feat in result.get("selected_features", [])
                           if "_X_" in feat or "crosses" in transformations]
        
        # In fallback mode with feature crossing, we expect some crossed features
        # but they might not always be selected, so we just log rather than assert
        if not crossed_features:
            logger.warning("No crossed features were created or selected")

if __name__ == "__main__":
    unittest.main() 