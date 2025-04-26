#!/usr/bin/env python3

import unittest
import requests
import pandas as pd
import numpy as np
import json
import time
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DATA_CLEANER_URL = os.getenv("DATA_CLEANER_URL", "http://localhost:5001")
HEALTH_ENDPOINT = "/health"
CLEAN_ENDPOINT = "/clean"

class DataCleanerDockerTest(unittest.TestCase):
    """Tests for the data_cleaner Docker service."""
    
    def setUp(self):
        """Setup before each test - verify the service is running."""
        self.service_url = DATA_CLEANER_URL
        logger.info(f"Testing data_cleaner at {self.service_url}")
        
        # Check if service is running
        try:
            response = requests.get(f"{self.service_url}{HEALTH_ENDPOINT}", timeout=5)
            response.raise_for_status()
            logger.info("Service is running")
        except requests.exceptions.RequestException as e:
            logger.error(f"Service is not available: {str(e)}")
            logger.error("Make sure the data_cleaner Docker container is running")
            self.skipTest("Data cleaner service is not available")
    
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
    
    def test_clean_basic_dataset(self):
        """Test cleaning a simple dataset with minimal issues."""
        # Create a simple dataset with some missing values
        data = {
            "age": [25, 30, None, 45, 50],
            "temperature": [98.6, 99.1, 97.8, None, 98.2],
            "gender": ["M", "F", "M", "F", None]
        }
        df = pd.DataFrame(data)
        
        # Convert to JSON for API request
        json_data = {
            "data": df.where(pd.notna(df), None).to_dict(orient="records"),
            "target_column": "temperature"
        }
        
        # Send request to clean the data
        response = requests.post(
            f"{self.service_url}{CLEAN_ENDPOINT}",
            json=json_data
        )
        
        # Validate response
        self.assertEqual(response.status_code, 200, f"Error: {response.text}")
        
        result = response.json()
        self.assertIn("cleaned_data", result)
        self.assertIn("cleaning_summary", result)
        
        # Convert back to DataFrame for validation
        cleaned_df = pd.DataFrame(result["cleaned_data"])
        
        # Check that there are no missing values in the cleaned data
        self.assertFalse(cleaned_df.isnull().any().any())
        
        # Check that all columns are now numeric
        for col in cleaned_df.columns:
            self.assertTrue(
                np.issubdtype(cleaned_df[col].dtype, np.number),
                f"Column {col} is not numeric: {cleaned_df[col].dtype}"
            )
    
    def test_clean_complex_dataset(self):
        """Test cleaning a more complex dataset with various data types."""
        # Create a dataset with mixed data types and more missing values
        data = {
            "age": [25, 30, None, 45, 50, None, 35],
            "temperature": [98.6, 99.1, 97.8, None, 98.2, 101.3, None],
            "gender": ["Male", "Female", "Male", "Female", None, "Male", "Female"],
            "smoker": ["Yes", "No", None, "Yes", "No", None, "No"],
            "bmi": [22.5, 25.1, None, 30.8, None, 19.2, 24.3],
        }
        df = pd.DataFrame(data)
        
        # Convert to JSON for API request
        json_data = {
            "data": df.where(pd.notna(df), None).to_dict(orient="records"),
            "target_column": "temperature"
        }
        
        # Send request to clean the data
        response = requests.post(
            f"{self.service_url}{CLEAN_ENDPOINT}",
            json=json_data
        )
        
        # Validate response
        self.assertEqual(response.status_code, 200, f"Error: {response.text}")
        
        result = response.json()
        self.assertIn("cleaned_data", result)
        self.assertIn("cleaning_summary", result)
        
        # Convert back to DataFrame for validation
        cleaned_df = pd.DataFrame(result["cleaned_data"])
        
        # Check that there are no missing values in the cleaned data
        self.assertFalse(cleaned_df.isnull().any().any())
        
        # Check that all columns are now numeric
        for col in cleaned_df.columns:
            self.assertTrue(
                np.issubdtype(cleaned_df[col].dtype, np.number),
                f"Column {col} is not numeric: {cleaned_df[col].dtype}"
            )
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        # Test with empty dataset
        empty_data = {"data": [], "target_column": "temperature"}
        response = requests.post(
            f"{self.service_url}{CLEAN_ENDPOINT}",
            json=empty_data
        )
        self.assertNotEqual(response.status_code, 200)
        
        # Test with missing target column
        invalid_data = {
            "data": [{"age": 25, "temperature": 98.6}],
            # No target_column specified
        }
        response = requests.post(
            f"{self.service_url}{CLEAN_ENDPOINT}",
            json=invalid_data
        )
        self.assertNotEqual(response.status_code, 200)
        
        # Test with invalid JSON
        response = requests.post(
            f"{self.service_url}{CLEAN_ENDPOINT}",
            data="This is not JSON",
            headers={"Content-Type": "application/json"}
        )
        self.assertNotEqual(response.status_code, 200)
    
    def test_consistency(self):
        """Test that the cleaner produces consistent results with the same data."""
        data = {
            "age": [25, 30, None, 45, 50],
            "temperature": [98.6, 99.1, 97.8, None, 98.2],
            "gender": ["M", "F", "M", "F", None]
        }
        df = pd.DataFrame(data)
        
        # Convert to JSON for API request
        json_data = {
            "data": df.where(pd.notna(df), None).to_dict(orient="records"),
            "target_column": "temperature"
        }
        
        # Send request twice to check consistency
        response1 = requests.post(
            f"{self.service_url}{CLEAN_ENDPOINT}",
            json=json_data
        )
        
        response2 = requests.post(
            f"{self.service_url}{CLEAN_ENDPOINT}",
            json=json_data
        )
        
        result1 = response1.json()
        result2 = response2.json()
        
        # Check if the cleaning results are consistent
        if "cleaned_data" in result1 and "cleaned_data" in result2:
            df1 = pd.DataFrame(result1["cleaned_data"])
            df2 = pd.DataFrame(result2["cleaned_data"])
            
            # Check if the dataframes have the same shape
            self.assertEqual(df1.shape, df2.shape)
            
            # Check if the values are similar (allowing for floating point differences)
            for col in df1.columns:
                np.testing.assert_allclose(
                    df1[col].values, 
                    df2[col].values,
                    rtol=1e-5, atol=1e-8,
                    err_msg=f"Column {col} has inconsistent values"
                )

if __name__ == "__main__":
    unittest.main() 