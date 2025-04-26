import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import json
import tempfile
import os
from io import BytesIO
from werkzeug.datastructures import FileStorage

# Import the modules to test
from app_tabular import (
    classification_training_status,
    get_classification_training_status,
    stop_classification_training
)

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'C', 'D', 'E'],
        'target': [0, 1, 0, 1, 0]
    })

@pytest.fixture
def mock_user():
    """Create a mock user for testing"""
    user = MagicMock()
    user.id = 1
    return user

def test_classification_training_status_initialization():
    """Test that classification_training_status is properly initialized"""
    assert isinstance(classification_training_status, dict)
    assert len(classification_training_status) == 0

def test_get_classification_training_status(mock_user):
    """Test getting training status"""
    # Set up test data
    classification_training_status[mock_user.id] = {
        'status': 'running',
        'progress': 50,
        'message': 'Training in progress'
    }
    
    # Test getting status
    status = get_classification_training_status()
    assert status['status'] == 'running'
    assert status['progress'] == 50
    assert status['message'] == 'Training in progress'
    
    # Test when no status exists
    classification_training_status.clear()
    status = get_classification_training_status()
    assert status['status'] == 'not_started'
    assert status['progress'] == 0
    assert status['message'] == 'No training in progress'

def test_stop_classification_training(mock_user):
    """Test stopping training"""
    # Set up test data
    classification_training_status[mock_user.id] = {
        'status': 'running',
        'progress': 50,
        'message': 'Training in progress'
    }
    
    # Test stopping training
    result = stop_classification_training()
    assert result['success'] is True
    assert result['message'] == 'Training stopped successfully'
    
    # Verify status was cleared
    assert mock_user.id not in classification_training_status
    
    # Test when no training is in progress
    result = stop_classification_training()
    assert result['success'] is False
    assert result['message'] == 'No training in progress to stop'

@patch('app_tabular.requests.post')
def test_api_predict_tabular(mock_post, sample_data):
    """Test tabular prediction API"""
    # Mock successful prediction response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'predictions': [0, 1, 0],
        'probabilities': [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]
    }
    mock_post.return_value = mock_response
    
    # Test prediction
    from app_tabular import api_predict_tabular
    result = api_predict_tabular()
    
    assert result.status_code == 200
    assert 'predictions' in result.json
    assert 'probabilities' in result.json

@patch('app_tabular.requests.post')
def test_api_extract_encodings(mock_post, sample_data):
    """Test encoding extraction API"""
    # Mock successful encoding response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'encodings': {
            'feature2': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        }
    }
    mock_post.return_value = mock_response
    
    # Test encoding extraction
    from app_tabular import api_extract_encodings
    result = api_extract_encodings()
    
    assert result.status_code == 200
    assert 'encodings' in result.json
    assert 'feature2' in result.json['encodings']

def test_data_validation(sample_data):
    """Test data validation functions"""
    # Test for missing values
    data_with_nan = sample_data.copy()
    data_with_nan.loc[0, 'feature1'] = np.nan
    
    # Test for invalid data types
    data_with_invalid = sample_data.copy()
    data_with_invalid['feature1'] = data_with_invalid['feature1'].astype(str)
    
    # Test for empty data
    empty_data = pd.DataFrame()
    
    # These tests would need to be implemented based on your actual validation logic
    # For now, they serve as placeholders for the validation tests you should implement
    assert len(sample_data) > 0, "Sample data should not be empty"
    assert not sample_data.isnull().any().any(), "Sample data should not contain null values"
    assert sample_data['feature1'].dtype in [np.int64, np.float64], "Feature1 should be numeric" 