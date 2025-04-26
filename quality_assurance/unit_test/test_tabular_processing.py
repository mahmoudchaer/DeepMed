import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from flask import Flask, jsonify
from flask_login import LoginManager, current_user, login_user
from app_tabular import (
    classification_training_status,
    get_classification_training_status,
    stop_classification_training,
    api_predict_tabular,
    api_extract_encodings
)

@pytest.fixture
def app():
    """Create a Flask app for testing"""
    app = Flask(__name__)
    app.secret_key = 'test_secret_key'
    login_manager = LoginManager()
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
        return MagicMock(id=user_id)
    
    return app

@pytest.fixture
def client(app):
    """Create a test client"""
    return app.test_client()

@pytest.fixture
def mock_user():
    """Create a mock user"""
    user = MagicMock()
    user.id = 'test_user'
    user.is_authenticated = True
    return user

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'C', 'D', 'E'],
        'target': [0, 1, 0, 1, 0]
    })

def test_classification_training_status_initialization():
    """Test training status initialization"""
    assert isinstance(classification_training_status, dict)
    assert len(classification_training_status) == 0

def test_get_classification_training_status(app, mock_user):
    """Test getting training status"""
    with app.test_request_context():
        # Log in the user
        login_user(mock_user)
        
        # Set up test data
        classification_training_status[mock_user.id] = {
            'status': 'running',
            'progress': 50,
            'message': 'Training in progress'
        }
        
        # Test getting status
        response = get_classification_training_status()
        assert response.status_code == 200
        data = response.get_json()
        assert data == {
            'status': 'running',
            'progress': 50,
            'message': 'Training in progress'
        }

def test_stop_classification_training(app, mock_user):
    """Test stopping training"""
    with app.test_request_context():
        # Log in the user
        login_user(mock_user)
        
        # Set up test data
        classification_training_status[mock_user.id] = {
            'status': 'running',
            'progress': 50,
            'message': 'Training in progress'
        }
        
        # Test stopping training
        response = stop_classification_training()
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'stopped'
        assert data['message'] == 'Training has been stopped.'
        assert classification_training_status[mock_user.id]['status'] == 'stopped'

@patch('app_tabular.requests.post')
def test_api_predict_tabular(mock_post, app, mock_user, sample_data):
    """Test tabular prediction API"""
    with app.test_request_context():
        # Log in the user
        login_user(mock_user)
        
        # Mock successful prediction response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'predictions': [0, 1, 0],
            'probabilities': [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]
        }
        mock_post.return_value = mock_response
        
        # Test prediction
        response, status_code = api_predict_tabular()
        assert status_code == 200
        assert isinstance(response, dict)
        assert 'predictions' in response
        assert 'probabilities' in response

@patch('app_tabular.requests.post')
def test_api_extract_encodings(mock_post, app, mock_user, sample_data):
    """Test encoding extraction API"""
    with app.test_request_context():
        # Log in the user
        login_user(mock_user)
        
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
        response, status_code = api_extract_encodings()
        assert status_code == 200
        assert isinstance(response, dict)
        assert 'encodings' in response
        assert 'feature2' in response['encodings'] 