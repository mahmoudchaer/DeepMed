import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, session, url_for, redirect
from flask_login import LoginManager, current_user, login_user
from werkzeug.datastructures import FileStorage
from io import BytesIO
import os
from app_tabular import (
    classification_training_status,
    get_classification_training_status,
    stop_classification_training,
    api_predict_tabular,
    api_extract_encodings,
    upload,
    training
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
    
    # Register routes needed for testing
    @app.route('/')
    def index():
        return "Index page"
    
    @app.route('/training')
    def training():
        return "Training page"
    
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

@pytest.fixture
def mock_model_package():
    """Create a mock model package file"""
    file = BytesIO(b'fake zip content')
    file.filename = 'model.zip'
    return FileStorage(file, filename='model.zip', content_type='application/zip')

@pytest.fixture
def mock_input_file(sample_data):
    """Create a mock input file"""
    csv_content = sample_data.to_csv(index=False)
    file = BytesIO(csv_content.encode())
    file.filename = 'input.csv'
    return FileStorage(file, filename='input.csv', content_type='text/csv')

@pytest.fixture
def mock_csv_file(sample_data):
    """Create a mock CSV file for upload"""
    csv_content = sample_data.to_csv(index=False)
    file = BytesIO(csv_content.encode())
    file.filename = 'test.csv'
    return FileStorage(file, filename='test.csv', content_type='text/csv')

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
        # Don't check classification_training_status as it's cleared by the function

@patch('app_tabular.requests.post')
def test_api_predict_tabular(mock_post, app, mock_user, mock_model_package, mock_input_file):
    """Test tabular prediction API"""
    with app.test_request_context():
        # Log in the user
        login_user(mock_user)
        
        # Set up request files
        request.files = {
            'model_package': mock_model_package,
            'input_file': mock_input_file
        }
        
        # Mock successful prediction response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'predictions': [0, 1, 0],
            'probabilities': [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]
        }
        mock_post.return_value = mock_response
        
        # Test prediction
        response = api_predict_tabular()
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, dict)
        assert 'predictions' in data
        assert 'probabilities' in data

@patch('app_tabular.requests.post')
def test_api_extract_encodings(mock_post, app, mock_user, mock_model_package):
    """Test encoding extraction API"""
    with app.test_request_context():
        # Log in the user
        login_user(mock_user)
        
        # Set up request files
        request.files = {
            'model_package': mock_model_package
        }
        
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
        response = api_extract_encodings()
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, dict)
        assert 'encodings' in data
        assert 'feature2' in data['encodings']

def test_api_predict_tabular_missing_files(app, mock_user):
    """Test tabular prediction API with missing files"""
    with app.test_request_context():
        # Log in the user
        login_user(mock_user)
        
        # Test without files
        response_data, status_code = api_predict_tabular()
        assert status_code == 400
        assert isinstance(response_data, dict)
        assert 'error' in response_data
        assert 'Both model package and input file are required' in response_data['error']

def test_api_extract_encodings_missing_file(app, mock_user):
    """Test encoding extraction API with missing file"""
    with app.test_request_context():
        # Log in the user
        login_user(mock_user)
        
        # Test without file
        response_data, status_code = api_extract_encodings()
        assert status_code == 400
        assert isinstance(response_data, dict)
        assert 'error' in response_data
        assert 'Model package file is required' in response_data['error']

@patch('app_tabular.requests.post')
def test_api_predict_tabular_invalid_files(mock_post, app, mock_user, mock_model_package, mock_input_file):
    """Test tabular prediction API with invalid files"""
    with app.app_context():
        with app.test_request_context():
            # Log in the user
            login_user(mock_user)
            
            # Set up invalid files
            mock_model_package.filename = 'model.txt'  # Invalid extension
            mock_input_file.filename = 'input.txt'  # Invalid extension
            
            request.files = {
                'model_package': mock_model_package,
                'input_file': mock_input_file
            }
            
            # Test with invalid files
            response_data, status_code = api_predict_tabular()
            assert status_code == 400
            assert isinstance(response_data, dict)
            assert 'error' in response_data
            assert 'Model package must be a ZIP archive' in response_data['error']

@patch('app_tabular.requests.post')
def test_api_extract_encodings_invalid_file(mock_post, app, mock_user, mock_model_package):
    """Test encoding extraction API with invalid file"""
    with app.app_context():
        with app.test_request_context():
            # Log in the user
            login_user(mock_user)
            
            # Set up invalid file
            mock_model_package.filename = 'model.txt'  # Invalid extension
            
            request.files = {
                'model_package': mock_model_package
            }
            
            # Test with invalid file
            response_data, status_code = api_extract_encodings()
            assert status_code == 400
            assert isinstance(response_data, dict)
            assert 'error' in response_data
            assert 'Model package must be a ZIP archive' in response_data['error']

@patch('app_tabular.requests.post')
def test_api_predict_tabular_service_unavailable(mock_post, app, mock_user, mock_model_package, mock_input_file):
    """Test tabular prediction API when service is unavailable"""
    with app.app_context():
        with app.test_request_context():
            # Log in the user
            login_user(mock_user)
            
            # Set up request files
            request.files = {
                'model_package': mock_model_package,
                'input_file': mock_input_file
            }
            
            # Mock service unavailable response
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_post.return_value = mock_response
            
            # Test with unavailable service
            response_data, status_code = api_predict_tabular()
            assert status_code == 503
            assert isinstance(response_data, dict)
            assert 'error' in response_data
            assert 'Tabular prediction service is not available' in response_data['error']

@patch('app_tabular.requests.post')
def test_api_extract_encodings_service_unavailable(mock_post, app, mock_user, mock_model_package):
    """Test encoding extraction API when service is unavailable"""
    with app.app_context():
        with app.test_request_context():
            # Log in the user
            login_user(mock_user)
            
            # Set up request files
            request.files = {
                'model_package': mock_model_package
            }
            
            # Mock service unavailable response
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_post.return_value = mock_response
            
            # Test with unavailable service
            response_data, status_code = api_extract_encodings()
            assert status_code == 503
            assert isinstance(response_data, dict)
            assert 'error' in response_data
            assert 'Tabular prediction service is not available' in response_data['error']

@patch('app_tabular.requests.post')
def test_upload_success(mock_post, app, mock_user, mock_csv_file):
    """Test successful file upload"""
    with app.app_context():
        with app.test_request_context():
            # Log in the user
            login_user(mock_user)
            
            # Set up request file
            request.files = {
                'file': mock_csv_file
            }
            
            # Mock successful data loading
            with patch('app_tabular.pd.read_csv') as mock_read_csv:
                mock_read_csv.return_value = pd.DataFrame({
                    'feature1': [1, 2, 3],
                    'feature2': ['A', 'B', 'C']
                })
                
                # Test upload
                response = upload()
                assert response.status_code == 302  # Redirect
                assert 'uploaded_file' in session
                assert 'file_stats' in session
                assert 'data_columns' in session

@patch('app_tabular.requests.post')
def test_upload_invalid_file(mock_post, app, mock_user):
    """Test upload with invalid file"""
    with app.app_context():
        with app.test_request_context():
            # Log in the user
            login_user(mock_user)
            
            # Set up invalid file
            invalid_file = BytesIO(b'invalid content')
            invalid_file.filename = 'test.txt'
            request.files = {
                'file': FileStorage(invalid_file, filename='test.txt', content_type='text/plain')
            }
            
            # Test upload
            response = upload()
            assert response.status_code == 302  # Redirect
            assert 'uploaded_file' not in session
            assert 'file_stats' not in session
            assert 'data_columns' not in session

@patch('app_tabular.requests.post')
def test_training_reset(mock_post, app, mock_user):
    """Test training page reset"""
    with app.app_context():
        with app.test_request_context():
            # Log in the user
            login_user(mock_user)
            
            # Set up session data
            session['uploaded_file'] = '/tmp/test.csv'
            session['file_stats'] = {'rows': 100, 'columns': 5}
            session['data_columns'] = ['col1', 'col2']
            
            # Set up request args
            request.args = {'new': '1'}
            
            # Test training reset
            response = training()
            assert response.status_code == 302  # Redirect
            assert 'uploaded_file' not in session
            assert 'file_stats' not in session
            assert 'data_columns' not in session 