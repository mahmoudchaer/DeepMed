import pytest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
import numpy as np
import json
import tempfile
import uuid
from pathlib import Path
from werkzeug.datastructures import FileStorage
from io import BytesIO
import requests
from flask import Flask, session

# Import the modules to test
from app_api import (
    allowed_file,
    check_file_size,
    load_data,
    clean_data_for_json,
    setup_temp_dir,
    get_temp_filepath,
    cleanup_session_files,
    check_services,
    save_to_temp_file,
    load_from_temp_file,
    check_session_size,
    is_service_available
)

@pytest.fixture
def app():
    """Create a Flask app for testing"""
    app = Flask(__name__)
    app.secret_key = 'test_secret_key'
    return app

@pytest.fixture
def client(app):
    """Create a test client"""
    return app.test_client()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def sample_csv():
    """Create a sample CSV file for testing"""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c'],
        'C': [1.1, 2.2, 3.3]
    })
    return df.to_csv(index=False).encode('utf-8')

@pytest.fixture
def sample_excel():
    """Create a sample Excel file for testing"""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c'],
        'C': [1.1, 2.2, 3.3]
    })
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return output

def test_allowed_file():
    """Test file extension validation"""
    assert allowed_file('test.csv') is True
    assert allowed_file('test.xlsx') is True
    assert allowed_file('test.xls') is True
    assert allowed_file('test.txt') is False
    assert allowed_file('test.pdf') is False
    assert allowed_file('test') is False

def test_check_file_size(sample_csv):
    """Test file size validation"""
    # Create a file object
    file = FileStorage(
        stream=BytesIO(sample_csv),
        filename='test.csv',
        content_type='text/csv'
    )
    
    # Test with default 2MB limit
    assert check_file_size(file) is True
    
    # Test with smaller limit (file is about 30 bytes)
    assert check_file_size(file, max_size_mb=0.0001) is True

def test_load_data_csv(sample_csv, temp_dir):
    """Test loading CSV data"""
    # Save sample CSV to temp file
    filepath = os.path.join(temp_dir, 'test.csv')
    with open(filepath, 'wb') as f:
        f.write(sample_csv)
    
    # Test loading
    result = load_data(filepath)
    df, metadata = result
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ['A', 'B', 'C']
    assert isinstance(metadata, dict)
    assert 'columns' in metadata
    assert 'rows' in metadata

def test_load_data_excel(sample_excel, temp_dir):
    """Test loading Excel data"""
    # Save sample Excel to temp file
    filepath = os.path.join(temp_dir, 'test.xlsx')
    with open(filepath, 'wb') as f:
        f.write(sample_excel.getvalue())
    
    # Test loading
    result = load_data(filepath)
    df, metadata = result
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ['A', 'B', 'C']
    assert isinstance(metadata, dict)
    assert 'columns' in metadata
    assert 'rows' in metadata

def test_clean_data_for_json():
    """Test data cleaning for JSON serialization"""
    # Test with normal data
    data = {'a': 1, 'b': 'test', 'c': [1, 2, 3]}
    assert clean_data_for_json(data) == data
    
    # Test with NaN values
    data = {'a': np.nan, 'b': np.inf, 'c': -np.inf}
    cleaned = clean_data_for_json(data)
    assert cleaned['a'] is None
    assert cleaned['b'] is None
    assert cleaned['c'] is None
    
    # Test with numpy arrays
    data = {'a': np.array([1, 2, 3])}
    cleaned = clean_data_for_json(data)
    assert np.array_equal(cleaned['a'], np.array([1, 2, 3]))

def test_setup_temp_dir(temp_dir):
    """Test temporary directory setup"""
    # Test with valid directory
    assert setup_temp_dir(temp_dir) == temp_dir
    
    # Test with invalid directory (should create it)
    invalid_dir = os.path.join(temp_dir, 'nonexistent', 'subdir')
    result = setup_temp_dir(invalid_dir)
    assert result == invalid_dir
    assert os.path.exists(invalid_dir)

def test_get_temp_filepath():
    """Test temporary filepath generation"""
    # Test with filename
    filepath = get_temp_filepath('test.csv')
    assert filepath.endswith('.csv')
    # The filename should be a UUID followed by the original filename
    filename = os.path.basename(filepath)
    assert filename.endswith('test.csv')
    assert len(filename.split('_')[0]) == 36  # UUID length
    
    # Test with extension only
    filepath = get_temp_filepath(extension='.csv')
    assert filepath.endswith('.csv')
    filename = os.path.basename(filepath)
    assert filename.endswith('.csv')
    assert len(filename.split('.')[0]) == 36  # UUID length

def test_cleanup_session_files(app):
    """Test session file cleanup"""
    with app.test_request_context():
        # Set up test session data
        session['test_file'] = '/tmp/test1.txt'
        session['another_file'] = '/tmp/test2.txt'
        
        # Test cleanup
        cleanup_session_files()
        
        # Verify session keys are still present (function doesn't modify session)
        assert 'test_file' in session
        assert 'another_file' in session

@patch('app_api.SERVICES', {
    "Test Service": {"url": "http://test", "endpoint": "/health"}
})
@patch('requests.get')
def test_check_services(mock_get):
    """Test service health checking"""
    # Mock successful response
    mock_get.return_value.status_code = 200
    
    # Test with all services up
    result = check_services()
    assert isinstance(result, dict)
    assert "Test Service" in result
    assert result["Test Service"] == "healthy"
    
    # Test with service down
    mock_get.return_value.status_code = 500
    result = check_services()
    assert result["Test Service"] == "unhealthy"

def test_check_session_size(app):
    """Test session size checking"""
    with app.test_request_context():
        # Set up test session data
        session['data'] = 'x' * 1000  # Small data
        
        # Test with default size limit
        result = check_session_size()
        assert result is None  # Function returns None
        
        # Test with custom size limit
        result = check_session_size(max_size=1000000)
        assert result is None  # Function returns None

@patch('requests.get')
def test_is_service_available(mock_get):
    """Test service availability checking"""
    # Mock successful response
    mock_get.return_value.status_code = 200
    assert is_service_available('http://test') is True
    
    # Mock failed response
    mock_get.return_value.status_code = 500
    assert is_service_available('http://test') is False
    
    # Mock timeout
    mock_get.side_effect = requests.exceptions.Timeout
    assert is_service_available('http://test') is False 