import pytest
from unittest.mock import patch, MagicMock
import os
import io
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Import the modules to test
import storage
import keyvault

@pytest.fixture(autouse=True)
def setup_storage():
    """Setup storage module before each test"""
    # Reset storage module globals
    storage.AZURESTORAGEACCOUNT = None
    storage.AZURESTORAGEKEY = None
    storage.AZURECONTAINER = None
    storage.blob_service_client = None

@pytest.fixture
def mock_azure_credentials():
    """Fixture to mock Azure credentials"""
    with patch('keyvault.getenv') as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: {
            'AZURESTORAGEACCOUNT': '503nstorageomar',
            'AZURESTORAGEKEY': 'testkey',
            'AZURECONTAINER': '503n'
        }.get(key, default)
        yield mock_getenv

@pytest.fixture
def mock_blob_service_client():
    """Fixture to mock BlobServiceClient"""
    with patch('storage.BlobServiceClient') as mock_client:
        # Create a mock instance
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        # Initialize storage
        storage.init_storage()
        yield mock_instance

@pytest.fixture
def mock_secret_client():
    """Fixture to mock SecretClient"""
    with patch('keyvault.SecretClient') as mock_client:
        # Create a mock instance
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        yield mock_instance

def test_get_secret_success(mock_secret_client):
    """Test successful secret retrieval from Key Vault"""
    # Setup mock
    mock_secret = MagicMock()
    mock_secret.value = "test_secret_value"
    mock_secret_client.get_secret.return_value = mock_secret
    
    # Test
    result = keyvault.get_secret("test_secret")
    assert result == "test_secret_value"
    mock_secret_client.get_secret.assert_called_once_with("test_secret")

def test_getenv_success(mock_secret_client):
    """Test getenv function success"""
    # Setup mock
    mock_secret = MagicMock()
    mock_secret.value = "test_env_value"
    mock_secret_client.get_secret.return_value = mock_secret
    
    # Test
    result = keyvault.getenv("TEST_ENV")
    assert result == "test_env_value"

def test_upload_to_blob_success(mock_azure_credentials, mock_blob_service_client):
    """Test successful blob upload"""
    # Setup mock
    mock_blob_client = MagicMock()
    mock_blob_service_client.get_blob_client.return_value = mock_blob_client
    
    # Test
    test_file = io.BytesIO(b"test content")
    result = storage.upload_to_blob(test_file, "test.txt")
    
    assert result == "https://503nstorageomar.blob.core.windows.net/503n/test.txt"
    mock_blob_client.upload_blob.assert_called_once_with(test_file, overwrite=True)

def test_upload_to_blob_failure(mock_azure_credentials, mock_blob_service_client):
    """Test blob upload failure"""
    # Setup mock to raise exception
    mock_blob_client = MagicMock()
    mock_blob_client.upload_blob.side_effect = Exception("Upload failed")
    mock_blob_service_client.get_blob_client.return_value = mock_blob_client
    
    # Test
    test_file = io.BytesIO(b"test content")
    result = storage.upload_to_blob(test_file, "test.txt")
    
    assert result is None

def test_download_blob_success(mock_azure_credentials, mock_blob_service_client):
    """Test successful blob download"""
    # Setup mock
    mock_blob_client = MagicMock()
    mock_download_stream = MagicMock()
    mock_download_stream.readall.return_value = b"test content"
    mock_blob_client.download_blob.return_value = mock_download_stream
    mock_blob_service_client.get_blob_client.return_value = mock_blob_client
    
    # Test
    result = storage.download_blob("https://503nstorageomar.blob.core.windows.net/503n/test.txt")
    
    assert result == b"test content"
    mock_blob_client.download_blob.assert_called_once()

def test_download_blob_to_file(mock_azure_credentials, mock_blob_service_client, tmp_path):
    """Test downloading blob to file"""
    # Setup mock
    mock_blob_client = MagicMock()
    mock_download_stream = MagicMock()
    mock_download_stream.readall.return_value = b"test content"
    mock_blob_client.download_blob.return_value = mock_download_stream
    mock_blob_service_client.get_blob_client.return_value = mock_blob_client
    
    # Test
    test_file = tmp_path / "test.txt"
    result = storage.download_blob(
        "https://503nstorageomar.blob.core.windows.net/503n/test.txt",
        str(test_file)
    )
    
    assert result is True
    assert test_file.read_bytes() == b"test content"

def test_delete_blob_success(mock_azure_credentials, mock_blob_service_client):
    """Test successful blob deletion"""
    # Setup mock
    mock_blob_client = MagicMock()
    mock_blob_service_client.get_blob_client.return_value = mock_blob_client
    
    # Test
    result = storage.delete_blob("test.txt")
    
    assert result is True
    mock_blob_client.delete_blob.assert_called_once()

def test_delete_blob_failure(mock_azure_credentials, mock_blob_service_client):
    """Test blob deletion failure"""
    # Setup mock to raise exception
    mock_blob_client = MagicMock()
    mock_blob_client.delete_blob.side_effect = Exception("Delete failed")
    mock_blob_service_client.get_blob_client.return_value = mock_blob_client
    
    # Test
    result = storage.delete_blob("test.txt")
    
    assert result is False

def test_get_blob_url(mock_azure_credentials):
    """Test getting blob URL"""
    result = storage.get_blob_url("test.txt")
    assert result == "https://503nstorageomar.blob.core.windows.net/503n/test.txt" 