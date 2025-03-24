import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Azure credentials from .env
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER")

# Check if Azure credentials are available
if not all([AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_CONTAINER]):
    logger.warning("Azure Storage credentials not found. Blob storage operations will not work.")
    
# Create a Blob Service Client only if credentials are available
blob_service_client = None
if all([AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_CONTAINER]):
    try:
        blob_service_client = BlobServiceClient(
            f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
            credential=AZURE_STORAGE_KEY
        )
        logger.info(f"Connected to Azure Blob Storage account: {AZURE_STORAGE_ACCOUNT}")
    except Exception as e:
        logger.error(f"Error connecting to Azure Blob Storage: {str(e)}")

def upload_to_blob(file, filename):
    """Uploads a file to Azure Blob Storage."""
    if blob_service_client is None:
        logger.error("Azure Blob Storage client not initialized. Cannot upload file.")
        return None
        
    try:
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER, blob=filename)
        blob_client.upload_blob(file, overwrite=True)
        logger.info(f"File '{filename}' uploaded successfully!")
        return f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/{AZURE_CONTAINER}/{filename}"
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return None

def get_blob_url(filename):
    """Gets the public URL of a blob file."""
    if blob_service_client is None:
        logger.error("Azure Blob Storage client not initialized.")
        return None
        
    return f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/{AZURE_CONTAINER}/{filename}"

def delete_blob(filename):
    """Deletes a file from Azure Blob Storage."""
    if blob_service_client is None:
        logger.error("Azure Blob Storage client not initialized. Cannot delete file.")
        return False
        
    try:
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER, blob=filename)
        blob_client.delete_blob()
        logger.info(f"File '{filename}' deleted successfully!")
        return True
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        return False
