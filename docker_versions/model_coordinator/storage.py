import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Azure credentials from .env
AZURE-STORAGE-ACCOUNT = os.getenv("AZURE-STORAGE-ACCOUNT")
AZURE-STORAGE-KEY = os.getenv("AZURE-STORAGE-KEY")
AZURE-CONTAINER = os.getenv("AZURE-CONTAINER")

# Check if Azure credentials are available
if not all([AZURE-STORAGE-ACCOUNT, AZURE-STORAGE-KEY, AZURE-CONTAINER]):
    logger.warning("Azure Storage credentials not found. Blob storage operations will not work.")
    
# Create a Blob Service Client only if credentials are available
blob_service_client = None
if all([AZURE-STORAGE-ACCOUNT, AZURE-STORAGE-KEY, AZURE-CONTAINER]):
    try:
        blob_service_client = BlobServiceClient(
            f"https://{AZURE-STORAGE-ACCOUNT}.blob.core.windows.net",
            credential=AZURE-STORAGE-KEY
        )
        logger.info(f"Connected to Azure Blob Storage account: {AZURE-STORAGE-ACCOUNT}")
    except Exception as e:
        logger.error(f"Error connecting to Azure Blob Storage: {str(e)}")

def upload_to_blob(file, filename):
    """Uploads a file to Azure Blob Storage."""
    if blob_service_client is None:
        logger.error("Azure Blob Storage client not initialized. Cannot upload file.")
        return None
        
    try:
        blob_client = blob_service_client.get_blob_client(container=AZURE-CONTAINER, blob=filename)
        blob_client.upload_blob(file, overwrite=True)
        logger.info(f"File '{filename}' uploaded successfully!")
        return f"https://{AZURE-STORAGE-ACCOUNT}.blob.core.windows.net/{AZURE-CONTAINER}/{filename}"
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return None

def get_blob_url(filename):
    """Gets the public URL of a blob file."""
    if blob_service_client is None:
        logger.error("Azure Blob Storage client not initialized.")
        return None
        
    return f"https://{AZURE-STORAGE-ACCOUNT}.blob.core.windows.net/{AZURE-CONTAINER}/{filename}"

def delete_blob(filename):
    """Deletes a file from Azure Blob Storage."""
    if blob_service_client is None:
        logger.error("Azure Blob Storage client not initialized. Cannot delete file.")
        return False
        
    try:
        blob_client = blob_service_client.get_blob_client(container=AZURE-CONTAINER, blob=filename)
        blob_client.delete_blob()
        logger.info(f"File '{filename}' deleted successfully!")
        return True
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        return False

def download_blob(blob_url, local_path=None):
    """Downloads a file from Azure Blob Storage using authentication.
    
    Args:
        blob_url (str): The full URL of the blob to download
        local_path (str, optional): The local path to save the blob to. If not provided,
                                  the function will return the blob data.
        
    Returns:
        bytes or bool: The contents of the blob as bytes if local_path is None,
                      otherwise True if the download was successful.
    """
    if blob_service_client is None:
        logger.error("Azure Blob Storage client not initialized. Cannot download file.")
        return None
    
    try:
        # Extract the blob name from the URL
        parts = blob_url.split('/')
        if len(parts) < 5:
            logger.error(f"Invalid blob URL format: {blob_url}")
            return None
            
        blob_name = '/'.join(parts[4:])  # Everything after the container name
        
        # Get a blob client for the specific blob
        blob_client = blob_service_client.get_blob_client(container=AZURE-CONTAINER, blob=blob_name)
        
        # Download the blob content
        if local_path:
            # Download directly to file
            with open(local_path, "wb") as file:
                download_stream = blob_client.download_blob()
                file.write(download_stream.readall())
            logger.info(f"Blob '{blob_name}' downloaded successfully to {local_path}!")
            return True
        else:
            # Return the blob data
            download_stream = blob_client.download_blob()
            blob_data = download_stream.readall()
            logger.info(f"Blob '{blob_name}' downloaded successfully!")
            return blob_data
    except Exception as e:
        logger.error(f"Error downloading blob: {str(e)}")
        return None
