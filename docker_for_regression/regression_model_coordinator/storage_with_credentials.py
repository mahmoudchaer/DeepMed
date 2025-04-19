import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import logging
import traceback
import sys
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Enable verbose logging if environment variable is set
if os.environ.get('VERBOSE_LOGGING', 'false').lower() in ('true', '1', 't', 'yes'):
    logging.getLogger().setLevel(logging.DEBUG)
    logger.debug("Verbose logging enabled")

# Load environment variables
load_dotenv()

# Get Azure credentials from .env
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER")

# === ENTER YOUR CREDENTIALS HERE AS FALLBACK ===
# Uncomment and fill these if environment variables aren't working
# FALLBACK_ACCOUNT = "503nstoragehsen"
# FALLBACK_KEY = "your-key-here"
# FALLBACK_CONTAINER = "deepmed-containers"

# Use fallback credentials if environment variables not set
if not AZURE_STORAGE_ACCOUNT:
    logger.warning("AZURE_STORAGE_ACCOUNT not found in environment, checking for fallback")
    try:
        if 'FALLBACK_ACCOUNT' in globals():
            AZURE_STORAGE_ACCOUNT = FALLBACK_ACCOUNT
            logger.info(f"Using fallback storage account: {AZURE_STORAGE_ACCOUNT}")
    except:
        pass

if not AZURE_STORAGE_KEY:
    logger.warning("AZURE_STORAGE_KEY not found in environment, checking for fallback")
    try:
        if 'FALLBACK_KEY' in globals():
            AZURE_STORAGE_KEY = FALLBACK_KEY 
            logger.info("Using fallback storage key")
    except:
        pass

if not AZURE_CONTAINER:
    logger.warning("AZURE_CONTAINER not found in environment, checking for fallback")
    try:
        if 'FALLBACK_CONTAINER' in globals():
            AZURE_CONTAINER = FALLBACK_CONTAINER
            logger.info(f"Using fallback container: {AZURE_CONTAINER}")
    except:
        pass

# Check if Azure credentials are available
if not all([AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_CONTAINER]):
    logger.warning("Azure Storage credentials not found. Blob storage operations will not work.")
    logger.warning("AZURE_STORAGE_ACCOUNT: %s", "Set" if AZURE_STORAGE_ACCOUNT else "Not set")
    logger.warning("AZURE_STORAGE_KEY: %s", "Set" if AZURE_STORAGE_KEY else "Not set (Note: Key is required!)")
    logger.warning("AZURE_CONTAINER: %s", "Set" if AZURE_CONTAINER else "Not set")
    
    # Print environment variables for debugging (masking sensitive data)
    env_vars = os.environ.copy()
    if 'AZURE_STORAGE_KEY' in env_vars:
        env_vars['AZURE_STORAGE_KEY'] = 'MASKED_FOR_SECURITY'
    logger.debug(f"Current environment variables: {json.dumps({k: v for k, v in env_vars.items() if 'AZURE' in k})}")
    
# Create a Blob Service Client only if credentials are available
blob_service_client = None
if all([AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_CONTAINER]):
    try:
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={AZURE_STORAGE_ACCOUNT};AccountKey={AZURE_STORAGE_KEY};EndpointSuffix=core.windows.net"
        logger.debug(f"Attempting to connect to Azure Blob Storage with account: {AZURE_STORAGE_ACCOUNT}")
        
        # Try both connection methods
        try:
            logger.debug("Trying connection with BlobServiceClient and account URL...")
            blob_service_client = BlobServiceClient(
                f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
                credential=AZURE_STORAGE_KEY
            )
        except Exception as url_err:
            logger.warning(f"Connection with URL failed: {str(url_err)}")
            logger.debug("Trying connection with connection string instead...")
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Verify connection by listing containers
        containers = list(blob_service_client.list_containers(max_results=10))
        container_names = [c.name for c in containers]
        logger.info(f"Connected to Azure Blob Storage account: {AZURE_STORAGE_ACCOUNT}")
        logger.info(f"Available containers: {container_names}")
        
        # Check if our container exists
        if AZURE_CONTAINER not in container_names:
            logger.warning(f"Container '{AZURE_CONTAINER}' not found in storage account. Available containers: {container_names}")
            
            # Attempt to create container if it doesn't exist
            try:
                logger.info(f"Attempting to create container '{AZURE_CONTAINER}'...")
                blob_service_client.create_container(AZURE_CONTAINER)
                logger.info(f"Container '{AZURE_CONTAINER}' created successfully!")
            except Exception as create_err:
                logger.error(f"Failed to create container: {str(create_err)}")
        else:
            logger.info(f"Container '{AZURE_CONTAINER}' found and accessible")
            
    except Exception as e:
        logger.error(f"Error connecting to Azure Blob Storage: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        blob_service_client = None

def upload_to_blob(file, filename):
    """Uploads a file to Azure Blob Storage."""
    if blob_service_client is None:
        logger.error("Azure Blob Storage client not initialized. Cannot upload file.")
        return None
        
    try:
        logger.info(f"Preparing to upload file '{filename}' to Azure Blob Storage")
        
        if hasattr(file, 'tell'):
            initial_position = file.tell()
            
        # Check file size
        if hasattr(file, 'seek') and hasattr(file, 'tell'):
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            logger.info(f"File size: {file_size} bytes")
        else:
            if hasattr(file, 'getvalue'):
                file_size = len(file.getvalue())
                logger.info(f"BytesIO file size: {file_size} bytes")
            else:
                logger.warning("Could not determine file size")
        
        # Create blob client
        logger.debug(f"Creating blob client for {filename} in container {AZURE_CONTAINER}")
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER, blob=filename)
        
        # Upload the file
        logger.debug("Starting blob upload...")
        upload_result = blob_client.upload_blob(file, overwrite=True)
        logger.info(f"Upload result: {upload_result}")
        
        # Create URL
        blob_url = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/{AZURE_CONTAINER}/{filename}"
        logger.info(f"File '{filename}' uploaded successfully to Azure Blob Storage!")
        logger.info(f"Blob URL: {blob_url}")
        
        # Reset file position if it supports seeking
        if hasattr(file, 'seek') and 'initial_position' in locals():
            file.seek(initial_position)
            
        return blob_url
    except Exception as e:
        logger.error(f"Error uploading file to Azure: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

def get_blob_url(filename):
    """Gets the public URL of a blob file."""
    if blob_service_client is None:
        logger.error("Azure Blob Storage client not initialized.")
        return None
    
    logger.info(f"Generating URL for blob: {filename}")
    blob_url = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/{AZURE_CONTAINER}/{filename}"
    return blob_url

def download_blob(blob_url, local_path=None):
    """Downloads a file from Azure Blob Storage using authentication."""
    if blob_service_client is None:
        logger.error("Azure Blob Storage client not initialized. Cannot download file.")
        return None
    
    try:
        logger.info(f"Attempting to download blob from URL: {blob_url}")
        
        # Extract the blob name from the URL
        parts = blob_url.split('/')
        if len(parts) < 5:
            logger.error(f"Invalid blob URL format: {blob_url}")
            return None
            
        blob_name = '/'.join(parts[4:])  # Everything after the container name
        logger.debug(f"Extracted blob name: {blob_name}")
        
        # Get a blob client for the specific blob
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER, blob=blob_name)
        
        # Download the blob content
        if local_path:
            # Download directly to file
            logger.debug(f"Downloading to local path: {local_path}")
            with open(local_path, "wb") as file:
                download_stream = blob_client.download_blob()
                content = download_stream.readall()
                file.write(content)
                
            # Verify download
            file_size = os.path.getsize(local_path)
            logger.info(f"Blob '{blob_name}' downloaded successfully to {local_path}! Size: {file_size} bytes")
            
            return True
        else:
            download_stream = blob_client.download_blob()
            blob_data = download_stream.readall()
            logger.info(f"Blob '{blob_name}' downloaded successfully!")
            return blob_data
    except Exception as e:
        logger.error(f"Error downloading file from Azure: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None 