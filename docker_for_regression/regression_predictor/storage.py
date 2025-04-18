import os
from dotenv import load_dotenv
import logging
import requests
import io
import time

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
    logger.warning("AZURE_STORAGE_ACCOUNT: %s", "Set" if AZURE_STORAGE_ACCOUNT else "Not set")
    logger.warning("AZURE_STORAGE_KEY: %s", "Set" if AZURE_STORAGE_KEY else "Not set")
    logger.warning("AZURE_CONTAINER: %s", "Set" if AZURE_CONTAINER else "Not set")

# Create function to handle blob operations
def get_sas_token():
    """Get a SAS token for Azure Blob Storage access"""
    if not all([AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_CONTAINER]):
        logger.error("Azure Storage credentials are missing. Cannot generate SAS token.")
        return None
    
    try:
        # Here we would normally use the Azure SDK to create a SAS token
        # For simplicity, we'll use direct shared key auth instead
        return AZURE_STORAGE_KEY
    except Exception as e:
        logger.error(f"Error generating SAS token: {str(e)}")
        return None

def download_blob(blob_url, local_path):
    """Downloads a file from Azure Blob Storage using SAS token.
    
    Args:
        blob_url (str): The full URL of the blob to download
        local_path (str): The local path to save the blob to
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    if not all([AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_CONTAINER]):
        logger.error("Azure Storage credentials are missing. Cannot download blob.")
        return False
    
    try:
        # Try first with direct download approach
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # First, try simple direct download with requests
                logger.info(f"Attempting direct download from {blob_url} (attempt {retry_count+1}/{max_retries})")
                
                if not blob_url.startswith("http"):
                    # Handle relative URLs by constructing the full URL
                    if blob_url.startswith("/"):
                        blob_url = blob_url[1:]  # Remove leading slash
                    blob_url = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/{AZURE_CONTAINER}/{blob_url}"
                
                headers = {}
                if AZURE_STORAGE_KEY:
                    # Use authorization with account key (not for production, but works for testing)
                    import base64
                    from datetime import datetime, timedelta
                    import hmac
                    import hashlib
                    
                    # This is a simplified SharedKey authentication - usually we'd use the Azure SDK
                    # Format the date for the header
                    date = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
                    headers['x-ms-date'] = date
                    headers['x-ms-version'] = '2020-04-08'
                    
                    # Extract the blob path
                    path_parts = blob_url.split("/")
                    container_index = path_parts.index(AZURE_CONTAINER)
                    blob_path = "/" + AZURE_CONTAINER + "/" + "/".join(path_parts[container_index+1:])
                    
                    logger.info(f"Blob path for authorization: {blob_path}")
                    
                    # String to sign
                    string_to_sign = (
                        f"GET\n\n\n\n\n\n\n\n\n\n\n\n"
                        f"x-ms-date:{date}\n"
                        f"x-ms-version:2020-04-08\n"
                        f"/{AZURE_STORAGE_ACCOUNT}{blob_path}"
                    )
                    
                    # Sign the string with the key
                    key = base64.b64decode(AZURE_STORAGE_KEY)
                    signature = base64.b64encode(
                        hmac.new(key, string_to_sign.encode('utf-8'), hashlib.sha256).digest()
                    ).decode('utf-8')
                    
                    # Set the authorization header
                    auth_header = f"SharedKey {AZURE_STORAGE_ACCOUNT}:{signature}"
                    headers['Authorization'] = auth_header
                
                # Make the request with the headers
                with requests.get(blob_url, headers=headers, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                # If we get here, the download was successful
                logger.info(f"Downloaded blob to {local_path}")
                return True
                
            except requests.RequestException as e:
                logger.warning(f"Request failed on attempt {retry_count+1}: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    logger.error(f"Failed to download blob after {max_retries} attempts: {str(e)}")
                    return False
            
            except Exception as e:
                logger.error(f"Unexpected error downloading blob: {str(e)}")
                return False
        
        return False
        
    except Exception as e:
        logger.error(f"Error in download_blob: {str(e)}")
        return False 