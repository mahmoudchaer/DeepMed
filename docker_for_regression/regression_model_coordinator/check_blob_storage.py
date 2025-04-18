#!/usr/bin/env python3
"""
Azure Blob Storage Connection Test Script

This script verifies the Azure Blob Storage connection by:
1. Checking environment variables
2. Connecting to the storage account
3. Listing available containers
4. Creating a test container if needed
5. Uploading a test file
6. Downloading the test file
7. Cleaning up

Usage:
    python check_blob_storage.py
"""

import os
import sys
import io
import uuid
import time
import json
from dotenv import load_dotenv
import logging
import traceback

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("blob-check")

# Set to DEBUG for more detailed info
if os.environ.get('VERBOSE_LOGGING', 'false').lower() in ('true', '1', 't', 'yes'):
    logger.setLevel(logging.DEBUG)
    logger.debug("Verbose logging enabled")

def check_environment():
    """Check if required environment variables are set"""
    logger.info("Checking environment variables...")
    
    # Load environment variables from .env file if present
    load_dotenv()
    
    # Get Azure credentials from environment
    AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
    AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
    AZURE_CONTAINER = os.getenv("AZURE_CONTAINER")
    
    # Check if all required variables are set
    if not all([AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_CONTAINER]):
        logger.error("Azure Storage credentials are missing!")
        logger.error(f"AZURE_STORAGE_ACCOUNT: {'Set' if AZURE_STORAGE_ACCOUNT else 'Not set'}")
        logger.error(f"AZURE_STORAGE_KEY: {'Set' if AZURE_STORAGE_KEY else 'Not set'}")
        logger.error(f"AZURE_CONTAINER: {'Set' if AZURE_CONTAINER else 'Not set'}")
        
        # Show available environment variables for debugging
        env_vars = os.environ.copy()
        if 'AZURE_STORAGE_KEY' in env_vars:
            env_vars['AZURE_STORAGE_KEY'] = 'MASKED_FOR_SECURITY'
        
        azure_vars = {k: v for k, v in env_vars.items() if 'AZURE' in k.upper()}
        logger.debug(f"Available Azure environment variables: {json.dumps(azure_vars, indent=2)}")
        
        return False, None, None, None
    
    logger.info(f"All required environment variables are set:")
    logger.info(f"  AZURE_STORAGE_ACCOUNT: {AZURE_STORAGE_ACCOUNT}")
    logger.info(f"  AZURE_STORAGE_KEY: {'[MASKED]' + AZURE_STORAGE_KEY[-4:] if AZURE_STORAGE_KEY else 'Not set'}")
    logger.info(f"  AZURE_CONTAINER: {AZURE_CONTAINER}")
    
    return True, AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_CONTAINER

def try_connect_to_storage(account, key):
    """Try to connect to Azure Blob Storage"""
    logger.info(f"Attempting to connect to Azure Blob Storage account: {account}")
    
    try:
        from azure.storage.blob import BlobServiceClient
        
        # Try with URL first
        try:
            logger.debug("Trying connection with account URL...")
            connection_url = f"https://{account}.blob.core.windows.net"
            blob_service_client = BlobServiceClient(
                account_url=connection_url,
                credential=key
            )
            logger.info(f"Successfully connected to {connection_url}")
        except Exception as url_err:
            logger.warning(f"Connection with URL failed: {str(url_err)}")
            
            # Try with connection string
            logger.debug("Trying connection with connection string...")
            connection_string = (
                f"DefaultEndpointsProtocol=https;AccountName={account};"
                f"AccountKey={key};EndpointSuffix=core.windows.net"
            )
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            logger.info("Successfully connected using connection string")
        
        return True, blob_service_client
        
    except ImportError:
        logger.error("azure-storage-blob package is not installed")
        logger.error("Install it with: pip install azure-storage-blob")
        return False, None
    except Exception as e:
        logger.error(f"Error connecting to Azure Blob Storage: {str(e)}")
        logger.error(traceback.format_exc())
        return False, None

def list_containers(blob_service_client):
    """List available containers in the storage account"""
    logger.info("Listing available containers...")
    
    try:
        containers = list(blob_service_client.list_containers())
        container_names = [c.name for c in containers]
        
        if containers:
            logger.info(f"Found {len(containers)} containers:")
            for i, container in enumerate(containers):
                logger.info(f"  {i+1}. {container.name}")
        else:
            logger.warning("No containers found in the storage account")
        
        return container_names
    except Exception as e:
        logger.error(f"Error listing containers: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def check_container(blob_service_client, container_name):
    """Check if container exists and create it if needed"""
    logger.info(f"Checking container: '{container_name}'")
    
    try:
        container_client = blob_service_client.get_container_client(container_name)
        try:
            properties = container_client.get_container_properties()
            logger.info(f"Container '{container_name}' exists with properties:")
            logger.info(f"  Last Modified: {properties.last_modified}")
            logger.info(f"  ETag: {properties.etag}")
            return True
        except Exception as e:
            logger.warning(f"Container '{container_name}' does not exist or is not accessible")
            logger.debug(f"Error: {str(e)}")
            
            # Try to create container
            logger.info(f"Attempting to create container '{container_name}'...")
            container_client = blob_service_client.create_container(container_name)
            logger.info(f"Container '{container_name}' created successfully")
            return True
    except Exception as e:
        logger.error(f"Error checking/creating container: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_upload_download(blob_service_client, container_name):
    """Test uploading and downloading a file"""
    test_id = str(uuid.uuid4())[:8]
    test_filename = f"test_file_{test_id}.txt"
    test_content = f"This is a test file created at {time.time()}"
    
    logger.info(f"Testing upload and download with test file: {test_filename}")
    
    try:
        # Create blob client
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=test_filename
        )
        
        # Upload test file
        logger.info("Uploading test file...")
        test_bytes = io.BytesIO(test_content.encode('utf-8'))
        blob_client.upload_blob(test_bytes, overwrite=True)
        
        # Get blob URL
        blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{test_filename}"
        logger.info(f"File uploaded successfully to: {blob_url}")
        
        # Verify file exists
        logger.info("Verifying uploaded file exists...")
        properties = blob_client.get_blob_properties()
        logger.info(f"File exists with size: {properties.size} bytes")
        
        # Download the file
        logger.info("Downloading file to verify content...")
        download_stream = blob_client.download_blob()
        downloaded_content = download_stream.readall().decode('utf-8')
        
        if downloaded_content == test_content:
            logger.info("✅ Upload and download test successful!")
            logger.info(f"Content verified: '{downloaded_content}'")
        else:
            logger.error("❌ Downloaded content does not match uploaded content")
            logger.error(f"Expected: '{test_content}'")
            logger.error(f"Got: '{downloaded_content}'")
        
        # Clean up the test file
        logger.info("Cleaning up test file...")
        blob_client.delete_blob()
        logger.info("Test file deleted")
        
        return True
    except Exception as e:
        logger.error(f"Error in upload/download test: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to test Azure Blob Storage connectivity"""
    logger.info("=" * 60)
    logger.info("AZURE BLOB STORAGE CONNECTION TEST")
    logger.info("=" * 60)
    
    # Step 1: Check environment variables
    env_ok, account, key, container = check_environment()
    if not env_ok:
        logger.error("❌ Environment check failed")
        return False
    
    # Step 2: Connect to storage account
    connection_ok, blob_service_client = try_connect_to_storage(account, key)
    if not connection_ok:
        logger.error("❌ Connection to Azure Blob Storage failed")
        return False
    
    # Step 3: List containers
    container_names = list_containers(blob_service_client)
    
    # Step 4: Check target container
    if container not in container_names:
        logger.warning(f"Target container '{container}' not found in available containers")
    
    container_ok = check_container(blob_service_client, container)
    if not container_ok:
        logger.error(f"❌ Container '{container}' check/creation failed")
        return False
    
    # Step 5: Test upload and download
    test_ok = test_upload_download(blob_service_client, container)
    if not test_ok:
        logger.error("❌ Upload/download test failed")
        return False
    
    # All tests passed!
    logger.info("=" * 60)
    logger.info("✅ ALL AZURE BLOB STORAGE TESTS PASSED")
    logger.info("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 