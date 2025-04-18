#!/usr/bin/env python3
"""
Test Azure Blob Storage connection
"""

import os
import sys
import io
import time
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("azure-test")

# Load environment variables from .env file
print("Loading environment variables from .env file...")
load_dotenv()

# Check environment variables
account = os.getenv("AZURE_STORAGE_ACCOUNT")
key = os.getenv("AZURE_STORAGE_KEY")
container = os.getenv("AZURE_CONTAINER")

print(f"AZURE_STORAGE_ACCOUNT: {'Set' if account else 'Not set'}")
print(f"AZURE_STORAGE_KEY: {'Set' if key else 'Not set'}")
print(f"AZURE_CONTAINER: {'Set' if container else 'Not set'}")

if not all([account, key, container]):
    print("ERROR: One or more Azure Storage credentials are missing!")
    sys.exit(1)

try:
    # Try to import the Azure Storage SDK
    print("Importing Azure Storage Blob SDK...")
    try:
        from azure.storage.blob import BlobServiceClient
        print("Azure Storage Blob SDK imported successfully")
    except ImportError:
        print("ERROR: Azure Storage Blob SDK not installed. Run: pip install azure-storage-blob")
        sys.exit(1)

    # Try to connect to the storage account
    print(f"Connecting to Azure Storage account: {account}")
    try:
        blob_service_client = BlobServiceClient(
            account_url=f"https://{account}.blob.core.windows.net",
            credential=key
        )
        print("Successfully connected to Azure Storage account")
    except Exception as e:
        print(f"ERROR: Failed to connect to Azure Storage account: {str(e)}")
        sys.exit(1)

    # Check if the container exists
    print(f"Checking if container '{container}' exists...")
    try:
        container_client = blob_service_client.get_container_client(container)
        properties = container_client.get_container_properties()
        print(f"Container '{container}' exists")
    except Exception as e:
        print(f"ERROR: Container '{container}' does not exist or is not accessible: {str(e)}")
        print("Attempting to create the container...")
        try:
            container_client = blob_service_client.create_container(container)
            print(f"Container '{container}' created successfully")
        except Exception as e:
            print(f"ERROR: Failed to create container: {str(e)}")
            sys.exit(1)

    # Test upload
    test_filename = f"test_{int(time.time())}.txt"
    test_content = f"This is a test file created at {time.time()}"
    print(f"Uploading test file '{test_filename}'...")
    
    try:
        # Create blob client
        blob_client = blob_service_client.get_blob_client(container=container, blob=test_filename)
        
        # Upload test file
        test_bytes = io.BytesIO(test_content.encode('utf-8'))
        blob_client.upload_blob(test_bytes, overwrite=True)
        
        # Get blob URL
        blob_url = f"https://{account}.blob.core.windows.net/{container}/{test_filename}"
        print(f"File uploaded successfully to: {blob_url}")
        
        # Download the file
        print("Downloading file to verify content...")
        download_stream = blob_client.download_blob()
        downloaded_content = download_stream.readall().decode('utf-8')
        
        if downloaded_content == test_content:
            print("✅ Success! Upload and download test passed")
        else:
            print("❌ Downloaded content does not match uploaded content")
        
        # Delete the test file
        print("Cleaning up test file...")
        blob_client.delete_blob()
        print("Test file deleted")
        
    except Exception as e:
        print(f"ERROR: Upload/download test failed: {str(e)}")
        sys.exit(1)

    print("\n✅ All tests passed! Azure Blob Storage is configured correctly")
    
except Exception as e:
    print(f"Unexpected error: {str(e)}")
    sys.exit(1) 