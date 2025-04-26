#!/usr/bin/env python3
"""
Docker Secrets Adapter for Azure Key Vault
This script loads secrets from Azure Key Vault and sets them as environment variables.
It should be imported at the start of each Docker container's main script.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("docker_secrets")

DEFAULT_SECRETS = {
    "OPENAI-API-KEY": "MISSING_KEY_PLEASE_INSTALL_AZURE_SDK",
    "AZURESTORAGEACCOUNT": "devstoreaccount1",
    "AZURESTORAGEKEY": "PLACEHOLDER_STORAGE_KEY",  # <-- Replace secret!
    "AZURECONTAINER": "models"
}


def setup_env():
    """Load secrets from keyvault.py and set them as environment variables"""
    logger.info("Setting up environment variables from Azure Key Vault")
    
    keyvault_success = False
    try:
        # Add the parent directory to sys.path to import keyvault
        sys.path.append('/app')
        
        # Try to import keyvault
        try:
            import keyvault
            logger.info("Successfully imported keyvault module")
            keyvault_success = True
        except ImportError as e:
            logger.error(f"Failed to import keyvault module: {str(e)}")
        
        # Only proceed with Key Vault if import succeeded
        if keyvault_success:
            # Define the secrets we need
            secret_keys = [
                "SECRETKEY",
                "MYSQLUSER",
                "MYSQLPASSWORD",
                "MYSQLHOST",
                "MYSQLPORT",
                "MYSQLDB",
                "OPENAI-API-KEY",
                "AZURESTORAGEACCOUNT",
                "AZURESTORAGEKEY",
                "AZURECONTAINER",
                "DEBUG"
            ]
            
            # Load each secret and set as environment variable
            for key in secret_keys:
                value = keyvault.getenv(key)
                if value:
                    logger.info(f"✅ {key}: Found and set as environment variable")
                    os.environ[key] = value
                else:
                    logger.warning(f"⚠️ {key}: Not found in Azure Key Vault")
        
    except Exception as e:
        logger.error(f"Error setting up environment variables from Key Vault: {str(e)}")
        keyvault_success = False
    
    # Use fallback values for critical variables if not set
    # This ensures services can at least start up
    critical_keys_missing = False
    for key, default_value in DEFAULT_SECRETS.items():
        if key not in os.environ or not os.environ[key]:
            logger.warning(f"Setting default value for {key} as fallback")
            os.environ[key] = default_value
            critical_keys_missing = True
    
    # Check if existing values are available in the environment
    logger.info("Current environment variables:")
    for key in DEFAULT_SECRETS.keys():
        # Don't log the actual API key value, just whether it exists
        if key in os.environ and os.environ[key]:
            masked_value = os.environ[key][:4] + "..." if len(os.environ[key]) > 4 else "***"
            logger.info(f"✅ {key} is set to {masked_value}")
        else:
            logger.warning(f"⚠️ {key} is not set")
            critical_keys_missing = True
    
    return keyvault_success and not critical_keys_missing

# Auto-run when imported
setup_successful = setup_env()
if setup_successful:
    logger.info("Successfully set up environment variables from Azure Key Vault")
else:
    logger.warning("Failed to set up some or all environment variables from Azure Key Vault") 