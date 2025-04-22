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
logger = logging.getLogger("docker-secrets")

def setup_env():
    """Load secrets from keyvault.py and set them as environment variables"""
    logger.info("Setting up environment variables from Azure Key Vault")
    
    try:
        # Add the parent directory to sys.path to import keyvault
        sys.path.append('/app')
        
        # Try to import keyvault
        try:
            import keyvault
            logger.info("Successfully imported keyvault module")
        except ImportError as e:
            logger.error(f"Failed to import keyvault module: {str(e)}")
            return False
        
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
        
        return True
    
    except Exception as e:
        logger.error(f"Error setting up environment variables: {str(e)}")
        return False

# Auto-run when imported
setup_successful = setup_env()
if setup_successful:
    logger.info("Successfully set up environment variables from Azure Key Vault")
else:
    logger.warning("Failed to set up some or all environment variables from Azure Key Vault") 