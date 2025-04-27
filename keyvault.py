from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Azure Key Vault URL
VAULT_URL = "https://kv503n.vault.azure.net/"

# Cache the secrets to avoid repeated calls to Key Vault
secret_cache = {}

@lru_cache(maxsize=128)
def get_secret(secret_name, default_value=None):
    """
    Get a secret from Azure Key Vault.
    
    Args:
        secret_name (str): The name of the secret to retrieve
        default_value (any, optional): Default value to return if the secret can't be retrieved
        
    Returns:
        str: The secret value or default value
    """
    # If already in cache, return it
    if secret_name in secret_cache:
        return secret_cache[secret_name]
    
    try:
        # Create a SecretClient using DefaultAzureCredential
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=VAULT_URL, credential=credential)
        
        # Get the secret
        secret = client.get_secret(secret_name).value
        
        # Cache it for future use
        secret_cache[secret_name] = secret
        
        return secret
    except Exception as e:
        logger.error(f"Error retrieving secret '{secret_name}' from Key Vault: {str(e)}")
        # Clear the cache for this secret to allow retries
        if secret_name in secret_cache:
            del secret_cache[secret_name]
        # Clear the lru_cache for this secret
        get_secret.cache_clear()
        return default_value

def getenv(key, default=None):
    """
    Drop-in replacement for os.getenv that uses Key Vault instead.
    
    Args:
        key (str): The environment variable name/Key Vault secret name
        default (any, optional): Default value to return if not found
        
    Returns:
        str: The secret value from Key Vault, or the default value if not found
    """
    # Get the secret directly from Key Vault since naming is consistent
    return get_secret(key, default) 