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
        return default_value

# Map from environment variable names to Key Vault secret names
# This makes the transition easier
SECRET_MAP = {
    # MySQL settings
    "MYSQL_USER": "MYSQL-USER",
    "MYSQL_PASSWORD": "MYSQL-PASSWORD",
    "MYSQL_HOST": "MYSQL-HOST",
    "MYSQL_PORT": "MYSQL-PORT",
    "MYSQL_DB": "MYSQL-DB",
    
    # Flask settings
    "SECRET_KEY": "SECRET-KEY",
    
    # OpenAI settings
    "OPENAI_API_KEY": "OPENAI-API-KEY",
    "CHROMA_PERSIST_DIR": "CHROMA-PERSIST-DIR",
    
    # Azure Storage
    "AZURE_STORAGE_ACCOUNT": "AZURE-STORAGE-ACCOUNT",
    "AZURE_STORAGE_KEY": "AZURE-STORAGE-KEY",
    "AZURE_CONTAINER": "AZURE-CONTAINER",
    
    # Debug settings
    "DEBUG": "DEBUG",
    
    # Service URLs
    "DATA_CLEANER_URL": "DATA-CLEANER-URL",
    "FEATURE_SELECTOR_URL": "FEATURE-SELECTOR-URL",
    "ANOMALY_DETECTOR_URL": "ANOMALY-DETECTOR-URL",
    "MODEL_COORDINATOR_URL": "MODEL-COORDINATOR-URL",
    "MEDICAL_ASSISTANT_URL": "MEDICAL-ASSISTANT-URL",
    "AUGMENTATION_SERVICE_URL": "AUGMENTATION-SERVICE-URL",
    "MODEL_TRAINING_SERVICE_URL": "MODEL-TRAINING-SERVICE-URL",
    "PIPELINE_SERVICE_URL": "PIPELINE-SERVICE-URL",
    "ANOMALY_DETECTION_SERVICE_URL": "ANOMALY-DETECTION-SERVICE-URL",
    "PREDICTOR_SERVICE_URL": "PREDICTOR-SERVICE-URL",
    "TABULAR_PREDICTOR_SERVICE_URL": "TABULAR-PREDICTOR-SERVICE-URL",
    "REGRESSION_DATA_CLEANER_URL": "REGRESSION-DATA-CLEANER-URL",
    "REGRESSION_FEATURE_SELECTOR_URL": "REGRESSION-FEATURE-SELECTOR-URL",
    "REGRESSION_MODEL_COORDINATOR_URL": "REGRESSION-MODEL-COORDINATOR-URL",
    "REGRESSION_PREDICTOR_SERVICE_URL": "REGRESSION-PREDICTOR-SERVICE-URL",
    "EMBEDDING_URL": "EMBEDDING-URL",
    "VECTOR_URL": "VECTOR-URL",
    "LLM_URL": "LLM-URL",
    "SYSTEM_PROMPT": "SYSTEM-PROMPT",
}

def getenv(key, default=None):
    """
    Drop-in replacement for os.getenv that uses Key Vault instead.
    
    Args:
        key (str): The environment variable name
        default (any, optional): Default value to return if not found
        
    Returns:
        str: The secret value from Key Vault, or the default value if not found
    """
    # Map the environment variable name to the Key Vault secret name
    secret_name = SECRET_MAP.get(key, key)
    
    # Get the secret from Key Vault
    return get_secret(secret_name, default) 