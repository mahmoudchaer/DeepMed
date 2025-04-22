# Azure Key Vault Integration

## Overview

The application has been migrated from using `.env` files to Azure Key Vault for secret management. This improves security by:

1. Centralizing secrets management
2. Eliminating the need for sensitive information in configuration files
3. Supporting role-based access control for secrets
4. Providing audit logs for secret access
5. Enabling automatic secret rotation

## Setting Up Azure Key Vault

### Prerequisites

- Azure Subscription
- Azure CLI installed 
- Appropriate permissions to create resources in Azure

### Create Key Vault with Azure CLI

```bash
# Login to Azure
az login

# Set variables
RESOURCE_GROUP="rg-deepmed"
LOCATION="eastus"
KEYVAULT_NAME="kv503n"  # Use your own unique name

# Create a resource group if it doesn't exist
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Key Vault
az keyvault create --name $KEYVAULT_NAME --resource-group $RESOURCE_GROUP --location $LOCATION --sku standard
```

### Add Secrets to Key Vault

```bash
# MySQL settings
az keyvault secret set --vault-name $KEYVAULT_NAME --name "MYSQL-USER" --value "root"
az keyvault secret set --vault-name $KEYVAULT_NAME --name "MYSQL-PASSWORD" --value "your-secure-password"
az keyvault secret set --vault-name $KEYVAULT_NAME --name "MYSQL-HOST" --value "localhost"
az keyvault secret set --vault-name $KEYVAULT_NAME --name "MYSQL-PORT" --value "3306"
az keyvault secret set --vault-name $KEYVAULT_NAME --name "MYSQL-DB" --value "deepmedver"

# Flask settings
az keyvault secret set --vault-name $KEYVAULT_NAME --name "SECRET-KEY" --value "your-secure-random-key"

# OpenAI API Key
az keyvault secret set --vault-name $KEYVAULT_NAME --name "OPENAI-API-KEY" --value "your-openai-api-key"

# Azure Storage
az keyvault secret set --vault-name $KEYVAULT_NAME --name "AZURE-STORAGE-ACCOUNT" --value "your-storage-account"
az keyvault secret set --vault-name $KEYVAULT_NAME --name "AZURE-STORAGE-KEY" --value "your-storage-key"
az keyvault secret set --vault-name $KEYVAULT_NAME --name "AZURE-CONTAINER" --value "your-container"
```

### Deploy App with Managed Identity

For Azure App Service deployments, use Managed Identity:

```bash
# Create App Service with Managed Identity
az appservice plan create --name plan-deepmed --resource-group $RESOURCE_GROUP --sku B1
az webapp create --name app-deepmed --resource-group $RESOURCE_GROUP --plan plan-deepmed --runtime "PYTHON:3.9"
az webapp identity assign --name app-deepmed --resource-group $RESOURCE_GROUP

# Get the managed identity principal ID
PRINCIPAL_ID=$(az webapp identity show --name app-deepmed --resource-group $RESOURCE_GROUP --query principalId --output tsv)

# Grant the managed identity access to Key Vault
az keyvault set-policy --name $KEYVAULT_NAME --resource-group $RESOURCE_GROUP --object-id $PRINCIPAL_ID --secret-permissions get list
```

## Implementation Details

### Key Vault Module

A central module `keyvault.py` has been created that provides:

- Connection to Azure Key Vault using DefaultAzureCredential
- A drop-in replacement for `os.getenv()` functionality with `keyvault.getenv()`
- LRU caching to avoid repeated calls to Key Vault

### Authentication

The application uses Azure's DefaultAzureCredential which provides a simplified authentication experience:

- When running in Azure, it automatically uses Managed Identity
- For local development, it can use environment variables, az CLI, or Visual Studio Code authentication

## Required Secrets in Key Vault

Only essential secrets and sensitive credentials are stored in Key Vault with the following mappings:

| Environment Variable | Key Vault Secret Name |
|----------------------|----------------------|
| MYSQL_USER | MYSQL-USER |
| MYSQL_PASSWORD | MYSQL-PASSWORD |
| MYSQL_HOST | MYSQL-HOST |
| MYSQL_PORT | MYSQL-PORT |
| MYSQL_DB | MYSQL-DB |
| SECRET_KEY | SECRET-KEY |
| OPENAI_API_KEY | OPENAI-API-KEY |
| AZURE_STORAGE_ACCOUNT | AZURE-STORAGE-ACCOUNT |
| AZURE_STORAGE_KEY | AZURE-STORAGE-KEY |
| AZURE_CONTAINER | AZURE-CONTAINER |
| TEST_USER_EMAIL | TEST-USER-EMAIL |
| TEST_USER_PASSWORD | TEST-USER-PASSWORD |

Service URLs are now hardcoded in the application with sensible default values.

## Configuration

The Key Vault URL is configured in `keyvault.py`:

```python
# Azure Key Vault URL
VAULT_URL = "https://kv503n.vault.azure.net/"
```

## Local Development

For local development, you can authenticate with Azure using:

1. Environment variables
   ```bash
   export AZURE_TENANT_ID=your-tenant-id
   export AZURE_CLIENT_ID=your-client-id
   export AZURE_CLIENT_SECRET=your-client-secret
   ```
   
   In Windows PowerShell:
   ```powershell
   $env:AZURE_TENANT_ID="your-tenant-id"
   $env:AZURE_CLIENT_ID="your-client-id"
   $env:AZURE_CLIENT_SECRET="your-client-secret"
   ```

2. Azure CLI
   ```bash
   az login
   ```

3. Visual Studio Code Azure extension

## Docker Deployment

For Docker deployment, the necessary Azure authentication environment variables are passed to the containers, and the keyvault.py module is mounted as a volume to ensure consistent access to secrets. 