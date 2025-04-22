# Azure Key Vault Integration

## Overview

The application has been migrated from using `.env` files to Azure Key Vault for secret management. This improves security by:

1. Centralizing secrets management
2. Eliminating the need for sensitive information in configuration files
3. Supporting role-based access control for secrets
4. Providing audit logs for secret access
5. Enabling automatic secret rotation

## Implementation Details

### Key Vault Module

A central module `keyvault.py` has been created that provides:

- Connection to Azure Key Vault using DefaultAzureCredential
- A drop-in replacement for `os.getenv()` functionality with `keyvault.getenv()`
- LRU caching to avoid repeated calls to Key Vault
- Mapping of environment variable names to Key Vault secret names

### Authentication

The application uses Azure's DefaultAzureCredential which provides a simplified authentication experience:

- When running in Azure, it automatically uses Managed Identity
- For local development, it can use environment variables, az CLI, or Visual Studio Code authentication

## Required Secrets in Key Vault

All secrets previously in `.env` are now stored in Key Vault with the following mappings:

| Environment Variable | Key Vault Secret Name |
|----------------------|----------------------|
| MYSQL_USER | MYSQL-USER |
| MYSQL_PASSWORD | MYSQL-PASSWORD |
| MYSQL_HOST | MYSQL-HOST |
| MYSQL_PORT | MYSQL-PORT |
| MYSQL_DB | MYSQL-DB |
| SECRET_KEY | SECRET-KEY |
| OPENAI_API_KEY | OPENAI-API-KEY |
| CHROMA_PERSIST_DIR | CHROMA-PERSIST-DIR |
| AZURE_STORAGE_ACCOUNT | AZURE-STORAGE-ACCOUNT |
| AZURE_STORAGE_KEY | AZURE-STORAGE-KEY |
| AZURE_CONTAINER | AZURE-CONTAINER |
| DEBUG | DEBUG |

Plus various service URLs.

## Configuration

The Key Vault URL is configured in `keyvault.py`:

```python
# Azure Key Vault URL
VAULT_URL = "https://kv503n.vault.azure.net/"
```

## Required Packages

The following packages were added to `requirements.txt`:

```
azure-identity==1.13.0
azure-keyvault-secrets==4.6.0
```

## Local Development

For local development, you can authenticate with Azure using:

1. Environment variables
   ```
   set AZURE_TENANT_ID=your-tenant-id
   set AZURE_CLIENT_ID=your-client-id
   set AZURE_CLIENT_SECRET=your-client-secret
   ```

2. Azure CLI
   ```
   az login
   ```

3. Visual Studio Code Azure extension

## Deployment

When deployed to Azure, the application will use Managed Identity to access Key Vault. Ensure your app service or other Azure resource has a managed identity enabled and granted access to Key Vault. 