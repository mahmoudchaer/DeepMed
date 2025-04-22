from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import ResourceNotFoundError

# Replace with your Key Vault URL
VAULT_URL = "https://kv503n.vault.azure.net/"

# Secrets you want to verify
expected_secrets = [
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

# Set up Key Vault client
credential = DefaultAzureCredential()
client = SecretClient(vault_url=VAULT_URL, credential=credential)

print("🔍 Checking secrets in Key Vault...\n")

for secret_name in expected_secrets:
    try:
        secret = client.get_secret(secret_name)
        print(f"✅ {secret_name}: Found")
    except ResourceNotFoundError:
        print(f"❌ {secret_name}: NOT FOUND")
    except Exception as e:
        print(f"⚠️  {secret_name}: Error - {e}")
