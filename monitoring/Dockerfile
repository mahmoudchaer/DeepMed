FROM python:3.9-slim

WORKDIR /app

# Install Docker CLI dependencies
RUN apt-get update && \
    apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy local files
COPY . .

# Create keyvault.py with a dummy implementation
RUN echo 'import os\nimport logging\nfrom functools import lru_cache\n\n# Configure logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)\n\n# Secret cache\nsecret_cache = {}\n\n@lru_cache(maxsize=128)\ndef get_secret(secret_name, default_value=None):\n    """Get a secret, fallback to environment variable or default"""\n    # If already in cache, return it\n    if secret_name in secret_cache:\n        return secret_cache[secret_name]\n    \n    # Try to get from environment variable\n    env_value = os.getenv(secret_name)\n    if env_value:\n        secret_cache[secret_name] = env_value\n        return env_value\n    \n    logger.warning(f"Secret {secret_name} not found, using default value")\n    return default_value\n\ndef getenv(key, default=None):\n    """Drop-in replacement for os.getenv that tries Key Vault first"""\n    return get_secret(key, default)' > keyvault.py

EXPOSE 5432

CMD ["python", "app.py"] 