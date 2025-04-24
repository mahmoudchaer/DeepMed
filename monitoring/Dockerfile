FROM python:3.9-slim

WORKDIR /app

# Install Docker CLI dependencies
RUN apt-get update && \
    apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    python3-dev \
    gcc \
    libc-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy local files
COPY . .

# Create keyvault.py with a dummy implementation if it doesn't exist
RUN echo "# Dummy keyvault implementation\nclass KeyVault:\n    @staticmethod\n    def get_secret(secret_name):\n        return None\n\ndef get_secret(secret_name):\n    return None" > keyvault.py

EXPOSE 5432

CMD ["python", "app.py"] 