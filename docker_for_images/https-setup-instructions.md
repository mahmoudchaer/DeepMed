# Setting Up HTTPS for the Pipeline Service

This document explains how to enable HTTPS for the pipeline service in your DeepMed application.

## 1. Files Modified

The following files have been modified:

- `docker_for_images/pipeline/app.py` - Added SSL support
- `docker_for_images/pipeline/Dockerfile` - Added certificate generation
- `docker_for_images/pipeline/generate_cert.py` - Script to generate self-signed certificates
- `docker_for_images/pipeline/requirements.txt` - Added cryptography package
- `app_images.py` - Updated to use HTTPS when connecting to the pipeline service

## 2. Deployment Instructions

### Option 1: Rebuild Docker Images

1. Rebuild your pipeline service Docker image:

```bash
# Navigate to your project folder
cd DeepMed

# Rebuild the pipeline service
docker-compose build pipeline-service
docker-compose up -d pipeline-service
```

### Option 2: Manual Setup

If you prefer to manually apply these changes on your running container:

1. Copy the certificate generation script to your container:

```bash
docker cp docker_for_images/pipeline/generate_cert.py your-pipeline-container:/app/
```

2. Install the cryptography package:

```bash
docker exec your-pipeline-container pip install cryptography
```

3. Generate certificates inside the container:

```bash
docker exec your-pipeline-container python /app/generate_cert.py
```

4. Restart the pipeline service:

```bash
docker restart your-pipeline-container
```

## 3. Troubleshooting

### Verify HTTPS is Working

You can check if HTTPS is working by accessing the health endpoint:

```bash
curl -k https://localhost:5025/health
```

The `-k` option allows insecure connections since we're using a self-signed certificate.

### Common Issues

1. **Certificate Issues**: If you see SSL certificate errors, ensure the certificates are properly generated in the `/app/certs/` directory in the container.

2. **Connection Refused**: Make sure the port 5025 is exposed with HTTPS enabled.

3. **Mixed Content Warnings**: If your main application uses HTTPS but calls some services over HTTP, browsers may block these requests. All services should use the same protocol.

## 4. Production Considerations

For a production environment, consider:

1. Using proper CA-verified certificates instead of self-signed ones
2. Using a reverse proxy like Nginx to handle SSL termination
3. Implementing proper certificate renewal processes

## 5. Reverting Changes

If you need to revert to HTTP:

1. In `app_images.py`, change the URL back to HTTP:
   ```python
   PIPELINE_SERVICE_URL = 'http://localhost:5025'
   ```

2. In your Docker container, set the environment variable:
   ```bash
   docker exec -e USE_HTTPS=false your-pipeline-container
   ```

3. Restart the container:
   ```bash
   docker restart your-pipeline-container
   ``` 