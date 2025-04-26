# Docker Container Tests

This directory contains tests for validating the Docker containers in the DeepMed project. These tests verify that each containerized service functions correctly and maintains its expected behavior when deployed.

## Prerequisites

- Python 3.8+
- Docker installed and running
- DeepMed containers built and running
- Required Python packages (see below)

## Required Packages

```
requests
pandas
numpy
rich
```

Install them with:
```bash
pip install requests pandas numpy rich
```

## Running Tests

### Run all Docker tests

```bash
python run_docker_tests.py
```

### Run a specific test

```bash
python run_docker_tests.py test_data_cleaner_docker
```

### Run tests for a specific service

```bash
python run_docker_tests.py --service data_cleaner
python run_docker_tests.py --service feature_selector
```

### Verbose output

```bash
python run_docker_tests.py -v
```

### Test against a different host

If your Docker containers are running on a different host:

```bash
python run_docker_tests.py --host 192.168.1.100
```

## Test Structure

Each Docker container has its own test file following the naming convention `test_<service_name>_docker.py`. The tests verify:

1. **Availability**: The service is up and running
2. **Health Check**: The health endpoint returns correct information
3. **Basic Functionality**: The service performs its core functions correctly
4. **Error Handling**: The service handles invalid inputs appropriately
5. **Consistency**: The service produces consistent results for the same inputs

## Available Tests

- `test_data_cleaner_docker.py`: Tests for the data cleaner service
- `test_feature_selector_docker.py`: Tests for the feature selector service

## Adding New Tests

To add tests for a new Docker service:

1. Create a new file named `test_<service_name>_docker.py`
2. Implement a test class extending `unittest.TestCase`
3. Add test methods for each aspect of functionality
4. Make sure the tests check both normal operation and error conditions
5. Update the SERVICE_CONFIG dictionary in `run_docker_tests.py` with the new service

Example configuration entry:
```python
SERVICE_CONFIG = {
    "data_cleaner": {"url": "http://localhost:5001", "env_var": "DATA_CLEANER_URL"},
    "feature_selector": {"url": "http://localhost:5002", "env_var": "FEATURE_SELECTOR_URL"},
    "new_service": {"url": "http://localhost:5003", "env_var": "NEW_SERVICE_URL"},
}
```

## Updating Docker Service URLs

Each test uses environment variables to determine the URL of the service being tested. The default pattern is:

```python
SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:<port>")
```

The environment variables are automatically set by the `run_docker_tests.py` script based on the configuration in the SERVICE_CONFIG dictionary. 