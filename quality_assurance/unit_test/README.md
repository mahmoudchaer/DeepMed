# Unit Tests

This directory contains unit tests for the DeepMed application. The tests focus on core functionality including Azure Blob Storage operations and Azure Key Vault integration.

## Setup

1. Install the test dependencies:
```bash
pip install -r requirements-test.txt
```

2. Make sure you have the main project dependencies installed (from the root requirements.txt)

## Running Tests

To run all tests:
```bash
pytest
```

To run tests with coverage report:
```bash
pytest --cov=storage --cov=keyvault
```

## Test Structure

The tests are organized as follows:

- `test_storage_and_keyvault.py`: Contains tests for Azure Blob Storage operations and Key Vault integration
  - Tests for blob upload, download, and deletion
  - Tests for secret retrieval from Key Vault
  - Tests for environment variable handling

## Mocking

The tests use mocking to avoid making actual calls to Azure services. The following are mocked:
- Azure Blob Storage client
- Azure Key Vault client
- Azure credentials

## Adding New Tests

When adding new tests:
1. Follow the existing pattern of using fixtures for mocking
2. Test both success and failure cases
3. Include clear docstrings explaining what each test verifies
4. Use appropriate assertions to verify behavior 