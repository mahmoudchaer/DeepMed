# DeepMed Health Check

This tool checks the health of all DeepMed Docker services by testing their health endpoints. It's designed to work both locally and as part of GitHub Actions workflows.

## Features

- Tests health endpoints for all DeepMed services (tabular data, image processing, chatbot, and monitoring)
- GitHub Actions integration for automated health checks
- Environment-specific configurations for dev, staging, and production
- Parallel health checking for faster results
- Color-coded status output
- Supports exporting results to JSON for further analysis

## Project Structure

```
quality_assurance/
├── health_check.py         # Main health check script
├── requirements.txt        # Python dependencies
├── config/
│   └── services-template.json  # Template for service configurations
└── scripts/
    └── generate_service_config.py  # Script to generate env-specific configs
```

## Local Usage

### Prerequisites

- Python 3.7+
- Required packages (see requirements.txt)

### Installation

```bash
pip install -r quality_assurance/requirements.txt
```

### Basic usage

```bash
python quality_assurance/health_check.py
```

### Command line options

```bash
python quality_assurance/health_check.py --timeout 5 --workers 20 --verbose --output results.json
```

Options:
- `--timeout`: Request timeout in seconds (default: 2)
- `--workers`: Maximum number of parallel workers (default: 10)
- `--verbose`: Show detailed progress during checks
- `--output`: Save results to a JSON file
- `--config`: Path to services configuration JSON file
- `--host`: Host override (replaces localhost in service URLs)
- `--github-actions`: Format output for GitHub Actions

## GitHub Actions Integration

The health check is configured to run automatically in GitHub Actions:

- Every 4 hours on a schedule
- On pushes to main/master/develop branches when specific directories are modified
- Manually via workflow dispatch with environment selection

### Workflow Configuration

The workflow is defined in `.github/workflows/health-check.yml` and includes:

1. Environment selection (dev, staging, prod)
2. Health check execution
3. Results parsing
4. Notifications on failure (Slack integration)

### Running manually

You can trigger the health check workflow manually from the GitHub Actions tab:

1. Go to the Actions tab in your GitHub repository
2. Select "Docker Services Health Check" workflow
3. Click "Run workflow"
4. Select the environment (dev, staging, prod)
5. Optionally adjust the timeout
6. Click "Run workflow"

### Viewing results

Health check results are available in the GitHub Actions logs and as a downloadable artifact.

## Environment Configuration

The script supports different environments through:

1. Host overrides: `--host` parameter replaces "localhost" in service URLs
2. Configuration files: Generate environment-specific service configurations

### Generating environment configs

Use the included script to generate environment-specific configurations:

```bash
python quality_assurance/scripts/generate_service_config.py --env staging --output staging-services.json
```

Then use the generated config:

```bash
python quality_assurance/health_check.py --config staging-services.json
```

## Extending

### Adding new services

To add new services to the health check:

1. Edit `config/services-template.json` to include the new services
2. Regenerate environment-specific configurations if needed

### Customizing for different environments

Modify the `scripts/generate_service_config.py` script to add environment-specific customizations:

```python
# Environment-specific customizations
if env == "staging":
    # Example: disable certain services in staging
    if "Experimental Services" in config:
        del config["Experimental Services"]
```

## Troubleshooting

### Common issues

1. **Connection errors**: Ensure services are running and accessible from the runner
2. **Timeout errors**: Increase timeout value for slower services
3. **GitHub Actions failure**: Check workflow logs for detailed error information

### Workflow failures

If the GitHub Actions workflow fails:

1. Check the workflow run logs for error details
2. Verify service configurations match the environment
3. Ensure proper network connectivity between GitHub Actions and your services 