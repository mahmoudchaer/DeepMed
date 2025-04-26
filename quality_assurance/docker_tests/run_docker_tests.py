#!/usr/bin/env python3

import os
import sys
import unittest
import argparse
import logging
from rich.console import Console
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Service configuration
SERVICE_CONFIG = {
    "data_cleaner": {"url": "http://localhost:5001", "env_var": "DATA_CLEANER_URL"},
    "feature_selector": {"url": "http://localhost:5002", "env_var": "FEATURE_SELECTOR_URL"},
    # Add more services as they are tested
}

def setup_environment(host=None):
    """Set up environment variables for services."""
    for service_name, config in SERVICE_CONFIG.items():
        if host:
            service_url = config["url"].replace("localhost", host)
        else:
            service_url = config["url"]
        
        os.environ[config["env_var"]] = service_url
        logger.info(f"Set {config['env_var']}={service_url}")

def discover_and_run_tests(specific_test=None, verbose=False):
    """Discover and run all Docker tests or a specific test."""
    console = Console()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If a specific test is specified, run only that test
    if specific_test:
        if not specific_test.endswith('.py'):
            specific_test += '.py'
        
        test_file = os.path.join(script_dir, specific_test)
        if not os.path.exists(test_file):
            console.print(f"[red]Test file not found: {test_file}[/red]")
            return False
        
        test_module = specific_test[:-3]  # Remove .py extension
        console.print(f"[yellow]Running specific test: {test_module}[/yellow]")
        
        # Add the current directory to sys.path to import the module
        sys.path.insert(0, script_dir)
        try:
            __import__(test_module)
            suite = unittest.defaultTestLoader.loadTestsFromName(test_module)
        except ImportError as e:
            console.print(f"[red]Error importing {test_module}: {str(e)}[/red]")
            return False
    else:
        # Discover all test_*.py files in the current directory
        console.print("[yellow]Discovering all Docker tests...[/yellow]")
        suite = unittest.defaultTestLoader.discover(script_dir, pattern="test_*.py")
    
    # Run the tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Display results in a nice table
    table = Table(title="Docker Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")
    
    table.add_row("Total Tests", str(result.testsRun))
    table.add_row("Passed", str(result.testsRun - len(result.errors) - len(result.failures)))
    table.add_row("Failed", str(len(result.failures)), style="red" if result.failures else "green")
    table.add_row("Errors", str(len(result.errors)), style="red" if result.errors else "green")
    
    console.print(table)
    
    # Return True if all tests passed
    return len(result.failures) == 0 and len(result.errors) == 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Docker container tests")
    
    parser.add_argument("test", nargs="?", help="Specific test to run (without .py extension)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--host", help="Override host for all services (default: localhost)")
    parser.add_argument("--service", help="Run tests only for a specific service")
    
    args = parser.parse_args()
    
    # Set up environment variables for service URLs
    setup_environment(args.host)
    
    # If service is specified, find test files for that service
    specific_test = args.test
    if args.service and not specific_test:
        # Find test files for the specified service
        service_name = args.service.lower()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_files = [f for f in os.listdir(script_dir) if f.startswith(f"test_{service_name}")]
        
        if not test_files:
            logger.error(f"No test files found for service: {service_name}")
            return 1
        
        logger.info(f"Found {len(test_files)} test file(s) for service {service_name}: {', '.join(test_files)}")
        
        # If there's just one, use it directly
        if len(test_files) == 1:
            specific_test = test_files[0]
    
    success = discover_and_run_tests(specific_test, args.verbose)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 