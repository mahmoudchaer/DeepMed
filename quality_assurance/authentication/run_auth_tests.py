#!/usr/bin/env python3
"""
Authentication Tests Runner for DeepMed

This script runs all authentication-related tests, including:
1. Direct database interaction tests (authentication_test.py)
2. Flask route tests (flask_auth_test.py)

Usage:
python run_auth_tests.py
"""

import os
import sys
import time
import subprocess
import logging
from datetime import datetime

# Get the absolute path to the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get path to project root (two directories up)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SCRIPT_DIR, "auth_tests_runner.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("auth_tests_runner")

# Define tests to run
TESTS = [
    {
        "name": "Database Authentication Tests",
        "script": os.path.join(SCRIPT_DIR, "authentication_test.py"),
        "description": "Tests direct database interactions for user authentication"
    },
    {
        "name": "Flask Authentication Tests",
        "script": os.path.join(SCRIPT_DIR, "flask_auth_test.py"),
        "description": "Tests Flask routes for user registration and login"
    }
]

def run_test(test):
    """Run a single test script and return the result"""
    logger.info(f"Running {test['name']}: {test['description']}")
    
    try:
        # Execute the test script as a subprocess
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, test['script']],
            capture_output=True,
            text=True,
            check=False
        )
        end_time = time.time()
        
        # Process the result
        if result.returncode == 0:
            status = "PASSED"
        else:
            status = "FAILED"
            
        duration = end_time - start_time
        
        # Log the result
        logger.info(f"{test['name']}: {status} (Duration: {duration:.2f}s)")
        logger.info(f"Output: {result.stdout}")
        
        if result.stderr:
            logger.error(f"Errors: {result.stderr}")
        
        return {
            "name": test['name'],
            "status": status,
            "duration": duration,
            "returncode": result.returncode,
            "output": result.stdout,
            "errors": result.stderr
        }
    except Exception as e:
        logger.error(f"Error running {test['name']}: {str(e)}")
        return {
            "name": test['name'],
            "status": "ERROR",
            "duration": 0,
            "returncode": -1,
            "output": "",
            "errors": str(e)
        }

def run_all_tests():
    """Run all configured tests and return results"""
    results = []
    
    for test in TESTS:
        result = run_test(test)
        results.append(result)
        
    return results

def generate_report(results):
    """Generate and print a summary report of all test results"""
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['status'] == "PASSED")
    failed_tests = sum(1 for r in results if r['status'] == "FAILED")
    error_tests = sum(1 for r in results if r['status'] == "ERROR")
    
    print("\n" + "="*60)
    print(f"AUTHENTICATION TESTS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"Total Test Suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Errors: {error_tests}")
    print(f"Success Rate: {(passed_tests / total_tests * 100):.2f}%")
    print("="*60)
    
    # Print details for each test
    for result in results:
        status_symbol = "✓" if result['status'] == "PASSED" else "✗"
        print(f"\n{status_symbol} {result['name']} - {result['status']}")
        print(f"  Duration: {result['duration']:.2f}s")
        
        # Print errors if any
        if result['errors']:
            print(f"  Errors:")
            for line in result['errors'].split('\n'):
                if line.strip():
                    print(f"    {line.strip()}")
    
    print("\nComplete logs available in the quality_assurance/authentication directory.")

if __name__ == "__main__":
    logger.info("Starting authentication tests runner")
    
    start_time = time.time()
    results = run_all_tests()
    end_time = time.time()
    
    total_duration = end_time - start_time
    logger.info(f"All tests completed in {total_duration:.2f} seconds")
    
    generate_report(results) 