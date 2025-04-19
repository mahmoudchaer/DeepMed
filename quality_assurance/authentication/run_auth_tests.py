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
    format='%(asctime)s - %(levelname)s - %(message)s',
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
    logger.info(f"Running {test['name']}")
    
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
        duration = end_time - start_time
        
        # Process the result
        if result.returncode == 0:
            status = "PASSED"
            logger.info(f"✓ {test['name']}: {status} ({duration:.2f}s)")
        else:
            status = "FAILED"
            logger.error(f"✗ {test['name']}: {status} ({duration:.2f}s)")
        
        # Only log stdout for failed tests or if it's short
        if status == "FAILED" or len(result.stdout) < 500:
            logger.debug(f"Output: {result.stdout.strip()}")
        
        if result.stderr:
            # Extract just the error messages from stderr
            error_lines = [line for line in result.stderr.split('\n') 
                           if 'ERROR' in line or 'Exception' in line]
            if error_lines:
                logger.error(f"Errors: {' | '.join(error_lines[:3])}")
                if len(error_lines) > 3:
                    logger.error(f"...and {len(error_lines) - 3} more errors")
        
        return {
            "name": test['name'],
            "status": status,
            "duration": duration,
            "returncode": result.returncode,
            "output": result.stdout,
            "errors": result.stderr
        }
    except Exception as e:
        logger.error(f"Failed to run {test['name']}: {str(e)}")
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
    print(f"Total Test Suites: {total_tests} | Passed: {passed_tests} | Failed: {failed_tests} | Errors: {error_tests}")
    print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")
    print("="*60)
    
    # Print details for failed tests only
    failed_results = [r for r in results if r['status'] != "PASSED"]
    if failed_results:
        print("\nFAILED TEST SUITES:")
        for result in failed_results:
            print(f"  ✗ {result['name']} - {result['status']} ({result['duration']:.2f}s)")
            
            # Print just the first few error lines
            if result['errors']:
                error_lines = [line for line in result['errors'].split('\n') 
                              if line.strip() and ('ERROR' in line or 'Exception' in line)][:3]
                if error_lines:
                    for line in error_lines:
                        print(f"    > {line.strip()}")
                    if len(error_lines) > 3:
                        print(f"    > ...and {len(error_lines) - 3} more errors")
    
    print("\nComplete logs available in the authentication directory")

if __name__ == "__main__":
    logger.info("Starting authentication tests")
    
    start_time = time.time()
    results = run_all_tests()
    end_time = time.time()
    
    total_duration = end_time - start_time
    logger.info(f"All tests completed in {total_duration:.2f}s")
    
    generate_report(results) 