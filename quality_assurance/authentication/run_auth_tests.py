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
import re
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

def extract_test_results(output):
    """Extract test result counts from the output"""
    # Look for the summary line with the format: "Total: X | Passed: Y | Failed: Z | Success: W%"
    summary_match = re.search(r"Total: (\d+) \| Passed: (\d+) \| Failed: (\d+)", output)
    if summary_match:
        total = int(summary_match.group(1))
        passed = int(summary_match.group(2))
        failed = int(summary_match.group(3))
        return {
            "total": total,
            "passed": passed,
            "failed": failed
        }
    return None

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
        
        # Extract test result counts from output
        test_results = extract_test_results(result.stdout)
        
        # Process the result
        if test_results:
            # Consider the test suite "PASSED" only if no individual tests failed
            status = "PASSED" if test_results["failed"] == 0 else "FAILED"
            count_info = f"({test_results['passed']}/{test_results['total']} tests passed)"
        else:
            # Fallback to just checking the return code
            status = "PASSED" if result.returncode == 0 else "FAILED"
            count_info = ""
        
        if status == "PASSED":
            logger.info(f"✓ {test['name']}: {status} {count_info} ({duration:.2f}s)")
        else:
            logger.error(f"✗ {test['name']}: {status} {count_info} ({duration:.2f}s)")
        
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
            "errors": result.stderr,
            "test_results": test_results
        }
    except Exception as e:
        logger.error(f"Failed to run {test['name']}: {str(e)}")
        return {
            "name": test['name'],
            "status": "ERROR",
            "duration": 0,
            "returncode": -1,
            "output": "",
            "errors": str(e),
            "test_results": None
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
    
    # Count total tests across all test suites
    individual_tests = {
        "total": sum(r.get('test_results', {}).get('total', 0) for r in results if r.get('test_results')),
        "passed": sum(r.get('test_results', {}).get('passed', 0) for r in results if r.get('test_results')),
        "failed": sum(r.get('test_results', {}).get('failed', 0) for r in results if r.get('test_results'))
    }
    
    print("\n" + "="*60)
    print(f"AUTHENTICATION TESTS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"Test Suites: {total_tests} | Passed: {passed_tests} | Failed: {failed_tests} | Errors: {error_tests}")
    
    if individual_tests["total"] > 0:
        print(f"Individual Tests: {individual_tests['total']} | Passed: {individual_tests['passed']} | Failed: {individual_tests['failed']}")
        print(f"Success Rate: {(individual_tests['passed'] / individual_tests['total'] * 100):.1f}%")
    else:
        print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")
    
    print("="*60)
    
    # Print details for failed tests only
    failed_results = [r for r in results if r['status'] != "PASSED"]
    if failed_results:
        print("\nFAILED TEST SUITES:")
        for result in failed_results:
            print(f"  ✗ {result['name']} - {result['status']} ({result['duration']:.2f}s)")
            
            # Extract failed test names from output
            failed_tests = re.findall(r"✗ ([^:]+): FAILED", result['output'])
            if failed_tests:
                print(f"    Failed tests: {', '.join(failed_tests[:5])}")
                if len(failed_tests) > 5:
                    print(f"    ...and {len(failed_tests) - 5} more")
            
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