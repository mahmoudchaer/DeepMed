#!/usr/bin/env python3
"""
Flask Authentication Quality Assurance Script for DeepMed

This script tests the Flask authentication routes by:
1. Creating a test client for the Flask application
2. Testing registration and login routes with various scenarios
3. Testing authentication requirements for protected routes

Usage:
python flask_auth_test.py
"""

import os
import sys
import random
import string
import json
import time
from datetime import datetime
import logging
from dotenv import load_dotenv
import urllib.parse

# Get the absolute path to the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get path to project root (two directories up)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# Add project root to path to import app and db modules
sys.path.append(PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SCRIPT_DIR, "flask_auth_test_results.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("flask_auth_test")

# Load environment variables from project root .env
env_path = os.path.join(PROJECT_ROOT, '.env')
logger.info(f"Loading .env from: {env_path}")
load_dotenv(env_path)

# Import app and db after environment variables are loaded
try:
    from app_api import app, db
    from db.users import User
except ImportError as e:
    logger.error(f"Import error: {str(e)}")
    sys.exit(1)

# Test results storage
test_results = {
    "total_tests": 0,
    "passed_tests": 0,
    "failed_tests": 0,
    "test_details": []
}

def record_test_result(test_name, passed, details=None):
    """Record the result of a test"""
    global test_results
    
    result = "PASSED" if passed else "FAILED"
    
    # Log at appropriate level
    if passed:
        logger.debug(f"✓ {test_name}: {result}")
        if details:
            logger.debug(f"  Details: {details}")
    else:
        logger.error(f"✗ {test_name}: {result}")
        if details:
            logger.error(f"  Details: {details}")
    
    test_results["total_tests"] += 1
    if passed:
        test_results["passed_tests"] += 1
    else:
        test_results["failed_tests"] += 1
    
    test_results["test_details"].append({
        "test_name": test_name,
        "result": result,
        "details": details,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def generate_random_credentials():
    """Generate random email and password for testing"""
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    email = f"test_user_{random_string}@example.com"
    password = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    name = f"Test User {random_string}"
    
    return {
        "email": email,
        "password": password,
        "name": name
    }

class FlaskAuthTester:
    """Class to test Flask authentication routes"""
    
    def __init__(self):
        """Initialize Flask test client"""
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
        
        # Patch app routes to handle missing endpoints during testing
        self._patch_flask_app()
        
        # Create test client
        self.client = app.test_client()
        
        # Store created test users for cleanup
        self.test_users = []
        
        logger.info("Flask test client initialized")
    
    def _patch_flask_app(self):
        """Patch Flask app to handle redirects to non-existent endpoints during testing"""
        # Add a dummy training route to handle redirects
        @app.route('/training')
        def dummy_training():
            return "Dummy training page for testing"
        
        # Add a dummy welcome route if it doesn't exist
        if not app.url_map.is_endpoint_expecting('welcome'):
            @app.route('/welcome')
            def dummy_welcome():
                return "Dummy welcome page for testing"
        
        logger.debug("Added dummy routes for testing")
    
    def register_user(self, user_data):
        """Test user registration"""
        try:
            # Send POST request to register route
            response = self.client.post('/register', data={
                'name': user_data['name'],
                'email': user_data['email'],
                'password': user_data['password']
            }, follow_redirects=True)
            
            # Check response
            if response.status_code == 200 and b'Account created successfully' in response.data:
                logger.debug(f"Registered: {user_data['email']}")
                self.test_users.append(user_data['email'])
                return True, "Registration successful"
            else:
                logger.warning(f"Registration failed: {user_data['email']}")
                return False, f"Status: {response.status_code}"
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            return False, str(e)
    
    def login_user(self, email, password):
        """Test user login"""
        try:
            # Send POST request to login route
            response = self.client.post('/login', data={
                'email': email,
                'password': password
            }, follow_redirects=False)  # Changed to not follow redirects to avoid training endpoint
            
            # First check if we got a redirect (302) which indicates successful login
            if response.status_code == 302:
                # Successful login should redirect
                logger.debug(f"Login successful: {email}")
                return True, "Login successful"
            
            # If we got a 200 but stayed on login page, login failed
            elif response.status_code == 200 and b'Login' in response.data:
                logger.debug(f"Login failed: {email}")
                return False, "Invalid credentials"
            else:
                logger.warning(f"Unexpected login response: {response.status_code}")
                return False, f"Status: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return False, str(e)
    
    def logout_user(self):
        """Test user logout"""
        try:
            # Send POST request to logout route instead of GET
            response = self.client.post('/logout', follow_redirects=False)
            
            # Successful logout should redirect to login page
            if response.status_code == 302 or response.status_code == 303:
                logger.debug("Logout successful")
                return True, "Logout successful"
            else:
                logger.warning(f"Logout failed: {response.status_code}")
                return False, f"Status: {response.status_code}"
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return False, str(e)
    
    def access_protected_route(self):
        """Test accessing a protected route"""
        try:
            # Try to access a protected route
            response = self.client.get('/welcome', follow_redirects=False)
            
            # If not logged in, should redirect to login page (302)
            if not self.is_authenticated():
                if response.status_code == 302 and '/login' in response.location:
                    logger.debug("Protected route redirected to login")
                    return True, "Correctly redirected to login"
                else:
                    logger.warning(f"No login redirect: {response.status_code}")
                    return False, f"Status: {response.status_code}"
            # If logged in, should be able to access the protected route (200)
            else:
                if response.status_code == 200:
                    logger.debug("Accessed protected route when authenticated")
                    return True, "Access successful"
                else:
                    logger.warning(f"Protected route access failed: {response.status_code}")
                    return False, f"Status: {response.status_code}"
        except Exception as e:
            logger.error(f"Protected route access error: {str(e)}")
            return False, str(e)
    
    def is_authenticated(self):
        """Check if client is authenticated"""
        try:
            # Try to access a route that requires authentication
            response = self.client.get('/welcome', follow_redirects=False)
            
            # If not redirected to login, user is authenticated
            return response.status_code != 302
        except Exception as e:
            logger.error(f"Auth check error: {str(e)}")
            return False
    
    def clean_up_test_users(self):
        """Clean up test users created during testing"""
        with app.app_context():
            try:
                # Find and delete all test users with synchronize_session='fetch' to fix the LIKE operator error
                deleted = db.session.query(User).filter(
                    User.email.like('test_user_%@example.com')
                ).delete(synchronize_session='fetch')
                
                db.session.commit()
                logger.info(f"Cleaned up {deleted} test user(s)")
                return True, deleted
            except Exception as e:
                db.session.rollback()
                logger.error(f"Cleanup error: {str(e)}")
                return False, str(e)

def run_tests():
    """Run all Flask authentication tests"""
    tester = None
    try:
        # Initialize Flask auth tester
        tester = FlaskAuthTester()
        
        # Generate random credentials for testing
        test_creds1 = generate_random_credentials()
        test_creds2 = generate_random_credentials()
        
        logger.info("Starting Flask authentication tests")
        
        # Test 1: Access protected route when not logged in
        test_name = "Access Protected Route When Not Logged In"
        success, message = tester.access_protected_route()
        record_test_result(test_name, success, message)
        
        # Test 2: Register a new user
        test_name = "Register New User"
        success, message = tester.register_user(test_creds1)
        record_test_result(test_name, success, message)
        
        # Test 3: Register with existing email
        test_name = "Register with Existing Email"
        success, message = tester.register_user(test_creds1)
        # This should fail because the email already exists
        record_test_result(test_name, not success, message)
        
        # Test 4: Login with valid credentials
        test_name = "Login with Valid Credentials"
        success, message = tester.login_user(test_creds1["email"], test_creds1["password"])
        record_test_result(test_name, success, message)
        
        # Test 5: Access protected route when logged in
        test_name = "Access Protected Route When Logged In"
        success, message = tester.access_protected_route()
        record_test_result(test_name, success, message)
        
        # Test 6: Logout user
        test_name = "Logout User"
        success, message = tester.logout_user()
        record_test_result(test_name, success, message)
        
        # Test 7: Login with invalid password
        test_name = "Login with Invalid Password"
        success, message = tester.login_user(test_creds1["email"], "wrong_password")
        # This should fail because the password is incorrect
        record_test_result(test_name, not success, message)
        
        # Test 8: Login with non-existent user
        test_name = "Login with Non-existent User"
        success, message = tester.login_user("nonexistent@example.com", "password")
        # This should fail because the user doesn't exist
        record_test_result(test_name, not success, message)
        
        # Test 9: Register another user
        test_name = "Register Second User"
        success, message = tester.register_user(test_creds2)
        record_test_result(test_name, success, message)
        
        # Test 10: Login with second user
        test_name = "Login with Second User"
        success, message = tester.login_user(test_creds2["email"], test_creds2["password"])
        record_test_result(test_name, success, message)
        
        # Test 11: Logout second user
        test_name = "Logout Second User"
        success, message = tester.logout_user()
        record_test_result(test_name, success, message)
        
        # Final Test: Clean up test users
        test_name = "Clean Up Test Users"
        success, message = tester.clean_up_test_users()
        record_test_result(test_name, success, f"Deleted {message} test users")
        
    except Exception as e:
        logger.error(f"Test error: {str(e)}")

def print_summary():
    """Print test summary"""
    passed = test_results['passed_tests']
    failed = test_results['failed_tests']
    total = test_results['total_tests']
    
    print("\n" + "="*50)
    print("FLASK AUTHENTICATION TEST SUMMARY")
    print("="*50)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Success: {(passed / total * 100):.1f}%")
    
    # Print details of failed tests only
    if failed > 0:
        print("\nFAILED TESTS:")
        for test in test_results['test_details']:
            if test['result'] == "FAILED":
                print(f"  ✗ {test['test_name']}: {test['details']}")
    
    print("="*50)
    print(f"Complete log: {os.path.join(SCRIPT_DIR, 'flask_auth_test_results.log')}")

if __name__ == "__main__":
    start_time = time.time()
    logger.info("Starting Flask authentication tests")
    
    try:
        with app.app_context():
            run_tests()
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Tests completed in {duration:.2f}s")
    
    print_summary() 