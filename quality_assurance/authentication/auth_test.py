#!/usr/bin/env python3
"""
Authentication Quality Assurance Script for DeepMed

This script tests the authentication functionality by:
1. Creating new users with random credentials
2. Testing login with those credentials
3. Testing various edge cases and error conditions
4. Cleaning up test users after the tests

Usage:
python auth_test.py
"""

import os
import sys
import random
import string
import time
import re
from datetime import datetime
import logging
import urllib.parse
import requests
from pathlib import Path

# Get the absolute path to the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get path to project root (two directories up)
PROJECT_ROOT = Path(__file__).parent.parent.parent
# Add project root to path to import modules
sys.path.append(str(PROJECT_ROOT))

# Import keyvault module
import keyvault

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SCRIPT_DIR, "auth_test_results.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("auth_test")

# Import db models
try:
    from db.users import db, User
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.exc import IntegrityError, SQLAlchemyError
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
    password = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=12))
    name = f"Test User {random_string}"
    
    return {
        "email": email,
        "password": password,
        "name": name,
        "first_name": name.split(' ')[0],
        "last_name": name.split(' ')[-1]
    }

class DatabaseTester:
    """Class to handle database authentication tests"""
    
    def __init__(self):
        """Initialize database connection"""
        # Get database connection parameters from Key Vault
        mysql_user = keyvault.getenv('MYSQL_USER')
        mysql_password = keyvault.getenv('MYSQL_PASSWORD')
        mysql_host = keyvault.getenv('MYSQL_HOST')
        mysql_port = keyvault.getenv('MYSQL_PORT')
        mysql_db = keyvault.getenv('MYSQL_DB')
        
        # Log DB connection info (without password)
        logger.info(f"Connecting to DB: {mysql_user}@{mysql_host}:{mysql_port}/{mysql_db}")
        
        # URL encode the password to handle special characters
        encoded_password = urllib.parse.quote_plus(mysql_password)
        
        # Create database connection string
        db_uri = f'mysql+pymysql://{mysql_user}:{encoded_password}@{mysql_host}:{mysql_port}/{mysql_db}'
        
        try:
            # Create engine and session
            self.engine = create_engine(db_uri)
            self.Session = sessionmaker(bind=self.engine)
            self.session = self.Session()
            logger.info("DB connection established")
            self.test_users = []
        except Exception as e:
            logger.error(f"DB connection failed: {str(e)}")
            raise
    
    def create_user(self, user_data):
        """Create a new user in the database"""
        try:
            # Create new user object
            new_user = User(
                email=user_data["email"],
                first_name=user_data["first_name"],
                last_name=user_data["last_name"]
            )
            new_user.set_password(user_data["password"])
            
            # Add to database and commit
            self.session.add(new_user)
            self.session.commit()
            
            # Store test user for cleanup
            self.test_users.append(user_data["email"])
            
            logger.debug(f"Created user: {user_data['email']}")
            return True, new_user.id
        except IntegrityError:
            self.session.rollback()
            logger.warning(f"User already exists: {user_data['email']}")
            return False, "User already exists"
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating user: {str(e)}")
            return False, str(e)
    
    def verify_login(self, email, password):
        """Verify login credentials"""
        try:
            # Find user by email
            user = self.session.query(User).filter_by(email=email).first()
            
            # Check if user exists and password is correct
            if user and user.check_password(password):
                logger.debug(f"Login successful: {email}")
                return True, user.id
            
            # User not found or password incorrect
            if not user:
                logger.debug(f"User not found: {email}")
                return False, "User not found"
            else:
                logger.debug(f"Invalid password: {email}")
                return False, "Invalid password"
        except Exception as e:
            logger.error(f"Login verification error: {str(e)}")
            return False, str(e)
    
    def get_user(self, email):
        """Get user details by email"""
        try:
            user = self.session.query(User).filter_by(email=email).first()
            if user:
                return True, {
                    "id": user.id,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "created_at": user.created_at
                }
            else:
                return False, "User not found"
        except Exception as e:
            logger.error(f"Error getting user: {str(e)}")
            return False, str(e)
    
    def update_user(self, email, updates):
        """Update user details"""
        try:
            user = self.session.query(User).filter_by(email=email).first()
            if not user:
                return False, "User not found"
            
            # Apply updates
            for key, value in updates.items():
                if key == 'password':
                    user.set_password(value)
                elif hasattr(user, key):
                    setattr(user, key, value)
            
            self.session.commit()
            logger.debug(f"Updated user: {email}")
            return True, "User updated"
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating user: {str(e)}")
            return False, str(e)
    
    def delete_user(self, email):
        """Delete a user by email"""
        try:
            user = self.session.query(User).filter_by(email=email).first()
            if not user:
                return False, "User not found"
            
            self.session.delete(user)
            self.session.commit()
            
            # Remove from test users list
            if email in self.test_users:
                self.test_users.remove(email)
                
            logger.debug(f"Deleted user: {email}")
            return True, "User deleted"
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting user: {str(e)}")
            return False, str(e)
    
    def clean_up_test_users(self):
        """Clean up test users created during testing"""
        try:
            # Find and delete all test users with synchronize_session='fetch' to fix the LIKE operator error
            deleted = self.session.query(User).filter(
                User.email.like('test_user_%@example.com')
            ).delete(synchronize_session='fetch')
            
            self.session.commit()
            self.test_users = []
            logger.info(f"Cleaned up {deleted} test user(s)")
            return True, deleted
        except Exception as e:
            self.session.rollback()
            logger.error(f"Cleanup error: {str(e)}")
            return False, str(e)
    
    def close(self):
        """Close database connection"""
        try:
            self.session.close()
            logger.debug("DB connection closed")
        except Exception as e:
            logger.error(f"Error closing DB connection: {str(e)}")

def run_tests():
    """Run all authentication tests"""
    tester = None
    try:
        # Initialize database tester
        tester = DatabaseTester()
        
        # Generate random credentials for testing
        test_creds1 = generate_random_credentials()
        test_creds2 = generate_random_credentials()
        
        logger.info("Starting authentication tests")
        
        # Test 1: Create a new user
        test_name = "Create New User"
        success, user_id = tester.create_user(test_creds1)
        record_test_result(test_name, success, f"User ID: {user_id if success else None}")
        
        # Test 2: Create a user with existing email
        test_name = "Create User with Existing Email"
        success, message = tester.create_user(test_creds1)
        # This should fail (return False) because the email already exists
        record_test_result(test_name, not success, message)
        
        # Test 3: Login with valid credentials
        test_name = "Login with Valid Credentials"
        success, message = tester.verify_login(test_creds1["email"], test_creds1["password"])
        record_test_result(test_name, success, message)
        
        # Test 4: Login with invalid password
        test_name = "Login with Invalid Password"
        success, message = tester.verify_login(test_creds1["email"], "wrong_password")
        # This should fail (return False) because the password is incorrect
        record_test_result(test_name, not success, message)
        
        # Test 5: Login with non-existent user
        test_name = "Login with Non-existent User"
        success, message = tester.verify_login("nonexistent@example.com", "password")
        # This should fail (return False) because the user doesn't exist
        record_test_result(test_name, not success, message)
        
        # Test 6: Create a second user
        test_name = "Create Second User"
        success, user_id = tester.create_user(test_creds2)
        record_test_result(test_name, success, f"User ID: {user_id if success else None}")
        
        # Test 7: Login with second user
        test_name = "Login with Second User"
        success, message = tester.verify_login(test_creds2["email"], test_creds2["password"])
        record_test_result(test_name, success, message)
        
        # Test 8: Get user details
        test_name = "Get User Details"
        success, details = tester.get_user(test_creds1["email"])
        record_test_result(test_name, success, "User details retrieved" if success else details)
        
        # Test 9: Update user
        test_name = "Update User Details"
        update_data = {
            "first_name": "Updated",
            "last_name": "Name"
        }
        success, message = tester.update_user(test_creds1["email"], update_data)
        record_test_result(test_name, success, message)
        
        # Test 10: Verify update
        test_name = "Verify User Update"
        success, details = tester.get_user(test_creds1["email"])
        update_verified = success and details.get("first_name") == "Updated" and details.get("last_name") == "Name"
        record_test_result(test_name, update_verified, "Update verified" if update_verified else "Update failed")
        
        # Test 11: Update password
        test_name = "Update User Password"
        new_password = "NewPassword123!"
        success, message = tester.update_user(test_creds1["email"], {"password": new_password})
        record_test_result(test_name, success, message)
        
        # Test 12: Verify new password
        test_name = "Verify Password Update"
        success, message = tester.verify_login(test_creds1["email"], new_password)
        record_test_result(test_name, success, message)
        
        # Test 13: Delete user
        test_name = "Delete User"
        success, message = tester.delete_user(test_creds1["email"])
        record_test_result(test_name, success, message)
        
        # Test 14: Verify user deletion
        test_name = "Verify User Deletion"
        success, details = tester.get_user(test_creds1["email"])
        record_test_result(test_name, not success, "User successfully deleted" if not success else "User still exists")
        
        # Test 15: Password handling - empty password
        test_name = "Create User with Empty Password"
        empty_pass_creds = generate_random_credentials()
        empty_pass_creds["password"] = ""
        success, message = tester.create_user(empty_pass_creds)
        # Should fail because password is empty
        record_test_result(test_name, not success, message)
        
        # Test 16: Email validation - invalid email format
        test_name = "Create User with Invalid Email"
        invalid_email_creds = generate_random_credentials()
        invalid_email_creds["email"] = "not_an_email"
        success, message = tester.create_user(invalid_email_creds)
        # Should fail because email format is invalid
        record_test_result(test_name, not success, message)
        
        # Final Test: Clean up test users
        test_name = "Clean Up Test Users"
        success, message = tester.clean_up_test_users()
        record_test_result(test_name, success, f"Deleted {message} test users")
        
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
    finally:
        # Close database connection
        if tester:
            tester.close()

def print_summary():
    """Print test summary"""
    passed = test_results['passed_tests']
    failed = test_results['failed_tests']
    total = test_results['total_tests']
    
    print("\n" + "="*50)
    print("AUTHENTICATION TEST SUMMARY")
    print("="*50)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Success: {(passed / total * 100):.1f}%")
    
    # Print details of failed tests only
    if failed > 0:
        print("\nFAILED TESTS:")
        for test in test_results['test_details']:
            if test['result'] == "FAILED":
                print(f"  ✗ {test['test_name']}: {test['details']}")
    
    print("="*50)
    print(f"Complete log: {os.path.join(SCRIPT_DIR, 'auth_test_results.log')}")

def test_user_login():
    """Test user login endpoint"""
    BASE_URL = "http://localhost:5000"  # Flask app URL
    
    # Use keyvault to get test credentials
    test_email = keyvault.getenv("TEST_USER_EMAIL", "test@example.com")
    test_password = keyvault.getenv("TEST_USER_PASSWORD", "password123")
    
    logger.info(f"Testing login with email: {test_email}")
    
    # Create a session to maintain cookies
    session = requests.Session()
    
    # Attempt login
    login_data = {
        "email": test_email,
        "password": test_password
    }
    
    try:
        response = session.post(f"{BASE_URL}/login", data=login_data)
        logger.info(f"Login response status: {response.status_code}")
        
        if response.status_code == 200 or response.status_code == 302:
            logger.info("Login successful!")
            return True
        else:
            logger.error(f"Login failed with status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error testing login: {str(e)}")
        return False

if __name__ == "__main__":
    start_time = time.time()
    logger.info("Starting authentication tests")
    
    try:
        run_tests()
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Tests completed in {duration:.2f}s")
    
    print_summary()

    result = test_user_login()
    exit(0 if result else 1) 