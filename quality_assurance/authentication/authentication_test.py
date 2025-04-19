#!/usr/bin/env python3
"""
Authentication Quality Assurance Script for DeepMed

This script tests the authentication functionality by:
1. Creating a new user with random credentials
2. Testing login with those credentials
3. Testing various edge cases and error conditions
4. Cleaning up test users after the tests

Usage:
python authentication_test.py
"""

import os
import sys
import random
import string
import time
from datetime import datetime
import logging
from dotenv import load_dotenv
import urllib.parse

# Get the absolute path to the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get path to project root (two directories up)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# Add project root to path to import db modules
sys.path.append(PROJECT_ROOT)

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

# Load environment variables from project root .env
env_path = os.path.join(PROJECT_ROOT, '.env')
logger.info(f"Loading .env from: {env_path}")
load_dotenv(env_path)

# Import db models after environment variables are loaded
from db.users import db, User
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

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

class DatabaseConnection:
    """Class to handle database connection and operations"""
    
    def __init__(self):
        """Initialize database connection"""
        # Get database connection parameters from environment variables
        mysql_user = os.getenv('MYSQL_USER')
        mysql_password = os.getenv('MYSQL_PASSWORD')
        mysql_host = os.getenv('MYSQL_HOST')
        mysql_port = os.getenv('MYSQL_PORT')
        mysql_db = os.getenv('MYSQL_DB')
        
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
    
    def clean_up_test_users(self):
        """Clean up test users created during testing"""
        try:
            # Find and delete all test users with synchronize_session='fetch' to fix the LIKE operator error
            deleted = self.session.query(User).filter(
                User.email.like('test_user_%@example.com')
            ).delete(synchronize_session='fetch')
            
            self.session.commit()
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
    db_conn = None
    try:
        # Initialize database connection
        db_conn = DatabaseConnection()
        
        # Generate random credentials for testing
        test_creds1 = generate_random_credentials()
        test_creds2 = generate_random_credentials()
        
        logger.info("Starting DB authentication tests")
        
        # Test 1: Create a new user
        test_name = "Create New User"
        success, user_id = db_conn.create_user(test_creds1)
        record_test_result(test_name, success, f"User ID: {user_id if success else None}")
        
        # Test 2: Create a user with existing email
        test_name = "Create User with Existing Email"
        success, message = db_conn.create_user(test_creds1)
        # This should fail (return False) because the email already exists
        record_test_result(test_name, not success, message)
        
        # Test 3: Login with valid credentials
        test_name = "Login with Valid Credentials"
        success, message = db_conn.verify_login(test_creds1["email"], test_creds1["password"])
        record_test_result(test_name, success, message)
        
        # Test 4: Login with invalid password
        test_name = "Login with Invalid Password"
        success, message = db_conn.verify_login(test_creds1["email"], "wrong_password")
        # This should fail (return False) because the password is incorrect
        record_test_result(test_name, not success, message)
        
        # Test 5: Login with non-existent user
        test_name = "Login with Non-existent User"
        success, message = db_conn.verify_login("nonexistent@example.com", "password")
        # This should fail (return False) because the user doesn't exist
        record_test_result(test_name, not success, message)
        
        # Test 6: Create a second user
        test_name = "Create Second User"
        success, user_id = db_conn.create_user(test_creds2)
        record_test_result(test_name, success, f"User ID: {user_id if success else None}")
        
        # Test 7: Login with second user
        test_name = "Login with Second User"
        success, message = db_conn.verify_login(test_creds2["email"], test_creds2["password"])
        record_test_result(test_name, success, message)
        
        # Test 8: Password handling - empty password
        test_name = "Create User with Empty Password"
        empty_pass_creds = generate_random_credentials()
        empty_pass_creds["password"] = ""
        success, message = db_conn.create_user(empty_pass_creds)
        # Should fail or handle gracefully
        passed = not success or isinstance(message, str)
        record_test_result(test_name, passed, message)
        
        # Test 9: Email validation - invalid email format
        test_name = "Create User with Invalid Email"
        invalid_email_creds = generate_random_credentials()
        invalid_email_creds["email"] = "not_an_email"
        success, message = db_conn.create_user(invalid_email_creds)
        # Should fail or handle gracefully
        record_test_result(test_name, not success, message)
        
        # Final Test: Clean up test users
        test_name = "Clean Up Test Users"
        success, message = db_conn.clean_up_test_users()
        record_test_result(test_name, success, f"Deleted {message} test users")
        
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
    finally:
        # Close database connection
        if db_conn:
            db_conn.close()

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