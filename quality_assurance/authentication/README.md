# Authentication Quality Assurance Tests

This directory contains scripts for testing the authentication system of the DeepMed application.

## Test Scripts

### 1. `authentication_test.py`

Tests direct database operations for user authentication:
- Creating new users
- Verifying login credentials
- Handling edge cases (existing emails, invalid credentials)
- Cleaning up test users

### 2. `flask_auth_test.py`

Tests Flask routes for user authentication:
- Registration route (`/register`)
- Login route (`/login`) 
- Logout route (`/logout`)
- Protected route access
- Session management

### 3. `run_auth_tests.py`

Runner script that executes both test scripts and generates a consolidated report.

## Setup

These scripts automatically locate the `.env` file at the project root (two directories above) to load database credentials and other environment variables. Make sure the `.env` file contains valid database credentials:

```
MYSQL_USER=root
MYSQL_PASSWORD=pass
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DB=deepmedver
```

## Running the Tests

To run all authentication tests:

```bash
# From the project root directory
python quality_assurance/authentication/run_auth_tests.py

# Or from the authentication directory
cd quality_assurance/authentication
python run_auth_tests.py
```

To run individual test scripts:

```bash
# From the project root directory
python quality_assurance/authentication/authentication_test.py
python quality_assurance/authentication/flask_auth_test.py

# Or from the authentication directory
cd quality_assurance/authentication
python authentication_test.py
python flask_auth_test.py
```

## Test Results

Test results are logged to the following files:
- `auth_test_results.log`: Database authentication test results
- `flask_auth_test_results.log`: Flask authentication test results
- `auth_tests_runner.log`: Combined test runner logs

All log files are created in the `quality_assurance/authentication` directory.

A summary report is also displayed in the console when tests complete.

## Test Coverage

The test scripts cover the following aspects of the authentication system:

1. **User Registration**
   - Creating a new account with valid credentials
   - Attempting to create an account with an existing email
   - Handling invalid email formats and empty passwords

2. **User Login**
   - Logging in with valid credentials
   - Attempting to log in with invalid credentials
   - Attempting to log in with a non-existent account

3. **Session Management**
   - Accessing protected routes when authenticated
   - Accessing protected routes when not authenticated
   - Logging out and verifying session termination

4. **Database Operations**
   - User creation in the database
   - Credential verification
   - Error handling and data integrity

## Adding New Tests

To add new test cases:
1. Add new test methods to the existing test classes
2. Update the `run_tests()` function to include your new tests
3. Make sure to use the `record_test_result()` function to log results

## Cleanup

All tests include cleanup procedures to remove test users from the database when tests complete. 