#!/usr/bin/env python3
"""
Database initialization script for DeepMed application
This script sets up the MySQL database and creates all required tables
"""

import sys
import os
from setup_db import setup_database, create_tables

def main():
    """Run the database initialization process"""
    print("===== DeepMed MySQL Database Initialization =====")
    
    # Check if mysql client is installed
    try:
        import pymysql
    except ImportError:
        print("ERROR: pymysql module not found. Make sure MySQL and pymysql are installed.")
        print("Run: pip install pymysql mysqlclient")
        return 1
    
    # Check for .env file
    if not os.path.exists(".env"):
        print("WARNING: No .env file found. Using default database settings.")
        print("Consider creating a .env file with your database credentials.")
    
    # Setup database
    print("\nStep 1: Creating MySQL database if needed...")
    if not setup_database():
        print("ERROR: Failed to setup database. Check your MySQL installation and credentials.")
        return 1
    
    # Create database tables
    print("\nStep 2: Creating database tables...")
    if not create_tables():
        print("ERROR: Failed to create database tables.")
        return 1
    
    print("\n✅ Database initialization completed successfully!")
    print("\nYou can now run the application with:")
    print("  python app_api.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 