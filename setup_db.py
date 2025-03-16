import os
from dotenv import load_dotenv
import pymysql
from flask import Flask
from users import db, User
import sys
import urllib.parse  # For URL encoding

# Load environment variables
load_dotenv()

def setup_database():
    """Setup MySQL database"""
    # Database connection parameters
    MYSQL_USER = os.getenv('MYSQL_USER')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
    MYSQL_HOST = os.getenv('MYSQL_HOST')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT'))
    MYSQL_DB = os.getenv('MYSQL_DB')
    
    print(f"Attempting to connect to MySQL with user: {MYSQL_USER}, host: {MYSQL_HOST}, port: {MYSQL_PORT}")
    
    # Try to create the database if it doesn't exist
    try:
        # Connect to MySQL server without specifying a database
        print("Connecting to MySQL server...")
        conn = pymysql.connect(
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            host=MYSQL_HOST,
            port=MYSQL_PORT
        )
        cursor = conn.cursor()
        
        # Check if database exists
        print(f"Checking if database '{MYSQL_DB}' exists...")
        cursor.execute(f"SHOW DATABASES LIKE '{MYSQL_DB}'")
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Creating database {MYSQL_DB}...")
            cursor.execute(f'CREATE DATABASE `{MYSQL_DB}`')
            print(f"Database {MYSQL_DB} created successfully!")
        else:
            print(f"Database {MYSQL_DB} already exists.")
        
        cursor.close()
        conn.close()
        
        print("Database setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error setting up database: {str(e)}", file=sys.stderr)
        return False

def create_tables():
    """Create database tables"""
    try:
        # Get database connection parameters
        MYSQL_USER = os.getenv('MYSQL_USER')
        MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
        MYSQL_HOST = os.getenv('MYSQL_HOST')
        MYSQL_PORT = os.getenv('MYSQL_PORT')
        MYSQL_DB = os.getenv('MYSQL_DB')
        
        # URL encode the password to handle special characters
        encoded_password = urllib.parse.quote_plus(MYSQL_PASSWORD)
        
        # Create a small Flask app to initialize the database
        print("Creating Flask app for database initialization...")
        app = Flask(__name__)
        db_uri = f"mysql+pymysql://{MYSQL_USER}:{encoded_password}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
        print(f"Database URI: mysql+pymysql://{MYSQL_USER}:****@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}")  # Don't print actual password
        app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        
        # Initialize the database with the app
        print("Initializing database with Flask-SQLAlchemy...")
        db.init_app(app)
        
        # Create tables within the app context
        print("Creating database tables...")
        with app.app_context():
            db.create_all()
            print("Database tables created successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error creating tables: {str(e)}", file=sys.stderr)
        return False

if __name__ == "__main__":
    print("Setting up DeepMedVer database with MySQL...")
    if setup_database():
        create_tables()
    print("Database setup complete!") 