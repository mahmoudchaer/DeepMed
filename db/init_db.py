#!/usr/bin/env python3
"""
Database initialization script for DeepMed application.
This script sets up the MySQL database and creates all required tables.
"""

import os
import sys
import urllib.parse  # For URL encoding
import pymysql
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
# Import the key vault module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import keyvault

# Initialize Flask App & SQLAlchemy
app = Flask(__name__)
db = SQLAlchemy()

def get_db_uri():
    """Generate MySQL connection URI from environment variables."""
    MYSQLUSER = keyvault.getenv("MYSQLUSER")
    MYSQLPASSWORD = keyvault.getenv("MYSQLPASSWORD")
    MYSQLHOST = keyvault.getenv("MYSQLHOST")
    MYSQLPORT = keyvault.getenv("MYSQLPORT")
    MYSQLDB = keyvault.getenv("MYSQLDB")

    # URL encode the password to handle special characters
    encoded_password = urllib.parse.quote_plus(MYSQLPASSWORD)

    return f"mysql+pymysql://{MYSQLUSER}:{encoded_password}@{MYSQLHOST}:{MYSQLPORT}/{MYSQLDB}"

def setup_database():
    """Setup MySQL database if it doesn't exist."""
    MYSQLUSER = keyvault.getenv("MYSQLUSER")
    MYSQLPASSWORD = keyvault.getenv("MYSQLPASSWORD")
    MYSQLHOST = keyvault.getenv("MYSQLHOST")
    MYSQLPORT = int(keyvault.getenv("MYSQLPORT"))
    MYSQLDB = keyvault.getenv("MYSQLDB")

    print(f"🔄 Connecting to MySQL server at {MYSQLHOST}:{MYSQLPORT} with user {MYSQLUSER}...")

    try:
        conn = pymysql.connect(
            user=MYSQLUSER,
            password=MYSQLPASSWORD,
            host=MYSQLHOST,
            port=MYSQLPORT
        )
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(f"SHOW DATABASES LIKE '{MYSQLDB}'")
        exists = cursor.fetchone()

        if not exists:
            print(f"🛠️ Creating database '{MYSQLDB}'...")
            cursor.execute(f"CREATE DATABASE `{MYSQLDB}`")
            print(f"✅ Database '{MYSQLDB}' created successfully!")
        else:
            print(f"✅ Database '{MYSQLDB}' already exists.")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error setting up database: {str(e)}", file=sys.stderr)
        return False

def create_tables():
    """Create required database tables."""
    try:
        print("🔄 Initializing Flask app for database setup...")
        app.config["SQLALCHEMY_DATABASE_URI"] = get_db_uri()
        app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
        db.init_app(app)

        # Define Models
        class User(UserMixin, db.Model):
            """User model for authentication."""
            __tablename__ = 'users'
            
            id = db.Column(db.Integer, primary_key=True)
            email = db.Column(db.String(120), unique=True, nullable=False)
            password_hash = db.Column(db.String(255), nullable=False)
            first_name = db.Column(db.String(50), nullable=True)
            last_name = db.Column(db.String(50), nullable=True)
            created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
            last_login = db.Column(db.DateTime, nullable=True)
            
            def set_password(self, password):
                """Create hashed password."""
                self.password_hash = generate_password_hash(password)
            
            def check_password(self, password):
                """Check hashed password."""
                return check_password_hash(self.password_hash, password)

        class TrainingRun(db.Model):
            """Table for tracking user training runs."""
            id = db.Column(db.Integer, primary_key=True, autoincrement=True)
            user_id = db.Column(db.Integer, nullable=False)
            run_name = db.Column(db.String(255), nullable=False)
            prompt = db.Column(db.Text, nullable=True)
            created_at = db.Column(db.DateTime, server_default=db.func.current_timestamp())

        class TrainingModel(db.Model):
            """Table for storing trained models from each run."""
            id = db.Column(db.Integer, primary_key=True, autoincrement=True)
            user_id = db.Column(db.Integer, nullable=False)
            run_id = db.Column(db.Integer, nullable=False)
            model_name = db.Column(db.String(255), nullable=False)
            model_url = db.Column(db.Text, nullable=False)
            file_name = db.Column(db.String(255), nullable=True)
            metric_name = db.Column(db.String(50), nullable=True)
            metric_value = db.Column(db.Float, nullable=True)
            created_at = db.Column(db.DateTime, server_default=db.func.current_timestamp())
            
        class PreprocessingData(db.Model):
            """Table for storing preprocessing data for each training run."""
            id = db.Column(db.Integer, primary_key=True, autoincrement=True)
            run_id = db.Column(db.Integer, nullable=False)
            user_id = db.Column(db.Integer, nullable=False)
            cleaner_config = db.Column(db.Text, nullable=True)  # JSON configuration for data cleaner
            feature_selector_config = db.Column(db.Text, nullable=True)  # JSON configuration for feature selector
            original_columns = db.Column(db.Text, nullable=True)  # Original dataset columns as JSON
            selected_columns = db.Column(db.Text, nullable=True)  # Selected columns after feature selection as JSON
            cleaning_report = db.Column(db.Text, nullable=True)  # Detailed report of cleaning operations
            created_at = db.Column(db.DateTime, server_default=db.func.current_timestamp())

        # Create tables
        print("🛠️ Creating database tables...")
        with app.app_context():
            db.create_all()
            print("✅ All required tables created successfully!")

    except Exception as e:
        print(f"❌ Error creating tables: {str(e)}", file=sys.stderr)

def setup_event_scheduler():
    """Create MySQL Event Scheduler for auto-deletion after 15 days."""
    try:
        MYSQLUSER = keyvault.getenv("MYSQLUSER")
        MYSQLPASSWORD = keyvault.getenv("MYSQLPASSWORD")
        MYSQLHOST = keyvault.getenv("MYSQLHOST")
        MYSQLPORT = int(keyvault.getenv("MYSQLPORT"))
        MYSQLDB = keyvault.getenv("MYSQLDB")

        conn = pymysql.connect(
            host=MYSQLHOST,
            user=MYSQLUSER,
            password=MYSQLPASSWORD,
            port=MYSQLPORT,
            database=MYSQLDB
        )
        cursor = conn.cursor()

        # Enable MySQL Event Scheduler
        cursor.execute("SET GLOBAL event_scheduler = ON;")

        # Create MySQL Event for Auto-Deletion
        cursor.execute("""
            CREATE EVENT IF NOT EXISTS delete_expired_files
            ON SCHEDULE EVERY 1 DAY
            DO
            BEGIN
                DELETE FROM training_models WHERE created_at < NOW() - INTERVAL 15 DAY;
                DELETE FROM data_cleaners WHERE created_at < NOW() - INTERVAL 15 DAY;
                DELETE FROM preprocessing_data WHERE created_at < NOW() - INTERVAL 15 DAY;
            END;
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("✅ MySQL Event Scheduler for 15-day cleanup created successfully!")

    except Exception as e:
        print(f"❌ Error setting up MySQL Event Scheduler: {str(e)}", file=sys.stderr)

def main():
    """Run the database initialization process."""
    print("===== DeepMed MySQL Database Initialization =====")
    
    # Setup database
    print("\nStep 1: Creating MySQL database if needed...")
    if not setup_database():
        return 1
    
    # Create tables
    print("\nStep 2: Creating database tables...")
    create_tables()
    
    # Setup MySQL Event Scheduler for auto-deletion
    print("\nStep 3: Setting up auto-deletion trigger...")
    setup_event_scheduler()
    
    print("\n✅ Database initialization completed successfully!")
    print("\nYou can now run the application with:")
    print("  python app_api.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
