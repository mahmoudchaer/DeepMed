#!/usr/bin/env python3
"""
Setup MySQL Event Scheduler for automatic deletion of old records.
Deletes expired models and data cleaners after 15 days.
"""

import os
import pymysql
import sys
from dotenv import load_dotenv

# Load .env from the parent directory
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PARENT_DIR, ".env"))

def setup_event_scheduler():
    """Create MySQL Event Scheduler for auto-deletion after 15 days"""
    try:
        # Get database connection parameters
        MYSQL-USER = os.getenv("MYSQL-USER")
        MYSQL-PASSWORD = os.getenv("MYSQL-PASSWORD")
        MYSQL-HOST = os.getenv("MYSQL-HOST")
        MYSQL-PORT = int(os.getenv("MYSQL-PORT"))
        MYSQL-DB = os.getenv("MYSQL-DB")

        # Connect to MySQL
        conn = pymysql.connect(
            host=MYSQL-HOST,
            user=MYSQL-USER,
            password=MYSQL-PASSWORD,
            port=MYSQL-PORT,
            database=MYSQL-DB
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
            END;
        """)

        # Commit changes and close connection
        conn.commit()
        cursor.close()
        conn.close()

        print("✅ MySQL Event Scheduler for 15-day cleanup created successfully!")

    except Exception as e:
        print(f"❌ Error setting up MySQL Event Scheduler: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    setup_event_scheduler()
