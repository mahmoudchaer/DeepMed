#!/usr/bin/env python3
"""
Setup MySQL Event Scheduler for automatic deletion of old records.
Deletes expired models and data cleaners after 15 days.
"""

import os
import pymysql
import sys

# Add docker_secrets import to load environment variables
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

try:
    import docker_secrets
    print("Successfully imported docker_secrets adapter for DB events")
except ImportError as e:
    print(f"Could not import docker_secrets adapter for DB events: {str(e)}")

def setup_event_scheduler():
    """Create MySQL Event Scheduler for auto-deletion after 15 days"""
    try:
        # Get database connection parameters
        MYSQLUSER = os.getenv("MYSQLUSER")
        MYSQLPASSWORD = os.getenv("MYSQLPASSWORD")
        MYSQLHOST = os.getenv("MYSQLHOST")
        MYSQLPORT = int(os.getenv("MYSQLPORT"))
        MYSQLDB = os.getenv("MYSQLDB")

        # Connect to MySQL
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
