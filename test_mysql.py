import pymysql
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get connection parameters from .env
user = os.getenv('MYSQL_USER', 'root')
password = os.getenv('MYSQL_PASSWORD', 'passs')
host = os.getenv('MYSQL_HOST', 'localhost')
port = int(os.getenv('MYSQL_PORT', '3306'))

print(f"Attempting MySQL connection with: user={user}, host={host}, port={port}")

try:
    # Try to connect
    conn = pymysql.connect(
        user=user,
        password=password,
        host=host,
        port=port
    )
    
    # If we get here, connection was successful
    print("✅ SUCCESS: Connected to MySQL server!")
    
    # Check if we can execute a query
    cursor = conn.cursor()
    cursor.execute("SELECT VERSION()")
    version = cursor.fetchone()
    print(f"MySQL Version: {version[0]}")
    
    # Close connection
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"❌ ERROR: Failed to connect to MySQL: {str(e)}")
    print("\nTroubleshooting tips:")
    print("1. Make sure MySQL service is running (services.msc)")
    print("2. Check that the password in .env is correct")
    print("3. Verify port 3306 is not blocked by firewall")
    print("4. Try connecting with MySQL command line: mysql -u root -p") 