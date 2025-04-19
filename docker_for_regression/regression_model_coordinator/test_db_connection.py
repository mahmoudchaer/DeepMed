import os
import sys
import logging
from sqlalchemy import create_engine, text
import urllib.parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Get database connection settings from environment or use defaults
MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', 'root')
MYSQL_PORT = os.environ.get('MYSQL_PORT', '3306')
MYSQL_DB = os.environ.get('MYSQL_DB', 'deepmed')

def test_connection(host):
    """Test database connection with a specific host"""
    try:
        # URL encode the password to handle special characters
        encoded_password = urllib.parse.quote_plus(str(MYSQL_PASSWORD))
        
        # Create connection string with the current host
        connection_string = f'mysql+pymysql://{MYSQL_USER}:{encoded_password}@{host}:{MYSQL_PORT}/{MYSQL_DB}'
        logger.info(f"Trying database connection with host: {host}")
        
        # Create the engine with the connection string
        db_engine = create_engine(connection_string, connect_args={'connect_timeout': 3})
        
        # Test the connection
        with db_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info(f"Connection successful to {host}! Result: {result.scalar()}")
            
            # Try to query tables
            try:
                tables = conn.execute(text("SHOW TABLES")).fetchall()
                logger.info(f"Tables in database: {[t[0] for t in tables]}")
            except Exception as e:
                logger.warning(f"Could not query tables: {str(e)}")
                
        return True
    except Exception as e:
        logger.error(f"Failed to connect with host {host}: {str(e)}")
        return False

def main():
    """Test database connections with multiple possible hosts"""
    # Try multiple common database hostnames used in Docker environments
    possible_hosts = [
        "host.docker.internal",  # Special Docker hostname that maps to the host
        "localhost", 
        "127.0.0.1",
        "db",                  # Common database service name
        "mysql",
        "mariadb",
        "database"
    ]
    
    # Add any custom host from environment
    custom_host = os.environ.get('MYSQL_HOST')
    if custom_host and custom_host not in possible_hosts:
        possible_hosts.insert(0, custom_host)
    
    success = False
    for host in possible_hosts:
        if test_connection(host):
            success = True
            logger.info(f"✅ Successfully connected to database at {host}")
            break
    
    if not success:
        logger.error("❌ Failed to connect to database after trying all possible hosts")
        sys.exit(1)
    else:
        logger.info("Database connection test completed successfully")

if __name__ == "__main__":
    main() 