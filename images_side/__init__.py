# images_side package
import logging
from flask import Flask

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask application instance
app = Flask(__name__)

# These global variables are exported for backward compatibility
from images_side.config import EEP_SERVICE_URL, ALTERNATIVE_EEP_URLS

# Import and expose services for backward compatibility
from images_side.services.health_service import check_model_service_health, check_eep_service_health
from images_side.services.model_service import train_model
from images_side.services.eep_service import augment_data, process_data

# Import routes to register them with app
from images_side.routes import api_routes

# This is for backward compatibility with app_api.py
# Export the train_model API function
from images_side.routes.api_routes import api_train_model 