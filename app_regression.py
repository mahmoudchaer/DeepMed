from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_login import login_required, current_user
import os
import pandas as pd
import numpy as np
import logging
import json
import time
from datetime import datetime

# Import common components from app_api.py
from app_api import app, DATA_CLEANER_URL, MODEL_COORDINATOR_URL, FEATURE_SELECTOR_URL
from app_api import is_service_available, get_temp_filepath, safe_requests_post, clean_data_for_json
from app_api import check_services, save_to_temp_file, load_from_temp_file, SafeJSONEncoder, logger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route('/train_regression')
@login_required
def train_regression():
    """Route for training regression models"""
    return render_template('train_regression.html') 