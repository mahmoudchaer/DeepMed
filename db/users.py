#!/usr/bin/env python3
"""
User model and database configuration for DeepMed application.
"""

import os
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import urllib.parse

# Load environment variables
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PARENT_DIR, ".env"))

# Initialize SQLAlchemy
db = SQLAlchemy()

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

# Model for tracking user training runs
class TrainingRun(db.Model):
    """Table for tracking user training runs."""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, nullable=False)
    run_name = db.Column(db.String(255), nullable=False)
    prompt = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.current_timestamp())

# Model for storing trained models
class TrainingModel(db.Model):
    """Table for storing trained models from each run."""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, nullable=False)
    run_id = db.Column(db.Integer, nullable=False)
    model_name = db.Column(db.String(255), nullable=False)
    model_url = db.Column(db.Text, nullable=False)
    file_name = db.Column(db.String(255), nullable=True)
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