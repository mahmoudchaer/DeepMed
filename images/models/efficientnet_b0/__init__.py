"""
EfficientNet-B0 Model Package for Medical Image Classification

This package provides functionality to train and use an EfficientNet-B0 model
for medical image classification.
"""

# Import main components to make them available from the package
from . import model
from . import api

from .api import init_app, efficientnet_bp
from .model import EfficientNetB0Classifier 