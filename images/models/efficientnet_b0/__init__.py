"""
EfficientNet-B0 Model for Medical Image Classification

This module provides an implementation of the EfficientNet-B0 model
for training on medical images and exporting the trained model.
"""

from .api import init_app, efficientnet_bp
from .model import EfficientNetB0Classifier 