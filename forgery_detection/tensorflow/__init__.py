"""
TensorFlow/Keras Implementation for Forgery Detection
======================================================
Complete solution for scientific image forgery detection competition.

Uses ResNet101V2 encoder with U-Net decoder architecture.
Built-in ImageNet weights - no external weight files needed!
"""

from .model import ForgeryDetector, build_model
from .losses import dice_loss, focal_loss, combined_loss, ForgeryLoss
from .dataset import ForgeryDataset, create_datasets, create_dataloaders

__version__ = '1.0.0'
__all__ = [
    'ForgeryDetector',
    'build_model', 
    'dice_loss',
    'focal_loss', 
    'combined_loss',
    'ForgeryLoss',
    'ForgeryDataset',
    'create_datasets',
    'create_dataloaders',
]
