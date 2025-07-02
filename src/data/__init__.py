"""Data processing module.

This module contains data processing functionality for the MvHo-IB framework:
- dataset: Custom dataset classes
- data_loader: Data loading and preprocessing functionality
"""

from .dataset import MultiViewDataset
from .data_loader import load_data, create_data_loaders

__all__ = [
    "MultiViewDataset",
    "load_data", 
    "create_data_loaders"
] 