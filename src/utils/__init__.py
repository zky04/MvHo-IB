"""Utility functions module.

This module contains various utility functions for the MvHo-IB framework:
- info_bottleneck: Information bottleneck related calculations
- evaluator: Model evaluation metrics calculation
- config_utils: Configuration file processing tools
"""

from .info_bottleneck import renyi_entropy, calculate_gram_matrix, compute_renyi_entropy
from .evaluator import evaluate_model
from .config_utils import load_config, validate_config, save_config

__all__ = [
    "renyi_entropy",
    "calculate_gram_matrix", 
    "compute_renyi_entropy",
    "evaluate_model",
    "load_config",
    "validate_config",
    "save_config"
] 