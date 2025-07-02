#!/usr/bin/env python3
"""Main entry point for the MvHo-IB training pipeline.

This script handles the complete training workflow including data loading,
model initialization, training, and evaluation.

Typical usage example:
    python main.py
    
    Or specify configuration file:
    python main.py --config custom_config.yaml
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data import load_data, create_data_loaders
from src.trainer import Trainer
from src.utils import load_config, validate_config


def setup_seed(seed: int) -> None:
    """Set random seed to ensure reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line argument object
    """
    parser = argparse.ArgumentParser(
        description="MvHo-IB: Multi-View High-Order Information Bottleneck Brain Disease Diagnosis Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Specify the GPU ID to use"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def setup_device(config: dict, gpu_id: int = None) -> torch.device:
    """Set computation device.
    
    Args:
        config: Configuration dictionary
        gpu_id: Specified GPU ID
        
    Returns:
        PyTorch device object
    """
    device_config = config.get("experiment", {}).get("device", "auto")
    
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        logging.info(f"Using specified GPU: {device}")
    elif device_config == "cpu":
        device = torch.device("cpu")
        logging.info("Using CPU device")
    elif device_config == "cuda" or device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"Using GPU device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA not available, falling back to CPU")
    else:
        device = torch.device(device_config)
        logging.info(f"Using device specified in configuration: {device}")
    
    return device


def main():
    """Main training pipeline."""
    # Setup logging
    setup_logging()
    logging.info("=" * 60)
    logging.info("MVHO-IB TRAINING PIPELINE STARTED")
    logging.info("=" * 60)
    
    try:
        # Load and validate configuration
        config_path = 'config.yaml'
        config = load_config(config_path)
        validate_config(config)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        if torch.cuda.is_available():
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logging.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load data
        logging.info("Loading and preprocessing data...")
        x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test = load_data(config)
        
        # Create data loaders
        batch_size = config['training']['batch_size']
        train_loader, val_loader, test_loader = create_data_loaders(
            x1_train, x2_train, y_train,
            x1_val, x2_val, y_val,
            x1_test, x2_test, y_test,
            batch_size=batch_size
        )
        
        # Initialize trainer
        trainer = Trainer(config, device)
        
        # Get example batches for model initialization
        sample_x1, sample_x2, _ = next(iter(train_loader))
        # Add channel dimension to sample_x2 for model initialization (same as in training)
        sample_x2 = sample_x2.unsqueeze(1)  # [B, D, H, W] -> [B, 1, D, H, W]
        trainer.setup_models(sample_x1, sample_x2)
        
        # Training
        logging.info("Starting model training...")
        trainer.train_model(train_loader, val_loader)
        
        # Testing
        logging.info("Starting model evaluation...")
        test_results = trainer.test_model(test_loader)
        
        # Save results
        trainer.save_results(test_results)
        
        logging.info("=" * 60)
        logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"Training pipeline failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 