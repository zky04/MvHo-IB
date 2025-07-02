"""Configuration utilities for the MvHo-IB project.

Handles loading, validation, and processing of configuration files.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file has invalid YAML syntax
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure and required fields.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_sections = ['dataset_name', 'datasets', 'training', 'information_bottleneck']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate dataset configuration
    dataset_name = config['dataset_name']
    if dataset_name not in config['datasets']:
        raise ValueError(f"Dataset '{dataset_name}' not found in datasets configuration")
    
    dataset_config = config['datasets'][dataset_name]
    required_dataset_fields = ['x1_path', 'x2_path', 'num_classes', 'num_regions']
    
    for field in required_dataset_fields:
        if field not in dataset_config:
            raise ValueError(f"Missing required dataset field: {field}")
    
    # For ADNI dataset, validate classification mode
    if dataset_name == 'ADNI' and 'classification_mode' in dataset_config:
        valid_modes = ['three_class', 'binary_ad_nc', 'binary_ad_mci', 'binary_mci_nc']
        mode = dataset_config['classification_mode']
        if mode not in valid_modes:
            raise ValueError(f"Invalid ADNI classification mode: {mode}. Valid modes: {valid_modes}")
    
    # Validate training configuration
    training_config = config['training']
    required_training_fields = ['learning_rate', 'num_epochs', 'batch_size']
    
    for field in required_training_fields:
        if field not in training_config:
            raise ValueError(f"Missing required training field: {field}")
    
    # Validate information bottleneck configuration
    ib_config = config['information_bottleneck']
    required_ib_fields = ['beta_gin', 'beta_cnn', 'use_ib']
    
    for field in required_ib_fields:
        if field not in ib_config:
            raise ValueError(f"Missing required information bottleneck field: {field}")
    
    logging.info("Configuration validation passed")
    return True


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        save_path: Path where to save the configuration
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logging.info(f"Configuration saved to: {save_path}")


def update_config_paths(config: Dict[str, Any], base_path: Optional[str] = None) -> Dict[str, Any]:
    """Update file paths in configuration to be absolute paths.
    
    Args:
        config: Configuration dictionary
        base_path: Base directory for relative paths (defaults to current directory)
        
    Returns:
        Updated configuration with absolute paths
    """
    if base_path is None:
        base_path = Path.cwd()
    else:
        base_path = Path(base_path)
    
    # Update dataset file paths
    datasets_config = config.get('datasets', {})
    for dataset_name, dataset_config in datasets_config.items():
        for path_key in ['x1_path', 'x2_path']:
            if path_key in dataset_config:
                file_path = dataset_config[path_key]
                if not Path(file_path).is_absolute():
                    absolute_path = base_path / file_path
                    config['datasets'][dataset_name][path_key] = str(absolute_path)
    
    return config 