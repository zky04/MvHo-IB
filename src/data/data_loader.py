"""Data loading utilities.

Handles loading and preprocessing of multi-view brain network data.
"""

import os
import logging
import torch
from typing import Dict, List, Tuple, Any
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

from ..data.dataset import MultiViewDataset, collate_fn


def _filter_adni_data(graph_list: List, 
                     o_matrices: torch.Tensor, 
                     labels: torch.Tensor,
                     dataset_config: Dict) -> Tuple[List, torch.Tensor, torch.Tensor]:
    """Filter ADNI data based on classification mode.
    
    Args:
        graph_list: List of graph data
        o_matrices: 3D tensor data
        labels: Label data (0: AD, 1: MCI, 2: NC)
        dataset_config: Dataset configuration
        
    Returns:
        Filtered (graph_list, o_matrices, labels)
    """
    classification_mode = dataset_config.get('classification_mode', 'three_class')
    
    if classification_mode == "three_class":
        # Keep original three-class classification
        logging.info("ADNI classification mode: AD vs MCI vs NC (three-class)")
        return graph_list, o_matrices, labels
    
    # Create label mapping and filter indices
    if classification_mode == "binary_ad_nc":
        # AD vs NC (remove MCI samples)
        valid_mask = (labels == 0) | (labels == 2)  # AD or NC
        new_labels = labels[valid_mask].clone()
        new_labels[new_labels == 2] = 1  # NC -> 1, AD remains 0
        logging.info("ADNI classification mode: AD vs NC (binary)")
        
    elif classification_mode == "binary_ad_mci":  
        # AD vs MCI (remove NC samples)
        valid_mask = (labels == 0) | (labels == 1)  # AD or MCI
        new_labels = labels[valid_mask].clone()  # AD=0, MCI=1
        logging.info("ADNI classification mode: AD vs MCI (binary)")
        
    elif classification_mode == "binary_mci_nc":
        # MCI vs NC (remove AD samples) 
        valid_mask = (labels == 1) | (labels == 2)  # MCI or NC
        new_labels = labels[valid_mask].clone()
        new_labels[new_labels == 1] = 0  # MCI -> 0
        new_labels[new_labels == 2] = 1  # NC -> 1
        logging.info("ADNI classification mode: MCI vs NC (binary)")
        
    else:
        raise ValueError(f"Unsupported ADNI classification mode: {classification_mode}")
    
    # Filter data
    valid_indices = torch.where(valid_mask)[0]
    filtered_graph_list = [graph_list[i] for i in valid_indices.cpu().numpy()]
    filtered_o_matrices = o_matrices[valid_mask]
    
    # Update labels in graph data
    for i, new_label in enumerate(new_labels):
        filtered_graph_list[i].y = new_label.unsqueeze(0)
    
    logging.info(f"Samples before filtering: {len(graph_list)}, after filtering: {len(filtered_graph_list)}")
    logging.info(f"New label distribution: {Counter(new_labels.tolist())}")
    
    return filtered_graph_list, filtered_o_matrices, new_labels


def load_data(config: Dict) -> Tuple[Any, ...]:
    """Load multi-view brain network data with train/validation/test splits.
    
    Args:
        config: Configuration dictionary containing dataset parameters
        
    Returns:
        Tuple containing train/val/test data splits:
        (x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test)
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get dataset configuration
        dataset_name = config['dataset_name']
        dataset_config = config['datasets'][dataset_name]
        
        # Get data paths
        x1_path = dataset_config['x1_path']
        x2_path = dataset_config['x2_path']
        
        logging.info(f"Loading dataset: {dataset_name}")
        logging.info(f"Graph data path: {x1_path}")
        logging.info(f"3D data path: {x2_path}")
        
        # Load x1 data (graph data)
        logging.info(f"Loading x1 data from {x1_path}...")
        if not os.path.exists(x1_path):
            raise FileNotFoundError(f"Graph data file not found: {x1_path}")
            
        graph_data_list = torch.load(x1_path, map_location=device, weights_only=False)
        logging.info(f"x1 data loaded successfully. Contains {len(graph_data_list)} samples.")
        
        # Load x2 data (3D tensor data)
        logging.info(f"Loading x2 data from {x2_path}...")
        if not os.path.exists(x2_path):
            raise FileNotFoundError(f"3D data file not found: {x2_path}")
            
        combined_dataset = torch.load(x2_path, map_location=device, weights_only=False)
        logging.info(f"x2 data loaded successfully.")
        
        # Extract x2 data components
        o_matrices = combined_dataset['o_matrices']  # [N, D, H, W] - keep in 4D format
        x2_labels = combined_dataset['labels']       # [N]
        x2_sample_ids = combined_dataset['sample_ids']  # [N]
        
        logging.info(f"O-matrices shape: {o_matrices.shape}")
        logging.info(f"Labels shape: {x2_labels.shape}")
        logging.info(f"Sample IDs shape: {x2_sample_ids.shape}")
        
        # Extract sample IDs and labels from x1 data
        x1_sample_ids = torch.tensor([data.sample_id for data in graph_data_list], 
                                   dtype=torch.long, device=device)
        x1_labels = torch.tensor([data.y.item() for data in graph_data_list], 
                               dtype=torch.long, device=device)
        
        # Sort data by sample_id to ensure alignment
        x1_sorted_indices = x1_sample_ids.argsort()
        graph_data_list_sorted = [graph_data_list[i] for i in x1_sorted_indices.cpu().numpy()]
        x1_labels_sorted = x1_labels[x1_sorted_indices]
        
        x2_sorted_indices = x2_sample_ids.argsort()
        o_matrices_sorted = o_matrices[x2_sorted_indices]
        x2_labels_sorted = x2_labels[x2_sorted_indices]
        x2_sample_ids_sorted = x2_sample_ids[x2_sorted_indices]
        
        # Verify sample IDs and labels match between x1 and x2
        if not torch.equal(x1_sample_ids[x1_sorted_indices], x2_sample_ids_sorted):
            raise ValueError("Sample IDs don't match between x1 and x2 data")
        if not torch.equal(x1_labels_sorted, x2_labels_sorted):
            raise ValueError("Labels don't match between x1 and x2 data")
        
        logging.info("Data alignment verified successfully")
        
        # Handle ADNI classification modes
        if dataset_name == "ADNI":
            graph_data_list_sorted, o_matrices_sorted, x2_labels_sorted = _filter_adni_data(
                graph_data_list_sorted, o_matrices_sorted, x2_labels_sorted, dataset_config
            )
        
        # Create train/validation/test splits
        num_samples = len(graph_data_list_sorted)
        indices = list(range(num_samples))
        labels_np = x2_labels_sorted.cpu().numpy()
        
        # First split: 90% for train+val, 10% for test
        temp_indices, test_indices, temp_labels, test_labels = train_test_split(
            indices, labels_np, test_size=0.1, stratify=labels_np, random_state=42)
        
        # Second split: 80% train, 20% validation from the remaining data
        train_indices, val_indices, train_labels, val_labels = train_test_split(
            temp_indices, temp_labels, test_size=0.2, stratify=temp_labels, random_state=42)
        
        # Convert labels to long tensors
        labels_sorted = x2_labels_sorted.long()
        
        # Create train split
        x1_train = [graph_data_list_sorted[i] for i in train_indices]
        x2_train = o_matrices_sorted[train_indices]
        y_train = labels_sorted[train_indices].long()
        
        # Create validation split
        x1_val = [graph_data_list_sorted[i] for i in val_indices]
        x2_val = o_matrices_sorted[val_indices]
        y_val = labels_sorted[val_indices].long()
        
        # Create test split
        x1_test = [graph_data_list_sorted[i] for i in test_indices]
        x2_test = o_matrices_sorted[test_indices]
        y_test = labels_sorted[test_indices].long()
        
        logging.info(f"Data splits created successfully:")
        logging.info(f"  Training samples: {len(x1_train)}")
        logging.info(f"  Validation samples: {len(x1_val)}")
        logging.info(f"  Test samples: {len(x1_test)}")
        
        return x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test
        
    except Exception as e:
        logging.error(f"Error in load_data: {e}")
        raise


def create_data_loaders(x1_train: List, x2_train: torch.Tensor, y_train: torch.Tensor,
                       x1_val: List, x2_val: torch.Tensor, y_val: torch.Tensor,
                       x1_test: List, x2_test: torch.Tensor, y_test: torch.Tensor,
                       batch_size: int = 32, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train/validation/test sets.
    
    Args:
        x1_train, x2_train, y_train: Training data
        x1_val, x2_val, y_val: Validation data  
        x1_test, x2_test, y_test: Test data
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MultiViewDataset(x1_train, x2_train, y_train)
    val_dataset = MultiViewDataset(x1_val, x2_val, y_val)
    test_dataset = MultiViewDataset(x1_test, x2_test, y_test)
    
    # Create data loaders
    # Don't use pin_memory when data is already on GPU to avoid conflicts
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Use 0 workers to avoid multiprocessing issues with CUDA data
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    logging.info(f"DataLoaders created with batch_size={batch_size}")
    
    return train_loader, val_loader, test_loader 