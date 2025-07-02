"""Multi-view dataset implementation.

Dataset class for handling graph and tensor data simultaneously.
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from typing import List


class MultiViewDataset(Dataset):
    """Dataset for storing and returning (x1, x2, label) multi-view data.
    
    Args:
        x1_list: List of graph data (PyG Data objects)
        x2_tensor: 3D tensor data
        labels: Labels tensor
    """
    
    def __init__(self, x1_list: List, x2_tensor: torch.Tensor, labels: torch.Tensor):
        self.x1_list = x1_list
        self.x2_tensor = x2_tensor
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x1 = self.x1_list[idx]
        x2 = self.x2_tensor[idx]
        y = self.labels[idx]
        return x1, x2, y


def collate_fn(batch):
    """Collate function for DataLoader to batch multiple samples.
    
    Args:
        batch: List of (x1, x2, y) tuples
        
    Returns:
        Tuple of (x1_batch, x2_batch, y_batch)
    """
    x1_list, x2_list, y_list = zip(*batch)
    
    # Batch graph data using PyG Batch
    x1_batch = Batch.from_data_list(x1_list)
    
    # Stack tensor data
    x2_batch = torch.stack(x2_list)
    
    # Convert labels to tensor
    y_batch = torch.tensor(y_list, dtype=torch.long)
    
    return x1_batch, x2_batch, y_batch 