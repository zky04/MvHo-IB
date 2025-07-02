"""GIN (Graph Isomorphism Network) model implementation.

Graph neural network for processing brain network graph structure data.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, BatchNorm
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F


class GINModel(nn.Module):
    """Graph Isomorphism Network model."""

    def __init__(self, 
                 num_features: int,
                 embedding_dim: int = 64,
                 hidden_dims: list = [128, 256, 512],
                 dropout_rate: float = 0.5) -> None:
        """Initialize GIN model."""
        super(GINModel, self).__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self._build_layers()
        
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

    def _build_layers(self) -> None:
        """Build network layers."""
        dims = [self.num_features] + self.hidden_dims
        
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            
            self.layers.append(GINConv(mlp))
            self.batch_norms.append(BatchNorm(out_dim))

    def get_num_parameters(self) -> int:
        """Get number of model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch) -> torch.Tensor:
        """Forward pass."""
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x = layer(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Graph-level pooling
        x = global_mean_pool(x, batch_idx)
        
        x = self.output_projection(x)
        
        return x 