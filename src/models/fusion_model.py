"""Multi-view feature fusion model.

For fusing feature outputs from GIN and Brain3DCNN models.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class FusionModel(nn.Module):
    """Multi-view feature fusion model."""

    def __init__(self,
                 input_size: int,
                 num_classes: int,
                 dropout_rate: float = 0.5) -> None:
        """Initialize fusion model."""
        super(FusionModel, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Feature fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification layer
        self.classifier = nn.Linear(input_size // 4, num_classes)

    def get_num_parameters(self) -> int:
        """Get number of model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, *embeddings) -> tuple:
        """Forward pass.
        
        Args:
            *embeddings: Feature embeddings from different models
            
        Returns:
            Tuple of fused features and classification logits
        """
        # Filter out None values
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        
        if not valid_embeddings:
            raise ValueError("At least one valid embedding input is required")
        
        # Feature concatenation
        if len(valid_embeddings) == 1:
            fused_features = valid_embeddings[0]
        else:
            fused_features = torch.cat(valid_embeddings, dim=1)
        
        # Feature fusion
        fused_features = self.fusion_network(fused_features)
        
        # Classification prediction
        logits = self.classifier(fused_features)
        
        return fused_features, logits

    def get_fusion_features(self, *embeddings: torch.Tensor) -> torch.Tensor:
        """Get fused features without classification.
        
        Args:
            *embeddings: Feature embedding tensors from different views
            
        Returns:
            Fused feature representation tensor
        """
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        
        if len(valid_embeddings) == 0:
            raise ValueError("At least one valid feature embedding must be provided")
        
        if len(valid_embeddings) == 1:
            return valid_embeddings[0]
        else:
            return torch.cat(valid_embeddings, dim=1) 