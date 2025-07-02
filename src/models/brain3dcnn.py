"""Brain 3D CNN model implementation.

3D convolutional neural network for processing higher-order interactions in brain networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class E2E3DBlock(nn.Module):
    """Edge-to-Edge 3D convolution block for processing higher-order interaction features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(E2E3DBlock, self).__init__()
        
        # Distribute output channels among the three directions
        # Make sure the total adds up to out_channels
        planes = out_channels // 3
        remainder = out_channels % 3
        
        planes_d = planes + (1 if remainder > 0 else 0)
        planes_h = planes + (1 if remainder > 1 else 0)  
        planes_w = planes
        
        self.conv_d = nn.Conv3d(in_channels, planes_d, (1, 3, 3), padding=(0, 1, 1))
        self.conv_h = nn.Conv3d(in_channels, planes_h, (3, 1, 3), padding=(1, 0, 1))
        self.conv_w = nn.Conv3d(in_channels, planes_w, (3, 3, 1), padding=(1, 1, 0))
        
        # The concatenated output will have exactly out_channels
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out_d = self.conv_d(x)
        out_h = self.conv_h(x)
        out_w = self.conv_w(x)
        
        # Concatenate the three directional features
        out = torch.cat([out_d, out_h, out_w], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


class Brain3DCNN(nn.Module):
    """3D CNN model for processing higher-order interactions in brain networks."""

    def __init__(self,
                 example_tensor: torch.Tensor,
                 embedding_dim: int = 64,
                 channels: tuple = (32, 64),
                 dropout_rate: float = 0.5) -> None:
        """Initialize Brain3DCNN model."""
        super(Brain3DCNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        
        # Get input dimensions from example tensor
        # With unsqueeze(1) in trainer, we expect shape: [batch_size, 1, 8, 116, 116]
        
        if len(example_tensor.shape) != 5:
            raise ValueError(f"Expected 5D tensor [batch, channels, depth, height, width], got {len(example_tensor.shape)}D: {example_tensor.shape}")
        
        batch_size, input_channels, depth, height, width = example_tensor.shape
        
        # For xiaorong-style data loading with unsqueeze(1):
        # - input_channels should be 1 (added by unsqueeze)
        # - depth should be 8 (original O-information matrices)
        # - height, width should be 116 (brain regions)
        
        self.input_channels = input_channels  # Should be 1
        self.input_depth = depth             # Should be 8
        self.spatial_dim = height            # Should be 116
        
        print(f"Brain3DCNN initialized:")
        print(f"  Input channels: {self.input_channels}")
        print(f"  Input depth: {self.input_depth}")
        print(f"  Spatial dimension: {self.spatial_dim}")
        
        # E2E 3D convolution layers - channels parameter refers to model architecture
        self.e2e_conv1 = E2E3DBlock(self.input_channels, channels[0])
        self.e2e_conv2 = E2E3DBlock(channels[0], channels[1])
        
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Calculate pooled feature dimensions
        pooled_size = channels[1]
        
        # E2N layer: Edge-to-Node aggregation
        # Map from pooled features to the depth dimension (8 O-information matrices)
        self.e2n = nn.Linear(pooled_size, self.input_depth)
        
        # N2G layer: Node-to-Graph aggregation
        self.n2g = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.input_depth, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, embedding_dim)
        )

    def get_num_parameters(self) -> int:
        """Get number of model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Validate input dimensions
        if len(x.shape) != 5:
            raise ValueError(f"Expected 5D input tensor [batch, channels, depth, height, width], got {len(x.shape)}D: {x.shape}")
        
        # E2E convolution
        x = self.e2e_conv1(x)
        x = self.e2e_conv2(x)
        
        # Global pooling
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # E2N aggregation
        x = self.e2n(x)
        x = F.relu(x)
        
        # N2G aggregation
        x = x.unsqueeze(2)
        x = self.n2g(x)
        x = x.squeeze(2)
        
        # Final embedding
        x = self.fc(x)
        
        return x 