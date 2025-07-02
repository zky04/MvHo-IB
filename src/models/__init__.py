"""Neural network models module.

This module contains all neural network models used in the MvHo-IB framework:
- gin_model: GIN-based graph neural network
- brain3dcnn: Specialized 3D convolutional neural network for brain imaging
- fusion_model: Multi-view feature fusion model
"""

from .gin_model import GINModel
from .brain3dcnn import Brain3DCNN, E2E3DBlock
from .fusion_model import FusionModel

__all__ = [
    "GINModel",
    "Brain3DCNN", 
    "E2E3DBlock",
    "FusionModel"
] 