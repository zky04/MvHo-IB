"""Information bottleneck utility functions.

This module implements the Rényi entropy calculation for the information bottleneck
principle in the MvHo-IB framework.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def calculate_gram_matrix(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Calculate Gaussian kernel Gram matrix.
    
    Uses RBF (Radial Basis Function) kernel to compute the Gram matrix of input data,
    which is used for subsequent Rényi entropy estimation.
    
    Args:
        x: Input feature tensor, shape [batch_size, feature_dim]
        sigma: Gaussian kernel bandwidth parameter, controls kernel function smoothness
        
    Returns:
        Gram matrix, shape [batch_size, batch_size]
        
    Note:
        Gram matrix K[i,j] = exp(-||x_i - x_j||^2 / sigma)
    """
    # Flatten input to 2D matrix
    x = x.view(x.shape[0], -1)
    
    # Compute L2 norm squared for each sample
    instances_norm = torch.sum(x ** 2, dim=-1).reshape((-1, 1))
    
    # Compute pairwise distances: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2<x_i, x_j>
    pairwise_distances = (-2 * torch.mm(x, x.t()) + 
                         instances_norm + 
                         instances_norm.t())
    
    # Apply Gaussian kernel function
    gram_matrix = torch.exp(-pairwise_distances / sigma)
    
    return gram_matrix


def renyi_entropy(x: torch.Tensor, 
                  sigma: float = 5.0, 
                  alpha: float = 1.01, 
                  epsilon: float = 1e-3) -> torch.Tensor:
    """Calculate matrix-based Rényi α-order entropy.
    
    Uses eigenvalue decomposition method to calculate Rényi entropy, which is
    a key metric for quantifying information compression in information bottleneck principle.
    
    Args:
        x: Input feature tensor, shape [batch_size, feature_dim]
        sigma: Gaussian kernel bandwidth parameter, affects entropy estimation smoothness
        alpha: Order of Rényi entropy, usually set to values close to 1 (e.g., 1.01)
        epsilon: Numerical stability parameter, prevents matrix singularity
        
    Returns:
        Rényi α-order entropy value, scalar tensor
        
    Raises:
        RuntimeError: When eigenvalue decomposition fails
        
    Note:
        Rényi α-order entropy is defined as: H_α(X) = 1/(1-α) * log(Σ λ_i^α)
        where λ_i are eigenvalues of the normalized Gram matrix.
    """
    # Convert to double precision for better numerical stability
    x = x.double()
    
    # Calculate Gram matrix
    k = calculate_gram_matrix(x, sigma)
    
    # Normalize Gram matrix
    trace_k = torch.trace(k)
    if trace_k == 0:
        trace_k = 1e-10  # Avoid division by zero
    k = k / trace_k
    
    # Add regularization term to ensure matrix is positive definite
    k += epsilon * torch.eye(k.size(0), dtype=torch.double, device=k.device)
    
    try:
        # Calculate eigenvalues (only need eigenvalues, not eigenvectors)
        eigenvalues = torch.linalg.eigh(k, UPLO='U')[0]
        # Ensure eigenvalues are positive
        eigenvalues = torch.abs(eigenvalues)
        
    except RuntimeError as e:
        # Fallback handling when eigenvalue decomposition fails
        print(f"Eigenvalue decomposition failed: {e}")
        return torch.tensor(0.0, device=x.device, dtype=torch.float32)
    
    # Calculate Rényi α-order entropy
    # H_α(X) = 1/(1-α) * log(Σ λ_i^α)
    eigenvalues_pow_alpha = eigenvalues ** alpha
    sum_eigenvalues_pow_alpha = torch.sum(eigenvalues_pow_alpha)
    
    # Use log2 to calculate entropy value (in bits)
    entropy = (1.0 / (1.0 - alpha)) * torch.log2(sum_eigenvalues_pow_alpha + 1e-10)
    
    return entropy.float()


def compute_renyi_entropy(embeddings: torch.Tensor, 
                         alpha: float = 1.01, 
                         sigma: float = 1.0) -> torch.Tensor:
    """Compute Rényi entropy for information bottleneck regularization.
    
    The Rényi entropy is used to measure the amount of information in feature
    embeddings and serves as a regularization term to prevent information
    bottleneck from becoming too narrow.
    
    Args:
        embeddings: Feature embeddings tensor of shape [batch_size, feature_dim]
        alpha: Rényi entropy parameter, must be > 1
        sigma: Gaussian kernel bandwidth parameter
        
    Returns:
        Scalar tensor representing the Rényi entropy value
        
    Raises:
        ValueError: If alpha <= 1
    """
    return renyi_entropy(embeddings, sigma=sigma, alpha=alpha)


def compute_mutual_information(embeddings1: torch.Tensor, 
                              embeddings2: torch.Tensor,
                              sigma: float = 1.0) -> torch.Tensor:
    """Compute mutual information between two sets of embeddings.
    
    This function estimates the mutual information using kernel density estimation
    with Gaussian kernels.
    
    Args:
        embeddings1: First set of embeddings [batch_size, dim1]
        embeddings2: Second set of embeddings [batch_size, dim2]
        sigma: Gaussian kernel bandwidth parameter
        
    Returns:
        Scalar tensor representing the mutual information estimate
    """
    if embeddings1.size(0) != embeddings2.size(0):
        raise ValueError("Embeddings must have the same batch size")
    
    batch_size = embeddings1.size(0)
    
    if batch_size < 2:
        return torch.tensor(0.0, device=embeddings1.device)
    
    # Concatenate embeddings for joint distribution
    joint_embeddings = torch.cat([embeddings1, embeddings2], dim=1)
    
    # Compute individual and joint entropies
    h1 = renyi_entropy(embeddings1, alpha=1.01, sigma=sigma)
    h2 = renyi_entropy(embeddings2, alpha=1.01, sigma=sigma)
    h_joint = renyi_entropy(joint_embeddings, alpha=1.01, sigma=sigma)
    
    # MI = H(X) + H(Y) - H(X,Y)
    mutual_info = h1 + h2 - h_joint
    
    return mutual_info


def mutual_information_loss(z1: torch.Tensor, 
                           z2: torch.Tensor,
                           sigma: float = 5.0,
                           alpha: float = 1.01) -> torch.Tensor:
    """Calculate mutual information loss between two feature representations.
    
    This function is used to maximize mutual information between different view features,
    promoting consistency in multi-view representation learning.
    
    Args:
        z1: Feature representation from first view
        z2: Feature representation from second view  
        sigma: Gaussian kernel bandwidth parameter
        alpha: Order of Rényi entropy
        
    Returns:
        Negative mutual information value (used as loss function)
    """
    # Calculate marginal entropies
    h_z1 = renyi_entropy(z1, sigma, alpha)
    h_z2 = renyi_entropy(z2, sigma, alpha)
    
    # Calculate joint entropy
    z_joint = torch.cat([z1, z2], dim=1)
    h_z1_z2 = renyi_entropy(z_joint, sigma, alpha)
    
    # Mutual information = H(Z1) + H(Z2) - H(Z1, Z2)
    mutual_info = h_z1 + h_z2 - h_z1_z2
    
    # Return negative value for use as loss (maximize mutual information = minimize negative mutual information)
    return -mutual_info
