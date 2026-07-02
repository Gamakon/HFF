"""
GPU-Accelerated HF1 Calculations

Provides GPU acceleration for HF1 fitness calculations when available,
with automatic fallback to optimized CPU implementation.
"""

import numpy as np
from typing import Optional, Tuple
import warnings

# Try to import GPU libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Check what's available
GPU_BACKEND = None
if TORCH_AVAILABLE and torch.cuda.is_available():
    GPU_BACKEND = 'torch'
elif CUPY_AVAILABLE:
    try:
        cp.cuda.Device()
        GPU_BACKEND = 'cupy'
    except:
        pass

def calculate_hf1_gpu_torch(objectives: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """Calculate HF1 using PyTorch GPU acceleration"""
    
    # Move to GPU
    device = torch.device('cuda')
    objectives_gpu = torch.from_numpy(objectives).float().to(device)
    
    # Global normalization
    f_min = torch.min(objectives_gpu, dim=0)[0]
    f_max = torch.max(objectives_gpu, dim=0)[0]
    f_range = f_max - f_min
    f_range[f_range == 0] = 1.0
    
    normalized = (objectives_gpu - f_min) / f_range
    
    # Fractional coordinate decomposition
    n_solutions, n_objectives = normalized.shape
    
    # Calculate total energy for each solution
    total_energy = torch.sum(normalized ** 2, dim=1, keepdim=True)
    
    # Avoid division by zero
    total_energy[total_energy == 0] = 1e-10
    
    # Project to hypersphere
    y = torch.sign(normalized) * torch.sqrt((normalized ** 2) / total_energy)
    
    # North pole reference
    north_pole = torch.ones(n_objectives, device=device) / np.sqrt(n_objectives)
    
    # Calculate angular distances
    dot_products = torch.matmul(y, north_pole)
    dot_products = torch.clamp(dot_products, -1.0, 1.0)
    angles = torch.acos(dot_products)
    
    # Apply alpha parameter
    hf1_scores = angles ** alpha
    
    # Move back to CPU
    return hf1_scores.cpu().numpy()

def calculate_hf1_gpu_cupy(objectives: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """Calculate HF1 using CuPy GPU acceleration"""
    
    # Move to GPU
    objectives_gpu = cp.asarray(objectives)
    
    # Global normalization
    f_min = cp.min(objectives_gpu, axis=0)
    f_max = cp.max(objectives_gpu, axis=0)
    f_range = f_max - f_min
    f_range[f_range == 0] = 1.0
    
    normalized = (objectives_gpu - f_min) / f_range
    
    # Fractional coordinate decomposition
    n_solutions, n_objectives = normalized.shape
    
    # Calculate total energy for each solution
    total_energy = cp.sum(normalized ** 2, axis=1, keepdims=True)
    
    # Avoid division by zero
    total_energy[total_energy == 0] = 1e-10
    
    # Project to hypersphere
    y = cp.sign(normalized) * cp.sqrt((normalized ** 2) / total_energy)
    
    # North pole reference
    north_pole = cp.ones(n_objectives) / cp.sqrt(n_objectives)
    
    # Calculate angular distances
    dot_products = cp.dot(y, north_pole)
    dot_products = cp.clip(dot_products, -1.0, 1.0)
    angles = cp.arccos(dot_products)
    
    # Apply alpha parameter
    hf1_scores = angles ** alpha
    
    # Move back to CPU
    return cp.asnumpy(hf1_scores)

def calculate_hf1_vectorized_cpu(objectives: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """Optimized CPU implementation using vectorization"""
    
    # Global normalization
    f_min = np.min(objectives, axis=0)
    f_max = np.max(objectives, axis=0)
    f_range = f_max - f_min
    f_range[f_range == 0] = 1.0
    
    normalized = (objectives - f_min) / f_range
    
    # Fractional coordinate decomposition
    n_solutions, n_objectives = normalized.shape
    
    # Vectorized energy calculation
    total_energy = np.sum(normalized ** 2, axis=1, keepdims=True)
    
    # Avoid division by zero
    total_energy[total_energy == 0] = 1e-10
    
    # Vectorized projection
    y = np.sign(normalized) * np.sqrt((normalized ** 2) / total_energy)
    
    # North pole reference
    north_pole = np.ones(n_objectives) / np.sqrt(n_objectives)
    
    # Vectorized dot product
    dot_products = np.dot(y, north_pole)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles = np.arccos(dot_products)
    
    # Apply alpha parameter
    hf1_scores = angles ** alpha
    
    return hf1_scores

def calculate_hf1_batch(objectives: np.ndarray, 
                       alpha: float = 2.0,
                       batch_size: int = 10000,
                       backend: Optional[str] = None) -> np.ndarray:
    """
    Calculate HF1 scores with automatic backend selection and batching.
    
    Args:
        objectives: Population objectives matrix
        alpha: Angular distance exponent
        batch_size: Process in batches to avoid memory issues
        backend: Force specific backend ('torch', 'cupy', 'cpu', or None for auto)
    
    Returns:
        HF1 scores array
    """
    
    n_solutions = objectives.shape[0]
    
    # Select backend
    if backend is None:
        backend = GPU_BACKEND if GPU_BACKEND else 'cpu'
    
    # Process in batches for large populations
    if n_solutions > batch_size:
        hf1_scores = np.zeros(n_solutions)
        for i in range(0, n_solutions, batch_size):
            end_idx = min(i + batch_size, n_solutions)
            batch = objectives[i:end_idx]
            
            if backend == 'torch' and TORCH_AVAILABLE:
                hf1_scores[i:end_idx] = calculate_hf1_gpu_torch(batch, alpha)
            elif backend == 'cupy' and CUPY_AVAILABLE:
                hf1_scores[i:end_idx] = calculate_hf1_gpu_cupy(batch, alpha)
            else:
                hf1_scores[i:end_idx] = calculate_hf1_vectorized_cpu(batch, alpha)
        
        return hf1_scores
    else:
        # Small population - process all at once
        if backend == 'torch' and TORCH_AVAILABLE:
            return calculate_hf1_gpu_torch(objectives, alpha)
        elif backend == 'cupy' and CUPY_AVAILABLE:
            return calculate_hf1_gpu_cupy(objectives, alpha)
        else:
            return calculate_hf1_vectorized_cpu(objectives, alpha)

def get_available_backends():
    """Return list of available computation backends"""
    backends = ['cpu']  # CPU always available
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        backends.append('torch')
        
    if CUPY_AVAILABLE:
        try:
            cp.cuda.Device()
            backends.append('cupy')
        except:
            pass
    
    return backends

# Module initialization message
if GPU_BACKEND:
    print(f"🚀 GPU acceleration available: {GPU_BACKEND}")
else:
    print("💻 Using optimized CPU implementation")