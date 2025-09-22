"""
GPU-adaptive matmul dispatcher with registry pattern.

This module provides a unified interface for matrix multiplication operations
that automatically dispatches to the most optimized implementation available
for the current GPU architecture.
"""

import torch
from typing import Tuple, Callable, List, Optional
from system import SYSTEM_CONFIG

# Import kernel implementations (T4 optimized and fallback only)
from . import _fallback_impl


class MatmulRegistry:
    """
    Registry for matmul kernels with automatic dispatch.
    
    Maintains a list of (condition, kernel) pairs where condition is a function
    that returns True if the kernel can be used, and kernel is the actual
    implementation function.
    """
    
    def __init__(self):
        self.kernels: List[Tuple[Callable[[], bool], Callable]] = []
        self.fallback_kernel: Optional[Callable] = None
        self._setup_registry()
    
    def _setup_registry(self):
        """Setup the kernel registry optimized for T4 GPU."""
        
        # T4 optimized kernels (highest priority)
        self.kernels.extend([
            # FP16 matmul for T4 (optimal for T4's tensor cores)
            (
                lambda: SYSTEM_CONFIG.architecture == "t4" and SYSTEM_CONFIG.has_tensor_cores,
                _fallback_impl.matmul_fp16
            ),
            # BF16 matmul for T4 (if supported)
            (
                lambda: SYSTEM_CONFIG.architecture == "t4" and SYSTEM_CONFIG.has_bf16_support,
                _fallback_impl.matmul_fp16  # Use FP16 as T4 doesn't have native BF16
            ),
        ])
        
        # Generic tensor core support for non-T4 GPUs
        self.kernels.extend([
            # FP16 for any GPU with tensor cores
            (
                lambda: SYSTEM_CONFIG.has_tensor_cores and SYSTEM_CONFIG.architecture != "t4",
                _fallback_impl.matmul_fp16
            ),
        ])
        
        # Set fallback kernel
        self.fallback_kernel = _fallback_impl.matmul_generic
    
    def get_best_kernel(self) -> Callable:
        """
        Find the best available kernel for the current system.
        
        Returns:
            The most optimized kernel function available
        """
        for condition, kernel in self.kernels:
            try:
                if condition():
                    return kernel
            except Exception as e:
                # If a condition fails, continue to next kernel
                print(f"Warning: Kernel condition failed: {e}")
                continue
        
        # Return fallback if no specialized kernel matches
        return self.fallback_kernel
    
    def get_all_available_kernels(self) -> List[Tuple[str, Callable]]:
        """
        Get all kernels that are available for the current system.
        
        Returns:
            List of (description, kernel_function) tuples
        """
        available = []
        
        for condition, kernel in self.kernels:
            try:
                if condition():
                    kernel_name = kernel.__name__
                    available.append((kernel_name, kernel))
            except Exception:
                continue
        
        # Always include fallback
        available.append(("fallback", self.fallback_kernel))
        
        return available


# Global registry instance
_registry = MatmulRegistry()


def matmul(x: torch.Tensor, w: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    GPU-adaptive matrix multiplication.
    
    Automatically dispatches to the most optimized implementation available
    for the current GPU architecture.
    
    Args:
        x: Input tensor [batch_size, seq_len, d_model]
        w: Weight tensor [d_model, d_out]
        *args: Additional arguments passed to the kernel
        **kwargs: Additional keyword arguments passed to the kernel
        
    Returns:
        Output tensor [batch_size, seq_len, d_out]
    """
    kernel = _registry.get_best_kernel()
    return kernel(x, w, *args, **kwargs)


def matmul_with_info(x: torch.Tensor, w: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, str]:
    """
    Matrix multiplication with kernel information.
    
    Returns both the result and the name of the kernel used.
    
    Args:
        x: Input tensor [batch_size, seq_len, d_model]
        w: Weight tensor [d_model, d_out]
        *args: Additional arguments passed to the kernel
        **kwargs: Additional keyword arguments passed to the kernel
        
    Returns:
        Tuple of (output_tensor, kernel_name)
    """
    available_kernels = _registry.get_all_available_kernels()
    kernel_name, kernel = available_kernels[0]  # Best kernel is first
    
    result = kernel(x, w, *args, **kwargs)
    return result, kernel_name


def get_available_kernels() -> List[str]:
    """
    Get list of available kernel names for the current system.
    
    Returns:
        List of kernel names that can be used
    """
    available = _registry.get_all_available_kernels()
    return [name for name, _ in available]


def print_kernel_info():
    """Print information about available kernels for debugging."""
    config = SYSTEM_CONFIG
    available = _registry.get_all_available_kernels()
    
    print("🔧 Available Matmul Kernels:")
    print(f"   Architecture: {config.architecture}")
    print(f"   Compute Capability: {config.capability}")
    print(f"   Available kernels:")
    
    for i, (name, kernel) in enumerate(available):
        status = "✅" if i == 0 else "📋"
        print(f"   {status} {name}: {kernel.__name__}")


# Convenience functions for specific use cases
def matmul_fp16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """FP16 matmul optimized for T4 GPU."""
    return _fallback_impl.matmul_fp16(x, w)


def matmul_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """BF16 matmul - falls back to FP16 for T4 GPU."""
    if SYSTEM_CONFIG.has_bf16_support:
        return _fallback_impl.matmul_fp16(x, w)  # Use FP16 as T4 doesn't have native BF16
    else:
        return matmul(x, w)


if __name__ == "__main__":
    print_kernel_info()
