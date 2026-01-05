"""
Numba-optimized NumPy utility functions for ComfyUI.
Uses JIT compilation for performance acceleration.
Python 3.14 & CUDA 13.x compatible.
"""

import numpy as np
from numba import njit, prange
import logging

# Check if CUDA is available
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
    if CUDA_AVAILABLE:
        logging.info("Numba CUDA support enabled")
except Exception as e:
    CUDA_AVAILABLE = False
    logging.info(f"Numba CUDA not available: {e}")


# =============================================================================
# Image Processing Functions (Optimized with Numba)
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def normalize_image_array(img_array, scale=255.0):
    """
    Fast normalization of image array.
    Args:
        img_array: Input array (H, W, C)
        scale: Scale factor (default 255.0)
    Returns:
        Normalized float32 array
    """
    result = np.empty_like(img_array, dtype=np.float32)
    height, width, channels = img_array.shape
    
    for h in prange(height):
        for w in range(width):
            for c in range(channels):
                result[h, w, c] = img_array[h, w, c] / scale
    
    return result


@njit(parallel=True, cache=True, fastmath=True)
def denormalize_image_array(img_array, scale=255.0):
    """
    Fast denormalization of image array.
    Args:
        img_array: Normalized float array (H, W, C)
        scale: Scale factor (default 255.0)
    Returns:
        Denormalized uint8 array
    """
    result = np.empty_like(img_array, dtype=np.uint8)
    height, width, channels = img_array.shape
    
    for h in prange(height):
        for w in range(width):
            for c in range(channels):
                val = img_array[h, w, c] * scale
                result[h, w, c] = np.clip(val, 0, 255)
    
    return result


@njit(parallel=True, cache=True)
def alpha_composite(rgb, track, alpha):
    """
    Fast alpha compositing of RGB and track with alpha channel.
    Args:
        rgb: RGB image array (H, W, 3)
        track: Track image array (H, W, 3)
        alpha: Alpha channel (H, W, 3)
    Returns:
        Composited image
    """
    height, width, _ = rgb.shape
    result = np.empty_like(rgb)
    
    for h in prange(height):
        for w in range(width):
            for c in range(3):
                a = alpha[h, w, c]
                result[h, w, c] = rgb[h, w, c] * (1 - a) + track[h, w, c] * a
    
    return result


@njit(cache=True)
def create_rotation_matrix_x(theta):
    """Create rotation matrix around X axis."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, cos_t, -sin_t],
                     [0.0, sin_t, cos_t]], dtype=np.float32)


@njit(cache=True)
def create_rotation_matrix_y(theta):
    """Create rotation matrix around Y axis."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array([[cos_t, 0.0, sin_t],
                     [0.0, 1.0, 0.0],
                     [-sin_t, 0.0, cos_t]], dtype=np.float32)


@njit(cache=True)
def create_rotation_matrix_z(theta):
    """Create rotation matrix around Z axis."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array([[cos_t, -sin_t, 0.0],
                     [sin_t, cos_t, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


@njit(cache=True)
def combined_rotation_matrix(theta_x, theta_y, theta_z):
    """
    Create combined rotation matrix from Euler angles.
    Args:
        theta_x, theta_y, theta_z: Rotation angles in radians
    Returns:
        Combined 3x3 rotation matrix
    """
    Rx = create_rotation_matrix_x(theta_x)
    Ry = create_rotation_matrix_y(theta_y)
    Rz = create_rotation_matrix_z(theta_z)
    
    # R = Rz @ Ry @ Rx
    temp = np.dot(Ry, Rx)
    R = np.dot(Rz, temp)
    
    return R


# =============================================================================
# Interpolation Functions
# =============================================================================

@njit(parallel=True, cache=True)
def linear_interpolate_steps(t_steps, num_steps):
    """
    Linear interpolation of time steps.
    Args:
        t_steps: Original time steps array
        num_steps: Number of new steps
    Returns:
        Interpolated array
    """
    n = len(t_steps)
    xs = np.linspace(0.0, 1.0, n)
    new_xs = np.linspace(0.0, 1.0, num_steps)
    
    result = np.empty(num_steps, dtype=t_steps.dtype)
    
    for i in prange(num_steps):
        x = new_xs[i]
        
        # Find interpolation indices
        if x <= xs[0]:
            result[i] = t_steps[0]
        elif x >= xs[-1]:
            result[i] = t_steps[-1]
        else:
            # Binary search for efficiency
            idx = np.searchsorted(xs, x)
            if idx >= n:
                idx = n - 1
            
            if idx == 0:
                result[i] = t_steps[0]
            else:
                # Linear interpolation
                x0, x1 = xs[idx - 1], xs[idx]
                y0, y1 = t_steps[idx - 1], t_steps[idx]
                t = (x - x0) / (x1 - x0)
                result[i] = y0 + t * (y1 - y0)
    
    return result


# =============================================================================
# Array Operations
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def fast_array_min_max(arr):
    """
    Fast computation of min and max along first axis.
    Args:
        arr: Input array (N, M)
    Returns:
        (min_values, max_values) each of shape (M,)
    """
    n, m = arr.shape
    min_vals = np.empty(m, dtype=arr.dtype)
    max_vals = np.empty(m, dtype=arr.dtype)
    
    for j in prange(m):
        min_val = arr[0, j]
        max_val = arr[0, j]
        
        for i in range(1, n):
            val = arr[i, j]
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
        
        min_vals[j] = min_val
        max_vals[j] = max_val
    
    return min_vals, max_vals


@njit(parallel=True, cache=True, fastmath=True)
def fast_clip_array(arr, min_val=0.0, max_val=1.0):
    """
    Fast array clipping.
    Args:
        arr: Input array
        min_val: Minimum value
        max_val: Maximum value
    Returns:
        Clipped array
    """
    result = np.empty_like(arr)
    flat = arr.flatten()
    result_flat = result.flatten()
    
    for i in prange(len(flat)):
        val = flat[i]
        if val < min_val:
            result_flat[i] = min_val
        elif val > max_val:
            result_flat[i] = max_val
        else:
            result_flat[i] = val
    
    return result.reshape(arr.shape)


@njit(parallel=True, cache=True, fastmath=True)
def fast_array_multiply_add(a, b, scale_a=1.0, scale_b=1.0):
    """
    Fast computation of: scale_a * a + scale_b * b
    Args:
        a, b: Input arrays (same shape)
        scale_a, scale_b: Scaling factors
    Returns:
        Result array
    """
    result = np.empty_like(a)
    flat_a = a.flatten()
    flat_b = b.flatten()
    result_flat = result.flatten()
    
    for i in prange(len(flat_a)):
        result_flat[i] = scale_a * flat_a[i] + scale_b * flat_b[i]
    
    return result.reshape(a.shape)


# =============================================================================
# Mask Operations
# =============================================================================

@njit(parallel=True, cache=True)
def create_dilation_kernel(size=3, center_val=1.0, edge_val=1.0):
    """
    Create dilation kernel.
    Args:
        size: Kernel size
        center_val: Center value
        edge_val: Edge value
    Returns:
        Kernel array
    """
    kernel = np.full((size, size), edge_val, dtype=np.float32)
    kernel[size//2, size//2] = center_val
    return kernel


@njit(parallel=True, cache=True)
def apply_kernel_2d(image, kernel):
    """
    Apply 2D convolution kernel to image.
    Args:
        image: Input 2D array (H, W)
        kernel: Kernel array (K, K)
    Returns:
        Filtered image
    """
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    
    result = np.zeros_like(image)
    
    for i in prange(pad_h, h - pad_h):
        for j in range(pad_w, w - pad_w):
            sum_val = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    sum_val += image[i - pad_h + ki, j - pad_w + kj] * kernel[ki, kj]
            result[i, j] = sum_val
    
    return result


# =============================================================================
# Statistics
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def fast_mean_std(arr):
    """
    Fast computation of mean and standard deviation.
    Args:
        arr: Input array
    Returns:
        (mean, std)
    """
    flat = arr.flatten()
    n = len(flat)
    
    # Compute mean
    sum_val = 0.0
    for i in prange(n):
        sum_val += flat[i]
    mean = sum_val / n
    
    # Compute std
    sum_sq = 0.0
    for i in prange(n):
        diff = flat[i] - mean
        sum_sq += diff * diff
    std = np.sqrt(sum_sq / n)
    
    return mean, std


# =============================================================================
# Utility Functions (Non-JIT for compatibility)
# =============================================================================

def prepare_array_for_numba(arr):
    """
    Prepare array for Numba processing (ensure C-contiguous).
    Args:
        arr: Input array
    Returns:
        C-contiguous array
    """
    if not arr.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(arr)
    return arr


def log_numba_info():
    """Log Numba configuration information."""
    logging.info(f"Numba-optimized NumPy functions loaded")
    logging.info(f"CUDA support: {'Enabled' if CUDA_AVAILABLE else 'Disabled'}")
    logging.info(f"Parallel execution: Enabled (CPU)")
    logging.info(f"Cache: Enabled")
    logging.info(f"Fast math: Enabled")


# Initialize logging
log_numba_info()
