"""
Numba-optimized NumPy utility functions for ComfyUI.

This module provides JIT-compiled utilities for image processing and array operations,
with graceful fallbacks to pure NumPy when Numba is unavailable. Optimized for NumPy 2.x
features including enhanced typing, array API compliance, and free-threaded Python support.

Attributes:
    USE_NUMBA: Boolean flag to enable/disable Numba JIT compilation.
    CUDA_AVAILABLE: Boolean indicating CUDA support availability.
"""

import os
import logging
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray, ArrayLike, DTypeLike

try:
    from numba import njit, prange
    NUMBA_IMPORTED = True
except ImportError as e:
    NUMBA_IMPORTED = False
    logging.warning(f"Numba not available, falling back to NumPy-only path: {e}")

    def njit(*args, **kwargs):
        """No-op decorator for Numba compatibility when unavailable."""
        def wrapper(func):
            return func
        return wrapper

    def prange(*args, **kwargs):
        return range(*args)

# CUDA detection
try:
    if NUMBA_IMPORTED:
        from numba import cuda
        CUDA_AVAILABLE = cuda.is_available()
        if CUDA_AVAILABLE:
            logging.info("Numba CUDA support enabled")
    else:
        CUDA_AVAILABLE = False
except Exception as e:
    CUDA_AVAILABLE = False
    logging.warning(f"Numba CUDA not available: {e}")

# Runtime flag for Numba usage
USE_NUMBA = NUMBA_IMPORTED and os.getenv("COMFY_USE_NUMBA", "1").lower() not in ("0", "false", "no")


# =============================================================================
# Image Processing Functions (Optimized with Numba)
# =============================================================================

def _normalize_image_array_numpy(img_array, scale=255.0):
    """Pure NumPy fallback for image normalization (NumPy 2.x style)."""
    return np.asarray(img_array, dtype=np.float32, copy=False) / scale


@njit(parallel=True, cache=True, fastmath=True)
def _normalize_image_array_numba(img_array, scale=255.0):
    result = np.empty_like(img_array, dtype=np.float32)
    height, width, channels = img_array.shape

    for h in prange(height):
        for w in range(width):
            for c in range(channels):
                result[h, w, c] = img_array[h, w, c] / scale

    return result


def normalize_image_array(
    img_array: ArrayLike,
    scale: float = 255.0
) -> NDArray[np.float32]:
    """Normalize image array to [0, 1] range.

    Uses Numba JIT compilation when available for performance, otherwise falls back
    to vectorized NumPy operations.

    Args:
        img_array: Input image array of shape (H, W, C) with uint8 dtype.
        scale: Scaling factor, typically 255.0 for uint8 images.

    Returns:
        Normalized float32 array in [0, 1] range.

    Raises:
        ValueError: If input array has invalid shape or dtype.
    """
    if img_array.ndim != 3:
        raise ValueError(f"Expected 3D array (H, W, C), got {img_array.ndim}D")
    if USE_NUMBA and img_array.ndim == 3:
        return _normalize_image_array_numba(img_array, scale)
    return _normalize_image_array_numpy(img_array, scale)


def _denormalize_image_array_numpy(img_array, scale=255.0):
    """Pure NumPy fallback for image denormalization with clipping (NumPy 2.x style)."""
    return np.clip(np.asarray(img_array, dtype=np.float32, copy=False) * scale, 0.0, 255.0).astype(np.uint8, copy=False)


@njit(parallel=True, cache=True, fastmath=True)
def _denormalize_image_array_numba(img_array, scale=255.0):
    result = np.empty_like(img_array, dtype=np.uint8)
    height, width, channels = img_array.shape

    for h in prange(height):
        for w in range(width):
            for c in range(channels):
                val = img_array[h, w, c] * scale
                # Manual clipping for Numba compatibility with scalars
                if val < 0.0:
                    val = 0.0
                elif val > 255.0:
                    val = 255.0
                result[h, w, c] = np.uint8(val)

    return result


def denormalize_image_array(
    img_array: ArrayLike,
    scale: float = 255.0
) -> NDArray[np.uint8]:
    """Denormalize image array to uint8 range with clipping.

    Uses Numba JIT compilation when available for performance, otherwise falls back
    to vectorized NumPy operations with safe clipping.

    Args:
        img_array: Input normalized float array of shape (H, W, C) in [0, 1].
        scale: Scaling factor, typically 255.0 for uint8 output.

    Returns:
        Denormalized uint8 array with values clipped to [0, 255].

    Raises:
        ValueError: If input array has invalid shape.
    """
    if img_array.ndim != 3:
        raise ValueError(f"Expected 3D array (H, W, C), got {img_array.ndim}D")
    if USE_NUMBA and img_array.ndim == 3:
        return _denormalize_image_array_numba(img_array, scale)
    return _denormalize_image_array_numpy(img_array, scale)


def _alpha_composite_numpy(rgb, track, alpha):
    """Pure NumPy alpha compositing fallback (NumPy 2.x vectorized)."""
    return np.asarray(rgb, dtype=np.float32, copy=False) * (1.0 - np.asarray(alpha, dtype=np.float32, copy=False)) + np.asarray(track, dtype=np.float32, copy=False) * np.asarray(alpha, dtype=np.float32, copy=False)


@njit(parallel=True, cache=True)
def _alpha_composite_numba(rgb, track, alpha):
    height, width, _ = rgb.shape
    result = np.empty_like(rgb)

    for h in prange(height):
        for w in range(width):
            for c in range(3):
                a = alpha[h, w, c]
                result[h, w, c] = rgb[h, w, c] * (1 - a) + track[h, w, c] * a

    return result


def alpha_composite(
    rgb: ArrayLike,
    track: ArrayLike,
    alpha: ArrayLike
) -> NDArray[np.float32]:
    """Perform alpha compositing of RGB and track images.

    Composites foreground (track) over background (rgb) using alpha channel.
    Uses Numba JIT when available for pixel-wise operations.

    Args:
        rgb: Background RGB image array (H, W, 3).
        track: Foreground RGB image array (H, W, 3).
        alpha: Alpha channel array (H, W, 3) in [0, 1].

    Returns:
        Composited float32 image array.

    Raises:
        ValueError: If input arrays have mismatched shapes.
    """
    if rgb.shape != track.shape or rgb.shape != alpha.shape:
        raise ValueError(f"Shape mismatch: rgb {rgb.shape}, track {track.shape}, alpha {alpha.shape}")
    if rgb.shape[-1] != 3:
        raise ValueError(f"Expected last dimension 3 for RGB, got {rgb.shape[-1]}")
    if USE_NUMBA and rgb.ndim == 3:
        return _alpha_composite_numba(rgb, track, alpha)
    return _alpha_composite_numpy(rgb, track, alpha)


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


@njit(parallel=True, cache=True)
def log_linear_interpolate_steps(t_steps, num_steps):
    """
    Log-linear interpolation of decreasing time steps.

    Args:
        t_steps: Original time steps array (assumed positive)
        num_steps: Number of new steps
    Returns:
        Interpolated array matching the original order
    """
    n = len(t_steps)
    xs = np.linspace(0.0, 1.0, n)
    new_xs = np.linspace(0.0, 1.0, num_steps)

    result = np.empty(num_steps, dtype=t_steps.dtype)

    # Precompute log of reversed steps to mirror legacy behavior
    log_rev = np.empty(n, dtype=np.float64)
    eps = np.finfo(np.float64).tiny
    for i in range(n):
        val = t_steps[n - 1 - i]
        if val < eps:
            val = eps
        log_rev[i] = np.log(val)

    for i in prange(num_steps):
        x = new_xs[i]

        if x <= xs[0]:
            log_val = log_rev[0]
        elif x >= xs[-1]:
            log_val = log_rev[-1]
        else:
            idx = np.searchsorted(xs, x)
            if idx >= n:
                idx = n - 1

            if idx == 0:
                log_val = log_rev[0]
            else:
                x0, x1 = xs[idx - 1], xs[idx]
                y0, y1 = log_rev[idx - 1], log_rev[idx]
                t = (x - x0) / (x1 - x0)
                log_val = y0 + t * (y1 - y0)

        # Reverse back to original ordering while exponentiating
        result[num_steps - 1 - i] = np.exp(log_val)

    return result


# =============================================================================
# Array Operations
# =============================================================================

def _fast_array_min_max_numpy(arr):
    return arr.min(axis=0), arr.max(axis=0)


@njit(parallel=True, cache=True, fastmath=True)
def _fast_array_min_max_numba(arr):
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


def fast_array_min_max(arr: ArrayLike) -> Tuple[NDArray, NDArray]:
    """Compute min and max along first axis efficiently.

    Uses Numba parallel loops when available for large arrays.

    Args:
        arr: Input 2D array of shape (N, M).

    Returns:
        Tuple of (min_values, max_values) each of shape (M,).

    Raises:
        ValueError: If input is not 2D.
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")
    if USE_NUMBA and arr.ndim >= 2:
        return _fast_array_min_max_numba(arr)
    return _fast_array_min_max_numpy(arr)


def _fast_clip_array_numpy(arr, min_val=0.0, max_val=1.0):
    return np.clip(np.asarray(arr, copy=False), min_val, max_val)


@njit(parallel=True, cache=True, fastmath=True)
def _fast_clip_array_numba(arr, min_val=0.0, max_val=1.0):
    shape = arr.shape
    flat = arr.flatten()
    result = np.empty(len(flat), dtype=arr.dtype)

    for i in prange(len(flat)):
        val = flat[i]
        if val < min_val:
            result[i] = min_val
        elif val > max_val:
            result[i] = max_val
        else:
            result[i] = val

    return result.reshape(shape)


def fast_clip_array(
    arr: ArrayLike,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> NDArray:
    """Clip array values to specified range efficiently.

    Uses Numba parallel operations when available.

    Args:
        arr: Input array to clip.
        min_val: Minimum value to clip to.
        max_val: Maximum value to clip to.

    Returns:
        Clipped array with same shape and dtype as input.

    Raises:
        ValueError: If min_val >= max_val.
    """
    if min_val >= max_val:
        raise ValueError(f"min_val {min_val} must be < max_val {max_val}")
    if USE_NUMBA:
        return _fast_clip_array_numba(arr, min_val, max_val)
    return _fast_clip_array_numpy(arr, min_val, max_val)


def _fast_array_multiply_add_numpy(a, b, scale_a=1.0, scale_b=1.0):
    return scale_a * np.asarray(a, copy=False) + scale_b * np.asarray(b, copy=False)


@njit(parallel=True, cache=True, fastmath=True)
def _fast_array_multiply_add_numba(a, b, scale_a=1.0, scale_b=1.0):
    shape = a.shape
    flat_a = a.flatten()
    flat_b = b.flatten()
    result = np.empty(len(flat_a), dtype=a.dtype)

    for i in prange(len(flat_a)):
        result[i] = scale_a * flat_a[i] + scale_b * flat_b[i]

    return result.reshape(shape)


def fast_array_multiply_add(a, b, scale_a=1.0, scale_b=1.0):
    """Linear blend with Numba when enabled, else NumPy."""
    if USE_NUMBA:
        return _fast_array_multiply_add_numba(a, b, scale_a, scale_b)
    return _fast_array_multiply_add_numpy(a, b, scale_a, scale_b)


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


def _apply_kernel_2d_numpy(image, kernel):
    """Vectorized 2D convolution fallback (zeroes borders to match legacy, NumPy 2.x style)."""
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(np.asarray(image, copy=False), ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    result = np.zeros((h, w), dtype=image.dtype)

    # Accumulate contributions
    for ki in range(kh):
        for kj in range(kw):
            result += kernel[ki, kj] * padded[ki:ki + h, kj:kj + w]

    # Zero borders to mirror legacy loop bounds
    if pad_h:
        result[:pad_h, :] = 0
        result[-pad_h:, :] = 0
    if pad_w:
        result[:, :pad_w] = 0
        result[:, -pad_w:] = 0

    return result


@njit(parallel=True, cache=True)
def _apply_kernel_2d_numba(image, kernel):
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


def apply_kernel_2d(image, kernel):
    """Apply kernel using Numba when enabled, else vectorized NumPy."""
    if USE_NUMBA:
        return _apply_kernel_2d_numba(image, kernel)
    return _apply_kernel_2d_numpy(image, kernel)


# =============================================================================
# Statistics
# =============================================================================

def _fast_mean_std_numpy(arr):
    arr_flat = np.asarray(arr, copy=False).flatten()
    return float(np.mean(arr_flat)), float(np.std(arr_flat))


@njit(parallel=True, cache=True, fastmath=True)
def _fast_mean_std_numba(arr):
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


def fast_mean_std(arr):
    """Mean/std with Numba when enabled, else NumPy."""
    if USE_NUMBA:
        return _fast_mean_std_numba(arr)
    return _fast_mean_std_numpy(arr)


# =============================================================================
# Utility Functions (Non-JIT for compatibility)
# =============================================================================

def prepare_array_for_numba(arr):
    """
    Prepare array for Numba processing (ensure C-contiguous, NumPy 2.x style).
    Args:
        arr: Input array
    Returns:
        C-contiguous array
    """
    arr = np.asarray(arr, copy=False)
    if not arr.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(arr)
    return arr


def log_numba_info():
    """Log Numba configuration information."""
    logging.info("Numba-optimized NumPy functions loaded")
    logging.info(f"Numba imported: {NUMBA_IMPORTED}")
    logging.info(f"Numba enabled (COMFY_USE_NUMBA): {USE_NUMBA}")
    logging.info(f"CUDA support: {'Enabled' if CUDA_AVAILABLE else 'Disabled'}")
    logging.info("Parallel execution: Enabled (CPU path)")
    logging.info("Cache: Enabled")
    logging.info("Fast math: Enabled where applicable")


# Initialize logging
log_numba_info()
