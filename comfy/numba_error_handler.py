"""
Error handling and logging utilities for Numba + NumPy operations.
Provides graceful fallback and comprehensive error tracking.
Python 3.14 & CUDA 13.x compatible.
"""

import logging
import functools
import traceback
import numpy as np
from typing import Callable, Any, Optional
import warnings

# Configure logging
logger = logging.getLogger(__name__)


class NumbaExecutionError(Exception):
    """Custom exception for Numba execution failures."""
    pass


class NumbaFallbackWarning(UserWarning):
    """Warning for Numba fallback to NumPy."""
    pass


def numba_safe_wrapper(fallback_func: Optional[Callable] = None, 
                       function_name: str = "unknown",
                       silent: bool = False):
    """
    Decorator for safe Numba function execution with automatic fallback.
    
    Args:
        fallback_func: NumPy fallback function to use if Numba fails
        function_name: Name for logging purposes
        silent: If True, suppress warnings (only log errors)
    
    Usage:
        @numba_safe_wrapper(fallback_func=numpy_version, function_name="fast_normalize")
        def numba_normalize(arr):
            # Numba optimized code
            pass
    """
    def decorator(numba_func: Callable) -> Callable:
        @functools.wraps(numba_func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                # Attempt Numba execution
                result = numba_func(*args, **kwargs)
                return result
                
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Log the error
                logger.error(
                    f"Numba execution failed for '{function_name}': "
                    f"{error_type}: {error_msg}"
                )
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                
                # Attempt fallback if available
                if fallback_func is not None:
                    if not silent:
                        warning_msg = (
                            f"Falling back to NumPy for '{function_name}'. "
                            f"Performance may be reduced."
                        )
                        warnings.warn(warning_msg, NumbaFallbackWarning)
                        logger.warning(warning_msg)
                    
                    try:
                        result = fallback_func(*args, **kwargs)
                        logger.info(f"NumPy fallback successful for '{function_name}'")
                        return result
                    except Exception as fallback_error:
                        logger.error(
                            f"NumPy fallback also failed for '{function_name}': "
                            f"{type(fallback_error).__name__}: {fallback_error}"
                        )
                        raise NumbaExecutionError(
                            f"Both Numba and NumPy fallback failed for '{function_name}'"
                        ) from fallback_error
                else:
                    # No fallback available
                    raise NumbaExecutionError(
                        f"Numba execution failed for '{function_name}' and no fallback provided"
                    ) from e
        
        return wrapper
    return decorator


def validate_numpy_array(arr: Any, 
                        name: str = "array",
                        dtype: Optional[type] = None,
                        ndim: Optional[int] = None,
                        shape: Optional[tuple] = None) -> np.ndarray:
    """
    Validate NumPy array with comprehensive checks.
    
    Args:
        arr: Input array to validate
        name: Name for error messages
        dtype: Expected dtype (optional)
        ndim: Expected number of dimensions (optional)
        shape: Expected shape (optional, use None for any size in dimension)
    
    Returns:
        Validated NumPy array
    
    Raises:
        TypeError: If input is not a valid array
        ValueError: If array doesn't meet requirements
    """
    # Check if it's array-like
    if not isinstance(arr, (np.ndarray, list, tuple)):
        raise TypeError(f"{name} must be array-like, got {type(arr)}")
    
    # Convert to NumPy array
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.asarray(arr)
        except Exception as e:
            raise ValueError(f"Cannot convert {name} to NumPy array: {e}")
    
    # Check dtype
    if dtype is not None and arr.dtype != dtype:
        logger.warning(
            f"{name} dtype is {arr.dtype}, expected {dtype}. "
            f"Attempting conversion."
        )
        try:
            arr = arr.astype(dtype)
        except Exception as e:
            raise ValueError(f"Cannot convert {name} to dtype {dtype}: {e}")
    
    # Check dimensions
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(
            f"{name} must be {ndim}-dimensional, got {arr.ndim}D array"
        )
    
    # Check shape
    if shape is not None:
        if len(shape) != arr.ndim:
            raise ValueError(
                f"{name} shape mismatch: expected {len(shape)} dimensions, "
                f"got {arr.ndim}"
            )
        for i, (expected, actual) in enumerate(zip(shape, arr.shape)):
            if expected is not None and expected != actual:
                raise ValueError(
                    f"{name} shape mismatch at dimension {i}: "
                    f"expected {expected}, got {actual}"
                )
    
    # Check for NaN/Inf
    if np.issubdtype(arr.dtype, np.floating):
        if np.any(np.isnan(arr)):
            logger.warning(f"{name} contains NaN values")
        if np.any(np.isinf(arr)):
            logger.warning(f"{name} contains Inf values")
    
    return arr


def ensure_contiguous(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """
    Ensure array is C-contiguous for Numba compatibility.
    
    Args:
        arr: Input array
        name: Name for logging
    
    Returns:
        C-contiguous array
    """
    if not arr.flags['C_CONTIGUOUS']:
        logger.debug(f"Converting {name} to C-contiguous layout")
        return np.ascontiguousarray(arr)
    return arr


def log_numba_performance(func_name: str, 
                         input_size: int,
                         execution_time: float,
                         speedup: Optional[float] = None):
    """
    Log performance metrics for Numba functions.
    
    Args:
        func_name: Function name
        input_size: Size of input data
        execution_time: Execution time in seconds
        speedup: Speedup factor vs NumPy (optional)
    """
    if speedup is not None:
        logger.info(
            f"Numba '{func_name}': {input_size} elements, "
            f"{execution_time:.4f}s, {speedup:.2f}x speedup"
        )
    else:
        logger.info(
            f"Numba '{func_name}': {input_size} elements, "
            f"{execution_time:.4f}s"
        )


class NumbaContext:
    """
    Context manager for Numba operations with automatic error handling.
    
    Usage:
        with NumbaContext("image_processing") as ctx:
            result = numba_function(data)
            ctx.log_success()
    """
    
    def __init__(self, operation_name: str, silent: bool = False):
        self.operation_name = operation_name
        self.silent = silent
        self.success = False
        
    def __enter__(self):
        logger.debug(f"Starting Numba operation: {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(
                f"Numba operation '{self.operation_name}' failed: "
                f"{exc_type.__name__}: {exc_val}"
            )
            # Don't suppress the exception
            return False
        
        if self.success and not self.silent:
            logger.debug(f"Numba operation '{self.operation_name}' completed successfully")
        
        return True
    
    def log_success(self):
        """Mark operation as successful."""
        self.success = True


def check_numba_availability() -> dict:
    """
    Check Numba and CUDA availability with detailed diagnostics.
    
    Returns:
        Dictionary with availability status and details
    """
    result = {
        "numba_available": False,
        "numba_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "parallel_enabled": False,
        "errors": []
    }
    
    try:
        import numba
        result["numba_available"] = True
        result["numba_version"] = numba.__version__
        
        # Check parallel support
        try:
            from numba import config
            result["parallel_enabled"] = config.NUMBA_NUM_THREADS > 0
        except Exception as e:
            result["errors"].append(f"Parallel check failed: {e}")
        
        # Check CUDA support
        try:
            from numba import cuda
            result["cuda_available"] = cuda.is_available()
            if result["cuda_available"]:
                result["cuda_version"] = cuda.runtime.get_version()
        except Exception as e:
            result["errors"].append(f"CUDA check failed: {e}")
            
    except ImportError as e:
        result["errors"].append(f"Numba import failed: {e}")
    
    return result


def log_system_info():
    """Log comprehensive system information for debugging."""
    info = check_numba_availability()
    
    logger.info("=== Numba System Information ===")
    logger.info(f"Numba available: {info['numba_available']}")
    if info['numba_version']:
        logger.info(f"Numba version: {info['numba_version']}")
    logger.info(f"CUDA available: {info['cuda_available']}")
    if info['cuda_version']:
        logger.info(f"CUDA version: {info['cuda_version']}")
    logger.info(f"Parallel execution: {'Enabled' if info['parallel_enabled'] else 'Disabled'}")
    
    if info['errors']:
        logger.warning("Errors encountered:")
        for error in info['errors']:
            logger.warning(f"  - {error}")
    
    logger.info("================================")


# Example usage patterns
def create_safe_numba_function(numba_impl: Callable, 
                              numpy_impl: Callable,
                              name: str) -> Callable:
    """
    Factory function to create safe Numba functions with fallback.
    
    Args:
        numba_impl: Numba-optimized implementation
        numpy_impl: NumPy fallback implementation
        name: Function name for logging
    
    Returns:
        Safe wrapped function
    """
    @functools.wraps(numba_impl)
    def safe_function(*args, **kwargs):
        with NumbaContext(name):
            try:
                # Validate inputs
                if len(args) > 0 and isinstance(args[0], np.ndarray):
                    args = list(args)
                    args[0] = ensure_contiguous(args[0], "input")
                
                # Try Numba
                result = numba_impl(*args, **kwargs)
                return result
                
            except Exception as e:
                logger.warning(
                    f"Numba '{name}' failed, falling back to NumPy: {e}"
                )
                result = numpy_impl(*args, **kwargs)
                return result
    
    return safe_function


# Initialize logging
if __name__ != "__main__":
    log_system_info()
