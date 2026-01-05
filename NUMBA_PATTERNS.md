# Numba Best Practices & Pattern Library

**Version**: 2.0  
**Date**: January 5, 2026  
**Status**: Production-Ready with ChromaDB Integration

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Design Patterns](#design-patterns)
3. [Error Handling](#error-handling)
4. [Logging Patterns](#logging-patterns)
5. [Type Annotations](#type-annotations)
6. [Usage Examples](#usage-examples)
7. [Performance Optimization](#performance-optimization)
8. [Testing Patterns](#testing-patterns)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Numba Integration

```python
from typing import Optional
import numpy as np
from numba import njit, prange
import logging

logger = logging.getLogger(__name__)

# Pattern: Simple Numba function with fallback
try:
    from comfy.numba_utils import normalize_image_array
    from comfy.numba_error_handler import check_numba_availability
    NUMBA_AVAILABLE = check_numba_availability()
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available, using NumPy fallback")

def process_image(img: np.ndarray) -> np.ndarray:
    """Process image with Numba acceleration (with fallback)."""
    if NUMBA_AVAILABLE and img.ndim == 3:
        try:
            return normalize_image_array(img)
        except Exception as e:
            logger.error(f"Numba failed: {e}, falling back to NumPy")
    
    # NumPy fallback
    return img.astype(np.float32) / 255.0
```

---

## Design Patterns

### Pattern 1: Safe Wrapper with Observability

**Use Case**: Production code requiring error handling + metrics

```python
from typing import Callable, Any, Optional
import numpy as np
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def numba_with_fallback(
    fallback_func: Callable,
    operation_name: str,
    enable_metrics: bool = True,
    enable_logging: bool = True
) -> Callable:
    """
    Decorator for Numba functions with automatic fallback and observability.
    
    Args:
        fallback_func: NumPy implementation to use if Numba fails
        operation_name: Name for logging/metrics
        enable_metrics: Record performance metrics
        enable_logging: Log errors and fallbacks
    
    Example:
        >>> def numpy_add(a, b):
        ...     return a + b
        >>> 
        >>> @numba_with_fallback(numpy_add, "fast_add")
        >>> @njit(parallel=True)
        >>> def numba_add(a, b):
        ...     return a + b
    """
    def decorator(numba_func: Callable) -> Callable:
        @wraps(numba_func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                # Try Numba version
                result = numba_func(*args, **kwargs)
                
                if enable_metrics:
                    # Record success metric
                    _record_success(operation_name, args)
                
                return result
                
            except Exception as e:
                if enable_logging:
                    logger.warning(
                        f"Numba {operation_name} failed: {e.__class__.__name__}: {e}. "
                        f"Falling back to NumPy implementation."
                    )
                
                # Fall back to NumPy
                result = fallback_func(*args, **kwargs)
                
                if enable_metrics:
                    _record_fallback(operation_name, str(e))
                
                return result
        
        return wrapper
    return decorator

def _record_success(operation: str, args: tuple):
    """Record successful Numba execution (stub for metrics integration)."""
    try:
        from comfy.metrics import MetricsCollector
        metrics = MetricsCollector.get_instance()
        
        array_size = args[0].size if len(args) > 0 and hasattr(args[0], 'size') else 0
        metrics.record_execution(
            operation=operation,
            function="numba",
            execution_time_ms=0,  # Filled by actual timer
            array_size=array_size,
            success=True
        )
    except ImportError:
        pass

def _record_fallback(operation: str, reason: str):
    """Record Numba fallback event."""
    try:
        from comfy.metrics import MetricsCollector
        metrics = MetricsCollector.get_instance()
        metrics.record_fallback(
            operation=operation,
            function="numba",
            reason=reason
        )
    except ImportError:
        pass
```

### Pattern 2: Context Manager for Batch Operations

**Use Case**: Processing multiple arrays with consistent error handling

```python
from contextlib import contextmanager
from typing import Generator, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

@contextmanager
def numba_batch_context(
    operation_name: str,
    fallback_mode: str = "numpy"
) -> Generator[dict, None, None]:
    """
    Context manager for batch Numba operations.
    
    Args:
        operation_name: Name for logging
        fallback_mode: "numpy" or "raise" on error
    
    Yields:
        dict: Context with 'process' function and stats
    
    Example:
        >>> with numba_batch_context("image_batch") as ctx:
        ...     results = [ctx['process'](img) for img in images]
        ...     print(f"Processed: {ctx['stats']}")
    """
    stats = {
        'total': 0,
        'numba_success': 0,
        'fallback_used': 0,
        'errors': []
    }
    
    def process_array(
        func: Callable,
        fallback: Callable,
        arr: np.ndarray
    ) -> Optional[np.ndarray]:
        """Process single array with error handling."""
        stats['total'] += 1
        
        try:
            result = func(arr)
            stats['numba_success'] += 1
            return result
            
        except Exception as e:
            stats['fallback_used'] += 1
            
            if fallback_mode == "numpy":
                logger.debug(f"{operation_name} fallback for array {stats['total']}: {e}")
                return fallback(arr)
            else:
                stats['errors'].append(str(e))
                raise
    
    context = {
        'process': process_array,
        'stats': stats
    }
    
    try:
        yield context
    finally:
        # Log summary
        logger.info(
            f"{operation_name} completed: "
            f"{stats['numba_success']}/{stats['total']} Numba, "
            f"{stats['fallback_used']} fallbacks, "
            f"{len(stats['errors'])} errors"
        )
```

### Pattern 3: Lazy Compilation with Warmup

**Use Case**: Avoid JIT overhead on first call

```python
import numpy as np
from numba import njit
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class NumbaFunction:
    """
    Wrapper for Numba functions with lazy compilation and warmup.
    
    Example:
        >>> fast_add = NumbaFunction(
        ...     numba_impl=numba_add_impl,
        ...     numpy_fallback=numpy_add,
        ...     warmup_shape=(100, 100)
        ... )
        >>> result = fast_add(a, b)  # Pre-compiled, no JIT overhead
    """
    
    def __init__(
        self,
        numba_impl: Callable,
        numpy_fallback: Callable,
        warmup_shape: Optional[tuple] = None,
        warmup_dtype: np.dtype = np.float32
    ):
        self.numba_func = numba_impl
        self.numpy_func = numpy_fallback
        self._compiled = False
        
        # Warmup compilation
        if warmup_shape:
            self._warmup(warmup_shape, warmup_dtype)
    
    def _warmup(self, shape: tuple, dtype: np.dtype):
        """Pre-compile Numba function with dummy data."""
        try:
            dummy = np.zeros(shape, dtype=dtype)
            _ = self.numba_func(dummy)
            self._compiled = True
            logger.info(f"Numba function warmed up with shape {shape}")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}, will use lazy compilation")
    
    def __call__(self, *args, **kwargs):
        """Execute function with fallback."""
        if not self._compiled:
            logger.debug("First call, JIT compilation in progress...")
        
        try:
            return self.numba_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Numba execution failed: {e}")
            return self.numpy_func(*args, **kwargs)

# Usage example
@njit(parallel=True, cache=True)
def _numba_normalize(arr):
    result = np.empty_like(arr, dtype=np.float32)
    for i in prange(arr.shape[0]):
        result[i] = arr[i] / 255.0
    return result

def _numpy_normalize(arr):
    return arr.astype(np.float32) / 255.0

# Pre-compile for common image sizes
normalize_image = NumbaFunction(
    numba_impl=_numba_normalize,
    numpy_fallback=_numpy_normalize,
    warmup_shape=(1920, 1080, 3),
    warmup_dtype=np.uint8
)
```

---

## Error Handling

### Comprehensive Error Handler Pattern

```python
from typing import Optional, Union, Tuple
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class NumbaErrorType(Enum):
    """Error categories for detailed handling."""
    COMPILATION = "compilation_error"
    TYPE_INFERENCE = "type_inference_error"
    RUNTIME = "runtime_error"
    ARRAY_CONTIGUITY = "array_contiguity_error"
    CUDA_MEMORY = "cuda_memory_error"
    UNSUPPORTED_OP = "unsupported_operation"

class NumbaErrorHandler:
    """
    Centralized error handling for Numba operations.
    
    Example:
        >>> handler = NumbaErrorHandler(log_level=logging.DEBUG)
        >>> result = handler.safe_execute(
        ...     numba_func=fast_process,
        ...     fallback_func=slow_process,
        ...     args=(image,),
        ...     operation_name="image_processing"
        ... )
    """
    
    def __init__(
        self,
        log_level: int = logging.INFO,
        raise_on_failure: bool = False,
        max_retries: int = 0
    ):
        self.log_level = log_level
        self.raise_on_failure = raise_on_failure
        self.max_retries = max_retries
        self.error_counts = {err.value: 0 for err in NumbaErrorType}
    
    def safe_execute(
        self,
        numba_func: Callable,
        fallback_func: Callable,
        args: tuple,
        kwargs: Optional[dict] = None,
        operation_name: str = "numba_operation"
    ) -> Tuple[Any, bool]:
        """
        Execute Numba function with comprehensive error handling.
        
        Returns:
            Tuple[result, used_numba]: (result, True if Numba succeeded)
        """
        kwargs = kwargs or {}
        
        for attempt in range(self.max_retries + 1):
            try:
                result = numba_func(*args, **kwargs)
                return result, True
                
            except Exception as e:
                error_type = self._classify_error(e)
                self.error_counts[error_type.value] += 1
                
                log_msg = (
                    f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"{error_type.value} - {e}"
                )
                
                if attempt < self.max_retries:
                    logger.log(self.log_level, f"{log_msg}. Retrying...")
                    continue
                else:
                    logger.log(self.log_level, f"{log_msg}. Using fallback.")
                    
                    if self.raise_on_failure:
                        raise
                    
                    # Execute fallback
                    try:
                        result = fallback_func(*args, **kwargs)
                        return result, False
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                        raise
    
    def _classify_error(self, error: Exception) -> NumbaErrorType:
        """Classify error for better diagnostics."""
        error_str = str(error).lower()
        
        if "typing" in error_str or "type inference" in error_str:
            return NumbaErrorType.TYPE_INFERENCE
        elif "compilation" in error_str or "cannot compile" in error_str:
            return NumbaErrorType.COMPILATION
        elif "contiguous" in error_str or "non-contiguous" in error_str:
            return NumbaErrorType.ARRAY_CONTIGUITY
        elif "cuda" in error_str and "memory" in error_str:
            return NumbaErrorType.CUDA_MEMORY
        elif "unsupported" in error_str or "not supported" in error_str:
            return NumbaErrorType.UNSUPPORTED_OP
        else:
            return NumbaErrorType.RUNTIME
    
    def get_error_summary(self) -> dict:
        """Get error statistics."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'by_type': dict(self.error_counts)
        }
```

---

## Logging Patterns

### Structured Logging for Numba Operations

```python
import logging
import time
from typing import Optional, Any
from dataclasses import dataclass, asdict
import json

@dataclass
class NumbaLogEntry:
    """Structured log entry for Numba operations."""
    timestamp: float
    operation: str
    function_name: str
    success: bool
    execution_time_ms: float
    array_shape: Optional[tuple] = None
    array_dtype: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    used_numba: bool = True
    
    def to_json(self) -> str:
        """Convert to JSON for structured logging."""
        return json.dumps(asdict(self), default=str)

class NumbaLogger:
    """
    Structured logger for Numba operations with performance tracking.
    
    Example:
        >>> logger = NumbaLogger("image_processing")
        >>> with logger.log_operation("normalize") as log_ctx:
        ...     result = normalize_image(img)
        ...     log_ctx.set_array_info(img)
    """
    
    def __init__(
        self,
        name: str,
        log_level: int = logging.INFO,
        log_to_file: bool = False,
        log_file_path: Optional[str] = None
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Console handler with formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        self.logger.addHandler(console_handler)
        
        # File handler for structured logs
        if log_to_file and log_file_path:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(
                logging.Formatter('%(message)s')  # JSON only
            )
            self.logger.addHandler(file_handler)
    
    @contextmanager
    def log_operation(
        self,
        operation_name: str,
        function_name: str = "numba_func"
    ) -> Generator['LogContext', None, None]:
        """
        Context manager for logging Numba operations.
        
        Example:
            >>> with logger.log_operation("normalize", "fast_normalize") as ctx:
            ...     result = fast_normalize(img)
            ...     ctx.set_array_info(img)
            ...     ctx.set_result_info(result)
        """
        ctx = LogContext(
            operation=operation_name,
            function_name=function_name,
            logger=self.logger
        )
        
        try:
            yield ctx
        finally:
            ctx.finalize()

class LogContext:
    """Context for operation logging."""
    
    def __init__(self, operation: str, function_name: str, logger: logging.Logger):
        self.operation = operation
        self.function_name = function_name
        self.logger = logger
        self.start_time = time.time()
        self.entry = NumbaLogEntry(
            timestamp=self.start_time,
            operation=operation,
            function_name=function_name,
            success=True,
            execution_time_ms=0.0
        )
    
    def set_array_info(self, arr: np.ndarray):
        """Record array information."""
        self.entry.array_shape = arr.shape
        self.entry.array_dtype = str(arr.dtype)
    
    def set_error(self, error: Exception, used_numba: bool = True):
        """Record error information."""
        self.entry.success = False
        self.entry.error_type = error.__class__.__name__
        self.entry.error_message = str(error)
        self.entry.used_numba = used_numba
    
    def finalize(self):
        """Finalize and log the entry."""
        self.entry.execution_time_ms = (time.time() - self.start_time) * 1000
        
        # Log structured JSON
        self.logger.info(self.entry.to_json())
        
        # Log human-readable summary
        status = "✓" if self.entry.success else "✗"
        self.logger.debug(
            f"{status} {self.operation}.{self.function_name}: "
            f"{self.entry.execution_time_ms:.2f}ms"
        )
```

---

## Type Annotations

### Type-Safe Numba Functions

```python
from typing import TypeVar, Union, Literal, Protocol
from typing_extensions import Annotated
import numpy as np
import numpy.typing as npt

# Type aliases for clarity
Float32Array = npt.NDArray[np.float32]
UInt8Array = npt.NDArray[np.uint8]
IntArray = npt.NDArray[np.int32]

# Image types
ImageArray = Annotated[
    Union[Float32Array, UInt8Array],
    "Image array with shape (H, W, C) or (H, W)"
]

# Generic array type
T = TypeVar('T', bound=np.generic)
ArrayLike = npt.NDArray[T]

class NumbaCallable(Protocol):
    """Protocol for Numba-compatible functions."""
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

# Type-annotated Numba functions
def normalize_image_array(
    img_array: UInt8Array,
    scale: float = 255.0
) -> Float32Array:
    """
    Normalize image array from uint8 to float32.
    
    Args:
        img_array: Input image (H, W, C) with dtype uint8
        scale: Scaling factor (default: 255.0)
    
    Returns:
        Normalized array with dtype float32 in range [0, 1]
    
    Raises:
        ValueError: If array is not 3D or not uint8
    
    Example:
        >>> img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> normalized = normalize_image_array(img)
        >>> assert normalized.dtype == np.float32
        >>> assert 0 <= normalized.min() <= normalized.max() <= 1
    """
    if img_array.ndim != 3:
        raise ValueError(f"Expected 3D array, got {img_array.ndim}D")
    if img_array.dtype != np.uint8:
        raise ValueError(f"Expected uint8, got {img_array.dtype}")
    
    result = np.empty_like(img_array, dtype=np.float32)
    height, width, channels = img_array.shape
    
    for h in prange(height):
        for w in range(width):
            for c in range(channels):
                result[h, w, c] = img_array[h, w, c] / scale
    
    return result

def process_batch(
    images: list[ImageArray],
    operation: Literal["normalize", "denormalize", "clip"] = "normalize"
) -> list[ImageArray]:
    """
    Process batch of images with type safety.
    
    Args:
        images: List of image arrays
        operation: Operation to perform
    
    Returns:
        Processed images with appropriate dtype
    """
    results: list[ImageArray] = []
    
    for img in images:
        if operation == "normalize":
            results.append(normalize_image_array(img))
        elif operation == "denormalize":
            results.append(denormalize_image_array(img))
        else:
            results.append(np.clip(img, 0, 255).astype(np.uint8))
    
    return results
```

---

## Usage Examples

### Example 1: Image Processing Pipeline

```python
from typing import List
import numpy as np
import logging
from numba import njit, prange

logger = logging.getLogger(__name__)

# Initialize Numba
try:
    from comfy.numba_utils import (
        normalize_image_array,
        denormalize_image_array,
        alpha_composite
    )
    from comfy.numba_error_handler import check_numba_availability
    NUMBA_AVAILABLE = check_numba_availability()
    logger.info(f"Numba status: {NUMBA_AVAILABLE}")
except ImportError as e:
    NUMBA_AVAILABLE = False
    logger.warning(f"Numba unavailable: {e}")

class ImageProcessor:
    """Image processor with Numba acceleration."""
    
    def __init__(self, use_numba: bool = True):
        self.use_numba = use_numba and NUMBA_AVAILABLE
        logger.info(f"ImageProcessor initialized (Numba: {self.use_numba})")
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        if self.use_numba and img.ndim == 3:
            try:
                return normalize_image_array(img.astype(np.uint8))
            except Exception as e:
                logger.error(f"Numba normalize failed: {e}")
        
        return img.astype(np.float32) / 255.0
    
    def denormalize(self, img: np.ndarray) -> np.ndarray:
        """Denormalize image to [0, 255] range."""
        if self.use_numba and img.ndim == 3:
            try:
                return denormalize_image_array(img.astype(np.float32), scale=255.0)
            except Exception as e:
                logger.error(f"Numba denormalize failed: {e}")
        
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    def blend(
        self,
        base: np.ndarray,
        overlay: np.ndarray,
        alpha: np.ndarray
    ) -> np.ndarray:
        """Alpha blend two images."""
        if self.use_numba:
            try:
                return alpha_composite(
                    base.astype(np.float32),
                    overlay.astype(np.float32),
                    alpha.astype(np.float32)
                )
            except Exception as e:
                logger.error(f"Numba blend failed: {e}")
        
        return base * (1 - alpha) + overlay * alpha

# Usage
processor = ImageProcessor(use_numba=True)
img = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)
normalized = processor.normalize(img)
denormalized = processor.denormalize(normalized)
```

### Example 2: Video Processing with Batch Operations

```python
from pathlib import Path
import numpy as np
from typing import Iterator
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Process video frames with Numba acceleration."""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        
        try:
            from comfy.numba_utils import fast_array_multiply_add
            self.numba_available = True
            self.multiply_add = fast_array_multiply_add
        except ImportError:
            self.numba_available = False
            logger.warning("Numba not available for video processing")
    
    def adjust_brightness_contrast(
        self,
        frames: List[np.ndarray],
        brightness: float = 0.0,
        contrast: float = 1.0
    ) -> List[np.ndarray]:
        """
        Adjust brightness and contrast of video frames.
        
        Args:
            frames: List of frame arrays
            brightness: Brightness adjustment [-1, 1]
            contrast: Contrast multiplier [0, 2]
        
        Returns:
            Adjusted frames
        """
        results = []
        
        for frame in frames:
            if self.numba_available:
                try:
                    # result = contrast * frame + brightness
                    adjusted = self.multiply_add(
                        frame, 
                        np.ones_like(frame) * brightness,
                        contrast,
                        1.0
                    )
                    results.append(np.clip(adjusted, 0, 255).astype(np.uint8))
                    continue
                except Exception as e:
                    logger.debug(f"Numba fallback: {e}")
            
            # NumPy fallback
            adjusted = contrast * frame + brightness
            results.append(np.clip(adjusted, 0, 255).astype(np.uint8))
        
        return results
    
    def process_video_batched(
        self,
        frame_generator: Iterator[np.ndarray],
        operation: Callable
    ) -> Iterator[np.ndarray]:
        """
        Process video in batches for efficiency.
        
        Args:
            frame_generator: Iterator yielding frames
            operation: Function to apply to each batch
        
        Yields:
            Processed frames
        """
        batch = []
        
        for frame in frame_generator:
            batch.append(frame)
            
            if len(batch) >= self.batch_size:
                # Process batch
                processed = operation(batch)
                yield from processed
                batch = []
        
        # Process remaining frames
        if batch:
            processed = operation(batch)
            yield from processed

# Usage
def load_frames(video_path: Path) -> Iterator[np.ndarray]:
    """Load frames from video (placeholder)."""
    # Actual video loading implementation
    for _ in range(100):
        yield np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

processor = VideoProcessor(batch_size=32)
frames = load_frames(Path("video.mp4"))
adjusted_frames = processor.process_video_batched(
    frames,
    lambda batch: processor.adjust_brightness_contrast(batch, brightness=10.0, contrast=1.2)
)
```

---

## Performance Optimization

### Benchmarking Pattern

```python
import time
import numpy as np
from typing import Callable, Dict
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    """Results from performance benchmark."""
    function_name: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    iterations: int
    speedup_vs_numpy: Optional[float] = None
    
    def __str__(self) -> str:
        speedup_str = f", {self.speedup_vs_numpy:.2f}x faster" if self.speedup_vs_numpy else ""
        return (
            f"{self.function_name}: {self.mean_time_ms:.3f}ms ± {self.std_time_ms:.3f}ms "
            f"(min: {self.min_time_ms:.3f}ms, max: {self.max_time_ms:.3f}ms, n={self.iterations}){speedup_str}"
        )

def benchmark_numba_vs_numpy(
    numba_func: Callable,
    numpy_func: Callable,
    test_data: np.ndarray,
    iterations: int = 100,
    warmup: int = 5
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark Numba vs NumPy implementation.
    
    Args:
        numba_func: Numba-optimized function
        numpy_func: NumPy baseline function
        test_data: Test array
        iterations: Number of iterations
        warmup: Warmup iterations (for JIT compilation)
    
    Returns:
        Dictionary with benchmark results
    
    Example:
        >>> results = benchmark_numba_vs_numpy(
        ...     numba_normalize,
        ...     numpy_normalize,
        ...     test_image,
        ...     iterations=100
        ... )
        >>> print(results['numba'])
        >>> print(results['numpy'])
    """
    print(f"Benchmarking with array shape {test_data.shape}, dtype {test_data.dtype}")
    
    # Warmup Numba (JIT compilation)
    print(f"Warming up Numba (JIT compilation)...")
    for _ in range(warmup):
        _ = numba_func(test_data)
    
    # Benchmark Numba
    print(f"Benchmarking Numba ({iterations} iterations)...")
    numba_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = numba_func(test_data)
        numba_times.append((time.perf_counter() - start) * 1000)
    
    # Benchmark NumPy
    print(f"Benchmarking NumPy ({iterations} iterations)...")
    numpy_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = numpy_func(test_data)
        numpy_times.append((time.perf_counter() - start) * 1000)
    
    # Calculate statistics
    numba_result = BenchmarkResult(
        function_name="Numba",
        mean_time_ms=np.mean(numba_times),
        std_time_ms=np.std(numba_times),
        min_time_ms=np.min(numba_times),
        max_time_ms=np.max(numba_times),
        iterations=iterations
    )
    
    numpy_result = BenchmarkResult(
        function_name="NumPy",
        mean_time_ms=np.mean(numpy_times),
        std_time_ms=np.std(numpy_times),
        min_time_ms=np.min(numpy_times),
        max_time_ms=np.max(numpy_times),
        iterations=iterations
    )
    
    # Calculate speedup
    speedup = numpy_result.mean_time_ms / numba_result.mean_time_ms
    numba_result.speedup_vs_numpy = speedup
    
    print("\n=== Results ===")
    print(numba_result)
    print(numpy_result)
    print(f"\nSpeedup: {speedup:.2f}x")
    
    return {
        'numba': numba_result,
        'numpy': numpy_result,
        'speedup': speedup
    }

# Example usage
if __name__ == "__main__":
    from comfy.numba_utils import normalize_image_array
    
    def numpy_normalize(arr):
        return arr.astype(np.float32) / 255.0
    
    # Test with different sizes
    sizes = [(100, 100, 3), (512, 512, 3), (1920, 1080, 3)]
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Testing size: {size}")
        print('='*60)
        
        test_img = np.random.randint(0, 256, size, dtype=np.uint8)
        results = benchmark_numba_vs_numpy(
            normalize_image_array,
            numpy_normalize,
            test_img,
            iterations=100
        )
```

---

## Testing Patterns

### Unit Testing Numba Functions

```python
import unittest
import numpy as np
from typing import Callable

class TestNumbaFunction(unittest.TestCase):
    """Base test class for Numba functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_image_small = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.test_image_large = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)
    
    def assert_arrays_close(
        self,
        actual: np.ndarray,
        expected: np.ndarray,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        msg: str = ""
    ):
        """Assert two arrays are close with detailed error message."""
        try:
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        except AssertionError as e:
            diff = np.abs(actual - expected)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            error_msg = (
                f"{msg}\n"
                f"Max difference: {max_diff}\n"
                f"Mean difference: {mean_diff}\n"
                f"Shape: actual={actual.shape}, expected={expected.shape}\n"
                f"Dtype: actual={actual.dtype}, expected={expected.dtype}"
            )
            raise AssertionError(error_msg) from e
    
    def test_numba_vs_numpy_equivalence(
        self,
        numba_func: Callable,
        numpy_func: Callable,
        test_array: np.ndarray
    ):
        """Test that Numba and NumPy produce equivalent results."""
        numba_result = numba_func(test_array)
        numpy_result = numpy_func(test_array)
        
        self.assert_arrays_close(
            numba_result,
            numpy_result,
            msg=f"Numba and NumPy results differ for {numba_func.__name__}"
        )
    
    def test_error_handling(
        self,
        func: Callable,
        invalid_inputs: list
    ):
        """Test error handling with invalid inputs."""
        for invalid_input in invalid_inputs:
            with self.subTest(input=invalid_input):
                with self.assertRaises((ValueError, TypeError)):
                    func(invalid_input)

# Concrete test example
class TestNormalizeImage(TestNumbaFunction):
    """Test image normalization function."""
    
    def test_normalize_small_image(self):
        """Test normalization on small image."""
        from comfy.numba_utils import normalize_image_array
        
        result = normalize_image_array(self.test_image_small)
        
        # Check dtype
        self.assertEqual(result.dtype, np.float32)
        
        # Check range
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))
        
        # Check shape
        self.assertEqual(result.shape, self.test_image_small.shape)
    
    def test_normalize_equivalence(self):
        """Test Numba vs NumPy equivalence."""
        from comfy.numba_utils import normalize_image_array
        
        def numpy_normalize(arr):
            return arr.astype(np.float32) / 255.0
        
        self.test_numba_vs_numpy_equivalence(
            normalize_image_array,
            numpy_normalize,
            self.test_image_small
        )
    
    def test_invalid_inputs(self):
        """Test error handling with invalid inputs."""
        from comfy.numba_utils import normalize_image_array
        
        invalid_inputs = [
            np.array([1, 2, 3]),  # 1D
            np.random.rand(10, 10).astype(np.float32),  # Wrong dtype
            None,
            "not an array"
        ]
        
        self.test_error_handling(normalize_image_array, invalid_inputs)

if __name__ == '__main__':
    unittest.main()
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "Type inference failed"

**Symptom**: `TypingError: Failed in nopython mode pipeline`

**Cause**: Numba cannot infer types for certain operations

**Solution**:
```python
# Bad: Numba can't infer np.clip on scalars
@njit
def bad_clip(arr):
    for i in range(len(arr)):
        arr[i] = np.clip(arr[i], 0, 255)  # ❌ Fails

# Good: Manual clipping
@njit
def good_clip(arr):
    for i in range(len(arr)):
        val = arr[i]
        if val < 0:
            arr[i] = 0
        elif val > 255:
            arr[i] = 255  # ✓ Works
```

#### Issue 2: "Non-contiguous array"

**Symptom**: Slower performance or errors with transposed arrays

**Solution**:
```python
# Check and fix contiguity
def ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure array is C-contiguous."""
    if not arr.flags['C_CONTIGUOUS']:
        logger.debug(f"Array not contiguous, creating copy")
        return np.ascontiguousarray(arr)
    return arr

# Use before Numba functions
img = ensure_contiguous(img.transpose(1, 0, 2))
result = numba_func(img)
```

#### Issue 3: "CUDA_ERROR_STUB_LIBRARY"

**Symptom**: Warning on startup about CUDA

**Impact**: None - benign warning

**Solution**: Ignore or suppress:
```python
import warnings
warnings.filterwarnings('ignore', message='.*CUDA_ERROR_STUB_LIBRARY.*')
```

---

## ChromaDB Integration

This documentation is stored in ChromaDB for efficient retrieval:

```python
from chromadb import Client

# Store patterns
client = Client()
collection = client.get_or_create_collection("numba_patterns")

collection.add(
    documents=[
        "Numba pattern: Safe wrapper with observability",
        "Numba pattern: Context manager for batch operations",
        "Numba error handling: Comprehensive error handler",
        "Numba logging: Structured logging with JSON",
        "Numba type annotations: Type-safe functions",
        "Numba usage: Image processing pipeline",
        "Numba performance: Benchmarking pattern",
        "Numba testing: Unit testing framework"
    ],
    ids=[f"pattern_{i}" for i in range(8)]
)
```

---

**Document Version**: 2.0  
**Last Updated**: January 5, 2026  
**Maintained By**: ComfyUI Development Team
