# NumPy Comprehensive Guide (2.4.0+)
*Last Updated: January 2030 - Based on NumPy 2.4.0 Documentation*

## 📋 Table of Contents
- [Installation & Version](#installation--version)
- [Core Array API](#core-array-api)
- [Type Annotations & Type Hints](#type-annotations--type-hints)
- [Broadcasting Patterns](#broadcasting-patterns)
- [Advanced Indexing](#advanced-indexing)
- [Performance Patterns](#performance-patterns)
- [dtype System](#dtype-system)
- [Memory Management](#memory-management)
- [Numba Integration](#numba-integration)
- [Common Pitfalls](#common-pitfalls)

---

## 🚀 Installation & Version

```bash
# Install latest NumPy (2.4.0+)
pip install "numpy>=2.4.0"

# Check version
python -c "import numpy as np; print(np.__version__)"
```

### Version Detection Pattern
```python
import numpy as np
from packaging import version

# Modern version check
if version.parse(np.__version__) >= version.parse("2.4.0"):
    print("NumPy 2.4+ features available")
    
# Array API compatibility
print(f"Array API version: {np.__array_api_version__}")  # 2024.12
```

---

## 🔧 Core Array API

### Array Creation (Jan 2030 Standards)

```python
import numpy as np
from numpy.typing import NDArray
from typing import Any

# Basic creation with type hints
arr1: NDArray[np.float64] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
arr2: NDArray[np.int32] = np.arange(10, dtype=np.int32)

# New ndmax parameter (NumPy 2.4+)
# Limit dimensionality when creating from nested sequences
nested = [[1, 2], [3, 4]]
arr_1d: NDArray[np.object_] = np.array(nested, dtype=object, ndmax=1)
# Result: array([list([1, 2]), list([3, 4])], dtype=object) with shape (2,)

# Common constructors with full signatures
zeros: NDArray[np.float64] = np.zeros(
    shape=(3, 4),
    dtype=np.float64,
    order='C'  # Row-major (C-style) layout
)

ones: NDArray[np.float32] = np.ones((2, 3), dtype=np.float32)

empty: NDArray[np.int64] = np.empty(
    shape=(5,),
    dtype=np.int64
)

# Identity matrices
identity: NDArray[np.float64] = np.eye(3, dtype=np.float64)

# Random generation (modern API)
rng = np.random.default_rng(seed=42)
random_floats: NDArray[np.float64] = rng.random((3, 4))
random_ints: NDArray[np.int64] = rng.integers(0, 100, size=(5,))
```

### Shape Manipulation

```python
# Reshape with validation
def safe_reshape(
    arr: NDArray[Any],
    new_shape: tuple[int, ...]
) -> NDArray[Any]:
    """Reshape with size validation."""
    if arr.size != np.prod(new_shape):
        raise ValueError(
            f"Cannot reshape array of size {arr.size} "
            f"into shape {new_shape} (size {np.prod(new_shape)})"
        )
    return arr.reshape(new_shape)

# Use np.reshape (returns copy) vs arr.reshape (may return view)
x = np.arange(12)
# Creates new copy (NumPy 2.4+)
y = np.reshape(x, (3, 4))  
# May return view
z = x.reshape(3, 4)

# Transpose operations
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
transposed = arr_2d.T  # Property access
transposed_alt = np.transpose(arr_2d, axes=(1, 0))

# Flatten vs ravel
flat_copy = arr_2d.flatten()  # Always returns copy
flat_view = arr_2d.ravel()    # Returns view when possible
```

---

## 🔤 Type Annotations & Type Hints

### Modern Type Hint Patterns (Python 3.14+)

```python
from numpy.typing import NDArray, ArrayLike, DTypeLike
from typing import Any, TypeVar
import numpy as np

# Generic type patterns
T = TypeVar('T', np.float64, np.float32, np.int32)

def process_array(
    data: ArrayLike,  # Accepts array-like inputs
    dtype: DTypeLike | None = None  # NumPy 2.4+ uses | instead of None
) -> NDArray[np.float64]:
    """
    Process array-like data.
    
    Args:
        data: Input array or array-like (list, tuple, etc.)
        dtype: Target dtype (None preserves input dtype)
        
    Returns:
        Processed array with float64 dtype
    """
    arr = np.asarray(data, dtype=dtype)
    return arr.astype(np.float64, casting='same_value')  # NumPy 2.3+

# Runtime signature introspection (NumPy 2.4+)
import inspect
sig = inspect.signature(process_array)
print(sig)  # Now works with most NumPy functions

# Shape-aware type hints
def matrix_multiply(
    a: NDArray[np.float64],  # Shape: (M, K)
    b: NDArray[np.float64],  # Shape: (K, N)
) -> NDArray[np.float64]:   # Shape: (M, N)
    """Type-safe matrix multiplication."""
    return np.matmul(a, b)

# Using default type parameter (NumPy 2.4+)
arr: NDArray = np.zeros(10)  # Inferred as NDArray[Any]
```

### StringDType Pattern (NumPy 2.4+)

```python
from numpy.dtypes import StringDType
from typing import Never

# StringDType is now a generic type
str_arr: NDArray[StringDType[Never]] = np.array(
    ["hello", "world"], 
    dtype=StringDType()
)

# With NA handling
str_arr_na: NDArray[StringDType[None]] = np.array(
    ["hello", None, "world"],
    dtype=StringDType(na_object=None)
)
```

---

## 📡 Broadcasting Patterns

### Broadcasting Rules (NumPy 2.4)

```python
"""
Broadcasting Rules:
1. Right-align shapes
2. Dimensions are compatible if:
   - They are equal, OR
   - One of them is 1
3. Missing dimensions assumed to be 1
"""

# Rule examples
def demonstrate_broadcasting():
    # Example 1: Scalar to array
    arr = np.array([1, 2, 3])    # shape: (3,)
    scalar = 2                    # shape: ()
    result = arr * scalar         # shape: (3,)
    # scalar broadcasts to shape (3,)
    
    # Example 2: 1D to 2D
    arr_2d = np.array([[1, 2, 3],
                       [4, 5, 6]])  # shape: (2, 3)
    arr_1d = np.array([10, 20, 30])  # shape: (3,)
    result = arr_2d + arr_1d          # shape: (2, 3)
    # arr_1d broadcasts from (3,) to (2, 3)
    
    # Example 3: Both need broadcasting
    a = np.array([[1], [2], [3]])     # shape: (3, 1)
    b = np.array([10, 20, 30])        # shape: (3,)
    result = a + b                     # shape: (3, 3)
    # a: (3, 1) → (3, 3)
    # b: (3,) → (1, 3) → (3, 3)

# 12 Essential Broadcasting Tricks

# 1. Row/Column operations
X = np.arange(12).reshape(3, 4)
row_bias = np.array([1, -1, 0, 2])     # (4,)
X_adj = X + row_bias                   # (3,4) + (4,)

col_scale = np.array([2, 10, 0.5])     # (3,)
X_scaled = X * col_scale[:, None]      # (3,4) * (3,1)

# 2. Center and standardize
X = np.random.randn(1000, 64)
mu = X.mean(axis=0, keepdims=True)     # (1, 64)
sigma = X.std(axis=0, keepdims=True) + 1e-8
X_normalized = (X - mu) / sigma        # Z-score normalization

# 3. Outer operations
a = np.array([1, 2, 3])
b = np.array([10, 20, 30, 40])
diff_matrix = a[:, None] - b[None, :]  # (3, 4) pairwise differences
outer_sum = np.add.outer(a, b)         # Explicit API

# 4. Pairwise Euclidean distances
X = np.random.randn(100, 10)  # 100 samples, 10 features
# ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
XX = np.sum(X**2, axis=1, keepdims=True)  # (100, 1)
YY = XX.T                                  # (1, 100)
XY = X @ X.T                               # (100, 100)
distances = np.sqrt(XX + YY - 2*XY)        # (100, 100)

# 5. Batch matrix operations
batch_A = np.random.randn(32, 4, 4)   # 32 matrices of 4x4
batch_B = np.random.randn(32, 4, 1)   # 32 vectors of 4x1
batch_result = batch_A @ batch_B      # Broadcasting in batch dimension

# 6. Conditional masking with broadcasting
data = np.random.randn(100, 50)
thresholds = np.array([0.5, 1.0, 1.5])[:, None]  # (3, 1)
mask = np.abs(data) > thresholds  # (3, 100, 50)

# 7. Grid creation
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 30)
X_grid, Y_grid = np.meshgrid(x, y, indexing='ij')  # (50, 30) each

# 8. Weighted sums
weights = np.array([0.2, 0.3, 0.5])    # (3,)
values = np.random.randn(100, 3)       # (100, 3)
weighted = values * weights             # Broadcasting weights
result = weighted.sum(axis=1)          # (100,)

# 9. Multi-dimensional indexing
indices = np.array([0, 2, 4])[:, None, None]  # (3, 1, 1)
data_3d = np.random.randn(10, 5, 8)            # (10, 5, 8)
selected = data_3d[indices]                    # (3, 5, 8)

# 10. Polynomial evaluation
x = np.linspace(-1, 1, 100)           # (100,)
coeffs = np.array([1, -2, 1])[:, None]  # (3, 1)
powers = np.arange(3)[:, None]          # (3, 1)
x_powers = x ** powers                  # (3, 100)
poly = (coeffs * x_powers).sum(axis=0)  # (100,)

# 11. Sliding window view (no copy)
from numpy.lib.stride_tricks import sliding_window_view
data = np.arange(10)
windows = sliding_window_view(data, window_shape=3)  # (8, 3)

# 12. Accumulate with broadcasting
cumsum_2d = np.cumsum(np.ones((5, 3)), axis=0) * np.arange(1, 4)
```

---

## 🎯 Advanced Indexing

### Basic vs Advanced Indexing

```python
# Basic indexing (returns view)
arr = np.arange(12).reshape(3, 4)
view = arr[1:, 2:]  # Slice indexing
view[0, 0] = 999    # Modifies original

# Advanced indexing (returns copy)
copy_arr = arr[[0, 2], [1, 3]]  # Fancy indexing
copy_arr[0] = 999  # Does NOT modify original

# Boolean indexing
mask = arr > 5
filtered = arr[mask]  # Returns copy

# Combining basic and advanced indexing
# Result depends on index contiguity
mixed1 = arr[1:, [2, 0, 1]]  # Shape: (2, 3)
mixed2 = arr[[0, 2], 1:]     # Shape: (2, 3)

# Integer array indexing patterns
rows = np.array([0, 1, 2])
cols = np.array([2, 0, 1])
result = arr[rows, cols]  # 1D result: [arr[0,2], arr[1,0], arr[2,1]]

# Broadcasting in indexing (NumPy 2.3+)
rows_2d = rows[:, None]      # (3, 1)
cols_1d = cols                # (3,)
result_2d = arr[rows_2d, cols_1d]  # (3, 3)

# Safe indexing helper
def safe_index(
    arr: NDArray[Any],
    indices: NDArray[np.int64],
    axis: int = 0
) -> NDArray[Any]:
    """Safe indexing with bounds checking."""
    if np.any(indices < 0) or np.any(indices >= arr.shape[axis]):
        raise IndexError(f"Indices out of bounds for axis {axis}")
    return np.take(arr, indices, axis=axis)
```

### Advanced Selection Patterns

```python
# Select with conditions
data = np.random.randn(100, 10)

# Multiple conditions
mask = (data > 0) & (data < 1)
filtered = data[mask]

# np.where for conditional selection
result = np.where(data > 0, data, 0)  # Clamp negative to 0

# np.select for multiple conditions
conditions = [data < -1, data > 1]
choices = [-1, 1]
result = np.select(conditions, choices, default=data)

# Structured indexing for sorting
indices = np.argsort(data, axis=0)  # Sort each column
sorted_data = np.take_along_axis(data, indices, axis=0)

# Top-k selection
k = 5
top_k_indices = np.argpartition(data, -k, axis=0)[-k:]
top_k_values = np.take_along_axis(data, top_k_indices, axis=0)
```

---

## ⚡ Performance Patterns

### Vectorization Examples

```python
from typing import Callable

# Anti-pattern: Python loop
def slow_sum(arr: NDArray[np.float64]) -> np.float64:
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]
    return total

# Fast pattern: Vectorized
def fast_sum(arr: NDArray[np.float64]) -> np.float64:
    return np.sum(arr)

# Vectorized computation patterns
def vectorized_operations():
    x = np.linspace(0, 10, 1000)
    
    # Mathematical operations (all vectorized)
    y1 = np.sin(x) + np.cos(x**2)
    y2 = np.exp(-x) * np.log(x + 1)
    y3 = np.sqrt(np.abs(x - 5))
    
    # Avoid Python loops
    # BAD:
    # result = [f(val) for val in x]
    
    # GOOD (if f is vectorizable):
    result = np.sin(x) * np.exp(-x)

# Use ufuncs for element-wise operations
def apply_ufunc(
    arr: NDArray[np.float64],
    func: Callable[[float], float]
) -> NDArray[np.float64]:
    """Apply numpy ufunc efficiently."""
    # Convert Python function to ufunc
    vectorized = np.vectorize(func, otypes=[np.float64])
    return vectorized(arr)

# Better: Use np.frompyfunc for permanent vectorization
sigmoid = np.frompyfunc(
    lambda x: 1 / (1 + np.exp(-x)),
    nin=1,
    nout=1
)
```

### Memory-Efficient Patterns

```python
# Use views instead of copies
large_array = np.random.randn(10000, 10000)

# GOOD: View (no copy)
view = large_array[::2, ::2]

# BAD: Copy
copy = large_array[::2, ::2].copy()

# In-place operations
arr = np.arange(10000, dtype=np.float64)
arr *= 2  # In-place, no extra memory
arr += 100  # In-place

# Out parameter pattern
result = np.empty_like(arr)
np.add(arr, 100, out=result)  # Write directly to result

# Generator pattern for large data
def chunked_processing(
    data: NDArray[Any],
    chunk_size: int = 1000
) -> NDArray[Any]:
    """Process large arrays in chunks."""
    n_chunks = len(data) // chunk_size
    results = []
    
    for i in range(n_chunks):
        chunk = data[i*chunk_size:(i+1)*chunk_size]
        processed = np.sqrt(chunk**2 + 1)  # Example operation
        results.append(processed)
    
    return np.concatenate(results)

# Use out=... for scalar results (NumPy 2.3+)
arr = np.array([1, 2, 3])
result = np.add(arr[0], arr[1], out=...)  # Ensures array result
```

### OpenMP Parallelization (NumPy 2.3+)

```python
# NumPy must be built with -Denable_openmp=true
import os

# Control number of threads
os.environ['OMP_NUM_THREADS'] = '4'

# Operations that benefit:
large = np.random.randn(10_000_000)
sorted_arr = np.sort(large)  # Parallel sort (if OpenMP enabled)
indices = np.argsort(large)  # Parallel argsort

# Check if OpenMP is enabled (approximate method)
import numpy as np
info = np.show_config()
# Look for "openmp" in build configuration
```

---

## 🔢 dtype System

### NumPy 2.4+ dtype Patterns

```python
# Modern dtype creation
dt1 = np.dtype(np.float64)
dt2 = np.dtype('f8')
dt3 = np.dtype('float64')

# Structured dtypes
person_dtype = np.dtype([
    ('name', 'U20'),
    ('age', np.int32),
    ('height', np.float32)
])

people = np.array([
    ('Alice', 30, 1.65),
    ('Bob', 25, 1.80)
], dtype=person_dtype)

# Access fields
names = people['name']

# dtype attributes
dt = np.dtype(np.float64)
print(dt.name)        # 'float64'
print(dt.itemsize)    # 8 (bytes)
print(dt.kind)        # 'f' (float)
print(dt.byteorder)   # '=' (native)

# Casting with 'same_value' (NumPy 2.4+)
arr = np.array([1.0, 2.5, 3.0])
try:
    # Will raise if values would change
    int_arr = arr.astype(np.int32, casting='same_value')
except ValueError as e:
    print(f"Casting would change values: {e}")

# Safe casting patterns
def safe_cast(
    arr: NDArray[Any],
    target_dtype: DTypeLike
) -> NDArray[Any]:
    """Cast with validation."""
    if not np.can_cast(arr.dtype, target_dtype, casting='safe'):
        raise TypeError(
            f"Cannot safely cast from {arr.dtype} to {target_dtype}"
        )
    return arr.astype(target_dtype)

# String dtypes (modern)
from numpy.dtypes import StringDType

# Variable-length strings
str_arr = np.array(['hello', 'world'], dtype=StringDType())

# Fixed-length strings (legacy)
fixed_str = np.array(['hello', 'world'], dtype='U10')  # Up to 10 chars
```

### finfo Improvements (NumPy 2.4+)

```python
# Modern finfo with accurate constants
info = np.finfo(np.float64)

print(f"Machine epsilon: {info.eps}")           # From C macros
print(f"Smallest normal: {info.smallest_normal}")
print(f"Smallest subnormal: {info.smallest_subnormal}")
print(f"Max value: {info.max}")
print(f"Min value: {info.min}")

# Derived attributes
print(f"Mantissa digits: {info.nmant}")
print(f"Exponent range: {info.minexp} to {info.maxexp}")

# Use for numerical stability
def numerically_stable_op(x: float) -> float:
    """Avoid underflow/overflow."""
    eps = np.finfo(np.float64).eps
    smallest = np.finfo(np.float64).smallest_normal
    
    if abs(x) < smallest:
        return 0.0
    return x / (1.0 + eps)
```

---

## 💾 Memory Management

### Memory Layout and Views

```python
# C-contiguous (row-major) vs F-contiguous (column-major)
arr_c = np.array([[1, 2, 3], [4, 5, 6]], order='C')
arr_f = np.array([[1, 2, 3], [4, 5, 6]], order='F')

print(arr_c.flags['C_CONTIGUOUS'])  # True
print(arr_f.flags['F_CONTIGUOUS'])  # True

# Check memory layout
def check_layout(arr: NDArray[Any]) -> str:
    """Determine array memory layout."""
    if arr.flags['C_CONTIGUOUS']:
        return 'C-contiguous (row-major)'
    elif arr.flags['F_CONTIGUOUS']:
        return 'F-contiguous (column-major)'
    else:
        return 'Non-contiguous'

# Stride manipulation (use with caution)
from numpy.lib.stride_tricks import as_strided

# Create sliding windows
def sliding_window(
    arr: NDArray[Any],
    window_size: int
) -> NDArray[Any]:
    """Create sliding windows without copying."""
    shape = (arr.shape[0] - window_size + 1, window_size)
    strides = (arr.strides[0], arr.strides[0])
    return as_strided(arr, shape=shape, strides=strides)

# Safe stride creation
from numpy.lib.stride_tricks import sliding_window_view
windows = sliding_window_view(np.arange(10), window_shape=3)

# Setting strides is deprecated (NumPy 2.4+)
# Use as_strided or sliding_window_view instead
```

### Copy vs View Detection

```python
def is_view(arr: NDArray[Any], potential_view: NDArray[Any]) -> bool:
    """Check if potential_view is a view of arr."""
    return np.shares_memory(arr, potential_view)

# Safe view creation
arr = np.arange(20).reshape(4, 5)
view1 = arr[1:3, 2:4]  # View (slicing)
view2 = arr[[0, 2]]     # Copy (fancy indexing)

print(is_view(arr, view1))  # True
print(is_view(arr, view2))  # False

# Ensure copy when needed
guaranteed_copy = arr.copy()

# Ensure contiguous array
contiguous = np.ascontiguousarray(arr)  # C-contiguous copy if needed
```

---

## 🔥 Numba Integration

### Numba-Compatible NumPy Patterns

```python
from numba import njit, prange
import numpy as np

# Basic numba function
@njit
def numba_sum(arr: np.ndarray) -> float:
    """Numba-accelerated sum."""
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]
    return total

# Parallel numba
@njit(parallel=True)
def parallel_square(arr: np.ndarray) -> np.ndarray:
    """Parallel element-wise square."""
    result = np.empty_like(arr)
    for i in prange(len(arr)):
        result[i] = arr[i] ** 2
    return result

# Matrix operations with numba
@njit
def numba_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Custom matrix multiplication."""
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    
    C = np.zeros((m, n), dtype=A.dtype)
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i, j] += A[i, p] * B[p, j]
    return C

# Type annotations with numba
from numba.types import float64, int64, Array

@njit(float64(float64[:], int64))
def typed_func(arr: np.ndarray, n: int) -> float:
    """Function with explicit numba types."""
    return np.sum(arr[:n])

# Supported NumPy operations in numba
@njit
def numba_compatible_ops(arr: np.ndarray) -> np.ndarray:
    """Common NumPy ops that work with numba."""
    # Math operations
    result = np.sin(arr) + np.cos(arr)
    result = np.exp(result) - np.log(np.abs(result) + 1)
    
    # Aggregations
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    
    # Array creation
    zeros = np.zeros_like(arr)
    ones = np.ones(len(arr))
    
    return result

# For more patterns, see NUMBA_PATTERNS.MD
```

---

## ⚠️ Common Pitfalls

### Anti-Patterns and Solutions

```python
# PITFALL 1: Unintended view modification
arr = np.arange(10)
view = arr[::2]
view[0] = 999  # Modifies arr!
# Solution: Always copy when independence is needed
safe_copy = arr[::2].copy()

# PITFALL 2: Shape broadcasting confusion
a = np.array([[1], [2], [3]])  # (3, 1)
b = np.array([10, 20, 30])      # (3,)
result = a + b  # (3, 3) - NOT (3, 1)!
# Solution: Use keepdims or explicit reshaping
result_intended = a + b[None, :]  # (3, 1) + (1, 3) → (3, 3)

# PITFALL 3: Integer division gotcha
arr_int = np.array([1, 2, 3], dtype=np.int32)
result = arr_int / 2  # Returns float64!
# Solution: Use integer division explicitly
int_result = arr_int // 2  # Floor division

# PITFALL 4: Comparing float arrays
a = np.array([0.1 + 0.2])
b = np.array([0.3])
equal = (a == b)  # Might be False due to floating point
# Solution: Use np.isclose or np.allclose
equal_close = np.isclose(a, b, rtol=1e-9, atol=1e-9)

# PITFALL 5: Mutating during iteration
arr = np.array([1, 2, 3, 4, 5])
# BAD: Modifying while iterating
# for i, val in enumerate(arr):
#     if val > 2:
#         arr[i] = 0  # Dangerous!
# Solution: Create new array or use masking
arr = np.where(arr > 2, 0, arr)

# PITFALL 6: np.sum axis confusion
arr_2d = np.array([[1, 2], [3, 4]])
row_sums = np.sum(arr_2d, axis=1)  # Sum across columns → (2,)
col_sums = np.sum(arr_2d, axis=0)  # Sum across rows → (2,)
# Remember: axis specifies which dimension to collapse

# PITFALL 7: Boolean indexing with wrong dtype
arr = np.array([1.5, 2.5, 3.5])
# BAD:
# mask = arr > 2  # Returns bool array
# result = arr[mask]  # This is fine
# mask[0] = 1  # BAD: Assigning int to bool!
# Solution: Be explicit about dtypes
mask = (arr > 2).astype(bool)

# PITFALL 8: Memory leak with large temporaries
# BAD:
# result = huge_array1 + huge_array2 + huge_array3  # Creates 2 temporaries
# Solution: Use out parameter
result = np.add(huge_array1, huge_array2)
np.add(result, huge_array3, out=result)

# PITFALL 9: np.unique unsorted results (NumPy 2.3+)
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
unique_vals = np.unique(arr)  # May not be sorted anymore!
# Solution: Explicitly sort if needed
unique_sorted = np.sort(np.unique(arr))
# Or use np.unique with sorted=True parameter (if available)

# PITFALL 10: Scalar vs 0-d array confusion
scalar = np.float64(3.14)
arr_0d = np.array(3.14)
print(scalar.shape)  # AttributeError!
print(arr_0d.shape)  # ()
# Solution: Use out=... to force array result (NumPy 2.3+)
result = np.add(scalar, 1.0, out=...)  # Returns array, not scalar
```

### Debugging Helpers

```python
def debug_array(arr: NDArray[Any], name: str = "array") -> None:
    """Print comprehensive array debug info."""
    print(f"\n=== {name} ===")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"Size: {arr.size} elements")
    print(f"Itemsize: {arr.itemsize} bytes")
    print(f"Total bytes: {arr.nbytes}")
    print(f"Ndim: {arr.ndim}")
    print(f"Flags:\n{arr.flags}")
    print(f"Strides: {arr.strides}")
    print(f"C-contiguous: {arr.flags['C_CONTIGUOUS']}")
    print(f"F-contiguous: {arr.flags['F_CONTIGUOUS']}")
    print(f"Writable: {arr.flags['WRITEABLE']}")
    print(f"Base is None: {arr.base is None}")

# Memory profiling helper
def memory_usage(arr: NDArray[Any]) -> str:
    """Get human-readable memory usage."""
    bytes_used = arr.nbytes
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_used < 1024:
            return f"{bytes_used:.2f} {unit}"
        bytes_used /= 1024
    return f"{bytes_used:.2f} TB"

# Validation helper
def validate_array(
    arr: NDArray[Any],
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: DTypeLike | None = None,
    check_finite: bool = True
) -> None:
    """Validate array properties."""
    if expected_shape and arr.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {arr.shape}")
    
    if expected_dtype and arr.dtype != expected_dtype:
        raise TypeError(f"Expected dtype {expected_dtype}, got {arr.dtype}")
    
    if check_finite and np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise ValueError("Array contains non-finite values (inf/nan)")
```

---

## 📚 References

### Official Documentation
- [NumPy 2.4 Release Notes](https://numpy.org/doc/stable/release/2.4.0-notes.html)
- [NumPy 2.3 Release Notes](https://numpy.org/devdocs/release/2.3.0-notes.html)
- [Array API Standard 2024.12](https://data-apis.org/array-api/latest/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [Broadcasting Rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)

### Type Hints Resources
- [Python typing module (3.14)](https://docs.python.org/3/library/typing.html)
- [NumPy Typing (numpy.typing)](https://numpy.org/doc/stable/reference/typing.html)
- [PEP 585 - Type Hinting Generics](https://peps.python.org/pep-0585/)

### Performance Resources
- [NumPy Performance Tips](https://numpy.org/doc/stable/user/c-info.beyond-basics.html)
- [SIMD Optimizations](https://numpy.org/doc/stable/reference/simd/index.html)
- [OpenMP Support](https://numpy.org/doc/stable/reference/threading.html)

### Ecosystem Integration
- [Numba Documentation](https://numba.pydata.org/)
- [SciPy Integration](https://scipy.github.io/devdocs/)
- [PyTorch NumPy Compatibility](https://pytorch.org/docs/stable/notes/numpy.html)

---

## 🔄 Version Migration

### Migrating from NumPy 1.x to 2.x

```python
# Import changes
from numpy.typing import NDArray, ArrayLike  # New location

# Deprecated aliases removed (use new names)
# OLD: np.in1d → NEW: np.isin
mask = np.isin([1, 2, 3], [2, 4])

# OLD: np.trapz → NEW: np.trapezoid
integral = np.trapezoid(y, x)

# String dtype changes
# OLD: dtype='U10' → NEW: StringDType() recommended
str_arr = np.array(['hello'], dtype=StringDType())

# Type hint updates
def func(arr: NDArray[np.float64]) -> NDArray[np.float64]:  # Type param now defaults
    return arr * 2

# Casting changes
# NEW: 'same_value' casting mode
arr.astype(np.int32, casting='same_value')

# Shape modification deprecated
# OLD: arr.shape = new_shape  
# NEW: arr = np.reshape(arr, new_shape)
```

---

## 💡 Best Practices Summary

1. **Always use type hints** with `NDArray[dtype]` for function signatures
2. **Prefer vectorization** over Python loops (10-100x faster)
3. **Use views** when possible to avoid unnecessary copies
4. **Leverage broadcasting** instead of explicit tiling
5. **Check for finite values** in numerical algorithms
6. **Use out= parameter** to avoid temporary allocations
7. **Profile before optimizing** - measure actual bottlenecks
8. **Keep arrays C-contiguous** for best performance
9. **Use numba for custom kernels** that can't be vectorized
10. **Validate array properties** at function boundaries
11. **Use keepdims=True** to preserve broadcasting compatibility
12. **Prefer `np.allclose`** over `==` for floating point comparisons
13. **Document array shapes** in docstrings and type hints
14. **Use structured dtypes** for heterogeneous data
15. **Check `np.shares_memory`** to detect unintended views

---

## 🏷️ Quick Reference Card

```python
# Array Creation
np.array([1, 2, 3])                    # From list
np.zeros((3, 4))                       # All zeros
np.ones((3, 4))                        # All ones
np.arange(start, stop, step)           # Range
np.linspace(start, stop, num)          # Evenly spaced
np.eye(n)                              # Identity matrix

# Shape Operations
arr.reshape(new_shape)                 # New shape
arr.flatten()                          # 1D copy
arr.ravel()                            # 1D view (when possible)
arr.T                                  # Transpose
np.expand_dims(arr, axis)              # Add dimension

# Indexing
arr[1, 2]                              # Single element
arr[1:3, :]                            # Slicing
arr[[0, 2], [1, 3]]                    # Fancy indexing
arr[arr > 0]                           # Boolean indexing

# Math Operations
np.sum(arr, axis=0)                    # Sum along axis
np.mean(arr)                           # Average
np.std(arr)                            # Standard deviation
np.dot(a, b) or a @ b                  # Matrix multiply
np.exp/log/sin/cos(arr)                # Element-wise math

# Comparisons
np.all(arr > 0)                        # All True?
np.any(arr > 0)                        # Any True?
np.allclose(a, b, rtol, atol)          # Close enough?
np.isclose(a, b)                       # Element-wise close

# Array API
arr.shape                              # Dimensions
arr.dtype                              # Data type
arr.size                               # Total elements
arr.ndim                               # Number of dimensions
arr.strides                            # Memory layout
```

---

**End of NumPy Comprehensive Guide**
*For Numba-specific patterns, see NUMBA_PATTERNS.MD*
*Last updated: January 2030 | NumPy 2.4.0+*
