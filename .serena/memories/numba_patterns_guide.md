# Numba Patterns & Best Practices Memory

**Last Updated**: January 5, 2026  
**Status**: Production-Ready  
**Location**: ChromaDB collection `numba_patterns` (13 documents)

## Quick Access

### ChromaDB Collection: `numba_patterns`

**Documents Stored** (13 patterns):
1. `pattern_safe_wrapper` - Safe wrapper with observability
2. `pattern_context_manager` - Context manager for batch operations
3. `pattern_lazy_compilation` - Lazy compilation with warmup
4. `error_handler_comprehensive` - Comprehensive error handling
5. `logging_structured` - Structured logging with JSON
6. `type_annotations` - Type-safe patterns
7. `usage_image_processing` - Image processing pipeline
8. `usage_video_processing` - Video processing batches
9. `performance_benchmarking` - Benchmarking framework
10. `testing_framework` - Unit testing patterns
11. `troubleshooting` - Common issues & solutions
12. `deployment_checklist` - Production deployment
13. `quick_reference` - Quick reference card

### Query Examples

```python
# Query ChromaDB for specific pattern
from chromadb import Client
client = Client()
collection = client.get_collection("numba_patterns")

# Find error handling patterns
results = collection.query(
    query_texts=["error handling fallback retry"],
    n_results=3
)

# Find performance optimization
results = collection.query(
    query_texts=["performance benchmarking speedup"],
    n_results=3
)
```

## Core Patterns Summary

### 1. Safe Wrapper Pattern
- **Use**: Production code with metrics
- **Features**: Auto fallback, metrics, logging
- **File**: NUMBA_PATTERNS.md lines 60-150

### 2. Context Manager Pattern
- **Use**: Batch processing with stats
- **Features**: Error tracking, summary logging
- **File**: NUMBA_PATTERNS.md lines 152-220

### 3. Error Handler Pattern
- **Use**: Centralized error management
- **Features**: 6 error categories, retries, stats
- **File**: NUMBA_PATTERNS.md lines 300-420

### 4. Structured Logging Pattern
- **Use**: Performance tracking with JSON
- **Features**: Context manager, detailed metrics
- **File**: NUMBA_PATTERNS.md lines 422-540

## File Locations

### Documentation
- NUMBA_PATTERNS.md - Comprehensive patterns guide (800+ lines)
- NUMBA_ERROR_HANDLING_GUIDE.md - Error handling (421 lines)
- NUMBA_UPGRADE_SUMMARY.md - Performance metrics (285 lines)

### Implementation
- comfy/numba_utils.py - 15+ JIT functions (383 lines)
- comfy/numba_error_handler.py - Error handling (503 lines)
- nodes.py - Image I/O (lines 1-30, 1614, 1687)
- comfy/utils.py - Lanczos (lines 20-37, 931)

## Performance Metrics

### Production Results
- Image normalization (1920x1080): 2-5x speedup
- Alpha compositing: 3-10x speedup
- Rotation matrices: 2-4x speedup
- Batch processing (32 images): 5-15x speedup

## Type Annotations

```python
Float32Array = npt.NDArray[np.float32]
UInt8Array = npt.NDArray[np.uint8]
ImageArray = Annotated[Union[Float32Array, UInt8Array], "Image (H, W, C)"]
```

## Common Patterns

### Graceful Fallback
```python
if NUMBA_AVAILABLE and arr.ndim == 3:
    try:
        return numba_func(arr)
    except Exception as e:
        logger.error(f"Numba failed: {e}")
return numpy_func(arr)
```

### Array Validation
```python
if arr.ndim != 3 or arr.dtype != np.uint8:
    raise ValueError(f"Invalid: ndim={arr.ndim}, dtype={arr.dtype}")
```

## Troubleshooting

1. **Type Inference Failed**: Use manual ops instead of np.clip
2. **Non-Contiguous Array**: Use np.ascontiguousarray()
3. **CUDA_ERROR_STUB_LIBRARY**: Benign warning, ignore
4. **Slow First Call**: Enable cache=True or use warmup
5. **Parallel Not Working**: Set NUMBA_THREADING_LAYER=tbb

## Testing Status

- Unit Tests: 20/20 passing (0.423s)
- ComfyUI Server: Running port 8188
- GPU: RTX 4090 24GB VRAM
- Numba: CPU parallel enabled

## Nodes Using Numba

1. SaveImage (nodes.py:1614) - denormalize_image_array
2. LoadImage (nodes.py:1687) - normalize_image_array
3. Lanczos (utils.py:931) - dual-path optimization
4. Dataset Export (nodes_dataset.py:188) - denormalize
5. Camera Trajectory (nodes_camera_trajectory.py:130) - rotation
6. Video Overlay (nodes_wanmove.py:190) - alpha_composite

---

**ChromaDB**: `numba_patterns` collection (13 documents)  
**Main File**: NUMBA_PATTERNS.md  
**Phase**: 10 complete (codebase-wide Numba upgrade)
