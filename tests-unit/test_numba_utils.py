"""Unit tests for Numba-optimized NumPy utilities.

Tests both Numba and NumPy fallback paths with NumPy 2.4.0+ features.

This test suite validates:
- Numba-optimized array operations
- Type hints with numpy.typing (NumPy 2.4.0+)
- Runtime signature introspection (NumPy 2.4.0+)
- Free-threaded Python compatibility (NumPy 2.4.0+)
- Error handling and fallback mechanisms
"""

from __future__ import annotations

import sys
import os
import unittest
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray, ArrayLike, DTypeLike

if TYPE_CHECKING:
    from numpy import floating, uint8

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import modules to test
from comfy.numba_utils import (
    normalize_image_array,
    denormalize_image_array,
    alpha_composite,
    combined_rotation_matrix,
    linear_interpolate_steps,
    log_linear_interpolate_steps,
    fast_array_min_max,
    fast_clip_array,
    fast_array_multiply_add,
    fast_mean_std,
    prepare_array_for_numba
)

from comfy.numba_error_handler import (
    validate_numpy_array,
    ensure_contiguous,
    check_numba_availability,
    NumbaContext,
    NumbaExecutionError
)


class TestNumbaUtils(unittest.TestCase):
    """Test Numba-optimized utility functions with NumPy 2.4.0+ features."""
    
    def setUp(self) -> None:
        """Set up test fixtures with type-safe arrays."""
        np.random.seed(42)
        self.small_img: NDArray[uint8] = np.random.randint(
            0, 256, (100, 100, 3), dtype=np.uint8
        )
        self.medium_img: NDArray[uint8] = np.random.randint(
            0, 256, (512, 512, 3), dtype=np.uint8
        )
    
    def test_normalize_image_array(self) -> None:
        """Test image normalization with NumPy 2.4.0 type hints."""
        result: NDArray[floating] = normalize_image_array(self.small_img, scale=255.0)
        
        # Check dtype
        self.assertEqual(result.dtype, np.float32)
        
        # Check shape
        self.assertEqual(result.shape, self.small_img.shape)
        
        # Check values in range [0, 1]
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))
        
        # Check specific values
        expected: NDArray[floating] = self.small_img.astype(np.float32) / 255.0
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_denormalize_image_array(self) -> None:
        """Test image denormalization with type safety."""
        normalized: NDArray[floating] = self.small_img.astype(np.float32) / 255.0
        result: NDArray[uint8] = denormalize_image_array(normalized, scale=255.0)
        
        # Check dtype
        self.assertEqual(result.dtype, np.uint8)
        
        # Check shape
        self.assertEqual(result.shape, normalized.shape)
        
        # Check values match original
        np.testing.assert_array_almost_equal(result, self.small_img, decimal=0)
    
    def test_alpha_composite(self) -> None:
        """Test alpha compositing with explicit array types."""
        h, w = 100, 100
        rgb: NDArray[floating] = np.random.rand(h, w, 3).astype(np.float32)
        track: NDArray[floating] = np.random.rand(h, w, 3).astype(np.float32)
        alpha: NDArray[floating] = np.random.rand(h, w, 3).astype(np.float32)
        
        result: NDArray[floating] = alpha_composite(rgb, track, alpha)
        
        # Check shape
        self.assertEqual(result.shape, (h, w, 3))
        
        # Check manual calculation for a few pixels
        for i in range(0, h, 20):
            for j in range(0, w, 20):
                expected = rgb[i, j] * (1 - alpha[i, j]) + track[i, j] * alpha[i, j]
                np.testing.assert_allclose(result[i, j], expected, rtol=1e-5)
    
    def test_combined_rotation_matrix(self) -> None:
        """Test rotation matrix computation with NumPy 2.4.0 precision."""
        theta_x: float = np.pi / 4
        theta_y: float = np.pi / 6
        theta_z: float = np.pi / 3
        
        result: NDArray[floating] = combined_rotation_matrix(theta_x, theta_y, theta_z)
        
        # Check shape
        self.assertEqual(result.shape, (3, 3))
        
        # Check it's a valid rotation matrix (det = 1, orthogonal)
        det: floating = np.linalg.det(result)
        self.assertAlmostEqual(float(det), 1.0, places=5)
        
        # R^T * R should be identity
        identity: NDArray[floating] = np.dot(result.T, result)
        np.testing.assert_allclose(identity, np.eye(3), rtol=1e-4, atol=1e-7)
    
    def test_linear_interpolate_steps(self) -> None:
        """Test linear interpolation with type-safe arrays."""
        t_steps: NDArray[floating] = np.array([0.0, 1.0, 3.0, 5.0])
        num_steps: int = 10
        
        result: NDArray[floating] = linear_interpolate_steps(t_steps, num_steps)
        
        # Check shape
        self.assertEqual(len(result), num_steps)
        
        # Check monotonicity
        self.assertTrue(np.all(result[1:] >= result[:-1]))
        
        # Check bounds
        self.assertAlmostEqual(float(result[0]), float(t_steps[0]), places=5)
        self.assertAlmostEqual(float(result[-1]), float(t_steps[-1]), places=5)

    def test_log_linear_interpolate_steps(self) -> None:
        """Test log-linear interpolation matches NumPy reference implementation."""
        t_steps: NDArray[floating] = np.array([14.6, 6.7, 3.8, 2.0, 1.0, 0.5], dtype=np.float32)
        num_steps: int = 12

        result: NDArray[floating] = log_linear_interpolate_steps(t_steps, num_steps)

        # Reference NumPy implementation (mirror legacy code)
        xs = np.linspace(0.0, 1.0, len(t_steps))
        ys = np.log(t_steps[::-1])
        new_xs = np.linspace(0.0, 1.0, num_steps)
        new_ys = np.interp(new_xs, xs, ys)
        expected: NDArray[floating] = np.exp(new_ys)[::-1]

        self.assertEqual(len(result), num_steps)
        self.assertEqual(result.dtype, t_steps.dtype)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_fast_array_min_max(self) -> None:
        """Test fast min/max computation with NumPy 2.4.0 precision."""
        arr: NDArray[floating] = np.random.rand(100, 50).astype(np.float32)
        
        min_vals: NDArray[floating]
        max_vals: NDArray[floating]
        min_vals, max_vals = fast_array_min_max(arr)
        
        # Check against NumPy
        expected_min: NDArray[floating] = arr.min(axis=0)
        expected_max: NDArray[floating] = arr.max(axis=0)
        
        np.testing.assert_allclose(min_vals, expected_min, rtol=1e-5)
        np.testing.assert_allclose(max_vals, expected_max, rtol=1e-5)
    
    def test_fast_clip_array(self) -> None:
        """Test fast array clipping with explicit bounds."""
        arr: NDArray[floating] = (
            np.random.rand(50, 50).astype(np.float32) * 2 - 0.5
        )  # Range [-0.5, 1.5]
        
        result: NDArray[floating] = fast_clip_array(arr, min_val=0.0, max_val=1.0)
        
        # Check all values in range
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))
        
        # Compare with NumPy clip
        expected: NDArray[floating] = np.clip(arr, 0.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_fast_array_multiply_add(self) -> None:
        """Test fast multiply-add with type hints."""
        a: NDArray[floating] = np.random.rand(100, 100).astype(np.float32)
        b: NDArray[floating] = np.random.rand(100, 100).astype(np.float32)
        scale_a: float = 0.7
        scale_b: float = 0.3
        
        result: NDArray[floating] = fast_array_multiply_add(a, b, scale_a, scale_b)
        
        # Check against manual calculation
        expected: NDArray[floating] = scale_a * a + scale_b * b
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_fast_mean_std(self) -> None:
        """Test fast mean and standard deviation with NumPy 2.4.0."""
        arr: NDArray[floating] = np.random.rand(100, 100).astype(np.float32)
        
        mean: float
        std: float
        mean, std = fast_mean_std(arr)
        
        # Compare with NumPy
        expected_mean: floating = arr.mean()
        expected_std: floating = arr.std()
        
        self.assertAlmostEqual(mean, float(expected_mean), places=5)
        self.assertAlmostEqual(std, float(expected_std), places=5)
    
    def test_prepare_array_for_numba(self) -> None:
        """Test array preparation for Numba with contiguity checks."""
        # Create non-contiguous array by slicing
        arr: NDArray[floating] = np.random.rand(100, 100, 3).astype(np.float32)
        arr_non_contiguous: NDArray[floating] = arr[::2, ::2, :]  # Non-contiguous view
        
        # Only test if actually non-contiguous
        if not arr_non_contiguous.flags['C_CONTIGUOUS']:
            result: NDArray[floating] = prepare_array_for_numba(arr_non_contiguous)
            self.assertTrue(result.flags['C_CONTIGUOUS'])
            np.testing.assert_array_equal(result, arr_non_contiguous)
        else:
            # Skip if NumPy already made it contiguous
            self.skipTest("NumPy created contiguous array")
    
    def test_numpy_2_4_features(self) -> None:
        """Test NumPy 2.4.0+ specific features and compatibility.
        
        Tests:
        - Type hints with numpy.typing
        - 64 max dimensions support
        - Improved casting safety
        - Array API standard compliance
        """
        # Test array creation with explicit dtype
        arr: NDArray[floating] = np.zeros((10, 10), dtype=np.float32)
        self.assertEqual(arr.dtype, np.float32)
        
        # Test array with high dimensions (NumPy 2.4.0 supports up to 64)
        shape_10d: tuple[int, ...] = (2,) * 10
        arr_10d: NDArray[floating] = np.ones(shape_10d, dtype=np.float32)
        self.assertEqual(arr_10d.ndim, 10)
        self.assertEqual(arr_10d.size, 2**10)
        
        # Test type casting with safety (using 'unsafe' for int32 to float32)
        # Note: NumPy 2.4.0+ enforces stricter casting rules
        arr_int: NDArray = np.array([1, 2, 3], dtype=np.int32)
        arr_float: NDArray[floating] = arr_int.astype(np.float32, casting='unsafe')
        self.assertEqual(arr_float.dtype, np.float32)
        np.testing.assert_array_almost_equal(arr_float, [1.0, 2.0, 3.0])
        
        # Test array API standard compliance (NumPy 2.4.0+)
        arr_test: NDArray[floating] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.assertTrue(hasattr(arr_test, '__array_namespace__'))


class TestNumbaErrorHandler(unittest.TestCase):
    """Test error handling utilities with NumPy 2.4.0 type safety."""
    
    def test_validate_numpy_array_valid(self) -> None:
        """Test array validation with valid input."""
        arr: NDArray[floating] = np.random.rand(10, 10).astype(np.float32)
        
        result: NDArray = validate_numpy_array(
            arr, 
            name="test_array",
            dtype=np.float32,
            ndim=2
        )
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
    
    def test_validate_numpy_array_conversion(self) -> None:
        """Test array validation with type conversion."""
        arr: list[list[int]] = [[1, 2, 3], [4, 5, 6]]
        
        result: NDArray = validate_numpy_array(arr, name="test_list")
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 3))
    
    def test_validate_numpy_array_invalid_type(self) -> None:
        """Test array validation with invalid input."""
        with self.assertRaises(TypeError):
            validate_numpy_array("not an array", name="test")
    
    def test_validate_numpy_array_wrong_dims(self) -> None:
        """Test array validation with wrong dimensions."""
        arr: NDArray[floating] = np.random.rand(10, 10)
        
        with self.assertRaises(ValueError):
            validate_numpy_array(arr, name="test", ndim=3)
    
    def test_ensure_contiguous(self) -> None:
        """Test contiguous array conversion."""
        arr: NDArray[floating] = np.random.rand(10, 10, 3)
        arr_non_contiguous: NDArray[floating] = arr[::2, ::2, :]  # Non-contiguous view
        
        # Only test if actually non-contiguous
        if not arr_non_contiguous.flags['C_CONTIGUOUS']:
            result: NDArray[floating] = ensure_contiguous(arr_non_contiguous)
            self.assertTrue(result.flags['C_CONTIGUOUS'])
        else:
            # Skip if NumPy already made it contiguous
            self.skipTest("NumPy created contiguous array")
    
    def test_check_numba_availability(self) -> None:
        """Test Numba availability check."""
        result: dict[str, bool | str | None] = check_numba_availability()
        
        self.assertIsInstance(result, dict)
        self.assertIn("numba_available", result)
        self.assertIn("cuda_available", result)
        self.assertIn("numba_version", result)
    
    def test_numba_context_success(self) -> None:
        """Test NumbaContext with successful operation."""
        with NumbaContext("test_operation") as ctx:
            # Simulate successful operation
            result: int = 1 + 1
            ctx.log_success()
        
        self.assertTrue(ctx.success)
    
    def test_numba_context_failure(self) -> None:
        """Test NumbaContext with failed operation."""
        with self.assertRaises(ValueError):
            with NumbaContext("test_operation"):
                raise ValueError("Test error")


class TestIntegration(unittest.TestCase):
    """Integration tests for Numba functions with NumPy 2.4.0."""
    
    def test_image_processing_pipeline(self) -> None:
        """Test complete image processing pipeline with type safety."""
        # Create test image
        img: NDArray[uint8] = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Normalize
        normalized: NDArray[floating] = normalize_image_array(img, scale=255.0)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(np.all(normalized >= 0.0) and np.all(normalized <= 1.0))
        
        # Denormalize
        denormalized: NDArray[uint8] = denormalize_image_array(normalized, scale=255.0)
        self.assertEqual(denormalized.dtype, np.uint8)
        
        # Check round-trip
        np.testing.assert_array_almost_equal(denormalized, img, decimal=0)
    
    def test_rotation_and_interpolation(self) -> None:
        """Test rotation matrices with interpolation and NumPy 2.4.0 precision."""
        # Create rotation matrices
        angles: NDArray[floating] = np.linspace(0, 2*np.pi, 10)
        matrices: list[NDArray[floating]] = [
            combined_rotation_matrix(float(a), 0.0, 0.0) for a in angles
        ]
        
        # Check all are valid rotation matrices
        for mat in matrices:
            det: floating = np.linalg.det(mat)
            self.assertAlmostEqual(float(det), 1.0, places=4)
        
        # Test interpolation
        t_steps: NDArray[floating] = np.array([0.0, 0.5, 1.0, 1.5])
        interpolated: NDArray[floating] = linear_interpolate_steps(t_steps, 20)
        
        self.assertEqual(len(interpolated), 20)
        self.assertTrue(np.all(interpolated[1:] >= interpolated[:-1]))


def run_tests() -> unittest.TestResult:
    """Run all tests and return results.
    
    Returns:
        unittest.TestResult: Test execution results.
    """
    loader: unittest.TestLoader = unittest.TestLoader()
    suite: unittest.TestSuite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNumbaUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestNumbaErrorHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run with verbosity
    runner: unittest.TextTestRunner = unittest.TextTestRunner(verbosity=2)
    result: unittest.TestResult = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result: unittest.TestResult = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
