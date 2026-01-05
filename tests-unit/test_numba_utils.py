"""
Unit tests for Numba-optimized NumPy utilities.
Tests both Numba and NumPy fallback paths.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import modules to test
from comfy.numba_utils import (
    normalize_image_array,
    denormalize_image_array,
    alpha_composite,
    combined_rotation_matrix,
    linear_interpolate_steps,
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
    """Test Numba-optimized utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.small_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.medium_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    def test_normalize_image_array(self):
        """Test image normalization."""
        result = normalize_image_array(self.small_img, scale=255.0)
        
        # Check dtype
        self.assertEqual(result.dtype, np.float32)
        
        # Check shape
        self.assertEqual(result.shape, self.small_img.shape)
        
        # Check values in range [0, 1]
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))
        
        # Check specific values
        expected = self.small_img.astype(np.float32) / 255.0
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_denormalize_image_array(self):
        """Test image denormalization."""
        normalized = self.small_img.astype(np.float32) / 255.0
        result = denormalize_image_array(normalized, scale=255.0)
        
        # Check dtype
        self.assertEqual(result.dtype, np.uint8)
        
        # Check shape
        self.assertEqual(result.shape, normalized.shape)
        
        # Check values match original
        np.testing.assert_array_almost_equal(result, self.small_img, decimal=0)
    
    def test_alpha_composite(self):
        """Test alpha compositing."""
        h, w = 100, 100
        rgb = np.random.rand(h, w, 3).astype(np.float32)
        track = np.random.rand(h, w, 3).astype(np.float32)
        alpha = np.random.rand(h, w, 3).astype(np.float32)
        
        result = alpha_composite(rgb, track, alpha)
        
        # Check shape
        self.assertEqual(result.shape, (h, w, 3))
        
        # Check manual calculation for a few pixels
        for i in range(0, h, 20):
            for j in range(0, w, 20):
                expected = rgb[i, j] * (1 - alpha[i, j]) + track[i, j] * alpha[i, j]
                np.testing.assert_allclose(result[i, j], expected, rtol=1e-5)
    
    def test_combined_rotation_matrix(self):
        """Test rotation matrix computation."""
        theta_x = np.pi / 4
        theta_y = np.pi / 6
        theta_z = np.pi / 3
        
        result = combined_rotation_matrix(theta_x, theta_y, theta_z)
        
        # Check shape
        self.assertEqual(result.shape, (3, 3))
        
        # Check it's a valid rotation matrix (det = 1, orthogonal)
        det = np.linalg.det(result)
        self.assertAlmostEqual(det, 1.0, places=5)
        
        # R^T * R should be identity
        identity = np.dot(result.T, result)
        np.testing.assert_allclose(identity, np.eye(3), rtol=1e-5)
    
    def test_linear_interpolate_steps(self):
        """Test linear interpolation."""
        t_steps = np.array([0.0, 1.0, 3.0, 5.0])
        num_steps = 10
        
        result = linear_interpolate_steps(t_steps, num_steps)
        
        # Check shape
        self.assertEqual(len(result), num_steps)
        
        # Check monotonicity
        self.assertTrue(np.all(result[1:] >= result[:-1]))
        
        # Check bounds
        self.assertAlmostEqual(result[0], t_steps[0], places=5)
        self.assertAlmostEqual(result[-1], t_steps[-1], places=5)
    
    def test_fast_array_min_max(self):
        """Test fast min/max computation."""
        arr = np.random.rand(100, 50).astype(np.float32)
        
        min_vals, max_vals = fast_array_min_max(arr)
        
        # Check against NumPy
        expected_min = arr.min(axis=0)
        expected_max = arr.max(axis=0)
        
        np.testing.assert_allclose(min_vals, expected_min, rtol=1e-5)
        np.testing.assert_allclose(max_vals, expected_max, rtol=1e-5)
    
    def test_fast_clip_array(self):
        """Test fast array clipping."""
        arr = np.random.rand(50, 50).astype(np.float32) * 2 - 0.5  # Range [-0.5, 1.5]
        
        result = fast_clip_array(arr, min_val=0.0, max_val=1.0)
        
        # Check all values in range
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))
        
        # Compare with NumPy clip
        expected = np.clip(arr, 0.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_fast_array_multiply_add(self):
        """Test fast multiply-add."""
        a = np.random.rand(100, 100).astype(np.float32)
        b = np.random.rand(100, 100).astype(np.float32)
        scale_a = 0.7
        scale_b = 0.3
        
        result = fast_array_multiply_add(a, b, scale_a, scale_b)
        
        # Check against manual calculation
        expected = scale_a * a + scale_b * b
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_fast_mean_std(self):
        """Test fast mean and standard deviation."""
        arr = np.random.rand(100, 100).astype(np.float32)
        
        mean, std = fast_mean_std(arr)
        
        # Compare with NumPy
        expected_mean = arr.mean()
        expected_std = arr.std()
        
        self.assertAlmostEqual(mean, expected_mean, places=5)
        self.assertAlmostEqual(std, expected_std, places=5)
    
    def test_prepare_array_for_numba(self):
        """Test array preparation for Numba."""
        # Create non-contiguous array
        arr = np.random.rand(100, 100, 3).astype(np.float32)
        arr_non_contiguous = arr.transpose(2, 0, 1).transpose(1, 2, 0)
        
        self.assertFalse(arr_non_contiguous.flags['C_CONTIGUOUS'])
        
        result = prepare_array_for_numba(arr_non_contiguous)
        
        self.assertTrue(result.flags['C_CONTIGUOUS'])
        np.testing.assert_array_equal(result, arr_non_contiguous)


class TestNumbaErrorHandler(unittest.TestCase):
    """Test error handling utilities."""
    
    def test_validate_numpy_array_valid(self):
        """Test array validation with valid input."""
        arr = np.random.rand(10, 10).astype(np.float32)
        
        result = validate_numpy_array(
            arr, 
            name="test_array",
            dtype=np.float32,
            ndim=2
        )
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
    
    def test_validate_numpy_array_conversion(self):
        """Test array validation with type conversion."""
        arr = [[1, 2, 3], [4, 5, 6]]
        
        result = validate_numpy_array(arr, name="test_list")
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 3))
    
    def test_validate_numpy_array_invalid_type(self):
        """Test array validation with invalid input."""
        with self.assertRaises(TypeError):
            validate_numpy_array("not an array", name="test")
    
    def test_validate_numpy_array_wrong_dims(self):
        """Test array validation with wrong dimensions."""
        arr = np.random.rand(10, 10)
        
        with self.assertRaises(ValueError):
            validate_numpy_array(arr, name="test", ndim=3)
    
    def test_ensure_contiguous(self):
        """Test contiguous array conversion."""
        arr = np.random.rand(10, 10, 3)
        arr_non_contiguous = arr.transpose(2, 0, 1).transpose(1, 2, 0)
        
        self.assertFalse(arr_non_contiguous.flags['C_CONTIGUOUS'])
        
        result = ensure_contiguous(arr_non_contiguous)
        
        self.assertTrue(result.flags['C_CONTIGUOUS'])
    
    def test_check_numba_availability(self):
        """Test Numba availability check."""
        result = check_numba_availability()
        
        self.assertIsInstance(result, dict)
        self.assertIn("numba_available", result)
        self.assertIn("cuda_available", result)
        self.assertIn("numba_version", result)
    
    def test_numba_context_success(self):
        """Test NumbaContext with successful operation."""
        with NumbaContext("test_operation") as ctx:
            # Simulate successful operation
            result = 1 + 1
            ctx.log_success()
        
        self.assertTrue(ctx.success)
    
    def test_numba_context_failure(self):
        """Test NumbaContext with failed operation."""
        with self.assertRaises(ValueError):
            with NumbaContext("test_operation"):
                raise ValueError("Test error")


class TestIntegration(unittest.TestCase):
    """Integration tests for Numba functions."""
    
    def test_image_processing_pipeline(self):
        """Test complete image processing pipeline."""
        # Create test image
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Normalize
        normalized = normalize_image_array(img, scale=255.0)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(np.all(normalized >= 0.0) and np.all(normalized <= 1.0))
        
        # Denormalize
        denormalized = denormalize_image_array(normalized, scale=255.0)
        self.assertEqual(denormalized.dtype, np.uint8)
        
        # Check round-trip
        np.testing.assert_array_almost_equal(denormalized, img, decimal=0)
    
    def test_rotation_and_interpolation(self):
        """Test rotation matrices with interpolation."""
        # Create rotation matrices
        angles = np.linspace(0, 2*np.pi, 10)
        matrices = [combined_rotation_matrix(a, 0, 0) for a in angles]
        
        # Check all are valid rotation matrices
        for mat in matrices:
            det = np.linalg.det(mat)
            self.assertAlmostEqual(det, 1.0, places=4)
        
        # Test interpolation
        t_steps = np.array([0.0, 0.5, 1.0, 1.5])
        interpolated = linear_interpolate_steps(t_steps, 20)
        
        self.assertEqual(len(interpolated), 20)
        self.assertTrue(np.all(interpolated[1:] >= interpolated[:-1]))


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNumbaUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestNumbaErrorHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
