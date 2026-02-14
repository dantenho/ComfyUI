"""Shared interpolation utilities with optional Numba acceleration."""

from __future__ import annotations

import numpy as np

try:
    from comfy.numba_utils import log_linear_interpolate_steps as _log_linear_interpolate_steps
    from comfy.numba_error_handler import is_numba_available

    _NUMBA_AVAILABLE = is_numba_available()
except Exception:
    _NUMBA_AVAILABLE = False
    _log_linear_interpolate_steps = None


def _log_linear_interpolate_fallback(t_steps: np.ndarray, num_steps: int) -> np.ndarray:
    """Pure NumPy log-linear interpolation matching legacy behavior."""
    safe_steps = np.clip(t_steps, np.finfo(np.float64).tiny, None)
    xs = np.linspace(0.0, 1.0, len(safe_steps))
    ys = np.log(safe_steps[::-1])

    new_xs = np.linspace(0.0, 1.0, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    return np.exp(new_ys)[::-1].copy()


def loglinear_interp(t_steps: np.ndarray, num_steps: int) -> np.ndarray:
    """Log-linear interpolate decreasing time steps with Numba fallback."""
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")

    arr = np.asarray(t_steps, dtype=np.float64)

    if _NUMBA_AVAILABLE and _log_linear_interpolate_steps is not None:
        # Numba implementation expects float32/float64 arrays
        arr_for_numba = arr.astype(np.float32)
        return _log_linear_interpolate_steps(arr_for_numba, num_steps)

    return _log_linear_interpolate_fallback(arr, num_steps)
