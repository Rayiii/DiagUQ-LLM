"""Compatibility helpers for NumPy API changes."""

from __future__ import annotations

from typing import Any


def safe_trapezoid(y: Any, x: Any = None, dx: float = 1.0, axis: int = -1):
    """Call NumPy trapezoidal integration across NumPy 1.x and 2.x.

    NumPy 2.x keeps ``numpy.trapezoid`` but may remove ``numpy.trapz``.
    Older NumPy releases expose ``numpy.trapz`` but not ``numpy.trapezoid``.
    """
    import numpy as np

    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y, x=x, dx=dx, axis=axis)
    trapz = getattr(np, "trapz", None)
    if trapz is not None:
        return trapz(y, x=x, dx=dx, axis=axis)
    raise RuntimeError("Neither numpy.trapezoid nor numpy.trapz is available.")
