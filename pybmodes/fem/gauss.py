"""Gauss quadrature points and weights for the interval [0, 1]."""

from __future__ import annotations

import numpy as np


def gauss_6pt() -> tuple[np.ndarray, np.ndarray]:
    """6-point Gauss-Legendre quadrature on [0, 1].

    Returns (points, weights), both shape (6,).
    Used for spatial element integration.
    """
    pts, wts = np.polynomial.legendre.leggauss(6)
    return (pts + 1.0) / 2.0, wts / 2.0


def gauss_5pt() -> tuple[np.ndarray, np.ndarray]:
    """5-point Gauss-Legendre quadrature on [0, 1].

    Returns (points, weights), both shape (5,).
    Not required for free-vibration analysis.
    """
    pts, wts = np.polynomial.legendre.leggauss(5)
    return (pts + 1.0) / 2.0, wts / 2.0
