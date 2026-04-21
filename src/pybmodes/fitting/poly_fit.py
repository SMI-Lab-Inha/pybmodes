"""Constrained 6th-order polynomial fit for ElastoDyn mode shapes.

ElastoDyn requires mode shapes of the form:
    φ(x) = C2·x² + C3·x³ + C4·x⁴ + C5·x⁵ + C6·x⁶
with constraint φ(1) = 1, i.e. C2+C3+C4+C5+C6 = 1.

The constraint is enforced analytically by substituting C6 = 1−C2−C3−C4−C5,
reducing to a 4-parameter unconstrained least-squares problem:
    φ(x) = C2·(x²−x⁶) + C3·(x³−x⁶) + C4·(x⁴−x⁶) + C5·(x⁵−x⁶) + x⁶
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PolyFitResult:
    """Polynomial fit coefficients and quality metrics for one mode component."""
    c2: float
    c3: float
    c4: float
    c5: float
    c6: float
    rms_residual: float   # RMS of (φ_poly(x) − φ_fem(x)) over all stations
    tip_slope: float      # dφ/dx at x=1: 2C2+3C3+4C4+5C5+6C6

    def coefficients(self) -> np.ndarray:
        """Return [C2, C3, C4, C5, C6] as a length-5 array."""
        return np.array([self.c2, self.c3, self.c4, self.c5, self.c6])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial at positions x ∈ [0, 1]."""
        return (self.c2 * x**2 + self.c3 * x**3 + self.c4 * x**4
                + self.c5 * x**5 + self.c6 * x**6)


def fit_mode_shape(
    span_loc: np.ndarray,
    displacement: np.ndarray,
) -> PolyFitResult:
    """Fit a constrained 6th-order polynomial to a normalised mode shape.

    The displacement is internally normalised to tip = 1 before fitting.
    Stations with x = 0 and x = 1 contribute zero residual automatically
    and may be included or omitted without affecting the result.

    Parameters
    ----------
    span_loc     : 1-D array of normalised span positions, x ∈ [0, 1].
    displacement : 1-D array of mode shape displacements at each station.
                   Need not be pre-normalised; the function divides by
                   displacement[-1] (tip value) before fitting.

    Returns
    -------
    PolyFitResult with c2..c6, rms_residual, and tip_slope.

    Raises
    ------
    ValueError if displacement[-1] is zero (degenerate tip value).
    """
    x = np.asarray(span_loc, dtype=float)
    y = np.asarray(displacement, dtype=float)

    tip_val = y[-1]
    if abs(tip_val) < 1e-30:
        raise ValueError("Tip displacement is effectively zero; cannot normalise.")
    y = y / tip_val

    # Reduced basis: each column is x^(k+2) - x^6 for k = 0..3 (C2..C5)
    A = np.column_stack([x**k - x**6 for k in range(2, 6)])
    b = y - x**6

    coeffs_r, *_ = np.linalg.lstsq(A, b, rcond=None)
    c2, c3, c4, c5 = coeffs_r
    c6 = 1.0 - c2 - c3 - c4 - c5

    # Quality metrics
    phi = c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5 + c6*x**6
    rms = float(np.sqrt(np.mean((phi - y) ** 2)))
    tip_slope = float(2*c2 + 3*c3 + 4*c4 + 5*c5 + 6*c6)

    return PolyFitResult(
        c2=float(c2), c3=float(c3), c4=float(c4), c5=float(c5), c6=float(c6),
        rms_residual=rms,
        tip_slope=tip_slope,
    )
