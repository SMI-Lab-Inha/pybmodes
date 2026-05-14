"""Tests for fitting/poly_fit.py — constrained 6th-order polynomial fit.

All tests use synthetic polynomial mode shapes constructed in-test.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fitting import PolyFitResult, fit_mode_shape


class TestPolyFitExact:
    """Polynomials that live exactly in the model space are recovered exactly."""

    def _x(self, n=21):
        return np.linspace(0.0, 1.0, n)

    def test_returns_poly_fit_result(self):
        x = self._x()
        result = fit_mode_shape(x, x**2)
        assert isinstance(result, PolyFitResult)

    def test_pure_x2(self):
        x = self._x()
        r = fit_mode_shape(x, x**2)
        assert r.c2 == pytest.approx(1.0, abs=1e-10)
        assert r.c3 == pytest.approx(0.0, abs=1e-10)
        assert r.c4 == pytest.approx(0.0, abs=1e-10)
        assert r.c5 == pytest.approx(0.0, abs=1e-10)
        assert r.c6 == pytest.approx(0.0, abs=1e-10)

    def test_pure_x6(self):
        x = self._x()
        r = fit_mode_shape(x, x**6)
        assert r.c6 == pytest.approx(1.0, abs=1e-10)

    def test_mixed_coefficients(self):
        x = self._x(51)
        c = np.array([0.3, 0.2, 0.1, 0.15, 0.25])
        y = sum(c[k] * x**(k + 2) for k in range(5))
        r = fit_mode_shape(x, y)
        assert r.c2 == pytest.approx(c[0], abs=1e-10)
        assert r.c3 == pytest.approx(c[1], abs=1e-10)
        assert r.c4 == pytest.approx(c[2], abs=1e-10)
        assert r.c5 == pytest.approx(c[3], abs=1e-10)
        assert r.c6 == pytest.approx(c[4], abs=1e-10)

    def test_constraint_satisfied(self):
        rng = np.random.default_rng(42)
        x = np.sort(rng.uniform(0.0, 1.0, 30))
        x = np.concatenate([[0.0], x, [1.0]])
        y = np.sin(np.pi * x / 2)
        r = fit_mode_shape(x, y)
        assert r.c2 + r.c3 + r.c4 + r.c5 + r.c6 == pytest.approx(1.0, abs=1e-12)

    def test_rms_zero_for_exact_poly(self):
        x = self._x(31)
        y = 0.4 * x**2 + 0.1 * x**3 + 0.2 * x**4 + 0.1 * x**5 + 0.2 * x**6
        r = fit_mode_shape(x, y)
        assert r.rms_residual == pytest.approx(0.0, abs=1e-12)

    def test_tip_slope_x2(self):
        x = self._x()
        r = fit_mode_shape(x, x**2)
        assert r.tip_slope == pytest.approx(2.0, abs=1e-10)

    def test_coefficients_array(self):
        x = self._x()
        r = fit_mode_shape(x, x**3)
        coeffs = r.coefficients()
        assert coeffs.shape == (5,)
        assert coeffs.sum() == pytest.approx(1.0, abs=1e-12)

    def test_evaluate_matches_direct(self):
        x = self._x()
        y = 0.5 * x**2 + 0.5 * x**4
        r = fit_mode_shape(x, y)
        x_test = np.linspace(0, 1, 5)
        direct = (r.c2 * x_test**2 + r.c3 * x_test**3 + r.c4 * x_test**4
                  + r.c5 * x_test**5 + r.c6 * x_test**6)
        assert r.evaluate(x_test) == pytest.approx(direct, abs=1e-14)

    def test_normalises_tip(self):
        x = self._x()
        y = 3.7 * x**2
        r = fit_mode_shape(x, y)
        assert r.c2 == pytest.approx(1.0, abs=1e-10)

    def test_zero_tip_raises(self):
        x = np.linspace(0, 1, 11)
        y = np.zeros_like(x)
        with pytest.raises(ValueError, match="zero"):
            fit_mode_shape(x, y)


# ===========================================================================
# Input validation — bad shapes, lengths, non-finite values, non-monotonic
# ===========================================================================

class TestFitModeShapeValidation:

    def test_empty_arrays_raise(self) -> None:
        with pytest.raises(ValueError, match="at least 2 stations"):
            fit_mode_shape(np.array([]), np.array([]))

    def test_single_station_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 stations"):
            fit_mode_shape(np.array([1.0]), np.array([1.0]))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="must have the same length"):
            fit_mode_shape(np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0]))

    def test_two_d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D"):
            fit_mode_shape(np.array([[0.0, 0.5]]), np.array([[0.0, 1.0]]))

    def test_nan_displacement_raises(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            fit_mode_shape(
                np.linspace(0.0, 1.0, 6),
                np.array([0.0, 0.1, np.nan, 0.4, 0.7, 1.0]),
            )

    def test_non_monotonic_span_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            fit_mode_shape(
                np.array([0.0, 0.3, 0.2, 0.6, 0.8, 1.0]),  # 0.2 < 0.3
                np.linspace(0.0, 1.0, 6),
            )

    def test_well_formed_input_still_fits(self) -> None:
        """Sanity: a well-formed input still produces a clean fit."""
        x = np.linspace(0.0, 1.0, 11)
        y = x ** 2  # exact polynomial → fits to machine precision
        fit = fit_mode_shape(x, y)
        assert fit.rms_residual < 1e-10
        assert fit.c2 == pytest.approx(1.0, abs=1e-10)
