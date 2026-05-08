"""Additional unit tests for :mod:`pybmodes.fitting.poly_fit`.

The existing ``test_fitting.py`` covers the core constraint and integration
tests. This module focuses on numerical edge cases and the ``PolyFitResult``
helper API.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fitting import PolyFitResult, fit_mode_shape

# ===========================================================================
# Numerical edge cases
# ===========================================================================

class TestNumericalEdges:

    def test_negative_tip_displacement(self):
        # A negative tip should still be normalised to 1 internally.
        x = np.linspace(0.0, 1.0, 11)
        y = -1.0 * x**2
        r = fit_mode_shape(x, y)
        assert r.c2 == pytest.approx(1.0, abs=1e-10)
        assert r.c2 + r.c3 + r.c4 + r.c5 + r.c6 == pytest.approx(1.0, abs=1e-12)

    def test_very_small_tip_below_threshold_raises(self):
        x = np.linspace(0.0, 1.0, 11)
        y = np.zeros_like(x)
        y[-1] = 1e-31      # below the 1e-30 threshold
        with pytest.raises(ValueError, match="zero"):
            fit_mode_shape(x, y)

    def test_very_small_tip_at_threshold_succeeds(self):
        # Barely above the threshold — should still fit.
        x = np.linspace(0.0, 1.0, 11)
        y = 1e-25 * x**3
        r = fit_mode_shape(x, y)
        assert r.c2 + r.c3 + r.c4 + r.c5 + r.c6 == pytest.approx(1.0, abs=1e-12)

    def test_two_point_fit_on_minimal_data(self):
        # Underdetermined system (2 stations, 4 free coefficients) — lstsq
        # should still return a result satisfying the constraint.
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        r = fit_mode_shape(x, y)
        assert r.c2 + r.c3 + r.c4 + r.c5 + r.c6 == pytest.approx(1.0, abs=1e-12)

    def test_nonuniform_x_spacing(self):
        # Non-uniform sampling should not break the fit; an exact polynomial
        # should still be recovered.
        rng = np.random.default_rng(7)
        x = np.sort(rng.uniform(0.0, 1.0, 30))
        x = np.concatenate([[0.0], x, [1.0]])
        y = 0.5 * x**3 + 0.5 * x**5
        r = fit_mode_shape(x, y)
        assert r.c3 == pytest.approx(0.5, abs=1e-10)
        assert r.c5 == pytest.approx(0.5, abs=1e-10)
        assert r.rms_residual == pytest.approx(0.0, abs=1e-12)

    def test_tip_slope_x6(self):
        # dphi/dx |_{x=1} for x^6 is 6.
        x = np.linspace(0.0, 1.0, 11)
        r = fit_mode_shape(x, x**6)
        assert r.tip_slope == pytest.approx(6.0, abs=1e-10)


# ===========================================================================
# PolyFitResult helper API
# ===========================================================================

class TestPolyFitResultAPI:

    def test_evaluate_at_zero_is_zero(self):
        # All terms are >= x^2, so phi(0) = 0 always.
        x = np.linspace(0.0, 1.0, 11)
        r = fit_mode_shape(x, x**3)
        assert r.evaluate(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_evaluate_at_one_is_one(self):
        # The constraint c2+...+c6 = 1 forces phi(1) = 1 exactly.
        x = np.linspace(0.0, 1.0, 11)
        r = fit_mode_shape(x, 0.5 * x**2 + 0.5 * x**4)
        assert r.evaluate(np.array([1.0]))[0] == pytest.approx(1.0, abs=1e-12)

    def test_evaluate_accepts_scalar_array(self):
        x = np.linspace(0.0, 1.0, 11)
        r = fit_mode_shape(x, x**2)
        out = r.evaluate(np.array([0.5]))
        assert out.shape == (1,)
        assert out[0] == pytest.approx(0.25, abs=1e-12)

    def test_coefficients_order_and_length(self):
        x = np.linspace(0.0, 1.0, 11)
        r = fit_mode_shape(x, x**4)
        coeffs = r.coefficients()
        assert coeffs.shape == (5,)
        # Coefficients are returned in order [c2, c3, c4, c5, c6]
        np.testing.assert_array_almost_equal(coeffs, [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_dataclass_fields(self):
        r = PolyFitResult(c2=1.0, c3=0.0, c4=0.0, c5=0.0, c6=0.0,
                           rms_residual=0.0, tip_slope=2.0, cond_number=0.0)
        assert r.c2 == 1.0
        assert r.tip_slope == 2.0
        assert r.cond_number == 0.0


# ===========================================================================
# RMS residual on poor data
# ===========================================================================

class TestRmsResidual:

    def test_rms_positive_for_non_polynomial(self):
        # sin(pi x / 2) is smooth but not a 6th-order polynomial; some residual
        # is expected (small but > 0).
        x = np.linspace(0.0, 1.0, 31)
        y = np.sin(np.pi * x / 2)
        r = fit_mode_shape(x, y)
        assert r.rms_residual > 0.0
        # The constrained fit (zero displacement and zero slope at root, then
        # sum-to-1 at the tip) cannot match sin(pi x / 2) exactly, but the
        # residual should still be modest.
        assert r.rms_residual < 0.05

    def test_rms_smaller_for_better_match(self):
        x = np.linspace(0.0, 1.0, 31)
        # An exact poly should have a near-zero residual.
        y_exact = 0.5 * x**2 + 0.5 * x**4
        r_exact = fit_mode_shape(x, y_exact)
        # An off-shape (e.g. sqrt(x)) is far less polynomial.
        y_bad = np.sqrt(x)
        r_bad = fit_mode_shape(x, y_bad)
        assert r_exact.rms_residual < r_bad.rms_residual
