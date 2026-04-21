"""Tests for fitting/poly_fit.py — constrained 6th-order polynomial fit."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pybmodes.elastodyn.params import (
    _TowerFamilySelectionConfig,
    _TowerModeCandidate,
    _score_tower_family,
    _select_tower_family,
    _tower_candidate,
    _tower_family_candidates,
)
from pybmodes.fitting import PolyFitResult, fit_mode_shape
from pybmodes.fem.normalize import NodeModeShape
from pybmodes.models import RotatingBlade, Tower

CERT_DIR = pathlib.Path(__file__).parent / "data" / "certtest"


class TestPolyFitExact:
    """Fit known polynomials that live exactly in the model space."""

    def _make_x(self, n: int = 21) -> np.ndarray:
        return np.linspace(0.0, 1.0, n)

    def test_returns_poly_fit_result(self):
        x = self._make_x()
        result = fit_mode_shape(x, x**2)
        assert isinstance(result, PolyFitResult)

    def test_pure_x2(self):
        # φ = x² → C2=1, rest=0
        x = self._make_x()
        r = fit_mode_shape(x, x**2)
        assert r.c2 == pytest.approx(1.0, abs=1e-10)
        assert r.c3 == pytest.approx(0.0, abs=1e-10)
        assert r.c4 == pytest.approx(0.0, abs=1e-10)
        assert r.c5 == pytest.approx(0.0, abs=1e-10)
        assert r.c6 == pytest.approx(0.0, abs=1e-10)

    def test_pure_x6(self):
        # φ = x⁶ → C6=1, rest=0
        x = self._make_x()
        r = fit_mode_shape(x, x**6)
        assert r.c6 == pytest.approx(1.0, abs=1e-10)
        assert r.c2 == pytest.approx(0.0, abs=1e-10)

    def test_mixed_coefficients(self):
        # φ = 0.3x² + 0.2x³ + 0.1x⁴ + 0.15x⁵ + 0.25x⁶
        x = self._make_x(51)
        c = np.array([0.3, 0.2, 0.1, 0.15, 0.25])
        y = sum(c[k] * x**(k+2) for k in range(5))
        r = fit_mode_shape(x, y)
        assert r.c2 == pytest.approx(c[0], abs=1e-10)
        assert r.c3 == pytest.approx(c[1], abs=1e-10)
        assert r.c4 == pytest.approx(c[2], abs=1e-10)
        assert r.c5 == pytest.approx(c[3], abs=1e-10)
        assert r.c6 == pytest.approx(c[4], abs=1e-10)

    def test_constraint_satisfied(self):
        # C2+C3+C4+C5+C6 must equal 1 for any input
        rng = np.random.default_rng(42)
        x = np.sort(rng.uniform(0.0, 1.0, 30))
        x = np.concatenate([[0.0], x, [1.0]])
        y = np.sin(np.pi * x / 2)  # arbitrary smooth shape
        r = fit_mode_shape(x, y)
        assert r.c2 + r.c3 + r.c4 + r.c5 + r.c6 == pytest.approx(1.0, abs=1e-12)

    def test_rms_zero_for_exact_poly(self):
        x = self._make_x(31)
        y = 0.4 * x**2 + 0.1 * x**3 + 0.2 * x**4 + 0.1 * x**5 + 0.2 * x**6
        r = fit_mode_shape(x, y)
        assert r.rms_residual == pytest.approx(0.0, abs=1e-12)

    def test_tip_slope_x2(self):
        # dφ/dx|_{x=1} for x² is 2
        x = self._make_x()
        r = fit_mode_shape(x, x**2)
        assert r.tip_slope == pytest.approx(2.0, abs=1e-10)

    def test_coefficients_array(self):
        x = self._make_x()
        r = fit_mode_shape(x, x**3)
        coeffs = r.coefficients()
        assert coeffs.shape == (5,)
        assert coeffs.sum() == pytest.approx(1.0, abs=1e-12)

    def test_evaluate_matches_direct(self):
        x = self._make_x()
        y = 0.5 * x**2 + 0.5 * x**4
        r = fit_mode_shape(x, y)
        x_test = np.linspace(0, 1, 5)
        direct = r.c2*x_test**2 + r.c3*x_test**3 + r.c4*x_test**4 + r.c5*x_test**5 + r.c6*x_test**6
        assert r.evaluate(x_test) == pytest.approx(direct, abs=1e-14)

    def test_normalises_tip(self):
        # Supply un-normalised data (tip value != 1) — should still work
        x = self._make_x()
        y = 3.7 * x**2   # tip value is 3.7
        r = fit_mode_shape(x, y)
        # After internal normalisation the result should equal pure x²
        assert r.c2 == pytest.approx(1.0, abs=1e-10)
        assert r.c3 + r.c4 + r.c5 + r.c6 == pytest.approx(0.0, abs=1e-10)

    def test_zero_tip_raises(self):
        x = np.linspace(0, 1, 11)
        y = np.zeros_like(x)
        with pytest.raises(ValueError, match="zero"):
            fit_mode_shape(x, y)


@pytest.mark.integration
class TestPolyFitBladeModes:
    """Smoke-test: fit the first 3 modes of CertTest01 blade."""

    @pytest.fixture(autouse=True)
    def compute(self):
        blade = RotatingBlade(CERT_DIR / "Test01_nonunif_blade.bmi")
        result = blade.run(n_modes=6)
        self.shapes = result.shapes

    def test_flap1_constraint(self):
        s = self.shapes[0]
        r = fit_mode_shape(s.span_loc, s.flap_disp)
        assert r.c2 + r.c3 + r.c4 + r.c5 + r.c6 == pytest.approx(1.0, abs=1e-12)

    def test_flap1_rms_low(self):
        s = self.shapes[0]
        r = fit_mode_shape(s.span_loc, s.flap_disp)
        assert r.rms_residual < 0.05   # < 5% RMS for a smooth mode shape

    def test_edge1_constraint(self):
        # First edge mode: largest lag displacement component
        # Mode ordering: flap1, edge1, flap2, edge2... — find first with significant lag
        for s in self.shapes[:4]:
            if abs(s.lag_disp[-1]) > abs(s.flap_disp[-1]):
                r = fit_mode_shape(s.span_loc, s.lag_disp)
                assert r.c2 + r.c3 + r.c4 + r.c5 + r.c6 == pytest.approx(1.0, abs=1e-12)
                return
        pytest.skip("No predominantly-edge mode found in first 4 modes")

    def test_all_modes_constraint(self):
        for s in self.shapes:
            disp = s.flap_disp
            if abs(disp[-1]) < 1e-6:
                disp = s.lag_disp
            r = fit_mode_shape(s.span_loc, disp)
            assert r.c2 + r.c3 + r.c4 + r.c5 + r.c6 == pytest.approx(1.0, abs=1e-12)


@pytest.mark.integration
class TestPolyFitTowerModes:
    """Smoke-test: fit the first 4 modes of CertTest03 tower."""

    @pytest.fixture(autouse=True)
    def compute(self):
        tower = Tower(CERT_DIR / "Test03_tower.bmi")
        result = tower.run(n_modes=4)
        self.shapes = result.shapes

    def test_fa1_constraint(self):
        s = self.shapes[0]
        r = fit_mode_shape(s.span_loc, s.flap_disp)
        assert r.c2 + r.c3 + r.c4 + r.c5 + r.c6 == pytest.approx(1.0, abs=1e-12)

    def test_fa1_rms_low(self):
        s = self.shapes[0]
        r = fit_mode_shape(s.span_loc, s.flap_disp)
        assert r.rms_residual < 0.05


class TestTowerModeClassification:
    """Direct tests for explicit tower-mode family classification."""

    def _shape(
        self,
        freq_hz: float,
        flap_disp: np.ndarray,
        lag_disp: np.ndarray,
        flap_slope: np.ndarray | None = None,
        lag_slope: np.ndarray | None = None,
    ) -> NodeModeShape:
        span = np.linspace(0.0, 1.0, len(flap_disp))
        return NodeModeShape(
            mode_number=1,
            freq_hz=freq_hz,
            span_loc=span,
            flap_disp=np.asarray(flap_disp, dtype=float),
            flap_slope=(
                np.zeros_like(span)
                if flap_slope is None
                else np.asarray(flap_slope, dtype=float)
            ),
            lag_disp=np.asarray(lag_disp, dtype=float),
            lag_slope=(
                np.zeros_like(span)
                if lag_slope is None
                else np.asarray(lag_slope, dtype=float)
            ),
            twist=np.zeros_like(span),
        )

    def _candidate(
        self,
        freq_hz: float,
        *,
        is_fa: bool,
        fa_rms: float,
        ss_rms: float,
        rms_residual: float,
    ) -> _TowerModeCandidate:
        shape = self._shape(
            freq_hz=freq_hz,
            flap_disp=[0.0, 0.2, 1.0] if is_fa else [0.0, 0.01, 0.02],
            lag_disp=[0.0, 0.01, 0.02] if is_fa else [0.0, 0.2, 1.0],
        )
        fit_disp = np.array([0.0, 0.2, 1.0], dtype=float)
        fit = PolyFitResult(
            c2=1.0,
            c3=0.0,
            c4=0.0,
            c5=0.0,
            c6=0.0,
            rms_residual=rms_residual,
            tip_slope=2.0,
        )
        return _TowerModeCandidate(
            shape=shape,
            fa_disp=fit_disp if is_fa else np.array([0.0, 0.01, 0.02], dtype=float),
            ss_disp=np.array([0.0, 0.01, 0.02], dtype=float) if is_fa else fit_disp,
            fa_rms=fa_rms,
            ss_rms=ss_rms,
            fit_disp=fit_disp,
            fit=fit,
            is_fa=is_fa,
        )

    def test_candidate_removes_root_rigid_motion_before_fitting(self):
        span = np.linspace(0.0, 1.0, 6)
        flap = 0.2 + 0.1 * span + span**2
        shape = self._shape(
            freq_hz=0.3,
            flap_disp=flap,
            lag_disp=np.zeros_like(span),
            flap_slope=np.full_like(span, 0.1),
        )

        candidate = _tower_candidate(shape)

        assert candidate.is_fa is True
        assert candidate.fit_disp[0] == pytest.approx(0.0)
        assert candidate.fit_disp[-1] == pytest.approx(1.0)

    def test_family_candidates_sort_by_frequency(self):
        c1 = _tower_candidate(self._shape(0.5, [0, 0.1, 0.4], [0, 0, 0]))
        c2 = _tower_candidate(self._shape(0.2, [0, 0.2, 1.0], [0, 0, 0]))
        family = _tower_family_candidates([c1, c2], is_fa=True)

        assert [cand.shape.freq_hz for cand in family] == pytest.approx([0.2, 0.5])

    def test_score_marks_good_fit_explicitly(self):
        good = self._candidate(0.2, is_fa=True, fa_rms=0.8, ss_rms=0.02, rms_residual=0.03)
        poor = self._candidate(0.4, is_fa=True, fa_rms=0.9, ss_rms=0.03, rms_residual=0.14)
        family = _tower_family_candidates([poor, good], is_fa=True)

        scores = _score_tower_family(
            family,
            is_fa=True,
            config=_TowerFamilySelectionConfig(good_fit_rms=0.09),
        )

        assert scores[0].family_rank == 1
        assert scores[0].fit_is_good is True
        assert scores[1].family_rank == 2
        assert scores[1].fit_is_good is False

    def test_select_family_prefers_first_good_higher_mode(self):
        first = self._candidate(0.2, is_fa=True, fa_rms=0.8, ss_rms=0.02, rms_residual=0.03)
        bad_second = self._candidate(0.4, is_fa=True, fa_rms=0.9, ss_rms=0.03, rms_residual=0.13)
        good_third = self._candidate(0.6, is_fa=True, fa_rms=0.7, ss_rms=0.02, rms_residual=0.04)

        fa1, fa2 = _select_tower_family(
            [bad_second, good_third, first],
            is_fa=True,
            config=_TowerFamilySelectionConfig(good_fit_rms=0.09),
        )

        assert fa1.shape.freq_hz == pytest.approx(0.2)
        assert fa2.shape.freq_hz == pytest.approx(0.6)

    def test_select_family_falls_back_to_lowest_residual_when_needed(self):
        first = self._candidate(0.2, is_fa=True, fa_rms=0.8, ss_rms=0.02, rms_residual=0.03)
        worse = self._candidate(0.4, is_fa=True, fa_rms=1.0, ss_rms=0.03, rms_residual=0.21)
        better = self._candidate(0.6, is_fa=True, fa_rms=0.9, ss_rms=0.03, rms_residual=0.12)

        fa1, fa2 = _select_tower_family(
            [better, first, worse],
            is_fa=True,
            config=_TowerFamilySelectionConfig(good_fit_rms=0.01),
        )

        assert fa1.shape.freq_hz == pytest.approx(0.2)
        assert fa2.fit.rms_residual <= worse.fit.rms_residual
