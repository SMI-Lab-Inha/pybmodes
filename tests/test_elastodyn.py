"""Tests for elastodyn/params.py and elastodyn/writer.py.

Smoke tests that validate the public API surface against synthetic mode
shapes constructed directly in-test, without any bundled reference data.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.elastodyn import (
    BladeElastoDynParams,
    TowerElastoDynParams,
    TowerSelectionReport,
    compute_blade_params,
    compute_tower_params,
    compute_tower_params_report,
    patch_dat,
)
from pybmodes.fem.normalize import NodeModeShape
from pybmodes.fitting.poly_fit import PolyFitResult
from pybmodes.models.result import ModalResult

# ===========================================================================
# Synthetic mode-shape helpers
# ===========================================================================

def _shape(*, mode_number, freq_hz, flap, lag, n=11):
    span = np.linspace(0.0, 1.0, n)
    return NodeModeShape(
        mode_number=mode_number,
        freq_hz=freq_hz,
        span_loc=span,
        flap_disp=np.asarray(flap, dtype=float),
        flap_slope=np.zeros_like(span),
        lag_disp=np.asarray(lag, dtype=float),
        lag_slope=np.zeros_like(span),
        twist=np.zeros_like(span),
    )


def _polynomial_shape(coeffs):
    """Build a span-loc-keyed displacement that lives exactly in the model space."""
    span = np.linspace(0.0, 1.0, 11)
    c2, c3, c4, c5, c6 = coeffs
    return c2 * span**2 + c3 * span**3 + c4 * span**4 + c5 * span**5 + c6 * span**6


# ===========================================================================
# compute_blade_params
# ===========================================================================

class TestBladeParams:

    @pytest.fixture
    def modal(self):
        # Two flap-dominated and one edge-dominated mode, smooth polynomials.
        flap1 = _polynomial_shape((1.0, 0.0, 0.0, 0.0, 0.0))     # pure x²
        flap2 = _polynomial_shape((-1.0, 0.0, 2.0, 0.0, 0.0))    # x² → x⁴ blend
        edge1 = _polynomial_shape((0.5, 0.0, 0.0, 0.0, 0.5))     # x² + x⁶
        zeros = np.zeros(11)
        shapes = [
            _shape(mode_number=1, freq_hz=0.5, flap=flap1, lag=zeros),
            _shape(mode_number=2, freq_hz=2.0, flap=flap2, lag=zeros),
            _shape(mode_number=3, freq_hz=3.5, flap=zeros, lag=edge1),
        ]
        return ModalResult(frequencies=np.array([0.5, 2.0, 3.5]), shapes=shapes)

    def test_returns_blade_params(self, modal):
        params = compute_blade_params(modal)
        assert isinstance(params, BladeElastoDynParams)

    def test_constraints(self, modal):
        params = compute_blade_params(modal)
        for fit in (params.BldFl1Sh, params.BldFl2Sh, params.BldEdgSh):
            total = fit.c2 + fit.c3 + fit.c4 + fit.c5 + fit.c6
            assert total == pytest.approx(1.0, abs=1e-12)

    def test_as_dict_keys(self, modal):
        d = compute_blade_params(modal).as_dict()
        assert set(d.keys()) == {
            f"{n}({k})"
            for n in ("BldFl1Sh", "BldFl2Sh", "BldEdgSh")
            for k in range(2, 7)
        }

    def test_insufficient_modes_raises(self):
        # Only one flap mode → cannot form BldFl2Sh
        flap1 = _polynomial_shape((1.0, 0.0, 0.0, 0.0, 0.0))
        edge1 = _polynomial_shape((0.5, 0.0, 0.0, 0.0, 0.5))
        zeros = np.zeros(11)
        modal = ModalResult(
            frequencies=np.array([0.5, 3.5]),
            shapes=[
                _shape(mode_number=1, freq_hz=0.5, flap=flap1, lag=zeros),
                _shape(mode_number=2, freq_hz=3.5, flap=zeros, lag=edge1),
            ],
        )
        with pytest.raises(ValueError, match="flap modes"):
            compute_blade_params(modal)


# ===========================================================================
# compute_tower_params / compute_tower_params_report
# ===========================================================================

class TestTowerParams:

    @pytest.fixture
    def modal(self):
        # Two FA-dominated + two SS-dominated modes, smooth polynomials.
        fa1 = _polynomial_shape((1.0, 0.0, 0.0, 0.0, 0.0))
        fa2 = _polynomial_shape((-1.0, 0.0, 2.0, 0.0, 0.0))
        ss1 = _polynomial_shape((0.5, 0.0, 0.0, 0.0, 0.5))
        ss2 = _polynomial_shape((0.0, 1.0, 0.0, 0.0, 0.0))
        zeros = np.zeros(11)
        shapes = [
            _shape(mode_number=1, freq_hz=0.4, flap=fa1, lag=zeros),
            _shape(mode_number=2, freq_hz=0.5, flap=zeros, lag=ss1),
            _shape(mode_number=3, freq_hz=2.0, flap=fa2, lag=zeros),
            _shape(mode_number=4, freq_hz=2.2, flap=zeros, lag=ss2),
        ]
        return ModalResult(frequencies=np.array([0.4, 0.5, 2.0, 2.2]), shapes=shapes)

    def test_returns_tower_params(self, modal):
        params = compute_tower_params(modal)
        assert isinstance(params, TowerElastoDynParams)

    def test_report_structure(self, modal):
        params, report = compute_tower_params_report(modal)
        assert isinstance(params, TowerElastoDynParams)
        assert isinstance(report, TowerSelectionReport)
        assert sum(m.selected for m in report.fa_family) == 2
        assert sum(m.selected for m in report.ss_family) == 2

    def test_constraints(self, modal):
        params = compute_tower_params(modal)
        for fit in (params.TwFAM1Sh, params.TwFAM2Sh,
                    params.TwSSM1Sh, params.TwSSM2Sh):
            total = fit.c2 + fit.c3 + fit.c4 + fit.c5 + fit.c6
            assert total == pytest.approx(1.0, abs=1e-12)

    def test_as_dict_keys(self, modal):
        d = compute_tower_params(modal).as_dict()
        assert set(d.keys()) == {
            f"{n}({k})"
            for n in ("TwFAM1Sh", "TwFAM2Sh", "TwSSM1Sh", "TwSSM2Sh")
            for k in range(2, 7)
        }


# ===========================================================================
# patch_dat round-trip on synthetic ElastoDyn templates
# ===========================================================================

_BLADE_TEMPLATE = """\
------- BLADE MODE SHAPES ------------------------------------------
          1   BldFl1Sh(2)   - Blade 1 flap mode 1, coeff of x^2
          0   BldFl1Sh(3)   - Blade 1 flap mode 1, coeff of x^3
          0   BldFl1Sh(4)   - Blade 1 flap mode 1, coeff of x^4
          0   BldFl1Sh(5)   - Blade 1 flap mode 1, coeff of x^5
          0   BldFl1Sh(6)   - Blade 1 flap mode 1, coeff of x^6
          1   BldFl2Sh(2)   - Blade 1 flap mode 2, coeff of x^2
          0   BldFl2Sh(3)   - Blade 1 flap mode 2, coeff of x^3
          0   BldFl2Sh(4)   - Blade 1 flap mode 2, coeff of x^4
          0   BldFl2Sh(5)   - Blade 1 flap mode 2, coeff of x^5
          0   BldFl2Sh(6)   - Blade 1 flap mode 2, coeff of x^6
          1   BldEdgSh(2)   - Blade 1 edge mode 1, coeff of x^2
          0   BldEdgSh(3)   - Blade 1 edge mode 1, coeff of x^3
          0   BldEdgSh(4)   - Blade 1 edge mode 1, coeff of x^4
          0   BldEdgSh(5)   - Blade 1 edge mode 1, coeff of x^5
          0   BldEdgSh(6)   - Blade 1 edge mode 1, coeff of x^6
"""


def _stub_blade_params():
    fit = PolyFitResult(c2=0.4, c3=0.2, c4=0.15, c5=0.15, c6=0.1,
                         rms_residual=0.01, tip_slope=2.5, cond_number=0.0)
    return BladeElastoDynParams(BldFl1Sh=fit, BldFl2Sh=fit, BldEdgSh=fit)


class TestPatchDat:

    def test_blade_patch_roundtrip(self, tmp_path):
        dat = tmp_path / "ElastoDyn.dat"
        dat.write_text(_BLADE_TEMPLATE, encoding="utf-8")
        params = _stub_blade_params()
        patch_dat(dat, params)
        text = dat.read_text(encoding="utf-8")
        d = params.as_dict()
        for line in text.splitlines():
            tokens = line.split()
            if len(tokens) >= 2 and tokens[1] in d:
                assert float(tokens[0]) == pytest.approx(d[tokens[1]], rel=1e-6)

    def test_missing_param_raises_keyerror(self, tmp_path):
        dat = tmp_path / "incomplete.dat"
        dat.write_text("nothing useful here\n", encoding="utf-8")
        params = _stub_blade_params()
        with pytest.raises(KeyError, match="BldFl1Sh"):
            patch_dat(dat, params)
