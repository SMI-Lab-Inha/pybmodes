"""Extra unit tests for :mod:`pybmodes.elastodyn`.

These complement ``test_elastodyn.py`` by covering classification helpers,
``patch_dat`` formatting / robustness, and the ``as_dict`` round-trip in
isolation from a full FEM solve.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from pybmodes.elastodyn import (
    BladeElastoDynParams,
    TowerElastoDynParams,
    patch_dat,
)
from pybmodes.elastodyn.params import (
    _is_fa_dominated,
    _remove_root_rigid_motion,
    _sorted_modes,
    _tower_candidate,
    _tower_family_candidates,
)
from pybmodes.fem.normalize import NodeModeShape
from pybmodes.fitting import PolyFitResult

# ===========================================================================
# Helpers
# ===========================================================================

def _shape(
    *,
    mode_number: int = 1,
    freq_hz: float = 1.0,
    flap: list[float] | np.ndarray = (0.0, 1.0),
    lag: list[float] | np.ndarray = (0.0, 0.0),
    flap_slope: list[float] | np.ndarray | None = None,
    lag_slope: list[float] | np.ndarray | None = None,
) -> NodeModeShape:
    flap = np.asarray(flap, dtype=float)
    lag = np.asarray(lag, dtype=float)
    span = np.linspace(0.0, 1.0, len(flap))
    return NodeModeShape(
        mode_number=mode_number,
        freq_hz=freq_hz,
        span_loc=span,
        flap_disp=flap,
        flap_slope=(np.zeros_like(span) if flap_slope is None
                    else np.asarray(flap_slope, dtype=float)),
        lag_disp=lag,
        lag_slope=(np.zeros_like(span) if lag_slope is None
                   else np.asarray(lag_slope, dtype=float)),
        twist=np.zeros_like(span),
    )


def _stub_blade_fit(c2: float = 1.0) -> PolyFitResult:
    others = (1.0 - c2) / 4.0
    return PolyFitResult(c2=c2, c3=others, c4=others, c5=others, c6=others,
                         rms_residual=0.01, tip_slope=2.0)


# ===========================================================================
# _is_fa_dominated
# ===========================================================================

class TestIsFaDominated:

    def test_pure_flap_is_fa(self):
        s = _shape(flap=[0.0, 0.0, 1.0], lag=[0.0, 0.0, 0.0])
        assert bool(_is_fa_dominated(s)) is True

    def test_pure_lag_is_not_fa(self):
        s = _shape(flap=[0.0, 0.0, 0.0], lag=[0.0, 0.0, 1.0])
        assert bool(_is_fa_dominated(s)) is False

    def test_tie_breaks_fa(self):
        # When |flap| == |lag|, the comparison uses >=, so it's flagged FA.
        s = _shape(flap=[0.0, 0.0, 0.7], lag=[0.0, 0.0, 0.7])
        assert bool(_is_fa_dominated(s)) is True

    def test_uses_tip_only(self):
        # Mid-span dominance shouldn't override a flap-dominant tip.
        s = _shape(flap=[0.0, 0.05, 1.0], lag=[0.0, 0.95, 0.0])
        assert bool(_is_fa_dominated(s)) is True


# ===========================================================================
# _sorted_modes
# ===========================================================================

class TestSortedModes:

    def test_returns_only_matching_modes(self):
        flap_mode = _shape(mode_number=1, flap=[0, 1.0], lag=[0, 0.0])
        edge_mode = _shape(mode_number=2, flap=[0, 0.0], lag=[0, 1.0])
        flaps = _sorted_modes([flap_mode, edge_mode], fa_dominated=True)
        edges = _sorted_modes([flap_mode, edge_mode], fa_dominated=False)
        assert [s.mode_number for s in flaps] == [1]
        assert [s.mode_number for s in edges] == [2]

    def test_preserves_input_order(self):
        # _sorted_modes filters but does not re-sort by frequency; preserve
        # the order in which the input modes appear.
        a = _shape(mode_number=1, freq_hz=2.0, flap=[0, 1], lag=[0, 0])
        b = _shape(mode_number=2, freq_hz=1.0, flap=[0, 1], lag=[0, 0])
        out = _sorted_modes([a, b], fa_dominated=True)
        assert [s.mode_number for s in out] == [1, 2]


# ===========================================================================
# _remove_root_rigid_motion
# ===========================================================================

class TestRemoveRootRigidMotion:

    def test_no_change_when_root_zero(self):
        x = np.linspace(0.0, 1.0, 5)
        y = x**2
        out = _remove_root_rigid_motion(x, y, np.zeros_like(x))
        np.testing.assert_allclose(out, y)

    def test_subtracts_constant_offset(self):
        x = np.linspace(0.0, 1.0, 5)
        y = 0.3 + x**2     # rigid translation
        out = _remove_root_rigid_motion(x, y, np.zeros_like(x))
        assert out[0] == pytest.approx(0.0)
        np.testing.assert_allclose(out, x**2, atol=1e-12)

    def test_subtracts_linear_slope(self):
        x = np.linspace(0.0, 1.0, 6)
        slope = np.full_like(x, 0.4)
        y = 0.4 * x + x**2       # slope*x at the root + bending
        out = _remove_root_rigid_motion(x, y, slope)
        assert out[0] == pytest.approx(0.0)
        np.testing.assert_allclose(out, x**2, atol=1e-12)


# ===========================================================================
# _tower_candidate / _tower_family_candidates
# ===========================================================================

class TestTowerCandidate:

    def test_picks_dominant_direction(self):
        # Flap dominates over lag -> candidate.is_fa should be True
        s = _shape(flap=[0.0, 0.5, 1.0], lag=[0.0, 0.05, 0.1])
        cand = _tower_candidate(s)
        assert cand.is_fa is True

    def test_picks_lag_when_dominant(self):
        s = _shape(flap=[0.0, 0.05, 0.1], lag=[0.0, 0.5, 1.0])
        cand = _tower_candidate(s)
        assert cand.is_fa is False

    def test_fit_normalised_to_tip(self):
        s = _shape(flap=[0.0, 0.5, 1.0], lag=[0.0, 0.0, 0.0])
        cand = _tower_candidate(s)
        # Constraint: c2+c3+c4+c5+c6 = 1 (normalised to tip)
        assert (cand.fit.c2 + cand.fit.c3 + cand.fit.c4
                + cand.fit.c5 + cand.fit.c6) == pytest.approx(1.0, abs=1e-12)


class TestTowerFamilyCandidates:

    def test_filters_by_direction_and_sorts_ascending(self):
        s1 = _shape(mode_number=1, freq_hz=0.5, flap=[0, 1], lag=[0, 0])
        s2 = _shape(mode_number=2, freq_hz=0.2, flap=[0, 1], lag=[0, 0])
        s3 = _shape(mode_number=3, freq_hz=0.3, flap=[0, 0], lag=[0, 1])
        cands = [_tower_candidate(s) for s in (s1, s2, s3)]
        fa = _tower_family_candidates(cands, is_fa=True)
        ss = _tower_family_candidates(cands, is_fa=False)
        assert [c.shape.mode_number for c in fa] == [2, 1]   # ascending freq
        assert [c.shape.mode_number for c in ss] == [3]


# ===========================================================================
# BladeElastoDynParams.as_dict round trip
# ===========================================================================

class TestBladeAsDict:

    def test_keys_and_count(self):
        p = BladeElastoDynParams(
            BldFl1Sh=_stub_blade_fit(),
            BldFl2Sh=_stub_blade_fit(),
            BldEdgSh=_stub_blade_fit(),
        )
        d = p.as_dict()
        assert len(d) == 15
        for ed_name in ("BldFl1Sh", "BldFl2Sh", "BldEdgSh"):
            for k in range(2, 7):
                assert f"{ed_name}({k})" in d

    def test_values_are_floats(self):
        p = BladeElastoDynParams(
            BldFl1Sh=_stub_blade_fit(),
            BldFl2Sh=_stub_blade_fit(),
            BldEdgSh=_stub_blade_fit(),
        )
        for v in p.as_dict().values():
            assert isinstance(v, float)


class TestTowerAsDict:

    def test_keys_and_count(self):
        fit = _stub_blade_fit()
        p = TowerElastoDynParams(
            TwFAM1Sh=fit, TwFAM2Sh=fit, TwSSM1Sh=fit, TwSSM2Sh=fit,
        )
        d = p.as_dict()
        assert len(d) == 20
        for ed_name in ("TwFAM1Sh", "TwFAM2Sh", "TwSSM1Sh", "TwSSM2Sh"):
            for k in range(2, 7):
                assert f"{ed_name}({k})" in d


# ===========================================================================
# patch_dat formatting and robustness
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


class TestPatchDatFormatting:

    @pytest.fixture
    def patched_text(self, tmp_path):
        fit = PolyFitResult(c2=0.4, c3=0.2, c4=0.15, c5=0.15, c6=0.1,
                             rms_residual=0.01, tip_slope=2.5)
        params = BladeElastoDynParams(
            BldFl1Sh=fit, BldFl2Sh=fit, BldEdgSh=fit,
        )
        f = tmp_path / "blade.dat"
        f.write_text(_BLADE_TEMPLATE, encoding='utf-8')
        patch_dat(f, params)
        return f.read_text(encoding='utf-8')

    def test_value_in_scientific_notation(self, patched_text):
        # Writer formats values as ' 0.4000000E+00' (space + 7-digit mantissa).
        # All BldFl1Sh(2) values should match this pattern.
        for line in patched_text.splitlines():
            if "BldFl1Sh(2)" in line:
                tokens = line.split()
                # The first token is the patched value. It must parse as float.
                v = float(tokens[0])
                assert v == pytest.approx(0.4)
                # And must look like scientific notation.
                assert re.match(r"-?\d\.\d{7}E[+-]\d{2}", tokens[0])
                return
        pytest.fail("BldFl1Sh(2) line not found")

    def test_missing_param_raises_keyerror(self, tmp_path):
        # File missing some of the required params -> KeyError mentions which.
        bad = tmp_path / "missing.dat"
        # Strip the BldEdgSh block entirely.
        truncated = "\n".join(
            line for line in _BLADE_TEMPLATE.splitlines()
            if "BldEdgSh" not in line
        ) + "\n"
        bad.write_text(truncated, encoding='utf-8')
        fit = _stub_blade_fit()
        params = BladeElastoDynParams(
            BldFl1Sh=fit, BldFl2Sh=fit, BldEdgSh=fit,
        )
        with pytest.raises(KeyError, match="BldEdgSh"):
            patch_dat(bad, params)

    def test_repeated_patch_preserves_values(self, tmp_path):
        # Running patch_dat twice should yield the same numerical coefficients,
        # even if leading whitespace shifts (the writer is not byte-idempotent).
        fit = _stub_blade_fit(c2=0.3)
        params = BladeElastoDynParams(
            BldFl1Sh=fit, BldFl2Sh=fit, BldEdgSh=fit,
        )
        f = tmp_path / "blade.dat"
        f.write_text(_BLADE_TEMPLATE, encoding='utf-8')
        patch_dat(f, params)
        patch_dat(f, params)
        # Final values must still match the requested coefficient.
        d = params.as_dict()
        for line in f.read_text(encoding='utf-8').splitlines():
            tokens = line.split()
            if len(tokens) >= 2 and tokens[1] in d:
                assert float(tokens[0]) == pytest.approx(d[tokens[1]], rel=1e-6)

    def test_preserves_unrelated_lines(self, tmp_path):
        # Lines that don't match a parameter name must be left untouched.
        fit = _stub_blade_fit()
        params = BladeElastoDynParams(
            BldFl1Sh=fit, BldFl2Sh=fit, BldEdgSh=fit,
        )
        f = tmp_path / "blade.dat"
        # Add a comment line at the top — must survive verbatim.
        marker = "# DO NOT EDIT: managed by tests\n"
        f.write_text(marker + _BLADE_TEMPLATE, encoding='utf-8')
        patch_dat(f, params)
        text = f.read_text(encoding='utf-8')
        assert text.startswith(marker)
