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
    _component_strength,
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

    def test_uses_spanwise_strength_not_tip_only(self):
        # A small lag tip value should not hide a lag-dominated mode when most
        # of the span is moving side-side.
        s = _shape(
            flap=[0.0, 0.05, 0.10, 1.0],
            lag=[0.0, 1.40, 1.20, 0.0],
        )
        assert bool(_is_fa_dominated(s)) is False

    def test_equal_strength_tie_breaks_by_tip(self):
        s = _shape(flap=[0.0, 1.0], lag=[0.0, -1.0])
        assert bool(_is_fa_dominated(s)) is True


class TestComponentStrength:

    def test_integrates_over_span(self):
        span = np.array([0.0, 0.5, 1.0])
        strength = _component_strength(span, np.ones_like(span) * 2.0)
        assert strength == pytest.approx(2.0)

    def test_sorts_span_before_integrating(self):
        span = np.array([1.0, 0.0, 0.5])
        disp = np.array([2.0, 2.0, 2.0])
        assert _component_strength(span, disp) == pytest.approx(2.0)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            _component_strength(np.array([0.0, 1.0]), np.array([1.0]))


class TestClassifierConsistency:
    """Blade and tower classifiers must agree on near-tie inputs."""

    def test_exact_tie_breaks_to_fa_for_both_paths(self):
        # Build a mode whose raw flap and lag displacements match exactly,
        # so spanwise strengths are equal and the tip-tiebreaker fires.
        # Both _is_fa_dominated (blade path) and _tower_candidate (tower path,
        # going through the shared classifier on rigid-body-removed disps)
        # must land on the same answer.
        from pybmodes.elastodyn.params import _classify_fa_dominant
        s = _shape(flap=[0.0, 0.5, 1.0], lag=[0.0, 0.5, 1.0])
        # Direct call to the shared rule
        assert bool(_classify_fa_dominant(s.span_loc, s.flap_disp, s.lag_disp)) is True
        # Blade path
        assert bool(_is_fa_dominated(s)) is True
        # Tower path (no rigid-body component here, so disps pass through)
        cand = _tower_candidate(s)
        assert bool(cand.is_fa) is True

    def test_near_tie_within_isclose_uses_tip(self):
        # Construct two displacement curves whose spanwise RMS strengths are
        # bit-identical (mirror images on uniform span), so the only thing
        # the classifier can decide on is the tip.  The FA tip is smaller
        # than the SS tip, so SS must win — and the test fails if the
        # implementation drops the tip-tiebreak.
        from pybmodes.elastodyn.params import (
            _classify_fa_dominant,
            _component_strength,
        )
        span = np.array([0.0, 0.5, 1.0])
        # Mirror pair: y² is the same multiset, so trapezoidal integration
        # gives equal areas on uniform span and identical spanwise strength.
        fa = np.array([0.5, 1.0, 0.2])
        ss = np.array([0.2, 1.0, 0.5])

        # Sanity-check: the strengths must really be (bit-)equal so the
        # tiebreak is the path that decides the classification.
        fa_strength = _component_strength(span, fa)
        ss_strength = _component_strength(span, ss)
        assert fa_strength == ss_strength
        assert np.isclose(fa_strength, ss_strength)
        assert abs(fa[-1]) < abs(ss[-1])  # tip favours SS

        # SS wins on the tip-tiebreak -> classifier returns False (not FA).
        assert bool(_classify_fa_dominant(span, fa, ss)) is False


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

    def test_sorts_by_ascending_frequency(self):
        # _sorted_modes must order the filtered modes by ascending freq_hz so
        # that compute_blade_params picks the genuine first/second flap modes
        # even when the caller hands shapes in arbitrary order.
        a = _shape(mode_number=1, freq_hz=2.0, flap=[0, 1], lag=[0, 0])
        b = _shape(mode_number=2, freq_hz=1.0, flap=[0, 1], lag=[0, 0])
        out = _sorted_modes([a, b], fa_dominated=True)
        assert [s.mode_number for s in out] == [2, 1]
        assert [s.freq_hz for s in out] == [1.0, 2.0]


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

    def test_uses_spanwise_strength_after_root_motion_removal(self):
        # Both components include rigid root motion.  After removing that affine
        # part, the side-side bending contribution dominates even though the FA
        # tip residual is larger in the raw shape.  The lag bending residual
        # bulges in the middle of the span and decays toward a small (but
        # non-zero) tip value — that's what catches a tip-only classifier.
        span = np.linspace(0.0, 1.0, 5)
        rigid = 10.0 + 5.0 * span
        s = _shape(
            flap=rigid + np.array([0.0, 0.05, 0.08, 0.10, 1.0]),
            lag=rigid + np.array([0.0, 1.4, 1.2, 0.8, 0.1]),
            flap_slope=np.full_like(span, 5.0),
            lag_slope=np.full_like(span, 5.0),
        )
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

    def test_preserves_crlf_line_endings(self, tmp_path):
        # OpenFAST .dat files on Windows ship with CRLF endings.  The writer
        # must not silently demote the patched lines to LF — that would yield
        # a file with mixed endings, which downstream parsers treat as a
        # corruption.
        fit = _stub_blade_fit()
        params = BladeElastoDynParams(
            BldFl1Sh=fit, BldFl2Sh=fit, BldEdgSh=fit,
        )
        crlf_template = _BLADE_TEMPLATE.replace("\n", "\r\n")
        f = tmp_path / "blade_crlf.dat"
        # write_bytes to bypass any newline translation by Python's text mode.
        f.write_bytes(crlf_template.encode("utf-8"))
        patch_dat(f, params)

        raw = f.read_bytes()
        # Every newline must still be \r\n; no bare \n that wasn't preceded by \r.
        assert b"\r\n" in raw
        bare_lf = sum(
            1 for i, b in enumerate(raw)
            if b == 0x0A and (i == 0 or raw[i - 1] != 0x0D)
        )
        assert bare_lf == 0, f"writer introduced {bare_lf} bare LF on a CRLF file"

    def test_preserves_lf_line_endings(self, tmp_path):
        # Symmetric guarantee: an LF-only file stays LF-only.
        fit = _stub_blade_fit()
        params = BladeElastoDynParams(
            BldFl1Sh=fit, BldFl2Sh=fit, BldEdgSh=fit,
        )
        f = tmp_path / "blade_lf.dat"
        f.write_bytes(_BLADE_TEMPLATE.encode("utf-8"))
        patch_dat(f, params)
        raw = f.read_bytes()
        assert b"\r\n" not in raw

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
