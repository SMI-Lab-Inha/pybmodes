"""Tests for the degenerate FA/SS eigenpair resolver in elastodyn.params.

The resolver detects consecutive modes whose relative frequency gap is below
``_DEGENERATE_FREQ_RTOL`` (1e-4) and rotates them inside their shared 2D
eigenspace so the first comes out FA-pure and the second SS-pure. This
removes a class of classifier ambiguity on perfectly symmetric structures
where the eigensolver's basis choice within the degenerate subspace is
arbitrary.
"""

from __future__ import annotations

import math
import pathlib
import warnings

import numpy as np
import pytest

from pybmodes.elastodyn.params import (
    _DEGENERATE_FREQ_RTOL,
    _is_degenerate_pair,
    _resolve_degenerate_pair,
    _rotate_degenerate_pairs,
    _rotate_shape_pair,
    _shape_participation,
    compute_tower_params_report,
)
from pybmodes.fitting.poly_fit import fit_mode_shape
from pybmodes.models import Tower

from ._synthetic_bmi import write_bmi, write_uniform_sec_props


# ---------------------------------------------------------------------------
# 1. Symmetric tower → degenerate pair → resolver returns clean FA/SS
# ---------------------------------------------------------------------------

def _build_symmetric_tower(tmp_path: pathlib.Path) -> pathlib.Path:
    """Tower with EI_FA == EI_SS exactly and no tip mass: degenerate by construction."""
    bmi_path = write_bmi(
        tmp_path / "sym.bmi",
        beam_type=2, radius=80.0, hub_rad=0.0,
        tow_support=0, sec_props_file="sym.dat",
    )
    write_uniform_sec_props(
        tmp_path / "sym.dat",
        n_secs=8, mass_den=5000.0,
        flp_stff=5.0e10, edge_stff=5.0e10,
    )
    return bmi_path


def test_degenerate_symmetric_tower(tmp_path):
    """Resolver undoes any rotation in the degenerate subspace and pins the
    polynomial coefficients regardless of the input basis."""
    bmi_path = _build_symmetric_tower(tmp_path)
    modal = Tower(bmi_path).run(n_modes=4)

    # The eigensolver pair {mode 1, mode 2} should be detected as degenerate.
    assert _is_degenerate_pair(modal.shapes[0], modal.shapes[1]), (
        f"modes 1, 2 freqs = {modal.shapes[0].freq_hz}, {modal.shapes[1].freq_hz} "
        f"— expected within {_DEGENERATE_FREQ_RTOL} relative gap"
    )

    # Fixed reference: resolve the pair as the eigensolver returned it.
    fa_ref, ss_ref, _ = _resolve_degenerate_pair(modal.shapes[0], modal.shapes[1])
    fa_fit_ref = fit_mode_shape(fa_ref.span_loc, fa_ref.flap_disp)
    ss_fit_ref = fit_mode_shape(ss_ref.span_loc, ss_ref.lag_disp)

    # Rotated shapes should be FA-pure / SS-pure (no torsion in this case).
    p_fa, _ = _shape_participation(fa_ref)
    _, p_ss = _shape_participation(ss_ref)
    assert p_fa > 0.99, f"FA-aligned p_FA = {p_fa:.4f}"
    assert p_ss > 0.99, f"SS-aligned p_SS = {p_ss:.4f}"

    # Polynomial fits should be near-exact for a clean Euler-Bernoulli mode.
    assert fa_fit_ref.rms_residual < 0.001
    assert ss_fit_ref.rms_residual < 0.001

    # Stability: rotate the same pair by an arbitrary angle to simulate a
    # different eigensolver basis choice, run the resolver, and verify the
    # polynomial coefficients land in the same place.
    for theta_artif_deg in (37.0, 91.0, -19.0):
        theta = math.radians(theta_artif_deg)
        rot_a, rot_b = _rotate_shape_pair(modal.shapes[0], modal.shapes[1], theta)
        fa, ss, _ = _resolve_degenerate_pair(rot_a, rot_b)

        # Same purity check after the artificial rotation pre-stage.
        p_fa_, _ = _shape_participation(fa)
        _, p_ss_ = _shape_participation(ss)
        assert p_fa_ > 0.99, (
            f"After {theta_artif_deg}° pre-rotation: p_FA = {p_fa_:.4f}"
        )
        assert p_ss_ > 0.99, (
            f"After {theta_artif_deg}° pre-rotation: p_SS = {p_ss_:.4f}"
        )

        # Coefficients invariant to the input basis (modulo overall sign,
        # which fit_mode_shape already absorbs by tip-normalisation).
        fa_fit = fit_mode_shape(fa.span_loc, fa.flap_disp)
        np.testing.assert_allclose(
            fa_fit.coefficients(), fa_fit_ref.coefficients(),
            atol=1e-6,
            err_msg=f"FA coefficients drifted under {theta_artif_deg}° pre-rotation",
        )


# ---------------------------------------------------------------------------
# 2. Asymmetric tower → resolver does nothing
# ---------------------------------------------------------------------------

def test_nondegenerate_pair_unchanged(tmp_path):
    """Asymmetric stiffness ⇒ frequency gap above threshold ⇒ no rotation."""
    bmi_path = write_bmi(
        tmp_path / "asym.bmi",
        beam_type=2, radius=80.0, hub_rad=0.0,
        tow_support=0, sec_props_file="asym.dat",
    )
    write_uniform_sec_props(
        tmp_path / "asym.dat",
        n_secs=8, mass_den=5000.0,
        flp_stff=5.0e10, edge_stff=5.0e11,  # 10× ratio
    )
    modal = Tower(bmi_path).run(n_modes=4)

    gap = (
        abs(modal.shapes[1].freq_hz - modal.shapes[0].freq_hz)
        / modal.shapes[0].freq_hz
    )
    assert gap > _DEGENERATE_FREQ_RTOL, (
        f"expected non-degenerate, got relative gap = {gap:.2e}"
    )

    rotated = _rotate_degenerate_pairs(modal.shapes)

    # Returned list must contain the original NodeModeShape *instances*
    # (no rotation applied, no new objects created).
    assert len(rotated) == len(modal.shapes)
    for orig, new in zip(modal.shapes, rotated):
        assert new is orig, (
            f"mode {orig.mode_number}: expected the same instance back, "
            f"got a fresh copy (rotation triggered when it shouldn't have)"
        )


# ---------------------------------------------------------------------------
# 3. Nearly-degenerate pair — gap just above threshold, classifier still works
# ---------------------------------------------------------------------------

def test_nearly_degenerate_pair(tmp_path):
    """A small but above-threshold stiffness asymmetry: resolver does NOT
    fire (correctly), but the participation classifier still picks FA / SS
    cleanly because the eigensolver returns nearly-pure modes once the
    degeneracy has been lifted."""
    bmi_path = write_bmi(
        tmp_path / "near.bmi",
        beam_type=2, radius=80.0, hub_rad=0.0,
        tow_support=0, sec_props_file="near.dat",
    )
    EI_FA = 5.0e10
    write_uniform_sec_props(
        tmp_path / "near.dat",
        n_secs=8, mass_den=5000.0,
        flp_stff=EI_FA, edge_stff=EI_FA * 1.0005,  # 0.05 % EI diff
    )
    modal = Tower(bmi_path).run(n_modes=4)

    gap = (
        abs(modal.shapes[1].freq_hz - modal.shapes[0].freq_hz)
        / modal.shapes[0].freq_hz
    )
    assert _DEGENERATE_FREQ_RTOL < gap, (
        f"frequency gap {gap:.2e} should be > threshold {_DEGENERATE_FREQ_RTOL}; "
        f"either tighten the EI ratio or relax the threshold"
    )

    # Resolver should not fire.
    rotated = _rotate_degenerate_pairs(modal.shapes)
    for orig, new in zip(modal.shapes, rotated):
        assert new is orig

    # Even without rotation the classifier should still hand back sensible
    # FA / SS polynomial fits — the asymmetry is enough for the eigensolver
    # to emit modes that are already FA-dominant and SS-dominant.
    params, report = compute_tower_params_report(modal)
    assert params.TwFAM1Sh.rms_residual < 0.01
    assert params.TwSSM1Sh.rms_residual < 0.01
    # The two selected first-family modes must be different mode numbers
    # (the classifier mustn't have collapsed them onto the same eigenmode).
    assert report.selected_fa_modes[0] != report.selected_ss_modes[0]


# ---------------------------------------------------------------------------
# 4. IEA-3.4 deck — live regression test for the original failure case
# ---------------------------------------------------------------------------

_IEA34_MAIN = (
    pathlib.Path(__file__).resolve().parents[1]
    / "docs/OpenFAST_files/IEA-3.4-130-RWT/openfast/IEA-3.4-130-RWT_ElastoDyn.dat"
)


@pytest.mark.skipif(
    not _IEA34_MAIN.is_file(),
    reason=f"IEA-3.4 ElastoDyn deck not present at {_IEA34_MAIN}",
)
def test_iea34_no_degeneracy_warning():
    """The IEA-3.4 modes 1-2 should resolve to clean FA/SS shapes with no
    warning emitted.

    The deck's tower is symmetric in stiffness; with a sufficiently large
    eigenvalue request scipy/LAPACK returns the degenerate pair at
    *identical* frequencies but in an arbitrarily-rotated basis (≈ 52/48
    FA/SS mix). We use ``n_modes=10`` here to match what the case-study
    script in ``cases/iea3mw_land/run.py`` exercises and to put the
    eigensolver on the full-solve path; smaller subset requests produce
    slightly-different numerical answers where the eigensolver lifts the
    degeneracy artificially and the modes come back already-separated.
    """
    tower = Tower.from_elastodyn(_IEA34_MAIN)
    modal = tower.run(n_modes=10)

    # Sanity: this deck is expected to produce a degenerate first pair
    # (gap exactly zero on a full solve — the structural model is symmetric
    # in stiffness and the rigid-RNA c.m. offset is small enough not to
    # lift the degeneracy at the FEM precision we operate at).
    assert _is_degenerate_pair(modal.shapes[0], modal.shapes[1]), (
        f"expected a degenerate first pair from the full solve; got "
        f"freqs {modal.shapes[0].freq_hz:.6f}, {modal.shapes[1].freq_hz:.6f}"
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        rotated = _rotate_degenerate_pairs(modal.shapes)
    runtime_warnings = [w for w in captured if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings, (
        "unexpected RuntimeWarning(s) from the resolver:\n  "
        + "\n  ".join(str(w.message) for w in runtime_warnings)
    )

    # First rotated shape should be FA-pure (was ~ 0.52 pre-rotation).
    p_fa, _ = _shape_participation(rotated[0])
    assert p_fa > 0.99, (
        f"After rotation, FA-aligned mode has p_FA = {p_fa:.4f} "
        f"(expected > 0.99). Pre-rotation values were ~ 0.522 / 0.478."
    )
    # And the second SS-pure.
    _, p_ss = _shape_participation(rotated[1])
    assert p_ss > 0.99, f"SS-aligned mode has p_SS = {p_ss:.4f}"
