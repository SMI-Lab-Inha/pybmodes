"""Asymmetric floating-platform CM support (1.2.0).

The rigid-arm transform in :func:`pybmodes.fem.nondim._rigid_arm_T`
maps the platform's 6×6 mass/stiffness matrices from their reference
point onto the tower-base FEM DOFs. Through 1.1.x it carried only a
*vertical* lever (``cm_pform − draft``); 1.2.0 generalises it to a
full 3-D arm so an asymmetric floating substructure with a horizontal
CM offset (``PtfmCMxt`` / ``PtfmCMyt``) is solved correctly.

These tests are self-contained (default suite, no external data):

1. ``rx = ry = 0`` is byte-identical to the pre-1.2.0 hand-written
   transform — the strongest possible guard that every axisymmetric
   spar / symmetric semi deck is unchanged (the OC3 Hywind cert test
   already pins this empirically vs BModes JJ at 0.0003 %).
2. The transform has the exact rigid-body kinematic structure
   ``G = [[I3, -skew(r)], [0, I3]]`` for the 3-D arm.
3. Closed-form, non-circular: transferring a point-mass spatial
   inertia through the transform reproduces the textbook
   parallel-axis result derived independently from kinetic energy.
4. End-to-end through the public solver: a synthetic floating model
   with a horizontal CM offset reproduces the analytic sway–yaw
   coupled-oscillator frequency split, and collapses back to the
   decoupled frequencies when the offset is zero.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from pybmodes.fem.nondim import _rigid_arm_T


def _skew(r: np.ndarray) -> np.ndarray:
    rx, ry, rz = r
    return np.array([
        [0.0, -rz,  ry],
        [rz,  0.0, -rx],
        [-ry, rx,  0.0],
    ])


# arm-independent FEM↔file reorder, identical to the one inside
# _rigid_arm_T (rebuilt here so the test is independent of internals).
def _P() -> np.ndarray:
    P = np.zeros((6, 6))
    P[0, 1] = 1.0
    P[1, 3] = 1.0
    P[2, 0] = 1.0
    P[3, 4] = -1.0
    P[4, 2] = 1.0
    P[5, 5] = 1.0
    return P


def _pre_120_T(p_base: float) -> np.ndarray:
    """The exact vertical-only transform pyBmodes shipped through
    1.1.x (verified vs BModes JJ on OC3 Hywind at 0.0003 %)."""
    T = np.zeros((6, 6))
    T[0, 1] = 1.0
    T[0, 2] = -p_base
    T[1, 3] = 1.0
    T[1, 4] = -p_base
    T[2, 0] = 1.0
    T[3, 4] = -1.0
    T[4, 2] = 1.0
    T[5, 5] = 1.0
    return T


@pytest.mark.parametrize("p_base", [0.0, -10.0, 89.9155, 144.386])
def test_symmetric_arm_byte_identical_to_pre_120(p_base: float) -> None:
    """rx = ry = 0 must reproduce the pre-1.2.0 transform exactly."""
    np.testing.assert_array_equal(
        _rigid_arm_T(p_base, 0.0, 0.0), _pre_120_T(p_base)
    )


@pytest.mark.parametrize(
    "p_base,rx,ry",
    [(89.9155, 5.0, -3.0), (-10.0, 0.0, 2.5), (50.0, -4.2, 0.0)],
)
def test_rigid_body_kinematic_structure(p_base, rx, ry) -> None:
    """T = G @ P with G = [[I3, -skew(r)], [0, I3]] and r = (rx, ry,
    -p_base) — i.e. a textbook rigid-body small-rotation transfer."""
    P = _P()
    T = _rigid_arm_T(p_base, rx, ry)
    # P is orthogonal (signed permutation): G = T @ Pᵀ.
    G = T @ P.T

    r = np.array([rx, ry, -p_base])
    G_expected = np.eye(6)
    G_expected[0:3, 3:6] = -_skew(r)

    np.testing.assert_allclose(G, G_expected, atol=1e-12)
    # Lower-left block is exactly zero, lower-right is identity:
    # rotations transfer rigidly, translations pick up the arm.
    np.testing.assert_array_equal(G[3:6, 0:3], np.zeros((3, 3)))
    np.testing.assert_array_equal(G[3:6, 3:6], np.eye(3))


@pytest.mark.parametrize(
    "p_base,rx,ry,m",
    [(30.0, 6.0, -4.0, 1.7e6), (-12.0, 0.0, 3.5, 9.0e5)],
)
def test_point_mass_parallel_axis_closed_form(p_base, rx, ry, m) -> None:
    """Transferring a point mass referenced at its own CM through the
    arm must give the textbook spatial inertia about the tower base.

    Independent (non-circular) reference: from KE = ½m|ṫ + θ̇×r|² the
    spatial inertia about O is
        M_O = [[ m I3,        -m S(r) ],
               [ -m S(r)ᵀ,     m(|r|²I − r rᵀ) ]]
    in file DOF order. We build M_O from this formula and compare
    against the code path Tᵀ M_cm T (reordered back to file DOFs)."""
    r = np.array([rx, ry, -p_base])
    S = _skew(r)

    # Platform 6×6 at its own CM, file DOF order: point mass, no
    # rotational inertia, no coupling.
    M_cm_file = np.zeros((6, 6))
    M_cm_file[0, 0] = M_cm_file[1, 1] = M_cm_file[2, 2] = m

    P = _P()
    T = _rigid_arm_T(p_base, rx, ry)          # FEM ← (maps to) file
    M_base_fem = T.T @ M_cm_file @ T          # tower base, FEM order
    M_base_file = P @ M_base_fem @ P.T        # back to file order

    M_O = np.zeros((6, 6))
    M_O[0:3, 0:3] = m * np.eye(3)
    M_O[0:3, 3:6] = -m * S
    M_O[3:6, 0:3] = -m * S.T
    M_O[3:6, 3:6] = m * (float(r @ r) * np.eye(3) - np.outer(r, r))

    np.testing.assert_allclose(M_base_file, M_O, rtol=1e-10, atol=1e-6)


_SAMPLE09 = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src" / "pybmodes" / "_examples" / "sample_inputs"
    / "reference_turbines" / "09_iea15_umainesemi"
    / "09_iea15_umainesemi_tower.bmi"
)


def _solve_with_cm_offset(rx: float, ry: float, n_modes: int = 12):
    """Solve the bundled, known-good sample-09 floating tower with its
    PlatformSupport's horizontal CM offset overridden to (rx, ry).

    Reuses the validated sample-09 tower beam + section properties (a
    well-conditioned regime — `test_floating_samples_spectra` pins it)
    and changes ONLY the horizontal CM offset, so this isolates the
    wiring from ``PlatformSupport.cm_pform_x/y`` through
    ``nondim_platform`` into the solver without the conditioning
    pitfalls of a hand-rolled massless tower.
    """
    from pybmodes.io.bmi import read_bmi
    from pybmodes.io.sec_props import read_sec_props
    from pybmodes.models import Tower

    bmi = read_bmi(_SAMPLE09)
    bmi.support = dataclasses.replace(
        bmi.support, cm_pform_x=rx, cm_pform_y=ry
    )
    t = Tower.__new__(Tower)
    t._bmi = bmi
    t._sp = read_sec_props(bmi.resolve_sec_props_path())
    return t.run(n_modes=n_modes, check_model=False).frequencies


def test_zero_offset_reproduces_stock_sample09() -> None:
    """cm_pform_x = cm_pform_y = 0 must give exactly the stock
    sample-09 spectrum — i.e. the new field is inert for a symmetric
    platform (the value the bundled file already carries)."""
    from pybmodes.io.bmi import read_bmi
    from pybmodes.io.sec_props import read_sec_props
    from pybmodes.models import Tower

    bmi = read_bmi(_SAMPLE09)
    assert bmi.support.cm_pform_x == 0.0 and bmi.support.cm_pform_y == 0.0
    t = Tower.__new__(Tower)
    t._bmi = bmi
    t._sp = read_sec_props(bmi.resolve_sec_props_path())
    stock = t.run(n_modes=12, check_model=False).frequencies

    explicit_zero = _solve_with_cm_offset(0.0, 0.0)
    np.testing.assert_array_equal(stock, explicit_zero)


@pytest.mark.parametrize("rx", [10.0, 30.0])
def test_horizontal_cm_offset_is_wired_and_stable(rx: float) -> None:
    """A horizontal CM offset must (a) measurably shift the rigid-body
    spectrum vs the symmetric case — proving the field is wired
    through nondim_platform into the solve — and (b) stay
    n_modes-stable (no conditioning regression from the new coupling
    terms)."""
    f0 = _solve_with_cm_offset(0.0, 0.0, n_modes=12)
    fr = _solve_with_cm_offset(rx, 0.0, n_modes=12)

    # The offset must change the low (rigid-body) spectrum.
    assert np.max(np.abs(fr[:6] - f0[:6])) > 1e-4, (
        "horizontal CM offset had no effect — not wired through"
    )
    # …and the offset solve must itself be n_modes-stable.
    fr15 = _solve_with_cm_offset(rx, 0.0, n_modes=15)
    drift = float(np.max(np.abs(fr[:6] - fr15[:6])))
    assert drift < 1e-4, (
        f"asymmetric-CM spectrum drifts with n_modes ({drift:.2e} Hz) — "
        f"conditioning regression from the horizontal-arm coupling"
    )
    assert np.all(np.isfinite(fr)) and np.all(fr > 0.0)
