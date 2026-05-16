"""WindIO composite-blade cross-section reduction (1.4.0, issue #35,
Phase 2).

`RotatingBlade.from_windio` will reduce a WindIO blade composite layup
to the 1-D distributed beam properties the FEM consumes, via a
PreComp-class thin-wall classical-lamination-theory (CLT) shear-flow
reduction (Bir 2006, NREL/TP-500-38929).

Validation ladder (built sub-phase by sub-phase; default-suite rungs
are self-contained, integration rungs anchor on the companion RWT
BeamDyn 6×6 decks):

* SP-1 — CLT laminate primitives vs closed-form (Jones, *Mechanics of
  Composite Materials*): reduced stiffness, ply transformation, ABD
  assembly, membrane condensation. **(this file, below)**
* SP-2 — section geometry: airfoil arc parameterisation, spanwise
  blend, region arc-band resolution across both WindIO dialects.
* SP-3/4 — single- then multi-cell reduction vs closed-form tube /
  box (exact) and the IEA-3.4/10/15/22 + NREL-5MW BeamDyn 6×6
  diagonals (integration; PreComp-class tolerances).
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.io._precomp.laminate import (
    PlyElastic,
    abd_matrices,
    material_plane_stress,
    membrane_condensed,
    reduced_stiffness,
    transform_reduced_stiffness,
)
from pybmodes.io._precomp.profile import Profile


def _circle_coords(n: int = 241, diameter: float = 1.0):
    """WindIO 'circular' airfoil: TE→suction→LE→pressure→TE, the exact
    convention IEA-15's `circular` airfoil uses (x = ½(1+cos α),
    y = ½ sin α scaled by diameter). Odd `n` puts the LE exactly on a
    vertex (α = π) so a symmetric shape's landmarks are exact."""
    a = np.linspace(0.0, 2.0 * np.pi, n)
    x = 0.5 * (1.0 + np.cos(a))
    y = 0.5 * np.sin(a) * diameter
    return x, y


def _ellipse_coords(n: int = 241, tc: float = 0.4):
    a = np.linspace(0.0, 2.0 * np.pi, n)
    return 0.5 * (1.0 + np.cos(a)), 0.5 * np.sin(a) * tc

# ---------------------------------------------------------------------------
# SP-1. CLT laminate primitives vs closed form (default; no external data)
# ---------------------------------------------------------------------------


def test_material_plane_stress_isotropic() -> None:
    """Isotropic: E1 == E2 == E, G defaults to E/2(1+nu), rho scalar."""
    p = material_plane_stress({"name": "steel", "E": 2.0e11, "nu": 0.3,
                               "rho": 7800.0})
    assert p.E1 == p.E2 == 2.0e11
    assert p.nu12 == 0.3
    assert p.G12 == pytest.approx(2.0e11 / (2.0 * 1.3))
    assert p.rho == 7800.0
    # Explicit G overrides the isotropic default.
    p2 = material_plane_stress({"name": "s", "E": 2.0e11, "nu": 0.3,
                                "G": 7.7e10, "rho": 7800.0})
    assert p2.G12 == 7.7e10


def test_material_plane_stress_orthotropic() -> None:
    """Orthotropic vectors → in-plane (E1,E2,G12,nu12) subset; rho scalar."""
    p = material_plane_stress({
        "name": "CarbonUD",
        "E": [1.14e11, 8.0e9, 8.0e9],
        "G": [5.0e9, 5.0e9, 3.0e9],
        "nu": [0.3, 0.3, 0.4],
        "rho": 1600.0,
    })
    assert p.E1 == 1.14e11 and p.E2 == 8.0e9
    assert p.G12 == 5.0e9 and p.nu12 == 0.3 and p.rho == 1600.0


def test_material_plane_stress_density_synonym() -> None:
    p = material_plane_stress({"name": "x", "E": 1.0e10, "density": 1200.0})
    assert p.rho == 1200.0


def test_reduced_stiffness_isotropic_closed_form() -> None:
    """Q reduces to the isotropic E/(1-ν²) form (Jones eq. 2.66)."""
    E, nu = 70.0e9, 0.33
    p = PlyElastic(E1=E, E2=E, G12=E / (2.0 * (1.0 + nu)), nu12=nu, rho=2700.0)
    Q = reduced_stiffness(p)
    assert Q[0, 0] == pytest.approx(E / (1.0 - nu**2))
    assert Q[1, 1] == pytest.approx(E / (1.0 - nu**2))
    assert Q[0, 1] == pytest.approx(nu * E / (1.0 - nu**2))
    assert Q[2, 2] == pytest.approx(E / (2.0 * (1.0 + nu)))
    assert Q[0, 2] == 0.0 and Q[1, 2] == 0.0


def test_transform_zero_angle_is_identity() -> None:
    p = PlyElastic(E1=1.4e11, E2=9.0e9, G12=5.0e9, nu12=0.3, rho=1600.0)
    Q = reduced_stiffness(p)
    np.testing.assert_allclose(transform_reduced_stiffness(Q, 0.0), Q)


def test_transform_ninety_degrees_swaps_axes() -> None:
    """A 90° ply swaps the 11/22 stiffnesses; no shear coupling at 90°."""
    p = PlyElastic(E1=1.4e11, E2=9.0e9, G12=5.0e9, nu12=0.3, rho=1600.0)
    Q = reduced_stiffness(p)
    Qb = transform_reduced_stiffness(Q, np.pi / 2.0)
    assert Qb[0, 0] == pytest.approx(Q[1, 1])
    assert Qb[1, 1] == pytest.approx(Q[0, 0])
    assert Qb[2, 2] == pytest.approx(Q[2, 2])
    assert Qb[0, 2] == pytest.approx(0.0, abs=1e-3)
    assert Qb[1, 2] == pytest.approx(0.0, abs=1e-3)


def test_transform_pm45_shear_coupling_cancels_in_balanced_pair() -> None:
    """+θ and −θ plies have equal/opposite Qbar16 (Jones §2.6) — the
    basis for a *balanced* laminate having A16 = 0."""
    p = PlyElastic(E1=1.4e11, E2=9.0e9, G12=5.0e9, nu12=0.3, rho=1600.0)
    Q = reduced_stiffness(p)
    qp = transform_reduced_stiffness(Q, np.deg2rad(45.0))
    qm = transform_reduced_stiffness(Q, np.deg2rad(-45.0))
    assert qp[0, 2] == pytest.approx(-qm[0, 2])
    assert qp[0, 0] == pytest.approx(qm[0, 0])  # even in θ


def test_abd_single_ply_closed_form() -> None:
    """Single centred ply: B = 0, A = Qbar·t, D = Qbar·t³/12."""
    p = PlyElastic(E1=70.0e9, E2=70.0e9, G12=70.0e9 / 2.6, nu12=0.3,
                   rho=2700.0)
    Q = reduced_stiffness(p)
    t = 0.004
    A, B, D = abd_matrices([(Q, t)])
    np.testing.assert_allclose(A, Q * t, rtol=1e-12)
    np.testing.assert_allclose(B, np.zeros((3, 3)), atol=1e-9)
    np.testing.assert_allclose(D, Q * t**3 / 12.0, rtol=1e-12)


def test_abd_symmetric_stack_has_zero_B() -> None:
    """A mid-plane-symmetric [0/90/0] stack → B ≈ 0 (extension-bending
    decoupled, Jones §4.3)."""
    p = PlyElastic(E1=1.4e11, E2=9.0e9, G12=5.0e9, nu12=0.3, rho=1600.0)
    Q = reduced_stiffness(p)
    q0 = transform_reduced_stiffness(Q, 0.0)
    q90 = transform_reduced_stiffness(Q, np.pi / 2.0)
    _, B, _ = abd_matrices([(q0, 0.001), (q90, 0.002), (q0, 0.001)])
    assert np.max(np.abs(B)) < 1e-6 * 1.4e11 * 0.001


def test_abd_balanced_symmetric_has_zero_A16() -> None:
    """A balanced symmetric [+45/−45]_s laminate → A16 = A26 = 0."""
    p = PlyElastic(E1=1.4e11, E2=9.0e9, G12=5.0e9, nu12=0.3, rho=1600.0)
    Q = reduced_stiffness(p)
    qp = transform_reduced_stiffness(Q, np.deg2rad(45.0))
    qm = transform_reduced_stiffness(Q, np.deg2rad(-45.0))
    t = 0.001
    A, B, _ = abd_matrices([(qp, t), (qm, t), (qm, t), (qp, t)])
    assert abs(A[0, 2]) < 1e-6 * A[0, 0]
    assert abs(A[1, 2]) < 1e-6 * A[1, 1]
    assert np.max(np.abs(B)) < 1e-6 * A[0, 0]


def test_membrane_condensed_symmetric_returns_A() -> None:
    """B = 0 ⇒ Atilde = A exactly (no wall-bending knock-down)."""
    p = PlyElastic(E1=1.4e11, E2=9.0e9, G12=5.0e9, nu12=0.3, rho=1600.0)
    Q = reduced_stiffness(p)
    q0 = transform_reduced_stiffness(Q, 0.0)
    q90 = transform_reduced_stiffness(Q, np.pi / 2.0)
    A, B, D = abd_matrices([(q0, 0.001), (q90, 0.002), (q0, 0.001)])
    np.testing.assert_array_equal(membrane_condensed(A, B, D), A)


def test_membrane_condensed_unsymmetric_knocks_A_down() -> None:
    """An unsymmetric [0/90] laminate (B ≠ 0): Atilde = A − B D⁻¹ B,
    symmetric and strictly softer than A in the axial term."""
    p = PlyElastic(E1=1.4e11, E2=9.0e9, G12=5.0e9, nu12=0.3, rho=1600.0)
    Q = reduced_stiffness(p)
    q0 = transform_reduced_stiffness(Q, 0.0)
    q90 = transform_reduced_stiffness(Q, np.pi / 2.0)
    A, B, D = abd_matrices([(q0, 0.002), (q90, 0.002)])
    assert np.max(np.abs(B)) > 0.0
    At = membrane_condensed(A, B, D)
    np.testing.assert_allclose(At, A - B @ np.linalg.solve(D, B), rtol=1e-12)
    np.testing.assert_allclose(At, At.T, rtol=1e-10)
    assert At[0, 0] < A[0, 0]


# ---------------------------------------------------------------------------
# SP-2. Airfoil profile geometry / nd_arc spine (default; no external data)
# ---------------------------------------------------------------------------


def test_profile_circle_normalisation_and_arc_spine() -> None:
    """A unit circle: chord-normalised, the nd_arc spine spans [0,1]
    with the LE at the half-perimeter, t/c = 1."""
    p = Profile.from_windio_coords(*_circle_coords())
    assert p.s[0] == pytest.approx(0.0)
    assert p.s[-1] == pytest.approx(1.0)
    assert np.all(np.diff(p.s) > 0.0)            # strictly monotone spine
    assert p.s_le == pytest.approx(0.5, abs=2e-3)
    assert p.tc == pytest.approx(1.0, abs=5e-3)
    assert p.xc.min() == pytest.approx(0.0, abs=1e-9)
    assert p.xc.max() == pytest.approx(1.0, abs=1e-9)


def test_profile_arc_to_xy_landmarks() -> None:
    """s=0 and s=1 are the TE; s=s_le is the LE (x≈0)."""
    p = Profile.from_windio_coords(*_circle_coords())
    x0, y0 = p.arc_to_xy(0.0)
    x1, y1 = p.arc_to_xy(1.0)
    xle, yle = p.arc_to_xy(p.s_le)
    assert (x0, x1) == (pytest.approx(1.0, abs=1e-6),
                        pytest.approx(1.0, abs=1e-6))
    assert y0 == pytest.approx(0.0, abs=1e-6)
    assert xle == pytest.approx(0.0, abs=2e-3)
    assert yle == pytest.approx(0.0, abs=2e-3)


def test_profile_arc_of_chord_anchors() -> None:
    """fixed:LE → x/c=0, fixed:TE → x/c=1 map to the right nd_arc on
    each surface (the resolver's anchor primitives)."""
    p = Profile.from_windio_coords(*_circle_coords())
    assert p.arc_of_chord(1.0, side="suction") == pytest.approx(0.0, abs=2e-3)
    assert p.arc_of_chord(0.0, side="suction") == pytest.approx(
        p.s_le, abs=2e-3)
    assert p.arc_of_chord(0.0, side="pressure") == pytest.approx(
        p.s_le, abs=2e-3)
    assert p.arc_of_chord(1.0, side="pressure") == pytest.approx(
        1.0, abs=2e-3)
    with pytest.raises(ValueError, match="suction.*pressure"):
        p.arc_of_chord(0.5, side="sideways")


def test_profile_blend_weight_semantics() -> None:
    """weight 0 → self, 1 → other, ½ → midway t/c (spanwise airfoil
    interpolation)."""
    circ = Profile.from_windio_coords(*_circle_coords())          # t/c 1.0
    ell = Profile.from_windio_coords(*_ellipse_coords(tc=0.4))     # t/c 0.4
    assert circ.blend(ell, 0.0).tc == pytest.approx(circ.tc, abs=5e-3)
    assert circ.blend(ell, 1.0).tc == pytest.approx(ell.tc, abs=5e-3)
    mid = circ.blend(ell, 0.5)
    assert mid.tc == pytest.approx(0.5 * (circ.tc + ell.tc), abs=5e-3)
    # The blended profile keeps a valid, monotone nd_arc spine.
    assert mid.s[0] == pytest.approx(0.0) and mid.s[-1] == pytest.approx(1.0)
    assert np.all(np.diff(mid.s) > 0.0)


def test_profile_rejects_degenerate() -> None:
    with pytest.raises(ValueError, match="zero chord|zero perimeter"):
        Profile.from_windio_coords(np.zeros(10), np.zeros(10))
