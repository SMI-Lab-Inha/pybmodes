"""Centrifugal-stiffening validation: spinning inextensible cable.

Closed-form reference (Bir 2009, AIAA 2009-1035 §III.B; equivalently the
inextensible-cable limit of an Euler-Bernoulli beam with vanishing flexural
stiffness):

    ω_k = Ω · √(k · (2k − 1))            (rad/s)

with corresponding mode shapes given by the odd-order Legendre polynomials
P_{2k-1}(x). The first three modal frequencies are

    ω_1 = Ω,   ω_2 = Ω · √6,   ω_3 = Ω · √15.

Bir's analytical solution implicitly assumes a *pinned-free* root BC:

    w(0) = 0           (deflection locked at the root)
    w'(0) free         (slope free — required by the Legendre solution)

This is **not** the standard cantilever BC pyBmodes uses for
``hub_conn=1`` (which clamps both deflection and slope at the root). To
enable this test, pyBmodes adds ``hub_conn=4`` — pinned-free — which
locks axial, v_disp, w_disp, and twist at the root while leaving the
bending slopes free.

This test exercises the centrifugal-stiffening path in pyBmodes' FEM
core: with EI driven far below the centrifugal contribution, the only
restoring mechanism is the axial tension induced by rotation.
Verifying ω_k = Ω·√(k(2k−1)) confirms that the per-element
centrifugal-tension assembly (``cfe``) and the omega² factor are wired
correctly under the pinned-free BC.

Reference: Bir, G. S. (2009), "Blades and Towers Modal Analysis Code
(BModes): Verification of Blade Modal Analysis Capability,"
AIAA 2009-1035, Table 2a (BModes vs. analytical, spinning uniform cable).
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fem.assembly import assemble
from pybmodes.fem.boundary import active_dof_indices
from pybmodes.fem.nondim import RM, make_params
from pybmodes.fem.normalize import extract_mode_shapes
from pybmodes.fem.solver import eigvals_to_hz, solve_modes

# Bir 2009 §III.B parameters
L_PHYS = 31.623
M_PHYS = 100.0

# EI is set far below the centrifugal contribution at the smallest tested
# Ω so the cable analytical limit (ω = Ω·√(k(2k-1))) is recovered to
# ≤ 0.5 % for the first three flap modes. Bir's BModes runs reach
# ~ 0.1 % on the same setup (Table 2a); 0.5 % leaves headroom for
# mesh-discretisation drift on small Ω.
EI_FLAP_PHYS = 1.0e3
NSELT = 30
HUB_CONN_PINNED_FREE = 4


def _omega_to_rpm(omega_radps: float) -> float:
    return omega_radps * 30.0 / np.pi


def _flap_freqs_radps(rot_rpm: float, n: int) -> np.ndarray:
    """Solve the FEM and return the lowest *n* flap-dominated cable
    frequencies [rad/s], using the pinned-free root BC."""
    nd = make_params(radius=L_PHYS, hub_rad=0.0, rot_rpm=rot_rpm)

    eiy_nd = EI_FLAP_PHYS / nd.ref4
    eiz_nd = 1.0e4 * eiy_nd          # lag stiff so its modes don't interleave
    gj_nd = 1.0e3 * eiy_nd           # torsion stiff
    eac_nd = 1.0e10 / nd.ref2        # axial stiff (uncoupled)
    rmas_nd = M_PHYS / RM

    eli = 1.0 / NSELT
    el = np.full(NSELT, eli)
    xb = np.array([1.0 - (i + 1) * eli for i in range(NSELT)])

    # Centrifugal tension at outboard face of element i, non-dim.
    # cfe[0] = 0 at the tip; cfe[i] accumulates contributions from
    # elements outboard of i. No tip mass on the cable.
    contrib = 0.5 * rmas_nd * ((xb + el) ** 2 - xb ** 2)
    cfe = np.empty(NSELT)
    cfe[0] = 0.0
    cfe[1:] = np.cumsum(contrib[:-1])

    eiy_arr = np.full(NSELT, eiy_nd)
    eiz_arr = np.full(NSELT, eiz_nd)
    gj_arr = np.full(NSELT, gj_nd)
    eac_arr = np.full(NSELT, eac_nd)
    rmas_arr = np.full(NSELT, rmas_nd)
    skm1_arr = np.full(NSELT, 1.0e-5)
    skm2_arr = np.full(NSELT, 1.0e-5)
    eg_arr = np.zeros(NSELT)
    ea_arr = np.zeros(NSELT)

    sec_loc = np.array([0.0, 1.0])
    str_tw = np.zeros(2)

    gk, gm, _ = assemble(
        nselt=NSELT, el=el, xb=xb, cfe=cfe,
        eiy=eiy_arr, eiz=eiz_arr, gj=gj_arr, eac=eac_arr,
        rmas=rmas_arr, skm1=skm1_arr, skm2=skm2_arr,
        eg=eg_arr, ea=ea_arr,
        omega2=nd.omega2,
        sec_loc=sec_loc, str_tw=str_tw,
        hub_conn=HUB_CONN_PINNED_FREE,
    )

    n_solve = 8 * n + 6
    eigvals, eigvecs = solve_modes(gk, gm, n_modes=n_solve)
    freqs_hz = eigvals_to_hz(eigvals, nd.romg)
    freqs_radps = 2.0 * np.pi * freqs_hz

    active = active_dof_indices(NSELT, hub_conn=HUB_CONN_PINNED_FREE)
    shapes = extract_mode_shapes(
        eigvecs=eigvecs, eigvals_hz=freqs_hz,
        nselt=NSELT, el=el, xb=xb,
        radius=L_PHYS, hub_rad=0.0, bl_len=L_PHYS,
        hub_conn=HUB_CONN_PINNED_FREE, active_dofs=active,
    )

    flap_freqs: list[float] = []
    for k, shape in enumerate(shapes):
        flap_n = float(np.dot(shape.flap_disp, shape.flap_disp))
        lag_n = float(np.dot(shape.lag_disp, shape.lag_disp))
        twist_n = float(np.dot(shape.twist, shape.twist))
        if flap_n > 4.0 * (lag_n + twist_n) and freqs_radps[k] > 1e-6:
            flap_freqs.append(freqs_radps[k])
        if len(flap_freqs) == n:
            break
    if len(flap_freqs) < n:
        raise RuntimeError(
            f"Could not find {n} flap-dominated cable modes; got "
            f"{flap_freqs} from {n_solve}-mode spectrum."
        )
    return np.asarray(flap_freqs)


def _analytical_cable_freqs(omega_radps: float, n: int) -> np.ndarray:
    """Bir 2009 Eq. (8): ω_k = Ω · √(k(2k-1)) for k = 1..n, in rad/s."""
    k = np.arange(1, n + 1, dtype=float)
    return omega_radps * np.sqrt(k * (2.0 * k - 1.0))


# Bir Table 2a covers Ω ∈ {0, 2, 6, 10, 15, 20, 25, 30} rad/s. Skip Ω=0
# (degenerate — gives zero frequencies under the pinned-free BC).
_OMEGA_TEST_RADPS = (2.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCableFirstFlapFrequency:
    """Mode 1: ω = Ω ; pyBmodes within 0.5 % across the Bir 2009 Ω sweep."""

    REL_TOL = 5e-3

    @pytest.mark.parametrize("omega_radps", _OMEGA_TEST_RADPS)
    def test_first_flap(self, omega_radps: float) -> None:
        f_fem = _flap_freqs_radps(_omega_to_rpm(omega_radps), n=1)
        f_ref = _analytical_cable_freqs(omega_radps, n=1)
        rel = abs(f_fem[0] - f_ref[0]) / f_ref[0]
        assert rel < self.REL_TOL, (
            f"Ω = {omega_radps} rad/s: FEM {f_fem[0]:.6f}, "
            f"analytical {f_ref[0]:.6f}, error {rel*100:.3f} %"
        )


class TestCableHigherFlapModes:
    """Modes 2 and 3: ω = Ω·√6 and Ω·√15."""

    REL_TOL = 5e-3

    @pytest.mark.parametrize("omega_radps", _OMEGA_TEST_RADPS)
    def test_first_three(self, omega_radps: float) -> None:
        f_fem = _flap_freqs_radps(_omega_to_rpm(omega_radps), n=3)
        f_ref = _analytical_cable_freqs(omega_radps, n=3)
        rel = np.abs(f_fem - f_ref) / f_ref
        assert np.all(rel < self.REL_TOL), (
            f"Ω = {omega_radps} rad/s: FEM {f_fem}, ref {f_ref}, "
            f"rel-err {rel}"
        )


class TestCableLinearScaling:
    """ω_k / Ω is independent of Ω — verifies the centrifugal stiffening
    formulation linearises in Ω as Bir 2009 Eq. (8) requires."""

    def test_first_mode_ratio(self) -> None:
        ratios = np.asarray([
            _flap_freqs_radps(_omega_to_rpm(omega), n=1)[0] / omega
            for omega in _OMEGA_TEST_RADPS
        ])
        # All ratios should equal 1.0 (mode 1) within 0.5 %.
        assert np.all(np.abs(ratios - 1.0) < 5e-3), ratios

    def test_second_mode_ratio(self) -> None:
        target = float(np.sqrt(6.0))
        ratios = np.asarray([
            _flap_freqs_radps(_omega_to_rpm(omega), n=2)[1] / omega
            for omega in _OMEGA_TEST_RADPS
        ])
        rel = np.abs(ratios - target) / target
        assert np.all(rel < 5e-3), (ratios, rel)
