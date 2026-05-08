"""Centrifugal-stiffening + tip-mass validation: rotating uniform blade
carrying a concentrated tip mass.

Closed-form reference values transcribed from Bir, G. S. (2010),
NREL/CP-500-47953, "Verification of BModes: Rotary Beam and Tower Modal
Analysis Code," Section 4 / Table 5.

Configuration (Bir 2010 §4):

    L          = 31.623 m
    m          = 100 kg/m
    EI_flap    = 1.0e8 N·m²
    EI_lag     = 1.0e9 N·m²
    GJ         = 1.0e5 N·m²
    m_tip      = 3162.3 kg          (μ ≡ m_tip / (m·L) = 1.0 exactly)
    BC         = cantilever
    nselt      = 15                 (matches Bir 2010 §4 "15 beam elements")

The reference column in Bir's Table 5 is itself analytical (Wright et al.
1982). Bir reports BModes matches it to 0.02 % on the worst row; we gate
at 0.1 % to leave a small margin for our 15-element mesh choice.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fem.assembly import assemble
from pybmodes.fem.boundary import active_dof_indices
from pybmodes.fem.nondim import RM, TipMassND, make_params
from pybmodes.fem.normalize import extract_mode_shapes
from pybmodes.fem.solver import eigvals_to_hz, solve_modes

# Bir 2010 §4 parameters
L_PHYS = 31.623
M_PHYS = 100.0
EI_FLAP = 1.0e8
EI_LAG = 1.0e9
GJ_PHYS = 1.0e5
M_TIP = 3162.3
NSELT = 15

# Bir 2010 Table 5 — analytical / BModes flap frequencies (rad/s).
# Each row: (Ω, flap-1, flap-2).  Both columns are identical in Bir's
# Table 5 except for one rounding in the last row (47.14 vs 47.15);
# we use the analytical value as the assertion target.
_BIR_TABLE_5 = np.array([
    (0.0,   1.557, 16.25),
    (1.0,   1.902, 16.76),
    (2.0,   2.670, 18.19),
    (3.0,   3.582, 20.35),
    (4.0,   4.543, 23.03),
    (5.0,   5.522, 26.04),
    (6.0,   6.509, 29.29),
    (7.0,   7.501, 32.70),
    (8.0,   8.495, 36.21),
    (9.0,   9.490, 39.80),
    (10.0, 10.49,  43.45),
    (11.0, 11.48,  47.14),
    (12.0, 12.48,  50.86),
])


def _omega_to_rpm(omega_radps: float) -> float:
    return omega_radps * 30.0 / np.pi


def _flap_freqs_radps(rot_rpm: float, n: int) -> np.ndarray:
    """Solve the FEM and return the lowest *n* flap-dominated modes [rad/s]."""
    nd = make_params(radius=L_PHYS, hub_rad=0.0, rot_rpm=rot_rpm)

    eiy_nd = EI_FLAP / nd.ref4
    eiz_nd = EI_LAG / nd.ref4
    gj_nd = GJ_PHYS / nd.ref4
    eac_nd = 1.0e10 / nd.ref2
    rmas_nd = M_PHYS / RM

    eli = 1.0 / NSELT
    el = np.full(NSELT, eli)
    xb = np.array([1.0 - (i + 1) * eli for i in range(NSELT)])

    # Centrifugal tension. Distributed-mass contribution per element:
    #     contrib_distr[i] = 0.5 · rmas · ((x_out)² − (x_in)²)  (non-dim)
    # Tip mass at r = L contributes a constant T_tip = M_TIP·L·Ω² that is
    # transmitted through every section inboard of the tip — i.e. it adds
    # the same constant offset to cfe at every element face including the
    # outboard face of the tip element (where the tip mass is anchored).
    # In non-dim: cfe_tip = (M_TIP / RM) · (L / radius²)·radius = M_TIP /
    # (RM · radius) since L = radius here. Generally: M_TIP_nd · x_tip.
    contrib_distr = 0.5 * rmas_nd * ((xb + el) ** 2 - xb ** 2)
    m_tip_nd = M_TIP / nd.ref_mr      # = M_TIP / (RM · radius)
    x_tip_nd = 1.0                    # tip is at non-dim radius 1.0
    cfe_tip = m_tip_nd * x_tip_nd
    cfe = np.empty(NSELT)
    cfe[0] = cfe_tip
    cfe[1:] = cfe_tip + np.cumsum(contrib_distr[:-1])

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

    # Tip mass at the cantilever tip — no offsets, no inertia. Equivalent
    # to the duck-typed _TopMass used in test_uniform_tower_analytical.py
    # but built directly as a non-dim TipMassND so we don't need the
    # nondim_tip_mass remapping path (which uses the tower-only literal
    # cm_loc/cm_axial convention).
    tip_nd = TipMassND(
        mass=M_TIP / nd.ref_mr,
        cm_loc=0.0, cm_axial=0.0,
        ixx=0.0, iyy=0.0, izz=0.0, ixy=0.0, iyz=0.0, izx=0.0,
    )

    gk, gm, _ = assemble(
        nselt=NSELT, el=el, xb=xb, cfe=cfe,
        eiy=eiy_arr, eiz=eiz_arr, gj=gj_arr, eac=eac_arr,
        rmas=rmas_arr, skm1=skm1_arr, skm2=skm2_arr,
        eg=eg_arr, ea=ea_arr,
        omega2=nd.omega2,
        sec_loc=sec_loc, str_tw=str_tw,
        tip_mass=tip_nd, hub_conn=1,
    )

    n_solve = 8 * n + 6
    eigvals, eigvecs = solve_modes(gk, gm, n_modes=n_solve)
    freqs_hz = eigvals_to_hz(eigvals, nd.romg)
    freqs_radps = 2.0 * np.pi * freqs_hz

    active = active_dof_indices(NSELT, hub_conn=1)
    shapes = extract_mode_shapes(
        eigvecs=eigvecs, eigvals_hz=freqs_hz,
        nselt=NSELT, el=el, xb=xb,
        radius=L_PHYS, hub_rad=0.0, bl_len=L_PHYS,
        hub_conn=1, active_dofs=active,
    )

    flap_freqs: list[float] = []
    for k, shape in enumerate(shapes):
        flap_norm = float(np.dot(shape.flap_disp, shape.flap_disp))
        lag_norm = float(np.dot(shape.lag_disp, shape.lag_disp))
        twist_norm = float(np.dot(shape.twist, shape.twist))
        if flap_norm > 4.0 * (lag_norm + twist_norm) and freqs_radps[k] > 1e-6:
            flap_freqs.append(freqs_radps[k])
        if len(flap_freqs) == n:
            break
    if len(flap_freqs) < n:
        raise RuntimeError(
            f"Could not find {n} flap-dominated modes; got {flap_freqs} "
            f"from {n_solve}-mode spectrum."
        )
    return np.asarray(flap_freqs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRotatingBladeWithTipMassFlap:
    """Bir 2010 Table 5 reference — flap modes 1 and 2, ≤ 0.1 %."""

    REL_TOL = 1e-3   # 0.1 %

    @pytest.mark.parametrize(("omega_radps", "f_ref_radps"), [
        (row[0], row[1]) for row in _BIR_TABLE_5
    ])
    def test_first_flap_frequency(self, omega_radps: float,
                                  f_ref_radps: float) -> None:
        f_fem = _flap_freqs_radps(_omega_to_rpm(omega_radps), n=1)
        rel = abs(f_fem[0] - f_ref_radps) / f_ref_radps
        assert rel < self.REL_TOL, (
            f"Ω = {omega_radps} rad/s, mode 1: FEM {f_fem[0]:.4f}, "
            f"ref {f_ref_radps:.4f} rad/s, error {rel*100:.3f} %"
        )

    @pytest.mark.parametrize(("omega_radps", "f_ref_radps"), [
        (row[0], row[2]) for row in _BIR_TABLE_5
    ])
    def test_second_flap_frequency(self, omega_radps: float,
                                   f_ref_radps: float) -> None:
        f_fem = _flap_freqs_radps(_omega_to_rpm(omega_radps), n=2)
        rel = abs(f_fem[1] - f_ref_radps) / f_ref_radps
        assert rel < self.REL_TOL, (
            f"Ω = {omega_radps} rad/s, mode 2: FEM {f_fem[1]:.4f}, "
            f"ref {f_ref_radps:.4f} rad/s, error {rel*100:.3f} %"
        )


class TestTipMassReducesNonRotatingFrequency:
    """At Ω = 0 the tip-mass case must give a *lower* first flap frequency
    than the bare-blade case at the same EI / m / L. Mass loading the tip
    softens the structure."""

    def test_first_flap_lower_with_tip_mass(self) -> None:
        # Bir 2009 Table 3a (no tip mass, Ω=0): flap-1 = 3.516 rad/s.
        # Bir 2010 Table 5 (with μ=1 tip mass, Ω=0): flap-1 = 1.557 rad/s.
        f_fem = _flap_freqs_radps(0.0, n=1)
        assert f_fem[0] < 3.516
        assert abs(f_fem[0] - 1.557) / 1.557 < 1e-3
