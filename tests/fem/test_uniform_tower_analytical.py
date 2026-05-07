"""Analytical validation: cantilever beam with concentrated tip mass.

Closed-form Euler-Bernoulli reference for the lowest bending frequency of a
uniform cantilever beam carrying a concentrated tip mass at its free end —
the canonical idealisation of a wind-turbine tower with a rotor-nacelle
assembly.

Frequency equation (Blevins, *Formulas for Natural Frequency and Mode
Shape*, Krieger 1979; Karnovsky & Lebed 2001):

    1 + cos(βL)·cosh(βL)
        − μ · βL · ( sin(βL)·cosh(βL) − cos(βL)·sinh(βL) ) = 0

with mass ratio  μ = m_tip / (ρA · L).  The lowest root βL is then mapped
to the natural frequency through

    ω = (βL)² · sqrt(EI / (ρA · L⁴)).

This test bypasses every bundled fixture and validates the FEM core on a
problem whose answer is fixed by the equation above and printed in standard
vibration handbooks.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from pybmodes.fem.assembly import assemble
from pybmodes.fem.nondim import RM, make_params, nondim_tip_mass
from pybmodes.fem.solver import eigvals_to_hz, solve_modes

L_PHYS   = 80.0          # tower height [m]
M_PHYS   = 5000.0        # mass per unit length [kg/m]  (heavy steel section)
EI_PHYS  = 5.0e10        # bending stiffness EI [N·m²]


# ---------------------------------------------------------------------------
# Closed-form reference
# ---------------------------------------------------------------------------

def _frequency_equation(bL: float, mu: float) -> float:
    """Blevins' frequency equation for cantilever + tip mass."""
    cb, sb = np.cos(bL), np.sin(bL)
    chb, shb = np.cosh(bL), np.sinh(bL)
    return 1.0 + cb * chb - mu * bL * (sb * chb - cb * shb)


def _analytical_first_freq_hz(mu: float) -> float:
    """Lowest natural frequency [Hz] for the given tip-mass ratio μ."""
    # μ = 0 recovers the clean-cantilever root βL = 1.8751040687.
    # As μ grows the root moves toward zero (more tip mass softens the beam).
    # Bracket (0.5, 2.5) safely contains the lowest root for all μ ≥ 0.
    bL = brentq(_frequency_equation, 0.5, 2.5, args=(mu,), xtol=1e-12)
    omega = (bL ** 2) * np.sqrt(EI_PHYS / (M_PHYS * L_PHYS ** 4))
    return omega / (2.0 * np.pi)


# ---------------------------------------------------------------------------
# FEM build
# ---------------------------------------------------------------------------

class _TopMass:
    """Duck-typed TipMassProps with no offsets and no inertia."""
    def __init__(self, mass: float) -> None:
        self.mass      = mass
        self.cm_offset = 0.0
        self.cm_axial  = 0.0
        self.ixx = self.iyy = self.izz = 0.0
        self.ixy = self.izx = self.iyz = 0.0


def _fem_first_freq_hz(nselt: int, m_tip: float) -> float:
    """First bending frequency from the FEM core."""
    nd = make_params(radius=L_PHYS, hub_rad=0.0, rot_rpm=0.0)

    eiy_nd  = EI_PHYS  / nd.ref4
    eiz_nd  = eiy_nd                       # axisymmetric tower
    gj_nd   = 1.0e-3 * eiy_nd              # tiny torsion stiffness
    eac_nd  = 1.0e10 / nd.ref2             # large axial stiffness; uncoupled
    rmas_nd = M_PHYS  / RM

    eli = 1.0 / nselt
    el  = np.full(nselt, eli)
    xb  = np.array([1.0 - (i + 1) * eli for i in range(nselt)])

    cfe = np.zeros(nselt)
    eiy_arr  = np.full(nselt, eiy_nd)
    eiz_arr  = np.full(nselt, eiz_nd)
    gj_arr   = np.full(nselt, gj_nd)
    eac_arr  = np.full(nselt, eac_nd)
    rmas_arr = np.full(nselt, rmas_nd)
    skm1_arr = np.full(nselt, 1.0e-5)
    skm2_arr = np.full(nselt, 1.0e-5)
    eg_arr   = np.zeros(nselt)
    ea_arr   = np.zeros(nselt)

    sec_loc = np.array([0.0, 1.0])
    str_tw  = np.zeros(2)

    tip_nd = nondim_tip_mass(_TopMass(m_tip), nd, beam_type=2, id_form=1, hub_conn=1)

    gk, gm, _ = assemble(
        nselt=nselt, el=el, xb=xb, cfe=cfe,
        eiy=eiy_arr, eiz=eiz_arr, gj=gj_arr, eac=eac_arr,
        rmas=rmas_arr, skm1=skm1_arr, skm2=skm2_arr,
        eg=eg_arr, ea=ea_arr,
        omega2=0.0,
        sec_loc=sec_loc, str_tw=str_tw,
        tip_mass=tip_nd, hub_conn=1,
    )
    eigvals, _ = solve_modes(gk, gm, n_modes=4)
    freqs_hz = eigvals_to_hz(eigvals, nd.romg)
    return float(freqs_hz[0])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestCantileverWithTipMass:

    NSELT   = 40
    REL_TOL = 5e-3   # 0.5 %

    @pytest.mark.parametrize("mu", [0.0, 0.5, 1.0, 2.0, 5.0])
    def test_first_frequency_matches_analytical(self, mu):
        """For tip-mass ratios spanning two orders of magnitude."""
        m_tip = mu * M_PHYS * L_PHYS
        f_fem = _fem_first_freq_hz(self.NSELT, m_tip)
        f_ref = _analytical_first_freq_hz(mu)
        rel = abs(f_fem - f_ref) / f_ref
        assert rel < self.REL_TOL, (
            f"μ={mu}: FEM {f_fem:.6f} Hz, analytical {f_ref:.6f} Hz, "
            f"error {rel*100:.3f}%"
        )

    def test_frequency_monotone_decreasing_with_tip_mass(self):
        """Adding tip mass lowers the first bending frequency."""
        prev = float("inf")
        for mu in (0.0, 0.5, 1.0, 2.0, 5.0):
            f = _fem_first_freq_hz(self.NSELT, mu * M_PHYS * L_PHYS)
            assert f < prev
            prev = f


# ---------------------------------------------------------------------------
# Sanity: μ = 0 case must reduce to the cantilever solution we already test
# in test_cantilever.py — independent corroboration of that benchmark.
# ---------------------------------------------------------------------------

class TestZeroTipMassReducesToCantilever:

    def test_first_frequency_matches_cantilever(self):
        f_with_zero_tip = _fem_first_freq_hz(40, m_tip=0.0)
        f_analytical    = _analytical_first_freq_hz(mu=0.0)
        # μ=0 gives βL = 1.875104407, the standard cantilever root.
        bL_canonical = 1.87510407
        f_canonical = (bL_canonical ** 2) * np.sqrt(
            EI_PHYS / (M_PHYS * L_PHYS ** 4)
        ) / (2.0 * np.pi)
        assert abs(f_with_zero_tip - f_canonical) / f_canonical < 5e-3
        assert abs(f_analytical - f_canonical) / f_canonical < 1e-6
