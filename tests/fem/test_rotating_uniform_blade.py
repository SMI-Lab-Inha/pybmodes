"""Centrifugal-stiffening validation: rotating uniform cantilever blade.

Closed-form reference values from Wright, Smith, Thresher & Wang (1982),
"Vibration Modes of Centrifugally Stiffened Beams," *Journal of Applied
Mechanics*, Vol. 104, March 1982. As tabulated in Bir, G. S. (2009),
AIAA 2009-1035 ("Blades and Towers Modal Analysis Code (BModes):
Verification of Blade Modal Analysis Capability"), Table 3a.

Configuration (Bir 2009 §III.C):

    L          = 31.623 m            length
    m          = 100 kg/m             mass per unit length
    EI_flap    = 1.0e8 N·m²           flap (out-of-plane) bending stiffness
    EI_lag     = 1.0e9 N·m²           lag (in-plane) bending stiffness
    GJ         = 1.0e5 N·m²           torsion stiffness
    BC         = cantilevered at the in-board end
    No twist, no offsets between elastic, tension, and mass-centroid axes.

Reference frequencies (rad/s) for the lowest three flap modes, transcribed
from Bir 2009 Table 3a (analytical column):

    Ω (rad/s) | flap-1     flap-2     flap-3
    --------- | --------   --------   --------
       0      |  3.516     22.035     61.697
       1      |  3.682     22.181     61.842
       2      |  4.137     22.615     62.273
       3      |  4.797     23.320     62.985
       4      |  5.585     24.273     63.967
       5      |  6.450     25.446     65.205
       6      |  7.360     26.809     66.684
       7      |  8.300     28.334     68.386
       8      |  9.257     29.995     70.293
       9      | 10.226     31.771     72.387
      10      | 11.202     33.640     74.649
      11      | 12.184     35.589     77.064
      12      | 13.170     37.603     79.615

Bir reports BModes matches these analytical values to ≤ 0.0013 % on the
worst mode. pyBmodes uses the same 15-DOF beam element formulation; we
gate at 0.5 % to leave headroom for mesh-discretisation drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fem.assembly import assemble
from pybmodes.fem.boundary import active_dof_indices
from pybmodes.fem.nondim import RM, make_params
from pybmodes.fem.normalize import extract_mode_shapes
from pybmodes.fem.solver import eigvals_to_hz, solve_modes

# Bir 2009 §III.C parameters
L_PHYS = 31.623
M_PHYS = 100.0
EI_FLAP = 1.0e8
EI_LAG = 1.0e9
GJ_PHYS = 1.0e5

# Wright 1982 / Bir 2009 Table 3a flap frequencies (rad/s).
# Each row: (Ω, ω_flap1, ω_flap2, ω_flap3).
_BIR_TABLE_3A = np.array([
    (0.0,   3.516, 22.035, 61.697),
    (1.0,   3.682, 22.181, 61.842),
    (2.0,   4.137, 22.615, 62.273),
    (3.0,   4.797, 23.320, 62.985),
    (4.0,   5.585, 24.273, 63.967),
    (5.0,   6.450, 25.446, 65.205),
    (6.0,   7.360, 26.809, 66.684),
    (7.0,   8.300, 28.334, 68.386),
    (8.0,   9.257, 29.995, 70.293),
    (9.0,  10.226, 31.771, 72.387),
    (10.0, 11.202, 33.640, 74.649),
    (11.0, 12.184, 35.589, 77.064),
    (12.0, 13.170, 37.603, 79.615),
])

NSELT = 20  # matches Bir 2009 §III.C ("a BModes model … using 20 beam elements")


def _omega_to_rpm(omega_radps: float) -> float:
    return omega_radps * 30.0 / np.pi


def _flap_freqs_radps(rot_rpm: float, n: int) -> np.ndarray:
    """Build the FEM model and return the lowest *n* flap frequencies [rad/s].

    With ``EI_lag = 10·EI_flap``, the lag-1 mode (≈ √10 · flap-1) lands
    between flap-1 and flap-2, and lag-2 between flap-3 and flap-4. The
    eigenvalue ordering therefore mixes flap and lag modes; we filter the
    spectrum down to flap-dominated modes by inspecting each eigenvector's
    flap-vs-lag content via :func:`extract_mode_shapes`.
    """
    nd = make_params(radius=L_PHYS, hub_rad=0.0, rot_rpm=rot_rpm)

    eiy_nd = EI_FLAP / nd.ref4
    eiz_nd = EI_LAG / nd.ref4
    gj_nd = GJ_PHYS / nd.ref4
    eac_nd = 1.0e10 / nd.ref2
    rmas_nd = M_PHYS / RM

    eli = 1.0 / NSELT
    el = np.full(NSELT, eli)
    xb = np.array([1.0 - (i + 1) * eli for i in range(NSELT)])

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
        sec_loc=sec_loc, str_tw=str_tw, hub_conn=1,
    )
    # Solve a generous superset; lag and torsion modes will be mixed in.
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

    # Flap-dominated <=> ‖flap_disp‖² ≫ ‖lag_disp‖² + ‖twist‖².
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

class TestRotatingUniformBladeFlap:
    """Bir 2009 / Wright 1982 reference frequencies, modes 1-3 vs Ω sweep."""

    REL_TOL = 5e-3   # 0.5 %

    @pytest.mark.parametrize(("omega_radps", "f_ref_radps"), [
        (row[0], row[1]) for row in _BIR_TABLE_3A
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
        (row[0], row[2]) for row in _BIR_TABLE_3A
    ])
    def test_second_flap_frequency(self, omega_radps: float,
                                   f_ref_radps: float) -> None:
        f_fem = _flap_freqs_radps(_omega_to_rpm(omega_radps), n=2)
        rel = abs(f_fem[1] - f_ref_radps) / f_ref_radps
        assert rel < self.REL_TOL, (
            f"Ω = {omega_radps} rad/s, mode 2: FEM {f_fem[1]:.4f}, "
            f"ref {f_ref_radps:.4f} rad/s, error {rel*100:.3f} %"
        )

    @pytest.mark.parametrize(("omega_radps", "f_ref_radps"), [
        (row[0], row[3]) for row in _BIR_TABLE_3A
    ])
    def test_third_flap_frequency(self, omega_radps: float,
                                  f_ref_radps: float) -> None:
        f_fem = _flap_freqs_radps(_omega_to_rpm(omega_radps), n=3)
        rel = abs(f_fem[2] - f_ref_radps) / f_ref_radps
        assert rel < self.REL_TOL, (
            f"Ω = {omega_radps} rad/s, mode 3: FEM {f_fem[2]:.4f}, "
            f"ref {f_ref_radps:.4f} rad/s, error {rel*100:.3f} %"
        )


class TestCentrifugalStiffeningMonotone:
    """Frequencies must increase monotonically with rotor speed
    (centrifugal stiffening always raises the natural frequencies)."""

    def test_first_flap_monotone(self) -> None:
        freqs = np.array([
            _flap_freqs_radps(_omega_to_rpm(omega), n=1)[0]
            for omega in _BIR_TABLE_3A[:, 0]
        ])
        # Strict monotone increase across the Ω sweep.
        assert np.all(np.diff(freqs) > 0.0), freqs

    def test_southwell_relation_first_mode(self) -> None:
        """Southwell coefficient: ω₁²(Ω) ≈ ω₁²(0) + K·Ω². Verify K is
        approximately 1.193 (Wright 1982, Table 1, row corresponding to
        β₁L = 1.875 cantilever); 5 % tolerance accommodates higher-order
        corrections at large Ω."""
        f0 = _flap_freqs_radps(0.0, n=1)[0]
        f12 = _flap_freqs_radps(_omega_to_rpm(12.0), n=1)[0]
        omega12 = 12.0
        k_southwell = (f12 ** 2 - f0 ** 2) / omega12 ** 2
        # Wright 1982 / Bir 2009 implied K ≈ (13.17² - 3.516²)/144 = 1.118.
        # The classical Southwell value for the 1st flap is K ≈ 1.193 with
        # higher-order corrections. The numerical Bir-table value is the
        # one to gate against.
        k_target = (13.170 ** 2 - 3.516 ** 2) / 12.0 ** 2
        rel = abs(k_southwell - k_target) / k_target
        assert rel < 0.01, (k_southwell, k_target, rel)
