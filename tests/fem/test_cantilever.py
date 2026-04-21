"""Uniform cantilever beam vs Euler-Bernoulli analytical natural frequencies.

Physics: fixed-free Euler-Bernoulli beam, no rotation, no structural twist.
L = 50 m, EI_flap = 1 e6 N·m², m = 100 kg/m.

Analytical:
    ω_n = (β_n L)² √(EI / (m L⁴))
    β_n L ≈ [1.87510, 4.69409, 7.85476, 10.99554, 14.13717]

Edge (EI_z = 10⁴ × EI_flap) and torsion (GJ large) modes are far above the
tested flap modes so the first n_modes eigenpairs from the FEM are unambiguously
flap bending modes.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fem.assembly import assemble
from pybmodes.fem.nondim import RM, ROMG, make_params
from pybmodes.fem.solver import eigvals_to_hz, solve_modes

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
L_PHYS = 50.0       # total beam length [m]
EI_PHYS = 1.0e6     # flap bending stiffness [N·m²]
M_PHYS = 100.0      # mass per unit length [kg/m]

# Euler-Bernoulli β_n L values for a clamped-free beam (first 5 modes)
_BETA_L = np.array([1.87510407, 4.69409113, 7.85475744, 10.99554073, 14.13716839])


def _analytical_frequencies(n: int = 5) -> np.ndarray:
    """First *n* natural frequencies [Hz] for the uniform cantilever."""
    alpha = np.sqrt(EI_PHYS / (M_PHYS * L_PHYS**4))
    return _BETA_L[:n] ** 2 * alpha / (2.0 * np.pi)


def _build_cantilever(nselt: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (gk, gm) for a uniform cantilever with *nselt* equal elements.

    Non-dimensionalised with make_params(L, hub_rad=0, rpm=0).
    Edge stiffness is set 10⁴ × larger than flap so the first several
    eigenpairs from solve_modes are all flap bending modes.
    """
    nd = make_params(radius=L_PHYS, hub_rad=0.0, rot_rpm=0.0)

    # Non-dim stiffness / mass
    eiy_nd  = EI_PHYS / nd.ref4          # flap (we compare these)
    eiz_nd  = 1.0e4 * eiy_nd             # edge: first edge mode >> flap mode 5
    gj_nd   = 1.0e3 * eiy_nd             # torsion: first torsion mode >> flap mode 5
    eac_nd  = 100.0                       # axial: first axial mode >> flap mode 5
    rmas_nd = M_PHYS / RM                 # = 10.0

    # Equal-length elements in radius-normalised coords [0, 1]
    eli = 1.0 / nselt
    el  = np.full(nselt, eli)
    # Inboard (root-side) end of each element, tip-to-root ordering:
    #   element 0 (tip):  inboard at 1 - eli
    #   element nselt-1 (root): inboard at 0
    xb = np.array([1.0 - (i + 1) * eli for i in range(nselt)])

    # Centrifugal tension: zero (non-rotating)
    cfe = np.zeros(nselt)

    # Per-element property arrays (uniform)
    eiy_arr  = np.full(nselt, eiy_nd)
    eiz_arr  = np.full(nselt, eiz_nd)
    gj_arr   = np.full(nselt, gj_nd)
    eac_arr  = np.full(nselt, eac_nd)
    rmas_arr = np.full(nselt, rmas_nd)
    # Small but non-zero rotational inertia prevents a singular mass matrix
    skm1_arr = np.full(nselt, 1.0e-5)
    skm2_arr = np.full(nselt, 1.0e-5)
    eg_arr   = np.zeros(nselt)           # no CG offset
    ea_arr   = np.zeros(nselt)           # no tension-centre offset

    # No structural twist
    sec_loc = np.array([0.0, 1.0])
    str_tw  = np.zeros(2)

    gk, gm, _ = assemble(
        nselt=nselt,
        el=el, xb=xb, cfe=cfe,
        eiy=eiy_arr, eiz=eiz_arr, gj=gj_arr, eac=eac_arr,
        rmas=rmas_arr, skm1=skm1_arr, skm2=skm2_arr,
        eg=eg_arr, ea=ea_arr,
        omega2=0.0,
        sec_loc=sec_loc, str_tw=str_tw,
        hub_conn=1,   # cantilever: all 6 root DOFs fixed
    )
    return gk, gm


# ---------------------------------------------------------------------------
# Helper: run FEM and return first *n* frequencies [Hz]
# ---------------------------------------------------------------------------

def _fem_frequencies(nselt: int, n: int) -> np.ndarray:
    gk, gm = _build_cantilever(nselt)
    eigvals, _ = solve_modes(gk, gm, n_modes=n)
    return eigvals_to_hz(eigvals, ROMG)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnalyticalFrequencies:
    """Sanity-check the analytical helper."""

    def test_first_mode_positive(self):
        f = _analytical_frequencies(1)
        assert f[0] > 0.0

    def test_frequencies_ascending(self):
        f = _analytical_frequencies(5)
        assert np.all(np.diff(f) > 0.0)

    def test_mode_1_value(self):
        # ω₁ = 1.8751² √(EI/(mL⁴)) / (2π) with L=50, EI=1e6, m=100
        expected = 1.87510407**2 * np.sqrt(1e6 / (100.0 * 50.0**4)) / (2.0 * np.pi)
        assert _analytical_frequencies(1)[0] == pytest.approx(expected, rel=1e-6)


class TestCantileverFlap:
    """First-five flap natural frequencies match Euler-Bernoulli within 0.5 %.

    Uses 40 elements — sufficient for sub-percent accuracy on the first five modes.
    """

    NSELT = 40
    N_MODES = 5

    def setup_method(self):
        self.freqs_fem = _fem_frequencies(self.NSELT, self.N_MODES)
        self.freqs_ref = _analytical_frequencies(self.N_MODES)

    @pytest.mark.parametrize("k", range(3))
    def test_mode_frequency(self, k: int):
        """Modes 1–3 are within 0.5 % of the analytical solution."""
        rel_err = abs(self.freqs_fem[k] - self.freqs_ref[k]) / self.freqs_ref[k]
        assert rel_err < 0.005, (
            f"Mode {k+1}: FEM {self.freqs_fem[k]:.6f} Hz, "
            f"analytical {self.freqs_ref[k]:.6f} Hz, "
            f"error {rel_err*100:.3f} %"
        )

    def test_modes_ascending(self):
        """FEM frequencies are in ascending order."""
        assert np.all(np.diff(self.freqs_fem) > 0.0)

    def test_mode4_within_1pct(self):
        rel_err = abs(self.freqs_fem[3] - self.freqs_ref[3]) / self.freqs_ref[3]
        assert rel_err < 0.01, (
            f"Mode 4: FEM {self.freqs_fem[3]:.6f} Hz, "
            f"analytical {self.freqs_ref[3]:.6f} Hz, "
            f"error {rel_err*100:.3f} %"
        )

    def test_mode5_within_2pct(self):
        """Higher modes converge more slowly; 2 % is acceptable at 40 elements."""
        rel_err = abs(self.freqs_fem[4] - self.freqs_ref[4]) / self.freqs_ref[4]
        assert rel_err < 0.02, (
            f"Mode 5: FEM {self.freqs_fem[4]:.6f} Hz, "
            f"analytical {self.freqs_ref[4]:.6f} Hz, "
            f"error {rel_err*100:.3f} %"
        )


class TestCantileverMeshConvergence:
    """Finer meshes converge monotonically toward the analytical solution.

    Hermite cubic elements are very accurate for the lowest modes even on coarse
    meshes (mode 1 error < 0.01 % at 5 elements), so convergence is tested on
    mode 5 where the discretisation error is clearly above floating-point noise
    across all mesh sizes considered here.
    """

    # Mode 5 shows clean h⁴ convergence for mesh sizes 5–40 elements.
    MODE_IDX = 4   # 0-based index of mode 5

    def _error(self, nselt: int) -> float:
        f_exact = _analytical_frequencies(5)[self.MODE_IDX]
        freqs = _fem_frequencies(nselt, 5)
        return abs(freqs[self.MODE_IDX] - f_exact) / f_exact

    def test_mode5_monotone_convergence(self):
        """Mode-5 error decreases strictly as mesh is refined: 5→10→20→40."""
        errors = {n: self._error(n) for n in (5, 10, 20, 40)}
        assert errors[5]  > errors[10],  "5→10 elements should reduce mode-5 error"
        assert errors[10] > errors[20], "10→20 elements should reduce mode-5 error"
        assert errors[20] > errors[40], "20→40 elements should reduce mode-5 error"

    def test_mode5_coarse_mesh_accuracy(self):
        """10-element mesh achieves < 1 % error on mode 5."""
        rel_err = self._error(10)
        assert rel_err < 0.01, f"10-element mode-5 error {rel_err*100:.3f} % > 1 %"

    def test_mode5_fine_mesh_accuracy(self):
        """40-element mesh achieves < 0.01 % error on mode 5."""
        rel_err = self._error(40)
        assert rel_err < 1e-4, f"40-element mode-5 error {rel_err*100:.4f} % > 0.01 %"

    def test_convergence_rate(self):
        """Hermite cubic elements converge at rate h⁴ for frequency.

        Doubling the mesh should reduce the frequency error by roughly a factor
        of 16 (h⁴).  We verify a factor of ≥ 5 between nselt=10 and nselt=20
        (conservative bound, avoids sensitivity to floating-point noise).
        """
        err_10 = self._error(10)
        err_20 = self._error(20)
        ratio = err_10 / err_20
        assert ratio > 5.0, (
            f"Expected convergence ratio > 5 (h⁴ → ~16), got {ratio:.2f}"
        )
