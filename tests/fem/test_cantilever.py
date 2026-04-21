"""Uniform cantilever beam vs Euler-Bernoulli analytical natural frequencies.

Physics: fixed-free Euler-Bernoulli beam, no rotation, no structural twist.
L = 50 m, EI_flap = 1e6 N*m^2, m = 100 kg/m.

Analytical:
    omega_n = (beta_n L)^2 * sqrt(EI / (m L^4))
    beta_n L ~= [1.87510, 4.69409, 7.85476, 10.99554, 14.13717]

Edge (EI_z = 10^4 x EI_flap) and torsion (GJ large) modes are far above the
tested flap modes so the first n_modes eigenpairs from the FEM are unambiguously
flap bending modes.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fem.assembly import assemble
from pybmodes.fem.nondim import RM, ROMG, make_params
from pybmodes.fem.solver import eigvals_to_hz, solve_modes

L_PHYS = 50.0
EI_PHYS = 1.0e6
M_PHYS = 100.0

_BETA_L = np.array([1.87510407, 4.69409113, 7.85475744, 10.99554073, 14.13716839])


def _analytical_frequencies(n: int = 5) -> np.ndarray:
    """First *n* natural frequencies [Hz] for the uniform cantilever."""
    alpha = np.sqrt(EI_PHYS / (M_PHYS * L_PHYS**4))
    return _BETA_L[:n] ** 2 * alpha / (2.0 * np.pi)


def _build_cantilever(nselt: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (gk, gm) for a uniform cantilever with *nselt* equal elements."""
    nd = make_params(radius=L_PHYS, hub_rad=0.0, rot_rpm=0.0)

    eiy_nd = EI_PHYS / nd.ref4
    eiz_nd = 1.0e4 * eiy_nd
    gj_nd = 1.0e3 * eiy_nd
    eac_nd = 100.0
    rmas_nd = M_PHYS / RM

    eli = 1.0 / nselt
    el = np.full(nselt, eli)
    xb = np.array([1.0 - (i + 1) * eli for i in range(nselt)])

    cfe = np.zeros(nselt)
    eiy_arr = np.full(nselt, eiy_nd)
    eiz_arr = np.full(nselt, eiz_nd)
    gj_arr = np.full(nselt, gj_nd)
    eac_arr = np.full(nselt, eac_nd)
    rmas_arr = np.full(nselt, rmas_nd)
    skm1_arr = np.full(nselt, 1.0e-5)
    skm2_arr = np.full(nselt, 1.0e-5)
    eg_arr = np.zeros(nselt)
    ea_arr = np.zeros(nselt)

    sec_loc = np.array([0.0, 1.0])
    str_tw = np.zeros(2)

    gk, gm, _ = assemble(
        nselt=nselt,
        el=el,
        xb=xb,
        cfe=cfe,
        eiy=eiy_arr,
        eiz=eiz_arr,
        gj=gj_arr,
        eac=eac_arr,
        rmas=rmas_arr,
        skm1=skm1_arr,
        skm2=skm2_arr,
        eg=eg_arr,
        ea=ea_arr,
        omega2=0.0,
        sec_loc=sec_loc,
        str_tw=str_tw,
        hub_conn=1,
    )
    return gk, gm


def _fem_frequencies(nselt: int, n: int) -> np.ndarray:
    """Run the FEM model and return the first *n* frequencies [Hz]."""
    gk, gm = _build_cantilever(nselt)
    eigvals, _ = solve_modes(gk, gm, n_modes=n)
    return eigvals_to_hz(eigvals, ROMG)


class TestAnalyticalFrequencies:
    """Sanity-check the analytical helper."""

    def test_first_mode_positive(self):
        f = _analytical_frequencies(1)
        assert f[0] > 0.0

    def test_frequencies_ascending(self):
        f = _analytical_frequencies(5)
        assert np.all(np.diff(f) > 0.0)

    def test_mode_1_value(self):
        expected = 1.87510407**2 * np.sqrt(1e6 / (100.0 * 50.0**4)) / (2.0 * np.pi)
        assert _analytical_frequencies(1)[0] == pytest.approx(expected, rel=1e-6)


class TestCantileverFlap:
    """First-five flap natural frequencies match Euler-Bernoulli within 0.5%."""

    NSELT = 40
    N_MODES = 5

    def setup_method(self):
        self.freqs_fem = _fem_frequencies(self.NSELT, self.N_MODES)
        self.freqs_ref = _analytical_frequencies(self.N_MODES)

    @pytest.mark.parametrize("k", range(3))
    def test_mode_frequency(self, k: int):
        """Modes 1-3 are within 0.5% of the analytical solution."""
        rel_err = abs(self.freqs_fem[k] - self.freqs_ref[k]) / self.freqs_ref[k]
        assert rel_err < 0.005, (
            f"Mode {k+1}: FEM {self.freqs_fem[k]:.6f} Hz, "
            f"analytical {self.freqs_ref[k]:.6f} Hz, "
            f"error {rel_err*100:.3f} %"
        )

    def test_modes_ascending(self):
        assert np.all(np.diff(self.freqs_fem) > 0.0)

    def test_mode4_within_1pct(self):
        rel_err = abs(self.freqs_fem[3] - self.freqs_ref[3]) / self.freqs_ref[3]
        assert rel_err < 0.01

    def test_mode5_within_2pct(self):
        rel_err = abs(self.freqs_fem[4] - self.freqs_ref[4]) / self.freqs_ref[4]
        assert rel_err < 0.02


class TestCantileverMeshConvergence:
    """Finer meshes converge monotonically toward the analytical solution."""

    MODE_IDX = 4

    def _error(self, nselt: int) -> float:
        f_exact = _analytical_frequencies(5)[self.MODE_IDX]
        freqs = _fem_frequencies(nselt, 5)
        return abs(freqs[self.MODE_IDX] - f_exact) / f_exact

    def test_mode5_monotone_convergence(self):
        """Mode-5 error decreases strictly as mesh is refined: 5->10->20->40."""
        errors = {n: self._error(n) for n in (5, 10, 20, 40)}
        assert errors[5] > errors[10]
        assert errors[10] > errors[20]
        assert errors[20] > errors[40]

    def test_mode5_coarse_mesh_accuracy(self):
        rel_err = self._error(10)
        assert rel_err < 0.01

    def test_mode5_fine_mesh_accuracy(self):
        rel_err = self._error(40)
        assert rel_err < 1e-4

    def test_convergence_rate(self):
        """Hermite cubic elements converge at rate h^4 for frequency."""
        err_10 = self._error(10)
        err_20 = self._error(20)
        ratio = err_10 / err_20
        assert ratio > 5.0, f"Expected convergence ratio > 5 (h^4 -> ~16), got {ratio:.2f}"
