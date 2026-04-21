"""FEM core tests for quadrature, connectivity, and CertTest integration."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pybmodes.fem.boundary import NEDOF, build_connectivity, n_free_dof
from pybmodes.fem.gauss import gauss_5pt, gauss_6pt
from pybmodes.io.bmi import read_bmi
from pybmodes.io.out_parser import read_out
from pybmodes.models._pipeline import run_fem


class TestGauss6pt:
    def setup_method(self):
        self.gqp, self.gqw = gauss_6pt()

    def test_n_points(self):
        assert len(self.gqp) == 6
        assert len(self.gqw) == 6

    def test_weights_sum_to_one(self):
        assert np.sum(self.gqw) == pytest.approx(1.0, rel=1e-12)

    def test_points_in_unit_interval(self):
        assert np.all(self.gqp > 0.0)
        assert np.all(self.gqp < 1.0)

    def test_symmetric_about_half(self):
        # Points and weights should be symmetric about 0.5.
        gqp_s = np.sort(self.gqp)
        assert gqp_s[0] + gqp_s[5] == pytest.approx(1.0, abs=1e-12)
        assert gqp_s[1] + gqp_s[4] == pytest.approx(1.0, abs=1e-12)
        assert gqp_s[2] + gqp_s[3] == pytest.approx(1.0, abs=1e-12)

    def test_integrates_quadratic_exactly(self):
        # Integral from 0 to 1 of x^2 dx = 1/3.
        gqp, gqw = gauss_6pt()
        result = np.sum(gqw * gqp ** 2)
        assert result == pytest.approx(1.0 / 3.0, rel=1e-10)

    def test_integrates_degree11_exactly(self):
        # 6-point rule is exact for degree <= 11; integral of x^11 dx = 1/12.
        gqp, gqw = gauss_6pt()
        result = np.sum(gqw * gqp ** 11)
        assert result == pytest.approx(1.0 / 12.0, rel=1e-8)


class TestGauss5pt:
    def test_weights_sum_to_one(self):
        _, gqw = gauss_5pt()
        assert np.sum(gqw) == pytest.approx(1.0, rel=1e-12)

    def test_n_points(self):
        gqp, gqw = gauss_5pt()
        assert len(gqp) == len(gqw) == 5


class TestConnectivity:
    def test_ngd_formula(self):
        for n in (1, 5, 12, 20):
            assert n_free_dof(n) == 9 * n

    def test_shape(self):
        indeg = build_connectivity(5)
        assert indeg.shape == (NEDOF, 5)

    def test_tip_element_axial_tip(self):
        # Local DOF 4 (0-based: 3) of element 0 (tip) maps to global DOF 1.
        indeg = build_connectivity(3)
        assert indeg[3, 0] == 1

    def test_root_dofs_zeroed(self):
        nselt = 4
        indeg = build_connectivity(nselt)
        for j in [0, 4, 5, 8, 9, 12]:
            assert indeg[j, nselt - 1] == 0, f"root DOF {j} not zeroed"

    def test_shared_node_flap_disp(self):
        nselt = 3
        indeg = build_connectivity(nselt)
        assert indeg[4, 0] == 11
        assert indeg[6, 1] == 11


CERT_DIR = pathlib.Path(__file__).parent / "data" / "certtest"
REF_DIR = CERT_DIR / "expected"


def _run_fem(bmi_path: pathlib.Path, n_modes: int = 20):
    """Run the shared production FEM pipeline and return modal frequencies."""
    bmi = read_bmi(bmi_path)
    return run_fem(bmi, n_modes=n_modes).frequencies


@pytest.mark.integration
class TestCertTest01Blade:
    """Test01: non-uniform rotating blade, no tip mass."""

    @pytest.fixture(autouse=True)
    def compute(self):
        bmi_path = CERT_DIR / "Test01_nonunif_blade.bmi"
        ref_path = REF_DIR / "Test01_nonunif_blade.out"
        self.freqs = _run_fem(bmi_path, n_modes=20)
        self.ref = read_out(ref_path)

    def test_mode1_freq(self):
        assert self.freqs[0] == pytest.approx(self.ref[0].frequency, rel=5e-3)

    def test_mode2_freq(self):
        assert self.freqs[1] == pytest.approx(self.ref[1].frequency, rel=5e-3)

    def test_mode3_freq(self):
        assert self.freqs[2] == pytest.approx(self.ref[2].frequency, rel=5e-3)

    def test_first_5_modes_within_05pct(self):
        ref_freqs = self.ref.frequencies()[:5]
        for k in range(5):
            assert self.freqs[k] == pytest.approx(ref_freqs[k], rel=5e-3), (
                f"mode {k+1}: got {self.freqs[k]:.4f} Hz, ref {ref_freqs[k]:.4f} Hz"
            )


@pytest.mark.integration
class TestCertTest02BladeWithTipMass:
    """Test02: non-uniform rotating blade with 40 kg tip mass."""

    @pytest.fixture(autouse=True)
    def compute(self):
        bmi_path = CERT_DIR / "Test02_blade_with_tip_mass.bmi"
        ref_path = REF_DIR / "Test02_blade_with_tip_mass.out"
        self.freqs = _run_fem(bmi_path, n_modes=20)
        self.ref = read_out(ref_path)

    def test_mode1_freq(self):
        assert self.freqs[0] == pytest.approx(self.ref[0].frequency, rel=5e-3)

    def test_mode2_freq(self):
        assert self.freqs[1] == pytest.approx(self.ref[1].frequency, rel=5e-3)

    def test_mode3_freq(self):
        assert self.freqs[2] == pytest.approx(self.ref[2].frequency, rel=5e-3)

    def test_first_5_modes_within_05pct(self):
        ref_freqs = self.ref.frequencies()[:5]
        for k in range(5):
            assert self.freqs[k] == pytest.approx(ref_freqs[k], rel=5e-3), (
                f"mode {k+1}: got {self.freqs[k]:.4f} Hz, ref {ref_freqs[k]:.4f} Hz"
            )


@pytest.mark.integration
class TestCertTest03Tower:
    """Test03: onshore tower, no rotation."""

    @pytest.fixture(autouse=True)
    def compute(self):
        bmi_path = CERT_DIR / "Test03_tower.bmi"
        ref_path = REF_DIR / "Test03_tower.out"
        self.freqs = _run_fem(bmi_path, n_modes=20)
        self.ref = read_out(ref_path)

    def test_mode1_freq(self):
        assert self.freqs[0] == pytest.approx(self.ref[0].frequency, rel=5e-3)

    def test_mode2_freq(self):
        assert self.freqs[1] == pytest.approx(self.ref[1].frequency, rel=5e-3)

    def test_first_4_modes_within_05pct(self):
        ref_freqs = self.ref.frequencies()[:4]
        for k in range(4):
            assert self.freqs[k] == pytest.approx(ref_freqs[k], rel=5e-3), (
                f"mode {k+1}: got {self.freqs[k]:.4f} Hz, ref {ref_freqs[k]:.4f} Hz"
            )


@pytest.mark.integration
class TestCertTest04WireTower:
    """Test04: tower with two tension-wire attachment sets."""

    @pytest.fixture(autouse=True)
    def compute(self):
        bmi_path = CERT_DIR / "Test04_wires_supported_tower.bmi"
        ref_path = REF_DIR / "Test04_wires_supported_tower.out"
        self.freqs = _run_fem(bmi_path, n_modes=20)
        self.ref = read_out(ref_path)

    def test_mode1_freq(self):
        assert self.freqs[0] == pytest.approx(self.ref[0].frequency, rel=5e-3)

    def test_mode2_freq(self):
        assert self.freqs[1] == pytest.approx(self.ref[1].frequency, rel=5e-3)

    def test_first_4_modes_within_05pct(self):
        ref_freqs = self.ref.frequencies()[:4]
        for k in range(4):
            assert self.freqs[k] == pytest.approx(ref_freqs[k], rel=5e-3), (
                f"mode {k+1}: got {self.freqs[k]:.4f} Hz, ref {ref_freqs[k]:.4f} Hz"
            )
