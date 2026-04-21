"""Tests for the models layer: RotatingBlade and Tower."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pybmodes.fem.normalize import NodeModeShape
from pybmodes.io.out_parser import read_out
from pybmodes.models import ModalResult, RotatingBlade, Tower

CERT_DIR     = pathlib.Path(__file__).parent / "data" / "certtest"
REF_DIR      = CERT_DIR / "expected"
OFFSHORE_DIR = pathlib.Path(__file__).parent / "data" / "offshore"


@pytest.mark.integration
class TestRotatingBlade:
    """RotatingBlade wraps CertTest01 (rotating blade, no tip mass)."""

    @pytest.fixture(autouse=True)
    def compute(self):
        blade = RotatingBlade(CERT_DIR / "Test01_nonunif_blade.bmi")
        self.result = blade.run(n_modes=20)
        self.ref    = read_out(REF_DIR / "Test01_nonunif_blade.out")

    def test_returns_modal_result(self):
        assert isinstance(self.result, ModalResult)

    def test_frequencies_array(self):
        assert isinstance(self.result.frequencies, np.ndarray)
        assert len(self.result.frequencies) == 20

    def test_shapes_list(self):
        assert len(self.result.shapes) == 20
        assert isinstance(self.result.shapes[0], NodeModeShape)

    def test_first_5_modes_within_05pct(self):
        ref_freqs = self.ref.frequencies()[:5]
        for k in range(5):
            assert self.result.frequencies[k] == pytest.approx(ref_freqs[k], rel=5e-3), \
                f"mode {k+1}: got {self.result.frequencies[k]:.4f} Hz, ref {ref_freqs[k]:.4f} Hz"

    def test_wrong_beam_type_raises(self):
        with pytest.raises(ValueError, match="beam_type=1"):
            RotatingBlade(CERT_DIR / "Test03_tower.bmi")


@pytest.mark.integration
class TestRotatingBladeWithTipMass:
    """RotatingBlade wraps CertTest02 (rotating blade, 40 kg tip mass)."""

    @pytest.fixture(autouse=True)
    def compute(self):
        blade = RotatingBlade(CERT_DIR / "Test02_blade_with_tip_mass.bmi")
        self.result = blade.run(n_modes=20)
        self.ref    = read_out(REF_DIR / "Test02_blade_with_tip_mass.out")

    def test_first_5_modes_within_05pct(self):
        ref_freqs = self.ref.frequencies()[:5]
        for k in range(5):
            assert self.result.frequencies[k] == pytest.approx(ref_freqs[k], rel=5e-3), \
                f"mode {k+1}: got {self.result.frequencies[k]:.4f} Hz, ref {ref_freqs[k]:.4f} Hz"


@pytest.mark.integration
class TestTower:
    """Tower wraps CertTest03 (onshore tower, top mass)."""

    @pytest.fixture(autouse=True)
    def compute(self):
        tower = Tower(CERT_DIR / "Test03_tower.bmi")
        self.result = tower.run(n_modes=20)
        self.ref    = read_out(REF_DIR / "Test03_tower.out")

    def test_returns_modal_result(self):
        assert isinstance(self.result, ModalResult)

    def test_first_4_modes_within_05pct(self):
        ref_freqs = self.ref.frequencies()[:4]
        for k in range(4):
            assert self.result.frequencies[k] == pytest.approx(ref_freqs[k], rel=5e-3), \
                f"mode {k+1}: got {self.result.frequencies[k]:.4f} Hz, ref {ref_freqs[k]:.4f} Hz"

    def test_wrong_beam_type_raises(self):
        with pytest.raises(ValueError, match="beam_type=2"):
            Tower(CERT_DIR / "Test01_nonunif_blade.bmi")


@pytest.mark.integration
class TestWireTower:
    """Tower wraps CertTest04 (onshore tower with tension wires)."""

    @pytest.fixture(autouse=True)
    def compute(self):
        tower = Tower(CERT_DIR / "Test04_wires_supported_tower.bmi")
        self.result = tower.run(n_modes=20)
        self.ref    = read_out(REF_DIR / "Test04_wires_supported_tower.out")

    def test_first_4_modes_within_05pct(self):
        ref_freqs = self.ref.frequencies()[:4]
        for k in range(4):
            assert self.result.frequencies[k] == pytest.approx(ref_freqs[k], rel=5e-3), \
                f"mode {k+1}: got {self.result.frequencies[k]:.4f} Hz, ref {ref_freqs[k]:.4f} Hz"


@pytest.mark.integration
class TestOC3Hywind:
    """Tower with floating spar platform (hub_conn=2 free-free, OC3Hywind)."""

    @pytest.fixture(autouse=True)
    def compute(self):
        tower = Tower(OFFSHORE_DIR / "OC3Hywind.bmi")
        self.result = tower.run(n_modes=20)
        self.ref    = read_out(OFFSHORE_DIR / "OC3Hywind.out")

    def test_returns_modal_result(self):
        assert isinstance(self.result, ModalResult)

    def test_first_4_modes_within_05pct(self):
        ref_freqs = self.ref.frequencies()[:4]
        for k in range(4):
            assert self.result.frequencies[k] == pytest.approx(ref_freqs[k], rel=5e-3), \
                f"mode {k+1}: got {self.result.frequencies[k]:.6f} Hz, ref {ref_freqs[k]:.6f} Hz"


@pytest.mark.integration
class TestCSMonopile:
    """Tower with bottom-fixed monopile (hub_conn=3 axial+torsion BC)."""

    @pytest.fixture(autouse=True)
    def compute(self):
        tower = Tower(OFFSHORE_DIR / "CS_Monopile.bmi")
        self.result = tower.run(n_modes=20)
        self.ref    = read_out(OFFSHORE_DIR / "CS_Monopile.out")

    def test_returns_modal_result(self):
        assert isinstance(self.result, ModalResult)

    def test_first_4_modes_within_05pct(self):
        ref_freqs = self.ref.frequencies()[:4]
        for k in range(4):
            assert self.result.frequencies[k] == pytest.approx(ref_freqs[k], rel=5e-3), \
                f"mode {k+1}: got {self.result.frequencies[k]:.6f} Hz, ref {ref_freqs[k]:.6f} Hz"
