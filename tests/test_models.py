"""Tests for the public models layer (RotatingBlade, Tower, ModalResult)."""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fem.normalize import NodeModeShape
from pybmodes.models import ModalResult, RotatingBlade, Tower

from ._synthetic_bmi import write_bmi, write_uniform_sec_props

# ===========================================================================
# Synthetic case builders
# ===========================================================================

def _make_blade_case(tmp_path):
    bmi = write_bmi(
        tmp_path / "blade.bmi",
        beam_type=1, radius=50.0, hub_rad=0.0,
        sec_props_file="blade_secs.dat",
    )
    write_uniform_sec_props(tmp_path / "blade_secs.dat")
    return bmi


def _make_tower_case(tmp_path):
    bmi = write_bmi(
        tmp_path / "tower.bmi",
        beam_type=2, radius=80.0, hub_rad=0.0,
        sec_props_file="tower_secs.dat",
        tow_support=0,
    )
    write_uniform_sec_props(tmp_path / "tower_secs.dat",
                             mass_den=5000.0, flp_stff=5.0e10,
                             edge_stff=5.0e10)
    return bmi


# ===========================================================================
# Constructor validation
# ===========================================================================

class TestRotatingBladeValidation:

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RotatingBlade(tmp_path / "does_not_exist.bmi")

    def test_tower_input_rejected(self, tmp_path):
        path = _make_tower_case(tmp_path)
        with pytest.raises(ValueError, match="beam_type=1"):
            RotatingBlade(path)

    def test_str_path_accepted(self, tmp_path):
        path = _make_blade_case(tmp_path)
        blade = RotatingBlade(str(path))
        assert blade is not None


class TestTowerValidation:

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Tower(tmp_path / "does_not_exist.bmi")

    def test_blade_input_rejected(self, tmp_path):
        path = _make_blade_case(tmp_path)
        with pytest.raises(ValueError, match="beam_type=2"):
            Tower(path)


# ===========================================================================
# Dataclass shapes
# ===========================================================================

class TestModalResult:

    def test_construct_directly(self):
        result = ModalResult(frequencies=np.array([0.5, 1.5]), shapes=[])
        assert result.frequencies.shape == (2,)
        assert result.shapes == []

    def test_field_types(self):
        result = ModalResult(frequencies=np.array([1.0]), shapes=[])
        assert isinstance(result.frequencies, np.ndarray)
        assert isinstance(result.shapes, list)


class TestNodeModeShape:

    def test_construct(self):
        n = 5
        span = np.linspace(0.0, 1.0, n)
        shape = NodeModeShape(
            mode_number=1, freq_hz=1.5, span_loc=span,
            flap_disp=np.zeros(n), flap_slope=np.zeros(n),
            lag_disp=np.zeros(n), lag_slope=np.zeros(n),
            twist=np.zeros(n),
        )
        assert shape.mode_number == 1
        assert shape.freq_hz == 1.5
        assert len(shape.span_loc) == n


# ===========================================================================
# Synthetic end-to-end smoke
# ===========================================================================

class TestRunSmokeBlade:

    def test_run_returns_modal_result(self, tmp_path):
        path = _make_blade_case(tmp_path)
        result = RotatingBlade(path).run(n_modes=3)
        assert isinstance(result, ModalResult)
        assert result.frequencies.shape == (3,)
        assert len(result.shapes) == 3
        assert all(isinstance(s, NodeModeShape) for s in result.shapes)

    def test_frequencies_positive_and_sorted(self, tmp_path):
        path = _make_blade_case(tmp_path)
        result = RotatingBlade(path).run(n_modes=4)
        assert np.all(result.frequencies > 0.0)
        assert np.all(np.diff(result.frequencies) >= 0)


class TestRunSmokeTower:

    def test_run_returns_modal_result(self, tmp_path):
        path = _make_tower_case(tmp_path)
        result = Tower(path).run(n_modes=3)
        assert isinstance(result, ModalResult)
        assert len(result.shapes) == 3

    def test_frequencies_positive(self, tmp_path):
        path = _make_tower_case(tmp_path)
        result = Tower(path).run(n_modes=3)
        assert np.all(result.frequencies > 0.0)
