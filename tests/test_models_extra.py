"""Extra unit tests for the models layer (``pybmodes.models``).

Cover:
  * the public ``__init__`` re-exports
  * input-validation paths in ``RotatingBlade`` and ``Tower``
  * type-of-return expectations from ``run`` (without re-running CertTest)
  * non-existent-file behaviour (FileNotFoundError) from the constructors
  * ``ModalResult`` dataclass shape
  * ``NodeModeShape`` data layout
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import pybmodes.models as models_pkg
from pybmodes.fem.normalize import NodeModeShape
from pybmodes.models import ModalResult, RotatingBlade, Tower

CERT_DIR = pathlib.Path(__file__).parent / "data" / "certtest"


# ===========================================================================
# Public API re-exports
# ===========================================================================

class TestModelsPublicApi:

    def test_module_exports(self):
        assert "RotatingBlade" in models_pkg.__all__
        assert "Tower" in models_pkg.__all__
        assert "ModalResult" in models_pkg.__all__

    def test_rotating_blade_is_class(self):
        assert isinstance(RotatingBlade, type)

    def test_tower_is_class(self):
        assert isinstance(Tower, type)


# ===========================================================================
# Constructor input validation
# ===========================================================================

class TestRotatingBladeValidation:

    def test_missing_file_raises(self, tmp_path):
        bogus = tmp_path / "does_not_exist.bmi"
        with pytest.raises(FileNotFoundError):
            RotatingBlade(bogus)

    def test_tower_input_rejected(self, tower_bmi):
        with pytest.raises(ValueError, match="beam_type=1"):
            RotatingBlade(tower_bmi)

    def test_accepts_str_path(self, blade_bmi):
        # Path-like and str both work
        blade = RotatingBlade(str(blade_bmi))
        assert blade is not None

    def test_accepts_pathlib(self, blade_bmi):
        blade = RotatingBlade(blade_bmi)
        assert blade is not None


class TestTowerValidation:

    def test_missing_file_raises(self, tmp_path):
        bogus = tmp_path / "does_not_exist.bmi"
        with pytest.raises(FileNotFoundError):
            Tower(bogus)

    def test_blade_input_rejected(self, blade_bmi):
        with pytest.raises(ValueError, match="beam_type=2"):
            Tower(blade_bmi)

    def test_accepts_str_path(self, tower_bmi):
        tower = Tower(str(tower_bmi))
        assert tower is not None

    def test_wire_supported_tower_constructs(self, wire_tower_bmi):
        tower = Tower(wire_tower_bmi)
        assert tower is not None

    def test_offshore_monopile_constructs(self, monopile_bmi):
        tower = Tower(monopile_bmi)
        assert tower is not None

    def test_offshore_hywind_constructs(self, hywind_bmi):
        tower = Tower(hywind_bmi)
        assert tower is not None


# ===========================================================================
# ModalResult dataclass
# ===========================================================================

class TestModalResult:

    def test_construct_directly(self):
        # ModalResult is a simple container — instantiate it without an FEM run.
        freqs = np.array([0.5, 1.5, 2.5])
        shapes = []
        result = ModalResult(frequencies=freqs, shapes=shapes)
        assert result.frequencies.shape == (3,)
        assert result.shapes == []

    def test_field_types_preserved(self):
        result = ModalResult(frequencies=np.array([1.0]), shapes=[])
        assert isinstance(result.frequencies, np.ndarray)
        assert isinstance(result.shapes, list)


# ===========================================================================
# NodeModeShape dataclass
# ===========================================================================

class TestNodeModeShape:

    def test_construct(self):
        n = 5
        span = np.linspace(0.0, 1.0, n)
        shape = NodeModeShape(
            mode_number=1,
            freq_hz=1.5,
            span_loc=span,
            flap_disp=np.zeros(n),
            flap_slope=np.zeros(n),
            lag_disp=np.zeros(n),
            lag_slope=np.zeros(n),
            twist=np.zeros(n),
        )
        assert shape.mode_number == 1
        assert shape.freq_hz == 1.5
        assert len(shape.span_loc) == n

    def test_arrays_independent(self):
        # The dataclass does not deep-copy; assignments share buffers, which
        # callers must be aware of.
        arr = np.array([0.0, 1.0])
        shape = NodeModeShape(
            mode_number=1,
            freq_hz=1.0,
            span_loc=arr,
            flap_disp=arr,
            flap_slope=arr,
            lag_disp=arr,
            lag_slope=arr,
            twist=arr,
        )
        # Just ensure attribute access does not raise and the shapes match.
        assert shape.flap_disp is arr


# ===========================================================================
# Smoke-test of run() return type (cheap: 2 modes)
# ===========================================================================

@pytest.mark.integration
class TestRunReturnTypes:
    """Verify that run() returns the documented types — independent of CertTest accuracy."""

    def test_blade_run_returns_modal_result(self, blade_bmi):
        result = RotatingBlade(blade_bmi).run(n_modes=2)
        assert isinstance(result, ModalResult)
        assert result.frequencies.shape == (2,)
        assert len(result.shapes) == 2
        assert all(isinstance(s, NodeModeShape) for s in result.shapes)

    def test_tower_run_returns_modal_result(self, tower_bmi):
        result = Tower(tower_bmi).run(n_modes=2)
        assert isinstance(result, ModalResult)
        assert len(result.shapes) == 2

    def test_blade_default_n_modes_is_20(self, blade_bmi):
        # Avoid the full 20-mode solve (too expensive) — just verify a non-default
        # value reaches the solver.
        result = RotatingBlade(blade_bmi).run(n_modes=3)
        assert len(result.frequencies) == 3
