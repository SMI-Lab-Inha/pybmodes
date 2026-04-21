"""Tower: high-level model for a wind-turbine tower."""

from __future__ import annotations

import pathlib

from pybmodes.io.bmi import read_bmi
from pybmodes.models._pipeline import run_fem
from pybmodes.models.result import ModalResult


class Tower:
    """Compute natural frequencies and mode shapes for a tower.

    Parameters
    ----------
    bmi_path : path to the .bmi input file (beam_type must be 2).
    """

    def __init__(self, bmi_path: str | pathlib.Path) -> None:
        self._bmi = read_bmi(bmi_path)
        if self._bmi.beam_type != 2:
            raise ValueError(
                f"Tower requires beam_type=2, got {self._bmi.beam_type}"
            )

    def run(self, n_modes: int = 20) -> ModalResult:
        """Solve the eigenvalue problem and return frequencies + mode shapes.

        Parameters
        ----------
        n_modes : number of modes to extract (default 20).
        """
        return run_fem(self._bmi, n_modes=n_modes)
