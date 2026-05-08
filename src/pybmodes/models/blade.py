"""RotatingBlade: high-level model for a rotating wind-turbine blade."""

from __future__ import annotations

import pathlib

from pybmodes.io.bmi import read_bmi
from pybmodes.io.sec_props import SectionProperties
from pybmodes.models._pipeline import run_fem
from pybmodes.models.result import ModalResult


class RotatingBlade:
    """Compute natural frequencies and mode shapes for a rotating blade.

    Parameters
    ----------
    bmi_path : path to the .bmi input file (beam_type must be 1).
    """

    def __init__(self, bmi_path: str | pathlib.Path) -> None:
        self._bmi = read_bmi(bmi_path)
        self._sp: SectionProperties | None = None
        if self._bmi.beam_type != 1:
            raise ValueError(
                f"RotatingBlade requires beam_type=1, got {self._bmi.beam_type}"
            )

    @classmethod
    def from_elastodyn(
        cls,
        main_dat_path: str | pathlib.Path,
        *,
        validate_coeffs: bool = False,
    ) -> "RotatingBlade":
        """Build a blade model from an OpenFAST ElastoDyn main ``.dat``.

        The blade .dat is resolved relative to the main file via
        ``BldFile(1)`` (or ``BldFile1`` in the IEA-RWT convention).
        Centrifugal stiffening uses ``RotSpeed`` from the main file.

        Parameters
        ----------
        main_dat_path :
            Path to the ElastoDyn main ``.dat`` file.
        validate_coeffs :
            If ``True``, run
            :func:`pybmodes.elastodyn.validate_dat_coefficients` after
            building the model and attach the result as
            ``self.coeff_validation``. Emits a ``UserWarning`` if any
            block fails or warns. Default ``False``.
        """
        from pybmodes.io.elastodyn_reader import (
            read_elastodyn_blade,
            read_elastodyn_main,
            to_pybmodes_blade,
        )
        from pybmodes.models.tower import _run_validation_and_warn

        main_dat_path = pathlib.Path(main_dat_path)
        main = read_elastodyn_main(main_dat_path)
        bld_path = main_dat_path.parent / main.bld_file[0]
        blade = read_elastodyn_blade(bld_path)

        bmi, sp = to_pybmodes_blade(main, blade)

        obj = cls.__new__(cls)
        obj._bmi = bmi
        obj._sp = sp
        obj.coeff_validation = None

        if validate_coeffs:
            obj.coeff_validation = _run_validation_and_warn(main_dat_path)

        return obj

    def run(self, n_modes: int = 20) -> ModalResult:
        """Solve the eigenvalue problem and return frequencies + mode shapes.

        Parameters
        ----------
        n_modes : number of modes to extract (must be >= 1; default 20).
        """
        if not isinstance(n_modes, int) or n_modes < 1:
            raise ValueError(f"n_modes must be a positive integer; got {n_modes!r}")
        return run_fem(self._bmi, n_modes=n_modes, sp=self._sp)
