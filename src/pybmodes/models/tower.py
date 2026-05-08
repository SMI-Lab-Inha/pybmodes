"""Tower: high-level model for a wind-turbine tower."""

from __future__ import annotations

import pathlib
import warnings

from pybmodes.io.bmi import read_bmi
from pybmodes.io.sec_props import SectionProperties
from pybmodes.models._pipeline import run_fem
from pybmodes.models.result import ModalResult


def _run_validation_and_warn(main_dat_path: pathlib.Path):
    """Validate coefficient blocks in an ElastoDyn deck and warn on issues.

    Helper shared by ``Tower.from_elastodyn`` and
    ``RotatingBlade.from_elastodyn``. Returns the
    :class:`~pybmodes.elastodyn.ValidationResult`. Emits a
    :class:`UserWarning` if the overall verdict is WARN or FAIL, with
    per-block details for FAIL.
    """
    from pybmodes.elastodyn.validate import validate_dat_coefficients

    result = validate_dat_coefficients(main_dat_path)
    failing = result.failing_blocks()
    warning = result.warning_blocks()

    if failing:
        details = "\n  ".join(
            f"{b.name}: file_rms={b.file_rms:.4f}, "
            f"pyB_rms={b.pybmodes_rms:.4f}, ratio={b.ratio:.0f}"
            for b in failing
        )
        warnings.warn(
            f"{result.summary}\n  {details}\n  "
            f"Run `pybmodes patch {main_dat_path}` to regenerate.",
            UserWarning,
            stacklevel=3,
        )
    elif warning:
        warnings.warn(result.summary, UserWarning, stacklevel=3)

    return result


class Tower:
    """Compute natural frequencies and mode shapes for a tower.

    Parameters
    ----------
    bmi_path : path to the .bmi input file (beam_type must be 2).
    """

    def __init__(self, bmi_path: str | pathlib.Path) -> None:
        self._bmi = read_bmi(bmi_path)
        self._sp: SectionProperties | None = None
        if self._bmi.beam_type != 2:
            raise ValueError(
                f"Tower requires beam_type=2, got {self._bmi.beam_type}"
            )

    @classmethod
    def from_bmi(cls, bmi_path: str | pathlib.Path) -> "Tower":
        """Build a tower model from a BModes-format ``.bmi`` deck.

        Equivalent to ``Tower(bmi_path)`` — exposed as an explicit
        classmethod so callers can pick the constructor by source format
        symmetrically with :meth:`from_elastodyn` and
        :meth:`from_elastodyn_with_subdyn`.

        The BMI parser already covers all four certtest configurations
        (cantilever blade, blade + tip mass, cantilever tower, tension-
        wire-supported tower) plus the offshore platform-support paths
        (``hub_conn`` ∈ {1, 2, 3}, ``tow_support`` ∈ {0, 1, 2}, with
        ``PlatformSupport`` carrying hydro / mooring / platform-inertia
        6×6 matrices). All of those flow through the standard FEM
        pipeline; this constructor is a thin handle.
        """
        return cls(bmi_path)

    @classmethod
    def from_elastodyn(
        cls,
        main_dat_path: str | pathlib.Path,
        *,
        validate_coeffs: bool = False,
    ) -> "Tower":
        """Build a tower model from an OpenFAST ElastoDyn main ``.dat``.

        The main file is parsed plus the tower file referenced via
        ``TwrFile`` and (when the path is resolvable) the first blade file
        referenced via ``BldFile(1)`` — the latter is read only to compute
        the rotor-mass contribution to the lumped tower-top assembly.

        Parameters
        ----------
        main_dat_path :
            Path to the ElastoDyn main ``.dat`` file.
        validate_coeffs :
            If ``True``, run
            :func:`pybmodes.elastodyn.validate_dat_coefficients` after
            building the model and attach the result as
            ``self.coeff_validation``. Emits a ``UserWarning`` if any
            block fails or warns. Default ``False`` so the standard
            constructor stays cheap.
        """
        from pybmodes.io.elastodyn_reader import (
            read_elastodyn_blade,
            read_elastodyn_main,
            read_elastodyn_tower,
            to_pybmodes_tower,
        )

        main_dat_path = pathlib.Path(main_dat_path)
        main = read_elastodyn_main(main_dat_path)
        tower = read_elastodyn_tower(main_dat_path.parent / main.twr_file)

        blade = None
        if main.bld_file[0]:
            bld_path = main_dat_path.parent / main.bld_file[0]
            if bld_path.is_file():
                blade = read_elastodyn_blade(bld_path)

        bmi, sp = to_pybmodes_tower(main, tower, blade=blade)

        obj = cls.__new__(cls)
        obj._bmi = bmi
        obj._sp = sp
        obj.coeff_validation = None

        if validate_coeffs:
            obj.coeff_validation = _run_validation_and_warn(main_dat_path)

        return obj

    @classmethod
    def from_elastodyn_with_subdyn(
        cls,
        main_dat_path: str | pathlib.Path,
        subdyn_dat_path: str | pathlib.Path,
    ) -> "Tower":
        """Build a combined pile + tower cantilever from an ElastoDyn deck
        plus a SubDyn substructure file.

        The pile geometry comes from the SubDyn file (joints + members +
        circular cross-section properties); the tower above the transition
        piece comes from the ElastoDyn main + tower files. The two are
        spliced into a single cantilever with a clamped base at the
        SubDyn reaction joint (no soil flexibility).

        Designed for OC3-style fixed-base monopiles. Does not handle soil
        springs, hydrodynamic added mass, or non-circular substructure
        members. See :func:`pybmodes.io.subdyn_reader.to_pybmodes_pile_tower`
        for the assembly details.
        """
        from pybmodes.io.elastodyn_reader import (
            read_elastodyn_blade,
            read_elastodyn_main,
            read_elastodyn_tower,
        )
        from pybmodes.io.subdyn_reader import read_subdyn, to_pybmodes_pile_tower

        main_dat_path = pathlib.Path(main_dat_path)
        subdyn_dat_path = pathlib.Path(subdyn_dat_path)

        main = read_elastodyn_main(main_dat_path)
        tower = read_elastodyn_tower(main_dat_path.parent / main.twr_file)
        subdyn = read_subdyn(subdyn_dat_path)

        blade = None
        if main.bld_file[0]:
            bld_path = main_dat_path.parent / main.bld_file[0]
            if bld_path.is_file():
                blade = read_elastodyn_blade(bld_path)

        bmi, sp = to_pybmodes_pile_tower(main, tower, subdyn, blade=blade)

        obj = cls.__new__(cls)
        obj._bmi = bmi
        obj._sp = sp
        return obj

    def run(self, n_modes: int = 20) -> ModalResult:
        """Solve the eigenvalue problem and return frequencies + mode shapes.

        Parameters
        ----------
        n_modes : number of modes to extract (must be >= 1; default 20).

        Warning
        -------
        ``n_modes`` affects the LAPACK solver path. For symmetric or
        nearly-symmetric towers (``EI_FA ≈ EI_SS`` and small RNA c.m.
        offset), use ``n_modes >= 6``. With ``n_modes <= 4``,
        ``scipy.linalg.eigh`` invokes a subset eigenvalue routine that
        can artificially lift the degeneracy of the 1st FA / SS bending
        pair — the modes come back at slightly different frequencies
        and pre-separated, which prevents the degenerate-pair resolver
        in :mod:`pybmodes.elastodyn.params` from triggering. The
        polynomial fits still succeed, but the FA / SS classification
        may flip relative to a full solve, and downstream
        ``compute_tower_params_report`` may select different modes for
        ``TwFAM1Sh`` / ``TwSSM1Sh`` between runs at different
        ``n_modes``.

        Minimum recommended: ``n_modes >= 6`` for reliable FA / SS
        classification on symmetric structures. The default of 20 is
        safely above this threshold.
        """
        if not isinstance(n_modes, int) or n_modes < 1:
            raise ValueError(f"n_modes must be a positive integer; got {n_modes!r}")
        return run_fem(self._bmi, n_modes=n_modes, sp=self._sp)
