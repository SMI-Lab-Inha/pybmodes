"""Tower: high-level model for a wind-turbine tower."""

from __future__ import annotations

import pathlib
import warnings
from typing import TYPE_CHECKING

from pybmodes.io.bmi import read_bmi
from pybmodes.io.sec_props import SectionProperties
from pybmodes.models._pipeline import run_fem
from pybmodes.models.result import ModalResult

if TYPE_CHECKING:
    from pybmodes.elastodyn.validate import ValidationResult


def _scan_platform_fields(dat_path: pathlib.Path) -> dict[str, float]:
    """Scan an ElastoDyn ``.dat`` for the eight platform scalars used to
    assemble a floating ``PlatformSupport`` (``PtfmMass`` /
    ``PtfmRIner`` / ``PtfmPIner`` / ``PtfmYIner`` / ``PtfmCMxt`` /
    ``PtfmCMyt`` / ``PtfmCMzt`` / ``PtfmRefzt``).

    The full ElastoDyn parser in :mod:`pybmodes.io._elastodyn` doesn't
    surface these (they're irrelevant for the cantilever path); this
    helper is a tiny shim used by :meth:`Tower.from_elastodyn_with_mooring`
    to avoid extending the main parser for a single use case. Missing
    fields default to ``0.0`` (the BMI consumer will fail downstream
    if a critical field is genuinely absent).
    """
    fields: dict[str, float] = {
        "PtfmMass": 0.0, "PtfmRIner": 0.0, "PtfmPIner": 0.0,
        "PtfmYIner": 0.0, "PtfmCMxt": 0.0, "PtfmCMyt": 0.0,
        "PtfmCMzt": 0.0, "PtfmRefzt": 0.0,
    }
    with pathlib.Path(dat_path).open(
        "r", encoding="utf-8", errors="replace",
    ) as fh:
        for raw in fh:
            parts = raw.split()
            if len(parts) < 2:
                continue
            value, label = parts[0], parts[1]
            if label in fields:
                try:
                    fields[label] = float(value)
                except ValueError:
                    pass
    return fields


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

    # Populated by ``from_elastodyn(..., validate_coeffs=True)``;
    # ``None`` when the constructor didn't run validation. Declared
    # at class scope so mypy sees the attribute on instances built
    # via ``cls.__new__(cls)`` (the from_elastodyn path bypasses
    # ``__init__``).
    coeff_validation: "ValidationResult | None" = None

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
    def from_elastodyn_with_mooring(
        cls,
        main_dat_path: str | pathlib.Path,
        moordyn_dat_path: str | pathlib.Path,
        hydrodyn_dat_path: str | pathlib.Path | None = None,
    ) -> "Tower":
        """Build a free-free floating tower model with a populated
        :class:`~pybmodes.io.bmi.PlatformSupport` block.

        Assembles the platform-support 6 × 6 matrices from three OpenFAST
        decks:

        - **Mooring stiffness** ``K_moor`` from a MoorDyn ``.dat`` (parsed
          via :class:`pybmodes.mooring.MooringSystem.from_moordyn` and
          linearised at zero offset).
        - **Hydrodynamic added mass** ``A_inf`` and **hydrostatic
          restoring** ``C_hst`` from a HydroDyn ``.dat`` (parsed via
          :class:`pybmodes.io.HydroDynReader`, which follows ``PotFile``
          to the WAMIT ``.1`` and ``.hst`` files). Optional — if
          ``hydrodyn_dat_path`` is omitted, both default to zero, so
          the resulting model couples only mooring + platform inertia.
        - **Platform inertia** from the ``PtfmMass`` / ``PtfmRIner`` /
          ``PtfmPIner`` / ``PtfmYIner`` / ``PtfmCM*`` / ``PtfmRefzt``
          scalars in the ElastoDyn main file; the 6 × 6 ``i_matrix`` is
          assembled with parallel-axis terms transferring rotational
          inertia from the platform CM to the body origin
          (``PtfmRefzt``).

        Sets ``hub_conn = 2`` (free-free floating base) and
        ``tow_support = 1`` (inline platform-support block).

        Notes
        -----
        For ElastoDyn polynomial-coefficient generation use the standard
        cantilever :meth:`Tower.from_elastodyn` instead — the polynomial
        ansatz lives in a clamped-base frame regardless of platform
        configuration (see ``cases/ECOSYSTEM_FINDING.md``). This method
        is for coupled-frequency prediction only.
        """
        import numpy as np

        from pybmodes.io._elastodyn.adapter import to_pybmodes_tower
        from pybmodes.io.bmi import PlatformSupport
        from pybmodes.io.elastodyn_reader import (
            read_elastodyn_blade,
            read_elastodyn_main,
            read_elastodyn_tower,
        )
        from pybmodes.mooring import MooringSystem

        main_dat_path = pathlib.Path(main_dat_path)
        moordyn_dat_path = pathlib.Path(moordyn_dat_path)

        main = read_elastodyn_main(main_dat_path)
        tower = read_elastodyn_tower(main_dat_path.parent / main.twr_file)
        blade = None
        if main.bld_file[0]:
            bld_path = main_dat_path.parent / main.bld_file[0]
            if bld_path.is_file():
                blade = read_elastodyn_blade(bld_path)
        bmi, sp = to_pybmodes_tower(main, tower, blade)

        ptfm = _scan_platform_fields(main_dat_path)

        moor_sys = MooringSystem.from_moordyn(moordyn_dat_path)
        K_moor = moor_sys.stiffness_matrix(np.zeros(6))

        A_inf = np.zeros((6, 6))
        C_hst = np.zeros((6, 6))
        if hydrodyn_dat_path is not None:
            from pybmodes.io.wamit_reader import HydroDynReader
            wamit = HydroDynReader(hydrodyn_dat_path).read_platform_matrices()
            A_inf = wamit.A_inf
            C_hst = wamit.C_hst

        M = ptfm["PtfmMass"]
        I_R = ptfm["PtfmRIner"]
        I_P = ptfm["PtfmPIner"]
        I_Y = ptfm["PtfmYIner"]
        # Parallel-axis transfer of rotational inertia from CM (at
        # PtfmCMzt) to the body origin (at PtfmRefzt). For OC3 the CM is
        # well below the reference point so ``dz`` is large and the
        # parallel-axis contribution can dwarf ``I_R`` / ``I_P``.
        dz = ptfm["PtfmCMzt"] - ptfm["PtfmRefzt"]
        i_mat = np.zeros((6, 6))
        i_mat[0, 0] = M
        i_mat[1, 1] = M
        i_mat[2, 2] = M
        i_mat[3, 3] = I_R + M * dz * dz
        i_mat[4, 4] = I_P + M * dz * dz
        i_mat[5, 5] = I_Y
        i_mat[0, 4] = +M * dz
        i_mat[4, 0] = +M * dz
        i_mat[1, 3] = -M * dz
        i_mat[3, 1] = -M * dz

        platform_support = PlatformSupport(
            draft=max(0.0, -ptfm["PtfmRefzt"]),
            cm_pform=ptfm["PtfmCMzt"],
            mass_pform=M,
            i_matrix=i_mat,
            ref_msl=ptfm["PtfmRefzt"],
            hydro_M=A_inf,
            hydro_K=C_hst,
            mooring_K=K_moor,
            distr_m_z=np.zeros(0),
            distr_m=np.zeros(0),
            distr_k_z=np.zeros(0),
            distr_k=np.zeros(0),
        )

        bmi.hub_conn = 2
        bmi.tow_support = 1
        bmi.support = platform_support

        obj = cls.__new__(cls)
        obj._bmi = bmi
        obj._sp = sp
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

    def run(
        self, n_modes: int = 20, *, check_model: bool = True
    ) -> ModalResult:
        """Solve the eigenvalue problem and return frequencies + mode shapes.

        Parameters
        ----------
        n_modes : number of modes to extract (must be >= 1; default 20).
        check_model : run :func:`pybmodes.checks.check_model` before the
            solve (default ``True``). WARN and ERROR findings are
            emitted as ``UserWarning``; INFO findings are silent (call
            ``pybmodes.checks.check_model(model)`` explicitly to see
            those). Pass ``check_model=False`` to skip the pre-solve
            checks for scripted callers that have already validated
            their inputs.

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
        if check_model:
            from pybmodes.checks import check_model as _check_model
            for w in _check_model(self, n_modes=n_modes):
                if w.severity != "INFO":
                    warnings.warn(str(w), UserWarning, stacklevel=2)
        return run_fem(self._bmi, n_modes=n_modes, sp=self._sp)
