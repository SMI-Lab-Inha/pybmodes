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
    """Scan an ElastoDyn ``.dat`` for the platform scalars used to
    assemble a floating ``PlatformSupport``.

    Two field groups are extracted:

    - **Geometry / inertia** (``PtfmMass``, ``PtfmRIner``,
      ``PtfmPIner``, ``PtfmYIner``, ``PtfmCMxt``, ``PtfmCMyt``,
      ``PtfmCMzt``, ``PtfmRefzt``) — feed the BMI's ``mass_pform``,
      ``cm_pform``, ``i_matrix``, ``ref_msl``.
    - **Additional linear platform stiffness**
      (``PtfmSurgeStiff``, ``PtfmSwayStiff``, ``PtfmHeaveStiff``,
      ``PtfmRollStiff``, ``PtfmPitchStiff``, ``PtfmYawStiff``) — these
      are ElastoDyn-side springs added on top of HydroDyn / MoorDyn
      contributions. ``PtfmYawStiff`` is how the OC3 spec carries the
      delta-line crowfoot's yaw spring (~ 9.83e7 N·m/rad), which is NOT
      in the MoorDyn ``.dat``. :meth:`Tower.from_elastodyn_with_mooring`
      folds them into the diagonal of ``mooring_K``.

    The full ElastoDyn parser in :mod:`pybmodes.io._elastodyn` doesn't
    surface these (they're irrelevant for the cantilever path); this
    helper is a tiny shim used by :meth:`Tower.from_elastodyn_with_mooring`
    to avoid extending the main parser for a single use case. Missing
    fields default to ``0.0``. Fortran-style D / d exponents
    (``7.466D+06``) are normalised to ``E`` before parsing.
    """
    # Every field this helper extracts is load-bearing: the inertia
    # scalars define the platform's rigid-body mass-matrix
    # contribution, and the ``Ptfm*Stiff`` springs carry restoring
    # contributions that are NOT in HydroDyn or MoorDyn (the OC3
    # delta-line crowfoot yaw spring at ~ 9.83e7 N·m/rad lives in
    # ``PtfmYawStiff`` only). A malformed value silently falling back
    # to 0.0 would produce a physically wrong floating model with no
    # warning, so we raise on any parse failure.
    fields: dict[str, float] = {
        "PtfmMass": 0.0, "PtfmRIner": 0.0, "PtfmPIner": 0.0,
        "PtfmYIner": 0.0, "PtfmCMxt": 0.0, "PtfmCMyt": 0.0,
        "PtfmCMzt": 0.0, "PtfmRefzt": 0.0,
        "PtfmSurgeStiff": 0.0, "PtfmSwayStiff": 0.0,
        "PtfmHeaveStiff": 0.0, "PtfmRollStiff": 0.0,
        "PtfmPitchStiff": 0.0, "PtfmYawStiff": 0.0,
    }
    from pybmodes.io.wamit_reader import _parse_fortran_float

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
                    fields[label] = _parse_fortran_float(value)
                except ValueError as err:
                    raise ValueError(
                        f"Malformed value for {label!r} in "
                        f"{dat_path}: {value!r} cannot be parsed as a "
                        f"float (even with Fortran-style D/d exponent "
                        f"normalisation). The platform model would be "
                        f"physically meaningless or silently lose a "
                        f"restoring contribution without this scalar."
                    ) from err
    return fields


def _platform_inertia_matrix(ptfm: dict[str, float]):
    """Assemble the platform 6×6 inertia matrix AT THE CM in
    **OpenFAST DOF order** ``[surge, sway, heave, roll, pitch, yaw]``
    from the ``Ptfm*`` scalars produced by :func:`_scan_platform_fields`.

    Diagonal-only — translation slots 0–2 carry ``PtfmMass``,
    rotation slots 3 / 4 / 5 carry ``PtfmRIner`` / ``PtfmPIner`` /
    ``PtfmYIner`` respectively. Cross-coupling terms (``[0,4]`` for
    surge-pitch, ``[1,3]`` for sway-roll) are zero on the at-CM
    matrix; the downstream :func:`pybmodes.fem.nondim.nondim_platform`
    applies the rigid-arm CM → tower-base transfer using
    ``cm_pform - draft``, so adding a parallel-axis term here would
    double-count (caught by a pre-1.0 review).

    The DOF order is the canonical convention documented in
    :mod:`pybmodes.coords` and consumed by ``nondim_platform``. A
    pre-1.0 review caught a latent swap (``PtfmPIner`` at slot 3,
    ``PtfmRIner`` at slot 4) that was invisible on OC3 — where roll
    and pitch inertia are equal by symmetry — but would silently
    mis-couple roll and pitch on any asymmetric semi or
    submersible. :func:`tests.test_mooring.test_platform_inertia_matrix_dof_order`
    pins the convention.
    """
    import numpy as np

    i_mat = np.zeros((6, 6))
    i_mat[0, 0] = ptfm["PtfmMass"]    # surge mass
    i_mat[1, 1] = ptfm["PtfmMass"]    # sway  mass
    i_mat[2, 2] = ptfm["PtfmMass"]    # heave mass
    i_mat[3, 3] = ptfm["PtfmRIner"]   # roll  inertia about CM (DOF 3)
    i_mat[4, 4] = ptfm["PtfmPIner"]   # pitch inertia about CM (DOF 4)
    i_mat[5, 5] = ptfm["PtfmYIner"]   # yaw   inertia about CM
    return i_mat


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
          scalars in the ElastoDyn main file. The 6 × 6 ``i_matrix`` is
          stored AT THE CM (no parallel-axis transfer); the downstream
          ``pybmodes.fem.nondim.nondim_platform`` applies the rigid-arm
          transform from CM to tower base using ``cm_pform - draft``.
          ``cm_pform`` and ``draft`` are written in BModes file
          convention (positive distance below MSL; signed draft with
          negative = base above MSL).

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
        # Free-base floating: use physically-scaled section properties.
        # The cantilever proxies (EA ≈ 1e6·EI) wreck the conditioning
        # of the global matrices and, on an asymmetric spar/semi
        # platform, collapse the soft rigid-body modes into an
        # n_modes-dependent degenerate cluster (v1.1.1; the bundled-
        # sample fix, extended here to the in-memory path).
        bmi, sp = to_pybmodes_tower(main, tower, blade, physical_sec_props=True)

        # The cantilever adapter sets ``bmi.radius`` to the flexible
        # tower length (``TowerHt − TowerBsHt``). The floating BMI
        # convention (matching the bundled OC3Hywind.bmi) is
        # ``radius = TowerHt`` paired with ``draft = -TowerBsHt`` so
        # ``radius + draft = flexible length`` after the nondim step
        # in :func:`pybmodes.fem.nondim.make_params`. Overriding the
        # radius here keeps the FEM beam length consistent with the
        # ``draft = -TowerBsHt`` assignment below. Pre-1.0 review.
        # caught this — without the override the flexible length came
        # out as ``TowerHt - 2·TowerBsHt`` (e.g. 67.6 m for OC3
        # instead of 77.6 m).
        bmi.radius = float(main.tower_ht)

        ptfm = _scan_platform_fields(main_dat_path)

        moor_sys = MooringSystem.from_moordyn(moordyn_dat_path)
        K_moor = moor_sys.stiffness_matrix(np.zeros(6))
        # ElastoDyn carries six scalar springs (``PtfmSurgeStiff``,
        # ``PtfmSwayStiff``, ``PtfmHeaveStiff``, ``PtfmRollStiff``,
        # ``PtfmPitchStiff``, ``PtfmYawStiff``) that act *in addition*
        # to whatever HydroDyn / MoorDyn provide at runtime. The OC3
        # delta-line crowfoot is conventionally folded into
        # ``PtfmYawStiff`` (~ 9.83e7 N·m/rad); without including these
        # the coupled-yaw frequency for an OC3-style deck would land
        # an order of magnitude low.
        #
        # DOF order assumption — verified via the canonical OC3Hywind.bmi
        # ``mooring_K[0,4] = -2.821e6`` matching Jonkman (2010) NREL/TP-
        # 500-47535 K_15 surge→pitch coupling: BMI rigid-body matrices
        # are in standard OpenFAST DOF order [surge, sway, heave, roll,
        # pitch, yaw] — the same order as ``MooringSystem.stiffness_matrix()``
        # and as the ElastoDyn ``Ptfm*Stiff`` enumeration below.
        # ``test_oc3hywind_mooring_K_cross_coupling_sign`` pins this
        # invariant so any future DOF-order regression fails loudly
        # rather than silently producing wrong physics on asymmetric
        # platforms.
        for axis, key in enumerate((
            "PtfmSurgeStiff", "PtfmSwayStiff", "PtfmHeaveStiff",
            "PtfmRollStiff", "PtfmPitchStiff", "PtfmYawStiff",
        )):
            K_moor[axis, axis] += ptfm[key]

        A_inf = np.zeros((6, 6))
        C_hst = np.zeros((6, 6))
        if hydrodyn_dat_path is not None:
            from pybmodes.io.wamit_reader import HydroDynReader
            wamit = HydroDynReader(hydrodyn_dat_path).read_platform_matrices()
            A_inf = wamit.A_inf
            C_hst = wamit.C_hst

        M = ptfm["PtfmMass"]
        i_mat = _platform_inertia_matrix(ptfm)

        # BModes file convention for these scalars (see the OC3 Hywind
        # sample BMI in ``src/pybmodes/_examples/sample_inputs/
        # reference_turbines/07_nrel5mw_oc3hywind_spar/``):
        #   ``draft``    — signed depth of the flexible-tower base
        #                  *below* MSL (positive = below; negative =
        #                  above). For OC3 the TP sits at +10 m above
        #                  MSL so ``draft = -10``.
        #   ``cm_pform`` — POSITIVE distance from MSL down to the
        #                  platform CM. For OC3 ``cm_pform = 89.9155``
        #                  (CM at z = −89.9155 in MSL frame).
        #   ``ref_msl``  — positive distance below MSL of the platform
        #                  reference point (usually 0).
        # ElastoDyn stores all three as signed z (positive = above MSL
        # via ``TowerBsHt``; negative = below MSL via ``PtfmCMzt``).
        # The sign flips below translate ElastoDyn → BModes convention.
        platform_support = PlatformSupport(
            draft=-float(main.tower_bs_ht),
            cm_pform=-ptfm["PtfmCMzt"],
            mass_pform=M,
            i_matrix=i_mat,
            ref_msl=-ptfm["PtfmRefzt"],
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
