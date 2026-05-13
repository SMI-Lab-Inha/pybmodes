"""Regression tests for the post-1.0 review findings.

Three bugs caught after the v1.0.0 release:

1. **Roll / pitch inertia swap in ``_platform_inertia_matrix``** (high).
   The DOF order is OpenFAST ``[surge, sway, heave, roll, pitch, yaw]``
   per :mod:`pybmodes.coords` and consumed by
   :func:`pybmodes.fem.nondim.nondim_platform`. The original code
   assigned ``PtfmPIner`` to slot 3 and ``PtfmRIner`` to slot 4 —
   invisible on OC3 (roll inertia equals pitch inertia by axisymmetry)
   but a real bug on any asymmetric semi or submersible.

2. **BMI ``sec_props_file`` Windows-path normalisation** (medium).
   The ElastoDyn parser already rewrites Windows-style backslashes to
   forward slashes in ``TwrFile`` / ``BldFile``; the BMI parser
   didn't, so a BMI authored on Windows with ``subdir\\props.dat``
   failed on Linux / macOS.

3. **Campbell tower "too few modes" defensive guard** (low / medium).
   The blade sweep already raised a friendly ``RuntimeError`` when
   the FEM solver returned fewer modes than requested; the tower
   path silently broadcast-failed with a cryptic shape error.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pytest

from pybmodes.models.tower import _platform_inertia_matrix
from tests._synthetic_bmi import write_bmi, write_uniform_sec_props

# ---------------------------------------------------------------------------
# #1 — Platform inertia matrix DOF order
# ---------------------------------------------------------------------------

class TestPlatformInertiaMatrixDofOrder:
    """``_platform_inertia_matrix`` must place ``PtfmRIner`` at slot 3
    (roll) and ``PtfmPIner`` at slot 4 (pitch) — the OpenFAST 6-DOF
    convention. Pre-1.0 review caught this latent swap.
    """

    def test_distinct_inertias_land_in_correct_slots(self) -> None:
        """OC3 has ``PtfmRIner == PtfmPIner`` by axisymmetry; an
        asymmetric semi or submersible has them distinct. We use
        deliberately distinct values to expose any swap.
        """
        # The full set of fields _scan_platform_fields returns.
        # ``Ptfm*Stiff`` and ``PtfmCM*`` aren't read by the matrix
        # assembler but are part of the contract — include them so
        # the test mirrors real upstream usage.
        ptfm = {
            "PtfmMass": 1.5e7,
            "PtfmRIner": 1.0e10,  # roll inertia
            "PtfmPIner": 3.0e10,  # pitch inertia — deliberately different
            "PtfmYIner": 5.0e10,  # yaw inertia
            "PtfmCMxt": 0.0, "PtfmCMyt": 0.0, "PtfmCMzt": -20.0,
            "PtfmRefzt": 0.0,
            "PtfmSurgeStiff": 0.0, "PtfmSwayStiff": 0.0,
            "PtfmHeaveStiff": 0.0, "PtfmRollStiff": 0.0,
            "PtfmPitchStiff": 0.0, "PtfmYawStiff": 0.0,
        }
        i_mat = _platform_inertia_matrix(ptfm)
        # Translational mass at slots 0–2.
        assert i_mat[0, 0] == ptfm["PtfmMass"], "surge mass slot"
        assert i_mat[1, 1] == ptfm["PtfmMass"], "sway mass slot"
        assert i_mat[2, 2] == ptfm["PtfmMass"], "heave mass slot"
        # Rotational inertias in OpenFAST DOF order: roll(3) /
        # pitch(4) / yaw(5). The earlier code had RIner ↔ PIner
        # swapped — this is the load-bearing assertion.
        assert i_mat[3, 3] == ptfm["PtfmRIner"], (
            "i_matrix[3, 3] (roll, DOF 3) must equal PtfmRIner"
        )
        assert i_mat[4, 4] == ptfm["PtfmPIner"], (
            "i_matrix[4, 4] (pitch, DOF 4) must equal PtfmPIner"
        )
        assert i_mat[5, 5] == ptfm["PtfmYIner"], "yaw slot"

    def test_diagonal_only_no_cross_coupling(self) -> None:
        """No surge↔pitch or sway↔roll cross-coupling on the at-CM
        matrix. The rigid-arm transfer to the tower base happens
        downstream in ``nondim_platform``; adding it here would
        double-count.
        """
        ptfm = {
            "PtfmMass": 1.0e7, "PtfmRIner": 1.0e9, "PtfmPIner": 2.0e9,
            "PtfmYIner": 3.0e9, "PtfmCMxt": 0.0, "PtfmCMyt": 0.0,
            "PtfmCMzt": -10.0, "PtfmRefzt": 0.0,
            "PtfmSurgeStiff": 0.0, "PtfmSwayStiff": 0.0,
            "PtfmHeaveStiff": 0.0, "PtfmRollStiff": 0.0,
            "PtfmPitchStiff": 0.0, "PtfmYawStiff": 0.0,
        }
        i_mat = _platform_inertia_matrix(ptfm)
        # Pull the off-diagonal mask and check every off-diagonal is 0.
        off_diag = i_mat - np.diag(np.diag(i_mat))
        assert np.all(off_diag == 0.0), "no cross-coupling on at-CM matrix"


# ---------------------------------------------------------------------------
# #4 — BMI sec_props_file Windows-path normalisation
# ---------------------------------------------------------------------------

def test_bmi_sec_props_file_normalises_windows_backslashes(
    tmp_path: pathlib.Path,
) -> None:
    """A BMI authored on Windows with ``subdir\\props.dat`` must
    resolve correctly on Linux / macOS. The BMI parser strips quotes
    and rewrites backslashes to forward slashes — same convention as
    the ElastoDyn parser in
    :func:`pybmodes.io._elastodyn.parser._normalise_subfile_path`.
    """
    from pybmodes.io.bmi import read_bmi

    subdir = tmp_path / "props_sub"
    subdir.mkdir()
    sec_props_path = subdir / "props.dat"
    write_uniform_sec_props(sec_props_path)
    bmi_path = tmp_path / "tower.bmi"
    # Author the deck with Windows-style backslashes in the sec_props
    # reference — the bug is that this used to be stored verbatim and
    # ``pathlib.Path("props_sub\\props.dat")`` on POSIX treats the
    # whole string as a single filename.
    write_bmi(
        bmi_path, beam_type=2, radius=90.0, hub_rad=0.0, hub_conn=1,
        sec_props_file=r"props_sub\props.dat",
        n_elements=10, tip_mass=200_000.0,
    )
    parsed = read_bmi(bmi_path)
    # Normalised form is stored on the dataclass.
    assert parsed.sec_props_file == "props_sub/props.dat"
    # ``resolve_sec_props_path`` lands on the real file regardless of
    # platform — pathlib joins ``props_sub/props.dat`` correctly on
    # both Windows and POSIX.
    resolved = parsed.resolve_sec_props_path()
    assert resolved.is_file(), f"resolved path {resolved} should exist"
    assert resolved == sec_props_path.resolve()


# ---------------------------------------------------------------------------
# #5 — Campbell tower "too few modes" defensive guard
# ---------------------------------------------------------------------------

@dataclass
class _StubModalResult:
    """Tiny stand-in for :class:`~pybmodes.models.result.ModalResult`
    that lets us simulate the rare general-eig fallback returning
    fewer modes than requested. The Campbell tower path only inspects
    ``frequencies`` and ``shapes`` from this object on the too-few-
    modes branch."""

    frequencies: np.ndarray
    shapes: list = None  # type: ignore[assignment]


def test_campbell_tower_too_few_modes_raises_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the FEM solver returns fewer modes than requested on the
    tower path, the sweep raises a friendly ``RuntimeError`` rather
    than letting ``np.broadcast_to`` fail with a cryptic shape error.
    Mirrors the existing defensive guard on the blade path.
    """
    from pybmodes import campbell as cb

    requested = 4

    def fake_run_fem(bmi, *, n_modes, sp):  # noqa: ARG001
        # Simulate the asymmetric-K / general-eig fallback returning
        # fewer modes than requested (NaN-dropped eigenvalues).
        return _StubModalResult(
            frequencies=np.array([0.5, 1.2]),  # 2 < requested 4
            shapes=[],
        )

    monkeypatch.setattr(cb, "run_fem", fake_run_fem)

    # _solve_tower_once takes (tower, n_modes, n_steps) — pass a
    # placeholder pair for the tower since fake_run_fem ignores both.
    @dataclass
    class _StubBMI:
        rot_rpm: float = 0.0

    with pytest.raises(RuntimeError, match="too few|only \\d+ of"):
        cb._solve_tower_once((_StubBMI(), None), requested, n_steps=5)
