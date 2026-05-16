"""Floating-platform rigid-body mode naming (1.3.0).

`ModalResult.mode_labels` names the platform rigid-body modes
(surge / sway / heave / roll / pitch / yaw) for a free-free floating
tower. These tests are self-contained (default suite): the bundled
samples are repo-shipped, same data-independence rule as
`test_floating_samples_spectra`.

Coverage:
1. Bundled floating samples (OC3 Hywind spar, IEA-15 UMaineSemi) —
   the six lowest modes are exactly the six platform DOFs; the
   flexible tower modes above them are unlabelled (`None`).
2. A cantilever / land sample — `mode_labels` is `None` entirely
   (no rigid-body modes; must not be mislabelled).
3. `classify_platform_modes` unit behaviour: a synthetic eigenvector
   whose base node moves in one platform DOF is named that DOF; modes
   beyond the rigid-body count are `None`.
4. `mode_labels` (including `None` entries) round-trips through the
   NPZ and JSON serialisers.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pybmodes.fem.boundary import NESH, active_dof_indices
from pybmodes.fem.platform_modes import classify_platform_modes
from pybmodes.models import Tower
from pybmodes.models.result import ModalResult

_SAMPLES = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src" / "pybmodes" / "_examples" / "sample_inputs"
    / "reference_turbines"
)
_DOF_SET = {"surge", "sway", "heave", "roll", "pitch", "yaw"}


@pytest.mark.parametrize("sample_id", [
    "07_nrel5mw_oc3hywind_spar",
    "09_iea15_umainesemi",
])
def test_floating_sample_rigid_modes_named(sample_id: str) -> None:
    """The six lowest modes of a bundled floating sample are exactly
    the six platform DOFs (each once); modes above are unlabelled."""
    bmi = _SAMPLES / sample_id / f"{sample_id}_tower.bmi"
    res = Tower(bmi).run(n_modes=12, check_model=False)

    assert res.mode_labels is not None
    assert len(res.mode_labels) == len(res.frequencies)

    first6 = res.mode_labels[:6]
    assert all(lbl is not None for lbl in first6), first6
    # Exactly one of each platform DOF among the rigid-body modes.
    assert set(first6) == _DOF_SET, first6
    # Flexible tower modes above the rigid-body cluster are not named.
    assert all(lbl is None for lbl in res.mode_labels[6:]), res.mode_labels[6:]


def test_cantilever_sample_not_mislabelled() -> None:
    """A clamped-base (land) sample has no rigid-body modes — the
    classifier must never run, so mode_labels stays None."""
    bmi = _SAMPLES / "01_nrel5mw_land" / "01_nrel5mw_land_tower.bmi"
    res = Tower(bmi).run(n_modes=6, check_model=False)
    assert res.mode_labels is None


def _one_dof_eigvecs(nselt: int, dof_local: int, n_modes: int) -> np.ndarray:
    """Build compact eigenvectors (free-free → active == all DOFs)
    where mode 0's base node moves purely in FEM base DOF
    ``dof_local`` (0=axial … 5=phi) and other modes are inert."""
    ndt = NESH * nselt + 6
    ev = np.zeros((ndt, n_modes))
    ev[NESH * nselt + dof_local, 0] = 1.0
    return ev


@pytest.mark.parametrize("dof_local,expected", [
    (0, "heave"),   # axial   → heave
    (1, "surge"),   # v_disp  → surge
    (2, "pitch"),   # v_slope → pitch
    (3, "sway"),    # w_disp  → sway
    (4, "roll"),    # w_slope → roll
    (5, "yaw"),     # phi     → yaw
])
def test_classifier_single_dof_unit(dof_local: int, expected: str) -> None:
    """A base node moving purely in one FEM DOF is named the matching
    platform DOF (pins the FEM→platform reorder)."""
    nselt = 4
    n_modes = 8
    ev = _one_dof_eigvecs(nselt, dof_local, n_modes)
    active = active_dof_indices(nselt, hub_conn=2)
    Mp = np.eye(6)  # identity metric: pure single-DOF motion is unambiguous
    labels = classify_platform_modes(ev, active, nselt, Mp)

    assert labels[0] == expected
    # Inert modes carry no energy → None; modes past the rigid count
    # are None regardless.
    assert all(lbl is None for lbl in labels[1:])


def test_mode_labels_roundtrip_npz_json(tmp_path) -> None:
    """mode_labels (with None entries) round-trips through both
    serialisers."""
    from pybmodes.fem.normalize import NodeModeShape

    span = np.linspace(0.0, 1.0, 5)
    shapes = [
        NodeModeShape(
            mode_number=i + 1, freq_hz=0.01 * (i + 1), span_loc=span,
            flap_disp=np.zeros(5), flap_slope=np.zeros(5),
            lag_disp=np.zeros(5), lag_slope=np.zeros(5), twist=np.zeros(5),
        )
        for i in range(4)
    ]
    res = ModalResult(
        frequencies=np.array([0.01, 0.02, 0.03, 0.5]),
        shapes=shapes,
        mode_labels=["surge", "sway", "yaw", None],
    )

    npz = tmp_path / "r.npz"
    res.save(npz)
    assert ModalResult.load(npz).mode_labels == ["surge", "sway", "yaw", None]

    js = tmp_path / "r.json"
    res.to_json(js)
    assert ModalResult.from_json(js).mode_labels == ["surge", "sway", "yaw", None]

    # A result without labels still round-trips with mode_labels None.
    plain = ModalResult(frequencies=np.array([1.0]), shapes=shapes[:1])
    plain.save(tmp_path / "p.npz")
    assert ModalResult.load(tmp_path / "p.npz").mode_labels is None
