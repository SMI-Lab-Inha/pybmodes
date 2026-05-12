"""Round-trip tests for ModalResult / CampbellResult serialisation.

Three named tests come straight from the spec
(``test_modal_result_round_trip_npz``,
``test_modal_result_round_trip_json``, ``test_campbell_csv_columns``)
plus a few support tests that gate metadata capture and the
``CampbellResult.save → load`` round-trip.
"""

from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
import pytest

from pybmodes.campbell import CampbellResult
from pybmodes.fem.normalize import NodeModeShape
from pybmodes.models.result import ModalResult

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_modal_result(n_modes: int = 3, n_nodes: int = 7) -> ModalResult:
    """Build a small in-memory ``ModalResult`` with deterministic values
    so round-trip equality checks are stable."""
    rng = np.random.default_rng(42)
    span = np.linspace(0.0, 1.0, n_nodes)
    shapes = [
        NodeModeShape(
            mode_number=k + 1,
            freq_hz=0.5 * (k + 1) ** 1.5,
            span_loc=span,
            flap_disp=rng.standard_normal(n_nodes),
            flap_slope=rng.standard_normal(n_nodes),
            lag_disp=rng.standard_normal(n_nodes),
            lag_slope=rng.standard_normal(n_nodes),
            twist=rng.standard_normal(n_nodes),
        )
        for k in range(n_modes)
    ]
    frequencies = np.array([s.freq_hz for s in shapes])
    participation = rng.uniform(0.0, 1.0, size=(n_modes, 3))
    participation /= participation.sum(axis=1, keepdims=True)
    fit_residuals = {
        "TwFAM1Sh": 1.0e-5,
        "TwFAM2Sh": 2.3e-3,
    }
    return ModalResult(
        frequencies=frequencies,
        shapes=shapes,
        participation=participation,
        fit_residuals=fit_residuals,
    )


def _make_campbell_result(n_steps: int = 5, n_modes: int = 4) -> CampbellResult:
    rng = np.random.default_rng(7)
    omega = np.linspace(0.0, 12.0, n_steps)
    freqs = rng.uniform(0.3, 5.0, size=(n_steps, n_modes))
    parts = rng.uniform(0.0, 1.0, size=(n_steps, n_modes, 3))
    parts /= parts.sum(axis=-1, keepdims=True)
    mac = np.full((n_steps, n_modes), np.nan)
    mac[1:, :2] = rng.uniform(0.95, 1.0, size=(n_steps - 1, 2))
    return CampbellResult(
        omega_rpm=omega,
        frequencies=freqs,
        labels=["1st flap", "1st edge", "1st tower FA", "1st tower SS"],
        participation=parts,
        n_blade_modes=2,
        n_tower_modes=2,
        mac_to_previous=mac,
    )


# ---------------------------------------------------------------------------
# Modal result — npz round trip (spec-named)
# ---------------------------------------------------------------------------

def _shapes_equal(a: list[NodeModeShape], b: list[NodeModeShape]) -> bool:
    if len(a) != len(b):
        return False
    for sa, sb in zip(a, b):
        if sa.mode_number != sb.mode_number:
            return False
        if not np.isclose(sa.freq_hz, sb.freq_hz):
            return False
        for attr in ("span_loc", "flap_disp", "flap_slope",
                     "lag_disp", "lag_slope", "twist"):
            if not np.allclose(getattr(sa, attr), getattr(sb, attr)):
                return False
    return True


def test_modal_result_round_trip_npz(tmp_path: pathlib.Path) -> None:
    """ModalResult.save -> ModalResult.load yields a value-equal record
    including frequencies, mode shapes, participation, fit residuals,
    and metadata."""
    result = _make_modal_result()
    out = tmp_path / "modes.npz"
    result.save(out, source_file="dummy.bmi")
    assert out.is_file()

    loaded = ModalResult.load(out)
    np.testing.assert_allclose(loaded.frequencies, result.frequencies)
    assert _shapes_equal(loaded.shapes, result.shapes)
    assert loaded.participation is not None
    np.testing.assert_allclose(loaded.participation, result.participation)
    assert loaded.fit_residuals == result.fit_residuals
    # Metadata block populated automatically and round-trippable.
    assert loaded.metadata is not None
    assert "pybmodes_version" in loaded.metadata
    assert "timestamp" in loaded.metadata
    assert loaded.metadata["source_file"] == "dummy.bmi"


def test_modal_result_round_trip_json(tmp_path: pathlib.Path) -> None:
    """ModalResult.to_json -> ModalResult.from_json yields a value-equal
    record. The JSON file is human-readable and includes the schema
    version key."""
    result = _make_modal_result()
    out = tmp_path / "modes.json"
    result.to_json(out, source_file="another.bmi")
    assert out.is_file()
    # Sanity: file is valid JSON with the expected top-level keys.
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    assert "metadata" in payload and "frequencies" in payload
    assert "shapes" in payload and len(payload["shapes"]) == len(result.shapes)
    # Round-trip equality.
    loaded = ModalResult.from_json(out)
    np.testing.assert_allclose(loaded.frequencies, result.frequencies)
    assert _shapes_equal(loaded.shapes, result.shapes)
    assert loaded.participation is not None
    np.testing.assert_allclose(loaded.participation, result.participation)
    assert loaded.fit_residuals == result.fit_residuals


def test_modal_result_metadata_capture(tmp_path: pathlib.Path) -> None:
    """Default metadata grab populates pybmodes_version + timestamp +
    (when present) git_hash."""
    result = _make_modal_result()
    out = tmp_path / "modes.npz"
    result.save(out)
    loaded = ModalResult.load(out)
    meta = loaded.metadata
    assert meta is not None
    assert isinstance(meta["pybmodes_version"], str)
    assert meta["pybmodes_version"]  # not empty
    assert "timestamp" in meta
    # git_hash is best-effort; either None or a hex-ish short SHA.
    if meta.get("git_hash") is not None:
        assert isinstance(meta["git_hash"], str)


def test_modal_result_no_participation_or_residuals_omitted(
    tmp_path: pathlib.Path,
) -> None:
    """When participation / fit_residuals are unset, they round-trip as
    None — i.e. saved archives don't have to carry them."""
    result = _make_modal_result()
    result.participation = None
    result.fit_residuals = None
    out = tmp_path / "modes.npz"
    result.save(out)
    loaded = ModalResult.load(out)
    assert loaded.participation is None
    assert loaded.fit_residuals is None


def test_modal_result_empty_result_round_trip_npz(tmp_path: pathlib.Path) -> None:
    """An empty solve result should save/load without inventing shapes."""
    result = ModalResult(frequencies=np.empty(0), shapes=[])
    out = tmp_path / "empty.npz"
    result.save(out)

    loaded = ModalResult.load(out)
    assert loaded.frequencies.shape == (0,)
    assert loaded.shapes == []
    assert loaded.metadata is not None


def test_modal_result_save_rejects_frequency_shape_mismatch(
    tmp_path: pathlib.Path,
) -> None:
    """Saving catches accidental truncation between frequencies and shapes."""
    result = _make_modal_result(n_modes=2)
    result.frequencies = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match=r"len\(frequencies\)=3 != len\(shapes\)=2"):
        result.save(tmp_path / "bad.npz")


# ---------------------------------------------------------------------------
# Campbell result — CSV columns (spec-named) + npz round-trip
# ---------------------------------------------------------------------------

def test_campbell_csv_columns(tmp_path: pathlib.Path) -> None:
    """CampbellResult.to_csv emits a header with 'rpm', one column per
    mode label, and one '<label>_mac' column per mode label. Data
    rows match the result arrays element-for-element."""
    result = _make_campbell_result()
    out = tmp_path / "campbell.csv"
    result.to_csv(out)
    assert out.is_file()

    with out.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    header, *data = rows

    # Expected column ordering.
    expected_freq_cols = result.labels
    expected_mac_cols = [f"{lbl}_mac" for lbl in result.labels]
    assert header[0] == "rpm"
    assert header[1:1 + len(expected_freq_cols)] == expected_freq_cols
    assert header[1 + len(expected_freq_cols):] == expected_mac_cols

    # Row count matches the sweep length.
    assert len(data) == result.omega_rpm.size

    # Spot-check one row's numerical content.
    row = data[2]
    assert float(row[0]) == pytest.approx(result.omega_rpm[2])
    n_modes = result.frequencies.shape[1]
    for k in range(n_modes):
        assert float(row[1 + k]) == pytest.approx(result.frequencies[2, k])


def test_campbell_save_load_round_trip(tmp_path: pathlib.Path) -> None:
    """CampbellResult.save -> CampbellResult.load yields a value-equal
    record (so the save method is honest about being round-trippable)."""
    result = _make_campbell_result()
    out = tmp_path / "campbell.npz"
    result.save(out, source_file="dummy.dat")
    loaded = CampbellResult.load(out)
    np.testing.assert_allclose(loaded.omega_rpm, result.omega_rpm)
    np.testing.assert_allclose(loaded.frequencies, result.frequencies)
    np.testing.assert_allclose(loaded.participation, result.participation)
    # MAC has NaN entries; use equal_nan for the round-trip check.
    np.testing.assert_allclose(
        loaded.mac_to_previous, result.mac_to_previous, equal_nan=True,
    )
    assert loaded.labels == result.labels
    assert loaded.n_blade_modes == result.n_blade_modes
    assert loaded.n_tower_modes == result.n_tower_modes


def test_campbell_csv_uses_nan_mac_when_shape_missing(tmp_path: pathlib.Path) -> None:
    """Older CampbellResult-like objects without MAC data still write stable columns."""
    result = _make_campbell_result(n_steps=2, n_modes=2)
    result.mac_to_previous = np.empty((0, 0))
    out = tmp_path / "campbell_no_mac.csv"
    result.to_csv(out)

    with out.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["rpm", *result.labels, *[f"{lbl}_mac" for lbl in result.labels]]
    assert rows[1][-2:] == ["nan", "nan"]
    assert rows[2][-2:] == ["nan", "nan"]
