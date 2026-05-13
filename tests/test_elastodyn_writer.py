"""Default-suite round-trip tests for :mod:`pybmodes.io._elastodyn.writer`.

The integration-marked ``tests/test_elastodyn_reader.py::test_*_semantic_roundtrip``
parametrise this against upstream r-test decks under ``docs/`` which
ship gitignored, so on a fresh clone with no upstream data the writer
gets zero default-suite coverage. This module covers the same parse →
emit → re-parse fixed point against the bundled reference decks at
``src/pybmodes/_examples/reference_decks/nrel5mw_land/`` — those ship
inside the wheel as package-data and are present in any source-checkout
or installed environment.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import tempfile

import numpy as np
import pytest

from pybmodes.cli import _resolve_examples_root
from pybmodes.io.elastodyn_reader import (
    read_elastodyn_blade,
    read_elastodyn_main,
    read_elastodyn_tower,
    write_elastodyn_blade,
    write_elastodyn_main,
    write_elastodyn_tower,
)

_DECK_DIR = (
    _resolve_examples_root() / "reference_decks" / "nrel5mw_land"
)
_MAIN = _DECK_DIR / "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
_TOWER = _DECK_DIR / "NRELOffshrBsline5MW_Tower.dat"
_BLADE = _DECK_DIR / "NRELOffshrBsline5MW_Blade.dat"

if not all(p.is_file() for p in (_MAIN, _TOWER, _BLADE)):
    pytest.skip(
        "bundled NREL 5MW reference deck not present — regenerate with "
        "`python scripts/build_reference_decks.py`",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Numpy-aware dataclass equality (mirrors test_elastodyn_reader.py)
# ---------------------------------------------------------------------------

def _eq(a: object, b: object) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        aa = np.asarray(a)
        bb = np.asarray(b)
        if aa.shape != bb.shape:
            return False
        if np.issubdtype(aa.dtype, np.floating) or np.issubdtype(bb.dtype, np.floating):
            return bool(np.allclose(aa, bb, rtol=1e-12, atol=1e-15, equal_nan=True))
        return bool(np.array_equal(aa, bb))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b)) or math.isclose(
            a, b, rel_tol=1e-12, abs_tol=1e-15,
        )
    return a == b


def _assert_dataclass_eq(a: object, b: object, *, ignore: tuple[str, ...]) -> None:
    assert type(a) is type(b)
    for fld in dataclasses.fields(a):  # type: ignore[arg-type]
        if fld.name in ignore:
            continue
        va, vb = getattr(a, fld.name), getattr(b, fld.name)
        assert _eq(va, vb), f"field {fld.name!r}: {va!r} != {vb!r}"


def _reparse_text(reader_fn, text: str):
    """Write text to a temp file and re-parse it. The writer emits
    ``latin-1`` content (matching the OpenFAST source encoding), so
    the temp file is written with the same encoding."""
    with tempfile.NamedTemporaryFile(
        "w", encoding="latin-1", suffix=".dat", delete=False,
    ) as f:
        f.write(text)
        path = pathlib.Path(f.name)
    try:
        return reader_fn(path)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Round-trip: parse → emit → reparse → equal
# ---------------------------------------------------------------------------

def test_main_semantic_roundtrip() -> None:
    """``write_elastodyn_main`` emits text that re-parses to a typed
    dataclass equal to the original on every field except raw-line
    metadata (``scalars``, ``section_dividers``, ``out_list`` text,
    ``header`` whitespace)."""
    parsed1 = read_elastodyn_main(_MAIN)
    text = write_elastodyn_main(parsed1)
    parsed2 = _reparse_text(read_elastodyn_main, text)
    _assert_dataclass_eq(
        parsed1, parsed2,
        ignore=(
            "source_file", "scalars", "out_list", "nodal_out_list",
            "section_dividers", "header",
        ),
    )


def test_tower_semantic_roundtrip() -> None:
    """Same round-trip on the tower file."""
    parsed1 = read_elastodyn_tower(_TOWER)
    text = write_elastodyn_tower(parsed1)
    parsed2 = _reparse_text(read_elastodyn_tower, text)
    _assert_dataclass_eq(
        parsed1, parsed2,
        ignore=("source_file", "section_dividers", "distr_header_lines", "header"),
    )


def test_blade_semantic_roundtrip() -> None:
    """Same round-trip on the blade file."""
    parsed1 = read_elastodyn_blade(_BLADE)
    text = write_elastodyn_blade(parsed1)
    parsed2 = _reparse_text(read_elastodyn_blade, text)
    _assert_dataclass_eq(
        parsed1, parsed2,
        ignore=("source_file", "section_dividers", "distr_header_lines", "header"),
    )


def test_writer_emits_to_path(tmp_path: pathlib.Path) -> None:
    """The ``path=`` argument writes the same text the function
    returns, ``latin-1``-encoded."""
    parsed = read_elastodyn_tower(_TOWER)
    out_path = tmp_path / "tower_roundtrip.dat"
    returned_text = write_elastodyn_tower(parsed, path=out_path)
    assert out_path.is_file()
    on_disk = out_path.read_text(encoding="latin-1")
    assert on_disk == returned_text
