"""Tests for ``pybmodes.report.generate_report``.

Three spec-named tests:

* ``test_report_md_contains_frequencies`` — markdown output includes
  every requested frequency from the supplied ``ModalResult``.
* ``test_report_html_is_valid`` — HTML output is a well-formed,
  self-contained HTML5 document with the expected top-level structure.
* ``test_report_csv_has_coefficient_columns`` — CSV output exposes
  the polynomial-coefficient columns (``C2``..``C6`` + ``rms_residual``
  + ``cond_number``) so spreadsheets can ingest the blocks directly.

Plus a CLI smoke test that runs the full ``pybmodes report`` flow on
a reference deck and confirms the output file lands on disk.
"""

from __future__ import annotations

import pathlib

import numpy as np

from pybmodes.fem.normalize import NodeModeShape
from pybmodes.fitting.poly_fit import PolyFitResult
from pybmodes.models.result import ModalResult
from pybmodes.report import generate_report


def _make_modal_result(n_modes: int = 4, n_nodes: int = 7) -> ModalResult:
    rng = np.random.default_rng(11)
    span = np.linspace(0.0, 1.0, n_nodes)
    shapes = [
        NodeModeShape(
            mode_number=k + 1,
            freq_hz=0.5 * (k + 1) + 0.1,
            span_loc=span,
            flap_disp=rng.standard_normal(n_nodes),
            flap_slope=rng.standard_normal(n_nodes),
            lag_disp=rng.standard_normal(n_nodes),
            lag_slope=rng.standard_normal(n_nodes),
            twist=rng.standard_normal(n_nodes),
        )
        for k in range(n_modes)
    ]
    freqs = np.array([s.freq_hz for s in shapes])
    return ModalResult(frequencies=freqs, shapes=shapes)


def _make_synthetic_polyfit(seed: int) -> PolyFitResult:
    """Build a synthetic ``PolyFitResult`` with deterministic numbers
    for the polynomial-coefficient section."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(-2.0, 2.0, size=5)
    return PolyFitResult(
        c2=float(c[0]), c3=float(c[1]), c4=float(c[2]),
        c5=float(c[3]), c6=float(c[4]),
        rms_residual=float(rng.uniform(1e-5, 1e-3)),
        tip_slope=float(rng.uniform(0.5, 3.0)),
        cond_number=float(rng.uniform(1e2, 1e4)),
    )


def _make_tower_params():
    from pybmodes.elastodyn.params import TowerElastoDynParams
    return TowerElastoDynParams(
        TwFAM1Sh=_make_synthetic_polyfit(1),
        TwFAM2Sh=_make_synthetic_polyfit(2),
        TwSSM1Sh=_make_synthetic_polyfit(3),
        TwSSM2Sh=_make_synthetic_polyfit(4),
    )


def _make_blade_params():
    from pybmodes.elastodyn.params import BladeElastoDynParams
    return BladeElastoDynParams(
        BldFl1Sh=_make_synthetic_polyfit(5),
        BldFl2Sh=_make_synthetic_polyfit(6),
        BldEdgSh=_make_synthetic_polyfit(7),
    )


# ---------------------------------------------------------------------------
# Spec-named tests
# ---------------------------------------------------------------------------

def test_report_md_contains_frequencies(tmp_path: pathlib.Path) -> None:
    """Markdown output lists every frequency from the result, formatted
    to 4 decimal places (the convention the renderer uses)."""
    result = _make_modal_result()
    out = tmp_path / "report.md"
    generate_report(result, out, format="md")
    text = out.read_text(encoding="utf-8")
    # Top-level header
    assert "# pyBmodes Modal Analysis Report" in text
    # Frequencies section header
    assert "## 3. Natural frequencies" in text
    # Every frequency value (4-dp rounded) appears in the body.
    for f in result.frequencies:
        assert f"{float(f):.4f}" in text, (
            f"frequency {f:.4f} Hz missing from markdown report:\n{text}"
        )


def test_report_html_is_valid(tmp_path: pathlib.Path) -> None:
    """HTML output is a well-formed HTML5 document with DOCTYPE,
    ``<html>`` root, ``<head>`` + ``<title>``, ``<body>``, and at
    least one ``<table>`` per section that has tabular data."""
    result = _make_modal_result()
    out = tmp_path / "report.html"
    generate_report(result, out, format="html")
    text = out.read_text(encoding="utf-8")

    # Top-level structure.
    assert text.startswith("<!DOCTYPE html>"), "missing DOCTYPE"
    assert '<html lang="en">' in text
    assert "<head>" in text and "</head>" in text
    assert "<body>" in text and "</body>" in text
    assert "<title>pyBmodes Modal Analysis Report</title>" in text
    assert "</html>" in text

    # Tables present for frequencies + classification at minimum.
    assert text.count("<table>") >= 2
    # Section headers rendered as <h2>.
    assert "<h2>1. Model summary</h2>" in text
    assert "<h2>3. Natural frequencies</h2>" in text

    # All <table> tags balanced.
    assert text.count("<table>") == text.count("</table>")
    assert text.count("<tr>") == text.count("</tr>")
    assert text.count("<td>") == text.count("</td>")


def test_report_csv_has_coefficient_columns(tmp_path: pathlib.Path) -> None:
    """CSV output exposes the polynomial-coefficient columns
    (``C2..C6, rms_residual, cond_number``) and groups rows by section
    so spreadsheet pivots work cleanly."""
    import csv

    result = _make_modal_result()
    out = tmp_path / "report.csv"
    generate_report(
        result, out, format="csv",
        tower_params=_make_tower_params(),
        blade_params=_make_blade_params(),
    )
    with out.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))

    # Two header rows expected: one for frequencies, one for coeffs,
    # separated by a blank row.
    assert rows, "CSV is empty"
    assert rows[0] == ["section", "mode", "frequency_hz", "period_s"]
    # Find the second header.
    coeff_header = ["section", "block", "C2", "C3", "C4", "C5", "C6",
                    "rms_residual", "cond_number"]
    assert coeff_header in rows, (
        f"coefficient header {coeff_header!r} missing from CSV; rows were:\n"
        + "\n".join(",".join(r) for r in rows[:20])
    )

    # Spot-check that the tower + blade block names appear in the
    # rows that follow the coefficient header.
    flat = {row[1] for row in rows if len(row) > 1}
    for name in ("TwFAM1Sh", "TwFAM2Sh", "TwSSM1Sh", "TwSSM2Sh",
                 "BldFl1Sh", "BldFl2Sh", "BldEdgSh"):
        assert name in flat, f"block {name} missing from CSV rows"


# ---------------------------------------------------------------------------
# Supporting test — CLI smoke
# ---------------------------------------------------------------------------

def test_report_cli_writes_md(tmp_path: pathlib.Path) -> None:
    """``pybmodes report <reference-deck> --format md --out X`` writes a
    non-empty markdown file containing the section headers we expect."""
    from pybmodes.cli import _resolve_examples_root
    deck = (
        _resolve_examples_root() / "reference_decks" / "nrel5mw_land"
        / "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
    )
    if not deck.is_file():
        import pytest
        pytest.skip(f"reference deck not present at {deck}")

    from pybmodes.cli import main as cli_main
    out_path = tmp_path / "report.md"
    rc = cli_main([
        "report", str(deck),
        "--format", "md",
        "--out", str(out_path),
        "--no-validate",  # keep the smoke fast; validate_dat does its own solve
        "--n-modes", "6",
    ])
    assert rc == 0
    assert out_path.is_file()
    text = out_path.read_text(encoding="utf-8")
    assert "# pyBmodes Modal Analysis Report" in text
    assert "## 1. Model summary" in text
    assert "## 3. Natural frequencies" in text
    # The CLI populates the source-file metadata.
    assert deck.name in text
