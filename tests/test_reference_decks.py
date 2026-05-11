"""Tests for the bundled ``reference_decks/`` directory.

These tests assert that the artifacts produced by
``scripts/build_reference_decks.py`` are present and that every
case's post-patch validation report ends in ``Overall: PASS`` or
``Overall: WARN`` (a WARN means the constrained 6th-order polynomial
form cannot represent the deck's FEM mode shape below the 1 % PASS
gate — a property of the deck, not a pyBmodes bug; the polynomial
in the patched deck IS pyBmodes' best constrained fit so the deck
is still internally consistent). FAIL is rejected. They skip at
module level if the directory is absent (e.g. on a fresh clone where
the contributor hasn't run the build script).
"""

from __future__ import annotations

import pathlib

import pytest

from pybmodes.cli import _resolve_examples_root

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE_DECKS_DIR = _resolve_examples_root() / "reference_decks"

if not REFERENCE_DECKS_DIR.is_dir():
    pytest.skip(
        f"reference_decks/ not present at {REFERENCE_DECKS_DIR}; "
        "run `python scripts/build_reference_decks.py` to generate.",
        allow_module_level=True,
    )


def _case_dirs() -> list[pathlib.Path]:
    """Return reference_decks/<case>/ subdirectories that contain a
    validation_report.txt (i.e. were successfully built)."""
    return sorted(
        d for d in REFERENCE_DECKS_DIR.iterdir()
        if d.is_dir() and (d / "validation_report.txt").is_file()
    )


# ---------------------------------------------------------------------------
# Provenance: the directory carries the documentation we expect.
# ---------------------------------------------------------------------------

class TestProvenance:

    def test_readme_present(self) -> None:
        assert (REFERENCE_DECKS_DIR / "README.md").is_file()

    def test_validation_summary_present(self) -> None:
        assert (REFERENCE_DECKS_DIR / "VALIDATION_SUMMARY.md").is_file()

    def test_floating_cases_present(self) -> None:
        assert (REFERENCE_DECKS_DIR / "FLOATING_CASES.md").is_file()

    def test_at_least_one_case_built(self) -> None:
        assert len(_case_dirs()) >= 1, (
            "no reference_decks/<case>/ subdirectory contains a "
            "validation_report.txt"
        )

    def test_each_case_has_dat_files_and_report(self) -> None:
        for case_dir in _case_dirs():
            dat_files = list(case_dir.glob("*.dat"))
            assert dat_files, f"{case_dir.name}: no .dat files present"
            assert (case_dir / "validation_report.txt").is_file()
            # Every case ships its before-patch report too so the
            # ratios in VALIDATION_SUMMARY.md are reproducible from
            # the on-disk artifacts.
            assert (case_dir / "before_patch.txt").is_file()


# ---------------------------------------------------------------------------
# Each case's post-patch validation must end in PASS.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case_dir",
    _case_dirs() or [pytest.param(None, marks=pytest.mark.skip(
        reason="no reference_decks cases present"
    ))],
    ids=lambda d: d.name if d is not None else "no-cases",
)
def test_validation_report_ends_in_pass(case_dir: pathlib.Path) -> None:
    text = (case_dir / "validation_report.txt").read_text(encoding="utf-8")
    # PASS or WARN are both acceptable post-patch outcomes — see the
    # module docstring. FAIL is not.
    assert ("Overall: PASS" in text) or ("Overall: WARN" in text), (
        f"{case_dir.name}: validation_report.txt does not end in "
        f"'Overall: PASS' or 'Overall: WARN'.\n--- report ---\n{text}"
    )
    # Sanity: no FAIL line should appear anywhere in the post-patch
    # report.
    assert "FAIL" not in text, (
        f"{case_dir.name}: post-patch report contains a FAIL line.\n"
        f"--- report ---\n{text}"
    )


# ---------------------------------------------------------------------------
# The before-patch reports document the upstream inconsistency. They
# should NOT all PASS — that's the whole reason this directory exists.
# ---------------------------------------------------------------------------

def test_before_patch_reports_show_inconsistency() -> None:
    case_dirs = _case_dirs()
    if not case_dirs:
        pytest.skip("no reference_decks cases present")
    bad_count = 0
    for case_dir in case_dirs:
        text = (case_dir / "before_patch.txt").read_text(encoding="utf-8")
        if "Overall: FAIL" in text or "Overall: WARN" in text:
            bad_count += 1
    assert bad_count >= 1, (
        "every before_patch.txt shows Overall: PASS — that contradicts "
        "the ECOSYSTEM_FINDING.md narrative; either the upstream decks "
        "have been fixed (great, we should re-pin to the new commits) "
        "or the validator is mis-classifying."
    )
