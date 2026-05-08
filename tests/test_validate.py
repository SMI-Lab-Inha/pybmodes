"""Tests for the ElastoDyn coefficient-consistency validator.

These tests use the bundled OpenFAST r-test NREL 5MW deck at
``docs/OpenFAST_files/r-test/glue-codes/openfast/5MW_Land_DLL_WTurb/``
because it is known to ship inconsistent polynomial blocks (the same
ratios documented in ``cases/ECOSYSTEM_FINDING.md``: ~2,100× / ~2,500×
on TwFAM2Sh / TwSSM2Sh). If the deck isn't present locally the tests
skip at module level — same convention as ``tests/test_certtest.py``
for offshore reference data.
"""

from __future__ import annotations

import pathlib
import shutil
import warnings

import pytest

from pybmodes.elastodyn import (
    CoeffBlockResult,
    ValidationResult,
    compute_blade_params,
    compute_tower_params,
    patch_dat,
    validate_dat_coefficients,
)
from pybmodes.models import RotatingBlade, Tower

# ---------------------------------------------------------------------------
# Deck locator — module-level skip if absent
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
NREL5MW_DAT = (
    REPO_ROOT / "docs" / "OpenFAST_files" / "r-test" / "glue-codes"
    / "openfast" / "5MW_Land_DLL_WTurb"
    / "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
)

if not NREL5MW_DAT.is_file():
    pytest.skip(
        f"NREL 5MW r-test deck not present at {NREL5MW_DAT}; "
        "validator tests need it locally.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nrel5mw_validation() -> ValidationResult:
    """Cached validation result for the unmodified r-test deck."""
    return validate_dat_coefficients(NREL5MW_DAT)


# ---------------------------------------------------------------------------
# Result-shape tests
# ---------------------------------------------------------------------------

class TestValidationResultShape:

    def test_seven_blocks_present(self, nrel5mw_validation):
        assert set(nrel5mw_validation.tower_results.keys()) == {
            "TwFAM1Sh", "TwFAM2Sh", "TwSSM1Sh", "TwSSM2Sh",
        }
        assert set(nrel5mw_validation.blade_results.keys()) == {
            "BldFl1Sh", "BldFl2Sh", "BldEdgSh",
        }

    def test_each_block_carries_expected_fields(self, nrel5mw_validation):
        for block in nrel5mw_validation.all_blocks().values():
            assert isinstance(block, CoeffBlockResult)
            assert block.file_rms >= 0.0
            assert block.pybmodes_rms >= 0.0
            assert block.verdict in {"PASS", "WARN", "FAIL"}
            assert len(block.file_coeffs) == 5
            assert len(block.pybmodes_coeffs) == 5

    def test_dat_path_is_absolute(self, nrel5mw_validation):
        assert nrel5mw_validation.dat_path.is_absolute()
        assert nrel5mw_validation.dat_path == NREL5MW_DAT.resolve()


# ---------------------------------------------------------------------------
# The known ecosystem-finding pattern: TwFAM2Sh and TwSSM2Sh fail at huge
# ratios on the bundled NREL 5MW r-test deck.
# ---------------------------------------------------------------------------

class TestNREL5MWKnownFailures:

    def test_overall_verdict_fails(self, nrel5mw_validation):
        assert nrel5mw_validation.overall == "FAIL"

    def test_tw_fam2sh_fails_with_huge_ratio(self, nrel5mw_validation):
        block = nrel5mw_validation.tower_results["TwFAM2Sh"]
        assert block.verdict == "FAIL"
        # Ecosystem-finding doc reports ratio ~2,116×; allow a wide
        # band so this remains stable across mesh-numeric drift.
        assert block.ratio > 500.0

    def test_tw_ssm2sh_fails_with_huge_ratio(self, nrel5mw_validation):
        block = nrel5mw_validation.tower_results["TwSSM2Sh"]
        assert block.verdict == "FAIL"
        assert block.ratio > 500.0

    def test_first_mode_blocks_pass(self, nrel5mw_validation):
        # 1st-mode coefficients fit the structural model within the
        # PASS threshold (file_rms < 1 %); only 2nd-mode blocks fail.
        assert nrel5mw_validation.tower_results["TwFAM1Sh"].verdict == "PASS"
        assert nrel5mw_validation.tower_results["TwSSM1Sh"].verdict == "PASS"

    def test_blade_blocks_pass(self, nrel5mw_validation):
        # NREL 5MW blade polynomials, despite being mildly inconsistent
        # with the structural inputs (ratios ~2.5×), have absolute file
        # RMS < 1 % — they classify as PASS under the per-block gate.
        for name in ("BldFl1Sh", "BldFl2Sh", "BldEdgSh"):
            assert nrel5mw_validation.blade_results[name].verdict == "PASS"

    def test_summary_mentions_failing_count(self, nrel5mw_validation):
        assert "do not represent" in nrel5mw_validation.summary


# ---------------------------------------------------------------------------
# Round-trip: validate -> patch -> validate -> PASS.
#
# Stages a copy of the deck (plus the blade subtree) under tmp_path so
# the original repo files remain pristine.
# ---------------------------------------------------------------------------

class TestPatchRoundTrip:

    @pytest.fixture
    def staged_deck(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Copy the 5MW deck + blade subtree into tmp_path, return main."""
        src_dir = NREL5MW_DAT.parent
        blade_src = (
            REPO_ROOT / "docs" / "OpenFAST_files" / "r-test"
            / "glue-codes" / "openfast" / "5MW_Baseline"
        )
        if not blade_src.is_dir():
            pytest.skip(f"5MW_Baseline directory not present at {blade_src}")

        # The deck's BldFile points to ../5MW_Baseline/...; mirror that
        # layout under tmp_path: tmp_path/5MW_Land_DLL_WTurb/...,
        # tmp_path/5MW_Baseline/...
        deck_dst = tmp_path / src_dir.name
        shutil.copytree(src_dir, deck_dst)
        shutil.copytree(blade_src, tmp_path / blade_src.name)
        return deck_dst / NREL5MW_DAT.name

    def test_round_trip(self, staged_deck: pathlib.Path) -> None:
        # Step 1: confirm staged deck still fails (sanity).
        before = validate_dat_coefficients(staged_deck)
        assert before.overall == "FAIL"

        # Step 2: patch tower + blade .dat files in place.
        tower_modal = Tower.from_elastodyn(staged_deck).run(n_modes=10)
        blade_modal = RotatingBlade.from_elastodyn(staged_deck).run(n_modes=10)
        tower_params = compute_tower_params(tower_modal)
        blade_params = compute_blade_params(blade_modal)

        from pybmodes.io.elastodyn_reader import read_elastodyn_main
        main = read_elastodyn_main(staged_deck)
        patch_dat(staged_deck.parent / main.twr_file, tower_params)
        patch_dat(staged_deck.parent / main.bld_file[0], blade_params)

        # Step 3: re-validate; everything passes now.
        after = validate_dat_coefficients(staged_deck)
        assert after.overall == "PASS"
        for block in after.all_blocks().values():
            assert block.verdict == "PASS"
            # ratio close to 1: file polynomial is the pyBmodes
            # polynomial after the patch, modulo the ~7 sig figs the
            # writer keeps. Allow 1 % ratio drift to absorb that.
            assert abs(block.ratio - 1.0) < 1.0e-2


# ---------------------------------------------------------------------------
# Tower.from_elastodyn(validate_coeffs=True) and the symmetric blade path.
# ---------------------------------------------------------------------------

class TestModelValidationFlag:

    def test_tower_validate_coeffs_attaches_result(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tower = Tower.from_elastodyn(NREL5MW_DAT, validate_coeffs=True)
        assert isinstance(tower.coeff_validation, ValidationResult)
        assert tower.coeff_validation.overall == "FAIL"

    def test_tower_validate_coeffs_emits_warning(self) -> None:
        with pytest.warns(UserWarning, match="do not represent"):
            Tower.from_elastodyn(NREL5MW_DAT, validate_coeffs=True)

    def test_tower_default_no_validation(self) -> None:
        tower = Tower.from_elastodyn(NREL5MW_DAT)
        assert tower.coeff_validation is None

    def test_blade_validate_coeffs_attaches_result(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            blade = RotatingBlade.from_elastodyn(
                NREL5MW_DAT, validate_coeffs=True
            )
        assert isinstance(blade.coeff_validation, ValidationResult)

    def test_blade_default_no_validation(self) -> None:
        blade = RotatingBlade.from_elastodyn(NREL5MW_DAT)
        assert blade.coeff_validation is None


# ---------------------------------------------------------------------------
# CLI smoke tests via direct main() invocation.
# ---------------------------------------------------------------------------

class TestCLI:

    def test_validate_command_exits_1_on_fail(self, capsys) -> None:
        from pybmodes.cli import main as cli_main
        rc = cli_main(["validate", str(NREL5MW_DAT)])
        assert rc == 1
        captured = capsys.readouterr()
        assert "pyBmodes coefficient validator" in captured.out
        assert "FAIL" in captured.out
        assert "TwFAM2Sh" in captured.out

    def test_validate_missing_file_exits_2(self, capsys) -> None:
        from pybmodes.cli import main as cli_main
        rc = cli_main(["validate", "/nonexistent/path.dat"])
        assert rc == 2
        captured = capsys.readouterr()
        assert "not found" in captured.err
