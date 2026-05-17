"""F4 regression: ``pybmodes patch`` rejects genuinely-contradictory
``--output`` / ``--output-dir`` input.

These are aliases. The locked CLI contract keeps silent-agree for
every previously-valid invocation (one flag, or both equal); the only
new behaviour is a clear exit-2 rejection when the two are given
*different* paths. The check is pure argument validation and runs
before any deck I/O, so no external/integration data is needed.
"""

from __future__ import annotations

from pybmodes.cli import main


def test_patch_conflicting_output_aliases_exit_2(capsys) -> None:
    rc = main([
        "patch", "no_such_deck.dat",
        "--output", "dirA", "--output-dir", "dirB",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--output and --output-dir were given different paths" in err


def test_patch_equal_output_aliases_not_a_conflict(capsys) -> None:
    """Same value through both aliases is NOT a conflict — it falls
    through to the normal path and fails later on the missing deck
    (exit 2) with a *file-not-found* message, never the alias one."""
    rc = main([
        "patch", "no_such_deck.dat",
        "--output", "same_dir", "--output-dir", "same_dir",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--output and --output-dir were given different paths" not in err
    assert "file not found" in err


def test_patch_single_output_alias_unaffected(capsys) -> None:
    """One flag alone never trips the conflict guard."""
    rc = main(["patch", "no_such_deck.dat", "--output-dir", "out"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--output and --output-dir were given different paths" not in err
    assert "file not found" in err
