"""Ratchet: every ``cases/*/run.py`` and case-study notebook must
compile cleanly under ``DeprecationWarning``-as-error.

The ``cases/`` tree is outside ruff's scope (``src/ tests/ scripts/``
per ``pyproject.toml``), so the W605 invalid-escape class of bug —
``"%CD%\\src"`` in a non-raw docstring — slipped past in earlier
revisions. Python 3.12 emits a ``SyntaxWarning`` for these and
3.14 will make them a hard ``SyntaxError``. This test compiles each
``run.py`` with ``warnings`` filter ``error`` to catch the regression
class without pulling ``cases/`` into ruff (the case-study scripts
are deliberately exploratory and we don't want full lint
conformance there).

If a future contributor adds a new case-study script and it carries
a ``\\s``-style invalid escape, this test fails at the source-file
location so the fix is obvious.
"""

from __future__ import annotations

import compileall
import pathlib
import warnings

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES_DIR = REPO_ROOT / "cases"

# Each ``cases/*/run.py`` script — the canonical exploratory entry
# points under ``cases/``.
_CASE_SCRIPTS = sorted(CASES_DIR.glob("*/run.py"))

# Sanity: at least one case script exists (the file is under source
# control), so an empty list signals a checkout problem rather than
# a vacuous pass.
if not _CASE_SCRIPTS:  # pragma: no cover — checkout-state guard
    pytest.skip(
        f"no cases/*/run.py scripts found under {CASES_DIR}; "
        "this normally indicates a checkout problem.",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "script", _CASE_SCRIPTS, ids=lambda p: p.parent.name,
)
def test_case_script_compiles_clean(script: pathlib.Path) -> None:
    """``compile(...)`` the script with ``warnings.filterwarnings('error')``
    so any ``SyntaxWarning`` (e.g. W605 ``\\s`` invalid escape) is
    promoted to a real ``SyntaxError`` and fails the test.

    The script is read as text and fed to ``compile``; we don't
    execute it (the case studies are slow and external-data-
    dependent), only verify it's syntactically clean.
    """
    source = script.read_text(encoding="utf-8")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=SyntaxWarning)
        # ``compile`` raises ``SyntaxError`` directly for genuinely
        # malformed code; the ``filterwarnings`` line promotes
        # ``SyntaxWarning`` (the W605 class) to a ``SyntaxWarning``
        # exception, which Python's deprecation policy will turn into
        # a hard error in 3.14.
        try:
            compile(source, str(script), "exec")
        except (SyntaxError, SyntaxWarning) as err:
            pytest.fail(
                f"{script} does not compile cleanly under "
                f"-W error: {err}"
            )


def test_compileall_walks_cases_directory_clean() -> None:
    """Belt-and-braces: ``compileall`` over the whole ``cases/`` tree
    so any ``.py`` we haven't enumerated above (helpers, future
    additions) gets caught too. ``quiet=1`` suppresses the "compiling
    …" chatter so the test output stays clean."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=SyntaxWarning)
        ok = compileall.compile_dir(
            str(CASES_DIR), quiet=1, force=True,
        )
    assert ok, (
        f"compileall.compile_dir({CASES_DIR}) reported failures; "
        f"see the captured output above for offending files."
    )
