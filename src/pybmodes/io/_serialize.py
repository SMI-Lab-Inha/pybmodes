"""Internal serialisation helpers shared by ``ModalResult`` and
``CampbellResult``.

Two output formats are supported:

* ``.npz`` — NumPy's compressed multi-array format. Arrays are stored
  as ``ndarray`` keys; scalar / dict metadata is stored as a single
  ``__meta__`` key holding a JSON-serialised dict (kept as a 0-d
  string array so ``np.load`` returns it without `allow_pickle`).
* ``.json`` — UTF-8 JSON with arrays serialised as nested lists. Used
  for ``ModalResult.to_json`` / ``from_json``.

A small ``_capture_metadata`` helper grabs the pyBmodes version, the
current UTC timestamp, the source file path (when supplied), and the
git HEAD hash of the working directory (best-effort, silently None
when ``git`` isn't installed or the cwd isn't a repo).
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess
from typing import Any

import numpy as np


def _capture_metadata(source_file: pathlib.Path | str | None = None) -> dict[str, Any]:
    """Build a metadata dict for embedding in a saved ``ModalResult`` or
    ``CampbellResult``.

    Fields:

    * ``pybmodes_version`` — :data:`pybmodes.__version__` at save time.
    * ``timestamp`` — UTC ISO-8601 timestamp at save time.
    * ``source_file`` — string form of ``source_file`` when supplied,
      otherwise ``None``.
    * ``git_hash`` — short SHA of the current ``git`` HEAD if the cwd
      is a git repo and ``git`` is available; otherwise ``None``.

    The function never raises; missing pieces become ``None``.
    """
    from pybmodes import __version__ as pybmodes_version

    meta: dict[str, Any] = {
        "pybmodes_version": pybmodes_version,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "source_file": str(source_file) if source_file is not None else None,
        "git_hash": _try_git_hash(),
    }
    return meta


def _try_git_hash() -> str | None:
    """Return the short git HEAD hash for the cwd, or ``None`` on any
    failure path (no git binary, cwd not a repo, subprocess timeout)."""
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError):
        return None
    out = cp.stdout.strip()
    return out or None


def _metadata_to_npz_value(meta: dict[str, Any]) -> np.ndarray:
    """Pack a metadata dict into a 0-d Unicode-string array suitable
    for storing as the ``__meta__`` key of an ``.npz`` archive.

    The dtype is ``np.str_`` (fixed-length unicode) rather than
    ``object`` — that's deliberate. The previous ``dtype=object``
    pickled the array contents, so loading the file back required
    ``allow_pickle=True``, which contradicted the module docstring's
    promise that metadata is "kept loadable without pickle". Pre-1.0
    review pass 4 surfaced the inconsistency.

    Files written by older pyBmodes versions used ``dtype=object``
    and are still loadable via the ``allow_pickle=True`` argument
    that ``ModalResult.load`` / ``CampbellResult.load`` continue to
    pass — the change is forward-only."""
    return np.array(json.dumps(meta, default=str), dtype=np.str_)


def _metadata_from_npz_value(arr: np.ndarray) -> dict[str, Any]:
    """Unpack the inverse of :func:`_metadata_to_npz_value`. ``arr``
    is typically a 0-d array as returned by ``np.load``. Handles both
    the new ``dtype=np.str_`` form and the legacy ``dtype=object``
    form left by older saves."""
    raw = arr.item() if hasattr(arr, "item") else str(arr)
    return dict(json.loads(raw))
