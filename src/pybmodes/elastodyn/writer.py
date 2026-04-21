"""Patch ElastoDyn .dat files with computed polynomial coefficients.

Each coefficient occupies a separate line of the form:
    <value>   <ParamName(k)>   - comment text
The writer finds each line by matching the parameter name token, then replaces
the leading value in-place, preserving indentation and all trailing text.
"""

from __future__ import annotations

import pathlib
import re

from pybmodes.elastodyn.params import BladeElastoDynParams, TowerElastoDynParams

_MISSING: list[str] = []   # sentinel for type checking


def patch_dat(
    path: str | pathlib.Path,
    params: BladeElastoDynParams | TowerElastoDynParams,
) -> None:
    """Patch named mode-shape coefficient lines in an ElastoDyn .dat file.

    Each parameter in *params* is located by searching for its exact name as a
    whitespace-delimited token.  The value on the same line (the token before
    the name) is replaced with the computed coefficient.  All other content
    (indentation, comment text) is left unchanged.

    Parameters
    ----------
    path   : path to the ElastoDyn .dat file (modified in place).
    params : BladeElastoDynParams or TowerElastoDynParams.

    Raises
    ------
    KeyError  if a required parameter name is not found in the file.
    """
    path = pathlib.Path(path)
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)

    replacements = params.as_dict()
    missing = list(replacements.keys())

    for i, line in enumerate(lines):
        for name in list(missing):
            # Match: optional whitespace, any value, whitespace, then the exact
            # parameter name as a whole token (word boundary or end-of-field).
            pattern = re.compile(
                r'^(\s*)\S+(\s+' + re.escape(name) + r')(\s.*)?$'
            )
            m = pattern.match(line.rstrip('\n\r'))
            if m:
                value = replacements[name]
                tail  = m.group(3) if m.group(3) is not None else ''
                new_line = f"{m.group(1)}{value: .7E}{m.group(2)}{tail}\n"
                lines[i] = new_line
                missing.remove(name)
                break

    if missing:
        raise KeyError(
            f"The following parameter names were not found in {path}:\n"
            + "\n".join(f"  {n}" for n in missing)
        )

    path.write_text(''.join(lines), encoding='utf-8')
