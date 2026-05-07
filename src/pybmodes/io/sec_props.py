"""Parser for beam section-properties files (.dat).

File structure (all units SI):
  Line 1 : title string
  Line 2 : n_secs  label  description
  Line 3 : blank
  Line 4 : column header
  Line 5 : column units
  Lines 6+: one row per spanwise station (13 space-separated values)
            trailing notes / blank lines after the data are ignored

Column order:
  span_loc  str_tw  tw_iner  mass_den  flp_iner  edge_iner
  flp_stff  edge_stff  tor_stff  axial_stff  cg_offst  sc_offst  tc_offst
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SectionProperties:
    """Spanwise section property table."""

    title: str
    n_secs: int

    # All arrays have length n_secs.
    span_loc:   np.ndarray   # normalised span station (0–1)
    str_tw:     np.ndarray   # structural twist (deg)
    tw_iner:    np.ndarray   # inertia twist (deg) — set to 0 for towers
    mass_den:   np.ndarray   # mass per unit length (kg/m)
    flp_iner:   np.ndarray   # flap/f-a mass moment of inertia per unit length (kg·m)
    edge_iner:  np.ndarray   # edge/s-s mass moment of inertia per unit length (kg·m)
    flp_stff:   np.ndarray   # flap/f-a bending stiffness EI (N·m²)
    edge_stff:  np.ndarray   # edge/s-s bending stiffness EI (N·m²)
    tor_stff:   np.ndarray   # torsion stiffness GJ (N·m²)
    axial_stff: np.ndarray   # axial stiffness EA (N)
    cg_offst:   np.ndarray   # CG offset from reference axis (m)
    sc_offst:   np.ndarray   # shear-centre offset from reference axis (m)
    tc_offst:   np.ndarray   # tension-centre offset from reference axis (m)

    source_file: Optional[pathlib.Path] = None


_N_COLS = 13  # expected number of data columns per row


def read_sec_props(path: str | pathlib.Path) -> SectionProperties:
    """Parse a section-properties .dat file."""
    path = pathlib.Path(path)
    lines = path.read_text(encoding='latin-1').splitlines()

    non_empty = [ln.rstrip() for ln in lines if ln.strip()]

    # The file is expected to start with a title line, an n_secs declaration,
    # column header, units row, and at least one data row.  An empty or
    # severely-truncated file would otherwise raise a bare IndexError that
    # buries the path; convert to a clear ValueError.
    if len(non_empty) < 5:
        raise ValueError(
            f"{path}: section-properties file is empty or truncated "
            f"(found {len(non_empty)} non-blank lines, need >= 5: "
            f"title, n_secs, header, units, at least one data row)"
        )

    # Line 0 of non_empty: title
    title = non_empty[0].strip()

    # Line 1: "n_secs  label  description"
    try:
        n_secs = int(non_empty[1].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"{path}: cannot parse n_secs from line 2 (got {non_empty[1]!r})"
        ) from exc

    # Lines 2 & 3: column header and units — skip
    # Line 4 onward: data rows until a line that cannot be parsed as numbers
    data_rows: list[list[float]] = []
    for ln in non_empty[4:]:
        tokens = ln.split()
        if len(tokens) < _N_COLS:
            break                          # trailing notes / blank separator
        try:
            row = [_parse_fortran_float(t) for t in tokens[:_N_COLS]]
        except ValueError:
            break
        data_rows.append(row)

    if len(data_rows) != n_secs:
        raise ValueError(
            f"{path}: expected {n_secs} data rows, found {len(data_rows)}"
        )

    arr = np.array(data_rows, dtype=float)  # (n_secs, 13)

    return SectionProperties(
        title=title,
        n_secs=n_secs,
        span_loc   = arr[:, 0],
        str_tw     = arr[:, 1],
        tw_iner    = arr[:, 2],
        mass_den   = arr[:, 3],
        flp_iner   = arr[:, 4],
        edge_iner  = arr[:, 5],
        flp_stff   = arr[:, 6],
        edge_stff  = arr[:, 7],
        tor_stff   = arr[:, 8],
        axial_stff = arr[:, 9],
        cg_offst   = arr[:, 10],
        sc_offst   = arr[:, 11],
        tc_offst   = arr[:, 12],
        source_file=path,
    )


def _parse_fortran_float(token: str) -> float:
    """Parse a float literal, handling D/d exponent notation."""
    return float(token.replace('d', 'e').replace('D', 'E'))
