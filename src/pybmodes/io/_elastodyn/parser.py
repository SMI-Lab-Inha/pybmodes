"""Line-driven parsers for the three ElastoDyn ``.dat`` file flavours.

Each entry point reads a path with ``latin-1`` (the encoding OpenFAST
sources use), splits into lines, and delegates to a per-flavour
parser. The parsers walk the file in line order, dispatching to
``_assign_*_field`` for typed scalars and ``_consume_*_distributed``
for the numeric tables.

Field-set discrepancies vs. the user-specified spec are documented in
``pybmodes.io.elastodyn_reader``'s public docstring.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np

from pybmodes.io._elastodyn.lex import (
    _canon_label,
    _is_section_divider,
    _parse_float,
    _split_label_index,
    _split_value_label,
    _strip_quotes,
)
from pybmodes.io._elastodyn.types import (
    ElastoDynBlade,
    ElastoDynMain,
    ElastoDynTower,
)

# ---------------------------------------------------------------------------
# Main file parser
# ---------------------------------------------------------------------------


def read_elastodyn_main(path: str | pathlib.Path) -> ElastoDynMain:
    """Parse a top-level ElastoDyn ``.dat`` file."""
    path = pathlib.Path(path)
    text = path.read_text(encoding="latin-1")
    return _parse_main(text.splitlines(), source_file=path)


def _parse_main(
    lines: list[str], source_file: Optional[pathlib.Path] = None
) -> ElastoDynMain:
    obj = ElastoDynMain(header="", title="", source_file=source_file)

    # Header is line 0 (file marker), title is the next non-empty line.
    if lines:
        obj.header = lines[0].rstrip()
    title_idx = 1
    while title_idx < len(lines) and not lines[title_idx].strip():
        title_idx += 1
    if title_idx < len(lines):
        obj.title = lines[title_idx].rstrip()
        i = title_idx + 1
    else:
        i = 1

    in_nodal_outlist = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if _is_section_divider(line):
            obj.section_dividers.append(line.rstrip())
            # An OutList ends with an "END" marker on its own line, but the
            # nodal-OutList opener is itself a "==" divider — track it.
            if "[optional section]" in line.lower() or stripped.startswith("======"):
                in_nodal_outlist = True
            i += 1
            continue

        # Detect OutList header (line literally containing the word "OutList"
        # as a label with an arrow comment).
        if "OutList" in stripped and stripped.split()[0] == "OutList":
            target = obj.nodal_out_list if in_nodal_outlist else obj.out_list
            target.append(line.rstrip())
            i += 1
            # Consume quoted-channel lines until END or end-of-file.
            while i < len(lines):
                ln = lines[i]
                target.append(ln.rstrip())
                if ln.strip().upper().startswith("END"):
                    i += 1
                    break
                i += 1
            continue

        parts = _split_value_label(line)
        if parts is None:
            i += 1
            continue
        value_str, label = parts
        canon = _canon_label(label)
        obj.scalars[label] = value_str

        # Dispatch to typed fields where we care about the value.
        try:
            _assign_main_field(obj, label, canon, value_str)
        except (ValueError, IndexError):
            # Stay tolerant — unknown/odd lines just live in scalars.
            pass

        i += 1

    return obj


def _assign_main_field(
    obj: ElastoDynMain, label: str, canon: str, value_str: str
) -> None:
    """Populate typed fields from ``label`` + raw value-string."""
    # Re-derive (canon, idx) here so the bare-digit BldFile1 form is handled.
    canon, idx = _split_label_index(label)

    if canon == "NumBl":
        obj.num_bl = int(value_str)
    elif canon == "TipRad":
        obj.tip_rad = _parse_float(value_str)
    elif canon == "HubRad":
        obj.hub_rad = _parse_float(value_str)
    elif canon == "PreCone" and idx is not None and 0 <= idx < 3:
        obj.pre_cone[idx] = _parse_float(value_str)
    elif canon == "HubCM":
        obj.hub_cm = _parse_float(value_str)
    elif canon == "OverHang":
        obj.overhang = _parse_float(value_str)
    elif canon == "ShftTilt":
        obj.shft_tilt = _parse_float(value_str)
    elif canon == "Twr2Shft":
        obj.twr2shft = _parse_float(value_str)
    elif canon == "TowerHt":
        obj.tower_ht = _parse_float(value_str)
    elif canon == "TowerBsHt":
        obj.tower_bs_ht = _parse_float(value_str)
    elif canon == "NacCMxn":
        obj.nac_cm_xn = _parse_float(value_str)
    elif canon == "NacCMyn":
        obj.nac_cm_yn = _parse_float(value_str)
    elif canon == "NacCMzn":
        obj.nac_cm_zn = _parse_float(value_str)
    elif canon == "RotSpeed":
        obj.rot_speed_rpm = _parse_float(value_str)
    elif canon == "TipMass" and idx is not None and 0 <= idx < 3:
        obj.tip_mass[idx] = _parse_float(value_str)
    elif canon == "HubMass":
        obj.hub_mass = _parse_float(value_str)
    elif canon == "HubIner":
        obj.hub_iner = _parse_float(value_str)
    elif canon == "GenIner":
        obj.gen_iner = _parse_float(value_str)
    elif canon == "NacMass":
        obj.nac_mass = _parse_float(value_str)
    elif canon == "NacYIner":
        obj.nac_y_iner = _parse_float(value_str)
    elif canon == "YawBrMass":
        obj.yaw_br_mass = _parse_float(value_str)
    elif canon == "BldFile":
        i_safe = idx if idx is not None else 0
        if 0 <= i_safe < 3:
            obj.bld_file[i_safe] = _strip_quotes(value_str)
    elif canon == "TwrFile":
        obj.twr_file = _strip_quotes(value_str)


# ---------------------------------------------------------------------------
# Tower file parser
# ---------------------------------------------------------------------------


def read_elastodyn_tower(path: str | pathlib.Path) -> ElastoDynTower:
    path = pathlib.Path(path)
    text = path.read_text(encoding="latin-1")
    return _parse_tower(text.splitlines(), source_file=path)


def _parse_tower(
    lines: list[str], source_file: Optional[pathlib.Path] = None
) -> ElastoDynTower:
    obj = ElastoDynTower(header="", title="", source_file=source_file)

    if lines:
        obj.header = lines[0].rstrip()
    title_idx = 1
    while title_idx < len(lines) and not lines[title_idx].strip():
        title_idx += 1
    if title_idx < len(lines):
        obj.title = lines[title_idx].rstrip()
        i = title_idx + 1
    else:
        i = 1

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if _is_section_divider(line):
            obj.section_dividers.append(line.rstrip())
            if "DISTRIBUTED" in stripped.upper():
                # Consume the distributed-properties block.
                i = _consume_tower_distributed(obj, lines, i + 1)
                continue
            i += 1
            continue

        parts = _split_value_label(line)
        if parts is None:
            i += 1
            continue
        value_str, label = parts
        canon = _canon_label(label)
        try:
            _assign_tower_field(obj, label, canon, value_str)
        except (ValueError, IndexError):
            pass
        i += 1

    return obj


_TOWER_DISTR_COL_MAP = {
    "HtFract": "ht_fract",
    "TMassDen": "t_mass_den",
    "TwFAStif": "tw_fa_stif",
    "TwSSStif": "tw_ss_stif",
    "TwFAIner": "tw_fa_iner",
    "TwSSIner": "tw_ss_iner",
    "TwFAcgOf": "tw_fa_cg_of",
    "TwSScgOf": "tw_ss_cg_of",
}


def _consume_tower_distributed(
    obj: ElastoDynTower, lines: list[str], i: int
) -> int:
    """Read the column-header lines and the n_rows × n_cols numeric block."""
    # Two header rows (column names, units) before the data.
    header_lines: list[str] = []
    while i < len(lines) and len(header_lines) < 2:
        ln = lines[i]
        if ln.strip():
            header_lines.append(ln.rstrip())
        i += 1
    obj.distr_header_lines = header_lines

    col_names = header_lines[0].split() if header_lines else []
    field_names = [_TOWER_DISTR_COL_MAP.get(c) for c in col_names]
    n_cols = len(col_names)

    # Read data rows up to the next section divider or non-numeric line.
    rows: list[list[float]] = []
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if _is_section_divider(ln):
            break
        toks = s.split()
        if len(toks) < n_cols:
            break
        try:
            rows.append([_parse_float(t) for t in toks[:n_cols]])
        except ValueError:
            break
        i += 1

    if rows:
        arr = np.asarray(rows, dtype=float)
        for col_idx, fname in enumerate(field_names):
            if fname is None:
                continue
            setattr(obj, fname, arr[:, col_idx].copy())
    return i


def _assign_tower_field(
    obj: ElastoDynTower, label: str, canon: str, value_str: str
) -> None:
    canon, idx = _split_label_index(label)

    if canon == "NTwInpSt":
        obj.n_tw_inp_st = int(value_str)
    elif canon == "TwrFADmp" and idx is not None and 0 <= idx < 2:
        obj.twr_fa_dmp[idx] = _parse_float(value_str)
    elif canon == "TwrSSDmp" and idx is not None and 0 <= idx < 2:
        obj.twr_ss_dmp[idx] = _parse_float(value_str)
    elif canon == "FAStTunr" and idx is not None and 0 <= idx < 2:
        obj.fa_st_tunr[idx] = _parse_float(value_str)
    elif canon == "SSStTunr" and idx is not None and 0 <= idx < 2:
        obj.ss_st_tunr[idx] = _parse_float(value_str)
    elif canon == "AdjTwMa":
        obj.adj_tw_ma = _parse_float(value_str)
    elif canon == "AdjFASt":
        obj.adj_fa_st = _parse_float(value_str)
    elif canon == "AdjSSSt":
        obj.adj_ss_st = _parse_float(value_str)
    elif canon == "TwFAM1Sh" and idx is not None and 2 <= idx + 1 <= 6:
        obj.tw_fa_m1_sh[idx - 1] = _parse_float(value_str)
    elif canon == "TwFAM2Sh" and idx is not None and 2 <= idx + 1 <= 6:
        obj.tw_fa_m2_sh[idx - 1] = _parse_float(value_str)
    elif canon == "TwSSM1Sh" and idx is not None and 2 <= idx + 1 <= 6:
        obj.tw_ss_m1_sh[idx - 1] = _parse_float(value_str)
    elif canon == "TwSSM2Sh" and idx is not None and 2 <= idx + 1 <= 6:
        obj.tw_ss_m2_sh[idx - 1] = _parse_float(value_str)


# ---------------------------------------------------------------------------
# Blade file parser
# ---------------------------------------------------------------------------


def read_elastodyn_blade(path: str | pathlib.Path) -> ElastoDynBlade:
    path = pathlib.Path(path)
    text = path.read_text(encoding="latin-1")
    return _parse_blade(text.splitlines(), source_file=path)


_BLADE_DISTR_COL_MAP = {
    "BlFract":   "bl_fract",
    "PitchAxis": "pitch_axis",
    "StrcTwst":  "strc_twst",
    "BMassDen":  "b_mass_den",
    "FlpStff":   "flp_stff",
    "EdgStff":   "edg_stff",
}


def _parse_blade(
    lines: list[str], source_file: Optional[pathlib.Path] = None
) -> ElastoDynBlade:
    obj = ElastoDynBlade(header="", title="", source_file=source_file)

    if lines:
        obj.header = lines[0].rstrip()
    title_idx = 1
    while title_idx < len(lines) and not lines[title_idx].strip():
        title_idx += 1
    if title_idx < len(lines):
        obj.title = lines[title_idx].rstrip()
        i = title_idx + 1
    else:
        i = 1

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if _is_section_divider(line):
            obj.section_dividers.append(line.rstrip())
            if "DISTRIBUTED" in stripped.upper():
                i = _consume_blade_distributed(obj, lines, i + 1)
                continue
            i += 1
            continue

        parts = _split_value_label(line)
        if parts is None:
            i += 1
            continue
        value_str, label = parts
        canon = _canon_label(label)
        try:
            _assign_blade_field(obj, label, canon, value_str)
        except (ValueError, IndexError):
            pass
        i += 1

    return obj


def _consume_blade_distributed(
    obj: ElastoDynBlade, lines: list[str], i: int
) -> int:
    header_lines: list[str] = []
    while i < len(lines) and len(header_lines) < 2:
        ln = lines[i]
        if ln.strip():
            header_lines.append(ln.rstrip())
        i += 1
    obj.distr_header_lines = header_lines

    col_names = header_lines[0].split() if header_lines else []
    field_names = [_BLADE_DISTR_COL_MAP.get(c) for c in col_names]
    n_cols = len(col_names)

    rows: list[list[float]] = []
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if _is_section_divider(ln):
            break
        toks = s.split()
        if len(toks) < n_cols:
            break
        try:
            rows.append([_parse_float(t) for t in toks[:n_cols]])
        except ValueError:
            break
        i += 1

    if rows:
        arr = np.asarray(rows, dtype=float)
        for col_idx, fname in enumerate(field_names):
            if fname is None:
                continue
            setattr(obj, fname, arr[:, col_idx].copy())
    return i


def _assign_blade_field(
    obj: ElastoDynBlade, label: str, canon: str, value_str: str
) -> None:
    canon, idx = _split_label_index(label)

    if canon == "NBlInpSt":
        obj.n_bl_inp_st = int(value_str)
    elif canon == "BldFlDmp" and idx is not None and 0 <= idx < 2:
        obj.bld_fl_dmp[idx] = _parse_float(value_str)
    elif canon == "BldEdDmp" and idx is not None and 0 <= idx < 1:
        obj.bld_ed_dmp[idx] = _parse_float(value_str)
    elif canon == "FlStTunr" and idx is not None and 0 <= idx < 2:
        obj.fl_st_tunr[idx] = _parse_float(value_str)
    elif canon == "AdjBlMs":
        obj.adj_bl_ms = _parse_float(value_str)
    elif canon == "AdjFlSt":
        obj.adj_fl_st = _parse_float(value_str)
    elif canon == "AdjEdSt":
        obj.adj_ed_st = _parse_float(value_str)
    elif canon == "BldFl1Sh" and idx is not None and 2 <= idx + 1 <= 6:
        obj.bld_fl1_sh[idx - 1] = _parse_float(value_str)
    elif canon == "BldFl2Sh" and idx is not None and 2 <= idx + 1 <= 6:
        obj.bld_fl2_sh[idx - 1] = _parse_float(value_str)
    elif canon == "BldEdgSh" and idx is not None and 2 <= idx + 1 <= 6:
        obj.bld_edg_sh[idx - 1] = _parse_float(value_str)
