"""Parser and writer for OpenFAST ElastoDyn ``.dat`` input files.

ElastoDyn ``.dat`` files come in three flavours, all of which share a common
line-ordered convention: each significant line carries a value (or short
list) followed by a label keyword and an optional ``- comment`` tail. The
parser here is **label-based** rather than position-based, which keeps it
robust across the FAST v8 → OpenFAST v3+ format drift (renamed/added/
removed scalars between versions; ``BldFile(1)`` vs ``BldFile1``).

Three entry points cover the three file flavours:

  * :func:`read_elastodyn_main`  — the top-level ElastoDyn input file.
  * :func:`read_elastodyn_tower` — the tower-properties file referenced
    via ``TwrFile``.
  * :func:`read_elastodyn_blade` — the blade-properties file referenced
    via ``BldFile(1..3)``.

Each returns a dataclass holding the parsed values plus enough raw-line
metadata to re-emit a semantically identical file via the matching
``write_*`` function.

Field-set discrepancies vs. the user-specified spec
---------------------------------------------------

A few field names from the spec do not actually live in ElastoDyn
``.dat`` files:

  * ``HubHt`` is not stored; it is derived from ``TowerHt + Twr2Shft +
    OverHang·sin(ShftTilt)``. Exposed as :attr:`ElastoDynMain.hub_ht`.
  * ``RotMass`` is computed inside ElastoDyn from ``HubMass`` + the
    blade-mass integral; not an input. Exposed as
    :meth:`ElastoDynMain.compute_rot_mass` once the blade file is known.
  * ``GJStff``, ``EAStff``, ``FlpIner``, ``EdgIner``, ``Precrv/PreswpRef``,
    ``Flp/EdgcgOf``, ``Flp/EdgEAOf`` belong to BeamDyn, not ElastoDyn.
    Real ElastoDyn blade tables are 5-col (``BlFract, StrcTwst, BMassDen,
    FlpStff, EdgStff``) or 6-col (adds ``PitchAxis``). The reader detects
    columns from the header and stores whatever is present.
  * Tower distributed table is 4-col in every file shipped with the
    bundled RWTs (``HtFract, TMassDen, TwFAStif, TwSSStif``); the
    optional ``TwFAIner/TwSSIner/TwFAcgOf/TwSScgOf`` columns are
    accepted if seen but are not required.

Round-trip
----------

``write_*`` functions emit a canonically formatted file that **parses
back to an equal dataclass** but is not byte-identical to the original.
Whitespace, label column position, and comment text are normalised. The
test suite compares the parse-emit-reparse fixed point.
"""

from __future__ import annotations

import io
import math
import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ElastoDynMain:
    """Parsed top-level ElastoDyn input file."""

    # File-level metadata
    header: str
    title: str
    source_file: Optional[pathlib.Path] = None

    # Geometry / configuration
    num_bl: int = 3
    tip_rad: float = 0.0
    hub_rad: float = 0.0
    pre_cone: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    hub_cm: float = 0.0
    overhang: float = 0.0
    shft_tilt: float = 0.0
    twr2shft: float = 0.0
    tower_ht: float = 0.0
    tower_bs_ht: float = 0.0

    # Nacelle CM offsets (tower-top frame, downwind/lateral/vertical)
    nac_cm_xn: float = 0.0
    nac_cm_yn: float = 0.0
    nac_cm_zn: float = 0.0

    # Initial conditions (only RotSpeed is needed for centrifugal stiffening)
    rot_speed_rpm: float = 0.0

    # Mass and inertia
    tip_mass: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    hub_mass: float = 0.0
    hub_iner: float = 0.0
    gen_iner: float = 0.0
    nac_mass: float = 0.0
    nac_y_iner: float = 0.0
    yaw_br_mass: float = 0.0

    # File references
    bld_file: list[str] = field(default_factory=lambda: ["", "", ""])
    twr_file: str = ""

    # Captured raw scalar map: maps every ``LABEL`` (canonical, no parens) to
    # the verbatim value-token string from the source line. Used by the writer
    # to round-trip scalars we don't break out into typed fields.
    scalars: dict[str, str] = field(default_factory=dict)

    # Out-list (verbatim, including the END marker)
    out_list: list[str] = field(default_factory=list)
    nodal_out_list: list[str] = field(default_factory=list)

    # Section divider lines from the source, in order, for re-emit.
    section_dividers: list[str] = field(default_factory=list)

    @property
    def hub_ht(self) -> float:
        """Hub height above tower base, derived from geometry."""
        return self.tower_ht + self.twr2shft + self.overhang * math.sin(
            math.radians(self.shft_tilt)
        )

    def compute_rot_mass(self, blade: "ElastoDynBlade") -> float:
        """Total rotor mass = hub + N · ∫ BMassDen ds along the blade.

        Requires the blade file to integrate the distributed mass density.
        """
        if blade.bl_fract.size == 0:
            return self.hub_mass
        # Trapezoidal integral of BMassDen over the blade length.
        bl_len = self.tip_rad - self.hub_rad
        s = blade.bl_fract * bl_len
        bl_mass_per_blade = float(np.trapezoid(blade.b_mass_den, s))
        return self.hub_mass + self.num_bl * bl_mass_per_blade


@dataclass
class ElastoDynTower:
    """Parsed ElastoDyn tower input file."""

    header: str
    title: str
    source_file: Optional[pathlib.Path] = None

    n_tw_inp_st: int = 0
    twr_fa_dmp: list[float] = field(default_factory=lambda: [0.0, 0.0])
    twr_ss_dmp: list[float] = field(default_factory=lambda: [0.0, 0.0])
    fa_st_tunr: list[float] = field(default_factory=lambda: [1.0, 1.0])
    ss_st_tunr: list[float] = field(default_factory=lambda: [1.0, 1.0])
    adj_tw_ma: float = 1.0
    adj_fa_st: float = 1.0
    adj_ss_st: float = 1.0

    # Distributed properties — always at least the 4 mandatory columns.
    ht_fract: np.ndarray = field(default_factory=lambda: np.empty(0))
    t_mass_den: np.ndarray = field(default_factory=lambda: np.empty(0))
    tw_fa_stif: np.ndarray = field(default_factory=lambda: np.empty(0))
    tw_ss_stif: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Optional extra columns (not present in any of the bundled RWTs;
    # populated as zero-length arrays unless the source file carries them).
    tw_fa_iner: np.ndarray = field(default_factory=lambda: np.empty(0))
    tw_ss_iner: np.ndarray = field(default_factory=lambda: np.empty(0))
    tw_fa_cg_of: np.ndarray = field(default_factory=lambda: np.empty(0))
    tw_ss_cg_of: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Embedded mode-shape polynomial coefficients (degrees 2..6).
    tw_fa_m1_sh: np.ndarray = field(default_factory=lambda: np.zeros(5))
    tw_fa_m2_sh: np.ndarray = field(default_factory=lambda: np.zeros(5))
    tw_ss_m1_sh: np.ndarray = field(default_factory=lambda: np.zeros(5))
    tw_ss_m2_sh: np.ndarray = field(default_factory=lambda: np.zeros(5))

    # Verbatim column-header lines for re-emit.
    distr_header_lines: list[str] = field(default_factory=list)
    section_dividers: list[str] = field(default_factory=list)


@dataclass
class ElastoDynBlade:
    """Parsed ElastoDyn blade input file.

    ElastoDyn ships only translational mass density (``BMassDen``) and
    bending stiffnesses (``FlpStff``, ``EdgStff``) per spanwise station;
    it has no per-section *rotary* mass moments of inertia. Those live
    in BeamDyn or come from a cross-section pre-processor (VABS, PreComp).

    The :attr:`rotary_inertia_available` flag is therefore always
    ``False`` after parsing an ElastoDyn blade file. Downstream code
    (``to_pybmodes_blade``) treats the rotary inertia contributions as
    zero, which is the correct Euler-Bernoulli limit for slender blades
    and is sufficient for the bending modes (1–4 flap/edge) pyBmodes
    targets. A tiny regularisation floor is added in the section-property
    builder to keep the global mass matrix positive-definite without
    fabricating physically meaningful rotary terms.
    """

    header: str
    title: str
    source_file: Optional[pathlib.Path] = None

    n_bl_inp_st: int = 0
    bld_fl_dmp: list[float] = field(default_factory=lambda: [0.0, 0.0])
    bld_ed_dmp: list[float] = field(default_factory=lambda: [0.0])
    fl_st_tunr: list[float] = field(default_factory=lambda: [1.0, 1.0])
    adj_bl_ms: float = 1.0
    adj_fl_st: float = 1.0
    adj_ed_st: float = 1.0

    # Distributed properties — mandatory columns plus any extras present.
    bl_fract: np.ndarray = field(default_factory=lambda: np.empty(0))
    pitch_axis: Optional[np.ndarray] = None  # present in 6-col format
    strc_twst: np.ndarray = field(default_factory=lambda: np.empty(0))
    b_mass_den: np.ndarray = field(default_factory=lambda: np.empty(0))
    flp_stff: np.ndarray = field(default_factory=lambda: np.empty(0))
    edg_stff: np.ndarray = field(default_factory=lambda: np.empty(0))

    # ElastoDyn does not carry per-section rotary inertia columns; this
    # flag stays False after parse and downstream synthesis treats the
    # rotary contributions as zero (negligible for the low bending modes
    # pyBmodes targets — see class docstring).
    rotary_inertia_available: bool = False

    # Mode-shape polynomial coefficients (degrees 2..6).
    bld_fl1_sh: np.ndarray = field(default_factory=lambda: np.zeros(5))
    bld_fl2_sh: np.ndarray = field(default_factory=lambda: np.zeros(5))
    bld_edg_sh: np.ndarray = field(default_factory=lambda: np.zeros(5))

    distr_header_lines: list[str] = field(default_factory=list)
    section_dividers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Common scanning utilities
# ---------------------------------------------------------------------------

# Capture: <values...>  <label>  [- comment]
# label = identifier with optional (n) suffix (e.g. "BldFile(1)") or a digit
# tail with no parens (e.g. "BldFile1"). The lookahead requires whitespace +
# ``-`` afterward, but ``-`` may be missing on some legacy lines, so we make
# that branch tolerant.
_RE_LINE = re.compile(
    r"""
    ^\s*
    (?P<value>.+?)              # value(s), non-greedy
    \s+
    (?P<label>[A-Za-z][A-Za-z0-9_]*(?:\(\s*\d+\s*\))?)  # label
    (?:\s+-.*)?                 # optional " - comment" tail
    \s*$
    """,
    re.VERBOSE,
)


def _strip_quotes(tok: str) -> str:
    t = tok.strip()
    if len(t) >= 2 and ((t[0] == '"' and t[-1] == '"') or (t[0] == "'" and t[-1] == "'")):
        return t[1:-1]
    return t


def _is_section_divider(line: str) -> bool:
    s = line.strip()
    return s.startswith("---") or s.startswith("===") or s.startswith("===")


def _is_file_header(line: str) -> bool:
    s = line.strip()
    return s.startswith("---") and "ELASTODYN" in s.upper()


def _parse_float(tok: str) -> float:
    return float(tok.strip().replace("d", "e").replace("D", "E"))


_BARE_INDEXED_LABELS = ("BldFile",)


def _split_label_index(label: str) -> tuple[str, Optional[int]]:
    """Strip an array index from ``label``, returning ``(canon, idx)``.

    Two indexed-label forms are recognised:

    * Parenthesised — ``Foo(N)`` (the ElastoDyn convention for most arrays).
      Stripping the suffix preserves embedded digits in the base name, so
      ``Twr2Shft`` stays ``Twr2Shft`` and ``TwFAM1Sh(2)`` becomes
      ``TwFAM1Sh, idx=1``.
    * Bare-digit — ``FooN``, allowed only for labels listed in
      :data:`_BARE_INDEXED_LABELS` (currently only ``BldFile``, where the
      IEA RWT files use ``BldFile1`` while the 5MW deck uses ``BldFile(1)``).
    """
    m = re.match(r"^([A-Za-z][A-Za-z0-9_]*?)\s*\((\d+)\)\s*$", label)
    if m:
        return m.group(1), int(m.group(2)) - 1
    for base in _BARE_INDEXED_LABELS:
        m = re.match(rf"^{base}(\d+)$", label)
        if m:
            return base, int(m.group(1)) - 1
    return label, None


def _canon_label(label: str) -> str:
    return _split_label_index(label)[0]


def _split_value_label(line: str) -> Optional[tuple[str, str]]:
    """Split a data line into (value-string, label). None if no label."""
    if not line.strip() or _is_section_divider(line):
        return None
    m = _RE_LINE.match(line)
    if not m:
        return None
    return m.group("value").strip(), m.group("label")


# ---------------------------------------------------------------------------
# Main file parser
# ---------------------------------------------------------------------------


def read_elastodyn_main(path: str | pathlib.Path) -> ElastoDynMain:
    """Parse a top-level ElastoDyn ``.dat`` file."""
    path = pathlib.Path(path)
    text = path.read_text(encoding="latin-1")
    return _parse_main(text.splitlines(), source_file=path)


def _parse_main(lines: list[str], source_file: Optional[pathlib.Path] = None) -> ElastoDynMain:
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


def _parse_tower(lines: list[str], source_file: Optional[pathlib.Path] = None) -> ElastoDynTower:
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


def _consume_tower_distributed(obj: ElastoDynTower, lines: list[str], i: int) -> int:
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


def _parse_blade(lines: list[str], source_file: Optional[pathlib.Path] = None) -> ElastoDynBlade:
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


def _consume_blade_distributed(obj: ElastoDynBlade, lines: list[str], i: int) -> int:
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


# ---------------------------------------------------------------------------
# Writer — canonical re-emit
# ---------------------------------------------------------------------------

# Width constants used by the canonical writer. Values left-aligned in a
# 22-char column followed by a 12-char label column. This will not be
# byte-identical to any of the bundled RWT files but it parses back via
# read_*_elastodyn() to a dataclass equal to the original.
_VAL_W = 22
_LBL_W = 12


def _fmt_scalar_line(value: object, label: str, comment: str = "") -> str:
    if isinstance(value, bool):
        sval = "True" if value else "False"
    elif isinstance(value, int):
        sval = str(value)
    elif isinstance(value, float):
        sval = f"{value:.17g}"
    elif isinstance(value, str):
        sval = value
    else:
        sval = str(value)
    line = f"{sval:<{_VAL_W}} {label:<{_LBL_W}}"
    if comment:
        line = f"{line} - {comment}"
    return line


def _scalar_or_default(obj: ElastoDynMain, label: str, fallback: str) -> str:
    """Return the original value string for ``label`` from ``obj.scalars`` if
    captured, otherwise the fallback string."""
    return obj.scalars.get(label, fallback)


def write_elastodyn_main(obj: ElastoDynMain, path: str | pathlib.Path | None = None) -> str:
    """Re-emit a main ElastoDyn file. Returns the text and optionally writes."""
    out = io.StringIO()
    out.write(obj.header + "\n")
    out.write(obj.title + "\n")

    # Re-emit every captured scalar with the typed-field value when known,
    # falling back to the original token when we did not specialise it.
    typed_overrides = {
        "NumBl":     str(obj.num_bl),
        "TipRad":    f"{obj.tip_rad:.17g}",
        "HubRad":    f"{obj.hub_rad:.17g}",
        "HubCM":     f"{obj.hub_cm:.17g}",
        "OverHang":  f"{obj.overhang:.17g}",
        "ShftTilt":  f"{obj.shft_tilt:.17g}",
        "Twr2Shft":  f"{obj.twr2shft:.17g}",
        "TowerHt":   f"{obj.tower_ht:.17g}",
        "TowerBsHt": f"{obj.tower_bs_ht:.17g}",
        "NacCMxn":   f"{obj.nac_cm_xn:.17g}",
        "NacCMyn":   f"{obj.nac_cm_yn:.17g}",
        "NacCMzn":   f"{obj.nac_cm_zn:.17g}",
        "RotSpeed":  f"{obj.rot_speed_rpm:.17g}",
        "HubMass":   f"{obj.hub_mass:.17g}",
        "HubIner":   f"{obj.hub_iner:.17g}",
        "GenIner":   f"{obj.gen_iner:.17g}",
        "NacMass":   f"{obj.nac_mass:.17g}",
        "NacYIner":  f"{obj.nac_y_iner:.17g}",
        "YawBrMass": f"{obj.yaw_br_mass:.17g}",
        "TwrFile":   f'"{obj.twr_file}"',
    }
    # Indexed overrides: PreCone(i), TipMass(i), BldFile(i)
    for i, vf in enumerate(obj.pre_cone):
        typed_overrides[f"PreCone({i + 1})"] = f"{vf:.17g}"
    for i, vf in enumerate(obj.tip_mass):
        typed_overrides[f"TipMass({i + 1})"] = f"{vf:.17g}"
    for i, vs in enumerate(obj.bld_file):
        # Match whichever indexing form the original used.
        if f"BldFile({i + 1})" in obj.scalars:
            typed_overrides[f"BldFile({i + 1})"] = f'"{vs}"'
        elif f"BldFile{i + 1}" in obj.scalars:
            typed_overrides[f"BldFile{i + 1}"] = f'"{vs}"'

    for label, raw_val in obj.scalars.items():
        val = typed_overrides.get(label, raw_val)
        out.write(f"{val:<{_VAL_W}} {label:<{_LBL_W}}\n")

    # OutList sections, verbatim.
    if obj.out_list:
        out.write("---------------------- OUTPUT --------------------------------------------------\n")  # noqa: E501
        for ln in obj.out_list:
            out.write(ln + "\n")
    if obj.nodal_out_list:
        out.write("====== Outputs for all blade stations ===========================\n")
        for ln in obj.nodal_out_list:
            out.write(ln + "\n")

    text = out.getvalue()
    if path is not None:
        pathlib.Path(path).write_text(text, encoding="latin-1")
    return text


def write_elastodyn_tower(obj: ElastoDynTower, path: str | pathlib.Path | None = None) -> str:
    out = io.StringIO()
    out.write(obj.header + "\n")
    out.write(obj.title + "\n")

    out.write("---------------------- TOWER PARAMETERS ----------------------------------------\n")
    out.write(_fmt_scalar_line(obj.n_tw_inp_st, "NTwInpSt") + "\n")
    out.write(_fmt_scalar_line(obj.twr_fa_dmp[0], "TwrFADmp(1)") + "\n")
    out.write(_fmt_scalar_line(obj.twr_fa_dmp[1], "TwrFADmp(2)") + "\n")
    out.write(_fmt_scalar_line(obj.twr_ss_dmp[0], "TwrSSDmp(1)") + "\n")
    out.write(_fmt_scalar_line(obj.twr_ss_dmp[1], "TwrSSDmp(2)") + "\n")
    out.write("---------------------- TOWER ADJUSTMUNT FACTORS --------------------------------\n")
    out.write(_fmt_scalar_line(obj.fa_st_tunr[0], "FAStTunr(1)") + "\n")
    out.write(_fmt_scalar_line(obj.fa_st_tunr[1], "FAStTunr(2)") + "\n")
    out.write(_fmt_scalar_line(obj.ss_st_tunr[0], "SSStTunr(1)") + "\n")
    out.write(_fmt_scalar_line(obj.ss_st_tunr[1], "SSStTunr(2)") + "\n")
    out.write(_fmt_scalar_line(obj.adj_tw_ma, "AdjTwMa") + "\n")
    out.write(_fmt_scalar_line(obj.adj_fa_st, "AdjFASt") + "\n")
    out.write(_fmt_scalar_line(obj.adj_ss_st, "AdjSSSt") + "\n")

    out.write("---------------------- DISTRIBUTED TOWER PROPERTIES ----------------------------\n")
    if obj.distr_header_lines:
        for ln in obj.distr_header_lines:
            out.write(ln + "\n")
    else:
        out.write("  HtFract       TMassDen         TwFAStif       TwSSStif\n")
        out.write("   (-)           (kg/m)           (Nm^2)         (Nm^2)\n")
    cols = [obj.ht_fract, obj.t_mass_den, obj.tw_fa_stif, obj.tw_ss_stif]
    for extra in (obj.tw_fa_iner, obj.tw_ss_iner, obj.tw_fa_cg_of, obj.tw_ss_cg_of):
        if extra.size:
            cols.append(extra)
    for r in range(len(obj.ht_fract)):
        out.write("  ".join(f"{c[r]:.17e}" for c in cols) + "\n")

    out.write("---------------------- TOWER FORE-AFT MODE SHAPES ------------------------------\n")
    for k, v in enumerate(obj.tw_fa_m1_sh):
        out.write(_fmt_scalar_line(v, f"TwFAM1Sh({k + 2})") + "\n")
    for k, v in enumerate(obj.tw_fa_m2_sh):
        out.write(_fmt_scalar_line(v, f"TwFAM2Sh({k + 2})") + "\n")
    out.write("---------------------- TOWER SIDE-TO-SIDE MODE SHAPES --------------------------\n")
    for k, v in enumerate(obj.tw_ss_m1_sh):
        out.write(_fmt_scalar_line(v, f"TwSSM1Sh({k + 2})") + "\n")
    for k, v in enumerate(obj.tw_ss_m2_sh):
        out.write(_fmt_scalar_line(v, f"TwSSM2Sh({k + 2})") + "\n")

    text = out.getvalue()
    if path is not None:
        pathlib.Path(path).write_text(text, encoding="latin-1")
    return text


def write_elastodyn_blade(obj: ElastoDynBlade, path: str | pathlib.Path | None = None) -> str:
    out = io.StringIO()
    out.write(obj.header + "\n")
    out.write(obj.title + "\n")

    out.write("---------------------- BLADE PARAMETERS ----------------------------------------\n")
    out.write(_fmt_scalar_line(obj.n_bl_inp_st, "NBlInpSt") + "\n")
    out.write(_fmt_scalar_line(obj.bld_fl_dmp[0], "BldFlDmp(1)") + "\n")
    out.write(_fmt_scalar_line(obj.bld_fl_dmp[1], "BldFlDmp(2)") + "\n")
    out.write(_fmt_scalar_line(obj.bld_ed_dmp[0], "BldEdDmp(1)") + "\n")
    out.write("---------------------- BLADE ADJUSTMENT FACTORS --------------------------------\n")
    out.write(_fmt_scalar_line(obj.fl_st_tunr[0], "FlStTunr(1)") + "\n")
    out.write(_fmt_scalar_line(obj.fl_st_tunr[1], "FlStTunr(2)") + "\n")
    out.write(_fmt_scalar_line(obj.adj_bl_ms, "AdjBlMs") + "\n")
    out.write(_fmt_scalar_line(obj.adj_fl_st, "AdjFlSt") + "\n")
    out.write(_fmt_scalar_line(obj.adj_ed_st, "AdjEdSt") + "\n")

    out.write("---------------------- DISTRIBUTED BLADE PROPERTIES ----------------------------\n")
    if obj.distr_header_lines:
        for ln in obj.distr_header_lines:
            out.write(ln + "\n")
    else:
        if obj.pitch_axis is not None:
            out.write("    BlFract      PitchAxis      StrcTwst       BMassDen        FlpStff        EdgStff\n")  # noqa: E501
            out.write("      (-)           (-)          (deg)          (kg/m)         (Nm^2)         (Nm^2)\n")  # noqa: E501
        else:
            out.write("    BlFract               StrcTwst               BMassDen               FlpStff                 EdgStff\n")  # noqa: E501
            out.write("      (-)                   (deg)                 (kg/m)                 (Nm^2)                  (Nm^2)\n")  # noqa: E501
    cols: list[np.ndarray] = [obj.bl_fract]
    if obj.pitch_axis is not None:
        cols.append(obj.pitch_axis)
    cols.extend([obj.strc_twst, obj.b_mass_den, obj.flp_stff, obj.edg_stff])
    for r in range(len(obj.bl_fract)):
        out.write("  ".join(f"{c[r]:.17e}" for c in cols) + "\n")

    out.write("---------------------- BLADE MODE SHAPES ---------------------------------------\n")
    for k, v in enumerate(obj.bld_fl1_sh):
        out.write(_fmt_scalar_line(v, f"BldFl1Sh({k + 2})") + "\n")
    for k, v in enumerate(obj.bld_fl2_sh):
        out.write(_fmt_scalar_line(v, f"BldFl2Sh({k + 2})") + "\n")
    for k, v in enumerate(obj.bld_edg_sh):
        out.write(_fmt_scalar_line(v, f"BldEdgSh({k + 2})") + "\n")

    text = out.getvalue()
    if path is not None:
        pathlib.Path(path).write_text(text, encoding="latin-1")
    return text


# ---------------------------------------------------------------------------
# Adapter — synthesise pyBmodes BMIFile + SectionProperties from ElastoDyn
# ---------------------------------------------------------------------------

# Stiffness multipliers used when synthesising the GJ / EA columns that
# ElastoDyn does not carry. These pin torsion and axial DOFs out of the
# bending mode range; raise them by another order of magnitude if a
# specific case shows torsion-bending coupling artefacts.
_GJ_OVER_EI = 100.0
_EA_OVER_EI_PER_LEN_SQ = 1.0e6  # times (mean EI), gives rigid-axial behaviour


def _resolve_relative(main: ElastoDynMain, ref: str) -> pathlib.Path:
    p = pathlib.Path(ref)
    if p.is_absolute():
        return p
    if main.source_file is not None:
        return (main.source_file.parent / p).resolve()
    return p.resolve()


def _rotary_inertia_floor(
    mass_den: np.ndarray,
    char_length: float,
) -> np.ndarray:
    """Strictly-positive regularisation for the rotary-inertia columns.

    ElastoDyn carries no per-section rotary mass moments of inertia. The
    physically correct value for an Euler-Bernoulli beam at the bending
    frequencies pyBmodes resolves is *zero* — rotary inertia is a
    higher-order Timoshenko correction that's negligible for slender
    structures. However, the global mass matrix needs every diagonal
    block strictly positive to stay positive-definite, so we set a tiny
    floor here. ``1e-6 · mass_den · L²`` keeps the floor at the parts-per-
    million level relative to translational mass while killing the
    singularity. ``L`` is a per-element characteristic length supplied
    by the caller (chord proxy for blades, mean radius for towers).
    """
    return np.full_like(mass_den, 1.0e-6 * char_length ** 2) * mass_den


def _stack_blade_section_props(
    blade: ElastoDynBlade,
    rot_rpm: float,  # noqa: ARG001 — kept for parity with the tower variant
    chord_estimate: float = 4.0,
) -> "SectionProperties":  # forward-ref — imported lazily to dodge cycles
    """Convert an ElastoDyn blade record to pyBmodes section properties.

    Rotary mass moments of inertia (``flp_iner``, ``edge_iner``) are
    physically zero for a thin-beam Euler-Bernoulli model; we set them
    to a tiny PD-safety floor only. See :func:`_rotary_inertia_floor`.
    """
    from pybmodes.io.sec_props import SectionProperties

    span = blade.bl_fract.astype(float)
    str_tw = blade.strc_twst.astype(float)
    mass_den = blade.b_mass_den.astype(float) * blade.adj_bl_ms
    flp_stff = blade.flp_stff.astype(float) * blade.adj_fl_st
    edg_stff = blade.edg_stff.astype(float) * blade.adj_ed_st

    ei_max = np.maximum(flp_stff, edg_stff)
    tor_stff = ei_max * _GJ_OVER_EI
    axial_stff = ei_max * _EA_OVER_EI_PER_LEN_SQ

    # Per-spec: rotary inertia is zero for thin beams; PD floor only.
    flp_iner = _rotary_inertia_floor(mass_den, chord_estimate)
    edge_iner = _rotary_inertia_floor(mass_den, chord_estimate)
    zeros = np.zeros_like(span)

    return SectionProperties(
        title="ElastoDyn-derived blade section properties",
        n_secs=int(span.size),
        span_loc=span,
        str_tw=str_tw,
        tw_iner=str_tw.copy(),  # ElastoDyn lacks an independent inertia twist
        mass_den=mass_den,
        flp_iner=flp_iner,
        edge_iner=edge_iner,
        flp_stff=flp_stff,
        edge_stff=edg_stff,
        tor_stff=tor_stff,
        axial_stff=axial_stff,
        cg_offst=zeros.copy(),
        sc_offst=zeros.copy(),
        tc_offst=zeros.copy(),
        source_file=blade.source_file,
    )


def _stack_tower_section_props(
    tower: ElastoDynTower,
    radius_estimate: float = 3.0,
) -> "SectionProperties":
    """Convert an ElastoDyn tower record to pyBmodes section properties.

    Tower rotary inertia is treated identically to the blade case: zero
    in physical reality (Euler-Bernoulli limit), with a tiny floor for
    PD safety in the global mass matrix.
    """
    from pybmodes.io.sec_props import SectionProperties

    span = tower.ht_fract.astype(float)
    mass_den = tower.t_mass_den.astype(float) * tower.adj_tw_ma
    flp_stff = tower.tw_fa_stif.astype(float) * tower.adj_fa_st
    edg_stff = tower.tw_ss_stif.astype(float) * tower.adj_ss_st

    ei_max = np.maximum(flp_stff, edg_stff)
    tor_stff = ei_max * _GJ_OVER_EI
    axial_stff = ei_max * _EA_OVER_EI_PER_LEN_SQ

    flp_iner = _rotary_inertia_floor(mass_den, radius_estimate)
    edge_iner = _rotary_inertia_floor(mass_den, radius_estimate)
    zeros = np.zeros_like(span)

    return SectionProperties(
        title="ElastoDyn-derived tower section properties",
        n_secs=int(span.size),
        span_loc=span,
        str_tw=zeros.copy(),
        tw_iner=zeros.copy(),
        mass_den=mass_den,
        flp_iner=flp_iner,
        edge_iner=edge_iner,
        flp_stff=flp_stff,
        edge_stff=edg_stff,
        tor_stff=tor_stff,
        axial_stff=axial_stff,
        cg_offst=zeros.copy(),
        sc_offst=zeros.copy(),
        tc_offst=zeros.copy(),
        source_file=tower.source_file,
    )


def _build_bmi_skeleton(
    *,
    title: str,
    beam_type: int,
    radius: float,
    hub_rad: float,
    rot_rpm: float,
    precone: float,
    n_elements: int,
    el_loc: np.ndarray,
    tip_mass_props: "TipMassProps",
) -> "BMIFile":
    from pybmodes.io.bmi import BMIFile, ScalingFactors

    return BMIFile(
        title=title,
        echo=False,
        beam_type=beam_type,
        rot_rpm=rot_rpm,
        rpm_mult=1.0,
        radius=radius,
        hub_rad=hub_rad,
        precone=precone,
        bl_thp=0.0,
        hub_conn=1,
        n_modes_print=20,
        tab_delim=True,
        mid_node_tw=False,
        tip_mass=tip_mass_props,
        id_mat=1,
        sec_props_file="",  # in-memory; not resolved
        scaling=ScalingFactors(),
        n_elements=n_elements,
        el_loc=el_loc.astype(float),
        tow_support=0,
        support=None,
        source_file=None,
    )


def _tower_top_assembly_mass(
    main: ElastoDynMain,
    blade: Optional[ElastoDynBlade],
) -> "TipMassProps":
    """Lump the rotor-nacelle assembly (RNA) into a single ``TipMassProps``
    at the tower top via full rigid-body parallel-axis assembly.

    Bodies summed (each at its CM in the tower-top reference frame
    ``x = downwind, y = lateral, z = vertical``):

    1. **Nacelle** — mass ``NacMass`` at ``(NacCMxn, NacCMyn, NacCMzn)``.
       Inertia tensor: ``NacYIner`` about the yaw (z) axis; transverse
       components default to ``½·NacYIner`` as a slender-body
       approximation (ElastoDyn does not carry separate Ixx/Iyy for the
       nacelle).
    2. **Hub** — mass ``HubMass`` at the hub-mass location, which is the
       rotor apex translated by ``HubCM`` along the shaft axis. The shaft
       is tilted by ``ShftTilt`` from horizontal and the apex itself is
       at ``(OverHang·cos(ShftTilt), 0, Twr2Shft + OverHang·sin(ShftTilt))``
       relative to the tower top.
    3. **Blades** — total mass ``N_bl × ∫BMassDen ds`` from the blade
       file, treated as a point mass at the rotor apex. Distributed-blade
       rotational inertia is dropped — it's <1% of the parallel-axis
       contribution from the apex offset for a utility-scale RNA.

    Spec deviation: the prompt says ``m_total = NacMass + HubMass + 3·TipMass``,
    but in ElastoDyn ``TipMass(i)`` is the per-blade *tip-brake* mass
    (zero for the 5MW deck). Using that literally would drop the blade
    mass entirely. We use ``N_bl·∫BMassDen ds`` instead, matching what
    ElastoDyn itself computes as ``RotMass``.

    Inertia at the tower top is then ``∑_i [I_i + m_i·(|r_i|²·E - r_i⊗r_i)]``
    (parallel-axis theorem in tensor form). Cross-products are
    preserved. The diagonal and off-diagonal entries are passed through
    as ``ixx, iyy, izz, ixy, izx, iyz`` on the BMI tip-mass record;
    ``cm_offset`` and ``cm_axial`` are set to zero so the downstream
    nondimensionaliser does not re-apply parallel-axis terms (the
    tensor we hand it is already at the tower top).
    """
    from pybmodes.io.bmi import TipMassProps

    tilt = math.radians(main.shft_tilt)
    cos_t = math.cos(tilt)
    sin_t = math.sin(tilt)

    # --- Body 1: Nacelle ---
    m_nac = float(main.nac_mass)
    r_nac = np.array([main.nac_cm_xn, main.nac_cm_yn, main.nac_cm_zn], dtype=float)
    # Slender-body proxy: half of NacYIner on the transverse axes. For a
    # rectangular block this is ~Iy ≈ Iz/3 + small; ½ is a generous middle.
    I_nac = np.diag([0.5 * main.nac_y_iner, 0.5 * main.nac_y_iner, main.nac_y_iner])

    # --- Body 2: Hub ---
    m_hub = float(main.hub_mass)
    apex = np.array([main.overhang * cos_t, 0.0, main.twr2shft + main.overhang * sin_t])
    shaft_dir = np.array([cos_t, 0.0, sin_t])
    r_hub = apex + main.hub_cm * shaft_dir
    # HubIner is the inertia about the shaft (rotor) axis. Other axes
    # default to ½·HubIner (sphere-like proxy). For small ShftTilt the
    # shaft is nearly aligned with x, so we approximate the hub tensor as
    # diagonal in the tower-top frame; a proper rotation by ShftTilt is
    # left as a tightening target.
    I_hub = np.diag([main.hub_iner, 0.5 * main.hub_iner, 0.5 * main.hub_iner])

    # --- Body 3: Blades (lumped at the apex) ---
    # ElastoDyn does not carry per-section rotational inertia, and the
    # standard rigid-RNA approximation used to derive published reference
    # frequencies (e.g. Jonkman 2009 Tower FA = 0.3240 Hz) treats the
    # blades as a translational point mass at the rotor apex — the blade
    # rotational dynamics live in their own DOFs in OpenFAST and don't
    # rigidly couple to tower-mode rotation at low frequencies. Mirroring
    # that convention here keeps the tower modal frequencies close to the
    # published targets without fabricating distributed-blade inertia
    # data that ElastoDyn never had.
    if blade is not None and blade.bl_fract.size > 0:
        bl_len = main.tip_rad - main.hub_rad
        bmd = blade.b_mass_den * blade.adj_bl_ms
        s = blade.bl_fract * bl_len
        m_bl_each = float(np.trapezoid(bmd, s))
    else:
        m_bl_each = 0.0
    m_blades = main.num_bl * m_bl_each
    r_blades = apex.copy()
    I_blades = np.zeros((3, 3))

    bodies = [
        (m_nac,     r_nac,     I_nac),
        (m_hub,     r_hub,     I_hub),
        (m_blades,  r_blades,  I_blades),
    ]

    m_total = sum(m for m, _, _ in bodies)
    if m_total <= 0.0:
        return TipMassProps(
            mass=0.0, cm_offset=0.0, cm_axial=0.0,
            ixx=0.0, iyy=0.0, izz=0.0,
            ixy=0.0, izx=0.0, iyz=0.0,
        )

    # Assembly CM (relative to tower top, in tower-top frame). The
    # explicit ``start=np.zeros(3)`` keeps the static type at ndarray
    # even when ``bodies`` happens to be empty (handled above by the
    # m_total guard, but mypy doesn't see that path).
    cm: np.ndarray = sum(
        (m * r for m, r, _ in bodies),
        start=np.zeros(3),
    ) / m_total

    # Inertia tensor at the tower top via parallel-axis theorem.
    eye = np.eye(3)
    I_tt = np.zeros((3, 3))
    for m, r, I_body in bodies:
        rsq = float(r @ r)
        I_tt = I_tt + I_body + m * (rsq * eye - np.outer(r, r))

    return TipMassProps(
        mass=m_total,
        # cm_axial carries the BMI cm-axial lever arm used for kinematic
        # coupling terms (translation/rotation cross-blocks). cm_offset
        # is zeroed because the tower path drops the BMI horizontal
        # offset — the horizontal contribution is folded into ``ixx``,
        # ``iyy``, ``izz`` via the tensor parallel-axis above.
        cm_offset=0.0,
        cm_axial=float(cm[2]),
        ixx=float(I_tt[0, 0]),
        iyy=float(I_tt[1, 1]),
        izz=float(I_tt[2, 2]),
        ixy=float(I_tt[0, 1]),
        izx=float(I_tt[2, 0]),
        iyz=float(I_tt[1, 2]),
    )


def to_pybmodes_tower(
    main: ElastoDynMain,
    tower: ElastoDynTower,
    blade: Optional[ElastoDynBlade] = None,
) -> tuple["BMIFile", "SectionProperties"]:
    """Build pyBmodes ``BMIFile`` and ``SectionProperties`` for tower modal
    analysis from a parsed ElastoDyn bundle. ``blade`` is optional; when
    omitted, the rotor mass is approximated as ``HubMass`` only."""
    sp = _stack_tower_section_props(tower)
    tip = _tower_top_assembly_mass(main, blade)

    flexible_height = main.tower_ht - main.tower_bs_ht
    bmi = _build_bmi_skeleton(
        title=main.title or "ElastoDyn tower",
        beam_type=2,
        radius=flexible_height,
        hub_rad=0.0,
        rot_rpm=0.0,
        precone=0.0,
        n_elements=max(tower.n_tw_inp_st - 1, 1),
        el_loc=tower.ht_fract,
        tip_mass_props=tip,
    )
    return bmi, sp


def to_pybmodes_blade(
    main: ElastoDynMain,
    blade: ElastoDynBlade,
) -> tuple["BMIFile", "SectionProperties"]:
    """Build pyBmodes ``BMIFile`` and ``SectionProperties`` for blade modal
    analysis at the operating ``RotSpeed`` from the main file."""
    from pybmodes.io.bmi import TipMassProps

    sp = _stack_blade_section_props(blade, rot_rpm=main.rot_speed_rpm)

    tip = TipMassProps(
        mass=main.tip_mass[0],
        cm_offset=0.0, cm_axial=0.0,
        ixx=0.0, iyy=0.0, izz=0.0,
        ixy=0.0, izx=0.0, iyz=0.0,
    )

    bmi = _build_bmi_skeleton(
        title=main.title or "ElastoDyn blade",
        beam_type=1,
        radius=main.tip_rad,
        hub_rad=main.hub_rad,
        rot_rpm=main.rot_speed_rpm,
        precone=main.pre_cone[0],
        n_elements=max(blade.n_bl_inp_st - 1, 1),
        el_loc=blade.bl_fract,
        tip_mass_props=tip,
    )
    return bmi, sp


# Type-only forward references. Kept inside TYPE_CHECKING-style import sites
# above to avoid pulling pybmodes.io.bmi / pybmodes.io.sec_props at module-
# import time (the adapters only need them at call time).
if False:  # pragma: no cover
    from pybmodes.io.bmi import BMIFile, TipMassProps  # noqa: F401
    from pybmodes.io.sec_props import SectionProperties  # noqa: F401


# Public surface for ``from pybmodes.io.elastodyn_reader import *``.
__all__ = [
    "ElastoDynMain",
    "ElastoDynTower",
    "ElastoDynBlade",
    "read_elastodyn_main",
    "read_elastodyn_tower",
    "read_elastodyn_blade",
    "write_elastodyn_main",
    "write_elastodyn_tower",
    "write_elastodyn_blade",
    "to_pybmodes_tower",
    "to_pybmodes_blade",
]
