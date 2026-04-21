"""Parser for .bmi main input files.

The .bmi format is line-ordered: values precede their labels.  The reader
follows the format conventions:
  ReadCom  → consume one line verbatim (section headers / blank lines)
  ReadStr  → consume one line, return it as a string
  ReadVar  → skip blanks, return first whitespace token of next non-blank line
  ReadAry  → skip blanks, return first N tokens of next non-blank line
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class TipMassProps:
    """Mass and inertia of the blade tip or tower-top concentrated mass."""
    mass: float        # kg
    cm_offset: float   # m  (transverse, along tip-section y-axis)
    cm_axial: float    # m  (axial offset along z)
    ixx: float         # kg·m²
    iyy: float
    izz: float
    ixy: float
    izx: float
    iyz: float


@dataclass
class ScalingFactors:
    """Multiplicative scaling factors applied to all section properties."""
    sec_mass: float = 1.0
    flp_iner: float = 1.0
    lag_iner: float = 1.0
    flp_stff: float = 1.0
    edge_stff: float = 1.0
    tor_stff: float = 1.0
    axial_stff: float = 1.0
    cg_offst: float = 1.0
    sc_offst: float = 1.0
    tc_offst: float = 1.0


@dataclass
class TensionWireSupport:
    """Tension wire (guy wire) support for land-based towers."""
    n_attachments: int
    n_wires: list[int]           # number of wires at each attachment
    node_attach: list[int]       # FE node numbers of attachment points
    wire_stiffness: list[float]  # N/m, one per attachment set
    th_wire: list[float]         # deg, angle w.r.t. tower axis


@dataclass
class PlatformSupport:
    """Offshore floating-platform or monopile support."""
    draft: float            # m, depth of tower base below MSL (negative = above MSL)
    cm_pform: float         # m, platform c.m. depth below MSL
    mass_pform: float       # kg
    i_matrix: np.ndarray    # (6,6) platform structural mass matrix at reference point
    ref_msl: float          # m, reference point depth below MSL
    hydro_M: np.ndarray     # (6,6) hydrodynamic added-mass matrix at reference point
    hydro_K: np.ndarray     # (6,6) hydrostatic stiffness matrix at reference point
    mooring_K: np.ndarray   # (6,6) mooring stiffness matrix at reference point
    distr_m_z: np.ndarray   # m, locations for distributed added mass
    distr_m: np.ndarray     # kg/m
    distr_k_z: np.ndarray   # m, locations for distributed stiffness
    distr_k: np.ndarray     # N/m²
    wires: Optional[TensionWireSupport] = None  # optional tension wires (BModes_JJ format)


@dataclass
class BMIFile:
    """All parameters parsed from a .bmi main input file."""

    # --- general ---
    title: str
    echo: bool
    beam_type: int          # 1 = blade, 2 = tower
    rot_rpm: float          # rpm (before rpm_mult applied)
    rpm_mult: float
    radius: float           # m (rotor tip radius or tower height)
    hub_rad: float          # m (hub radius or tower rigid-base height)
    precone: float          # deg
    bl_thp: float           # deg, blade pitch setting
    hub_conn: int           # 1: cantilevered, 2: free-free, 3: axial+torsion only
    n_modes_print: int
    tab_delim: bool
    mid_node_tw: bool

    # --- tip / tower-top mass ---
    tip_mass: TipMassProps

    # --- section properties ---
    id_mat: int
    sec_props_file: str     # path as given in file (may be relative)

    # --- scaling ---
    scaling: ScalingFactors

    # --- FE discretization ---
    n_elements: int                         # nselt
    el_loc: np.ndarray                      # normalized, length n_elements+1

    # --- tower support (beam_type == 2 only) ---
    tow_support: int = 0                    # 0: none, 1: wires, 2: platform
    support: Optional[TensionWireSupport | PlatformSupport] = None

    # --- provenance ---
    source_file: Optional[pathlib.Path] = None

    def resolve_sec_props_path(self) -> pathlib.Path:
        """Return absolute path to the section properties file."""
        p = pathlib.Path(self.sec_props_file)
        if p.is_absolute():
            return p
        if self.source_file is not None:
            return (self.source_file.parent / p).resolve()
        return p.resolve()


# ---------------------------------------------------------------------------
# Internal reader
# ---------------------------------------------------------------------------

class _LineReader:
    """Stateful line-by-line reader following .bmi file format conventions."""

    def __init__(self, lines: list[str]):
        # Preserve all lines including blank ones; strip only CRLF endings and
        # inline ! comments.
        processed = []
        for raw in lines:
            line = raw.rstrip('\r\n')
            # Strip inline ! comment (but not inside quoted strings)
            bang = _find_comment_start(line)
            if bang >= 0:
                line = line[:bang]
            processed.append(line)
        self._lines = processed
        self._pos = 0

    # ReadCom: advance one line verbatim (no blank-skip)
    def read_com(self) -> None:
        self._pos += 1

    # ReadStr: return raw stripped content of next line
    def read_str(self) -> str:
        line = self._lines[self._pos]
        self._pos += 1
        return line.strip()

    # ReadVar: skip blanks, return first token of next non-blank line
    def read_var(self) -> str:
        self._skip_blanks()
        line = self._lines[self._pos]
        self._pos += 1
        tokens = line.split()
        if not tokens:
            raise ValueError(f"Empty data line at position {self._pos}")
        return tokens[0]

    # ReadAry: skip blanks, return first n tokens of next non-blank line
    def read_ary(self, n: int) -> list[str]:
        self._skip_blanks()
        line = self._lines[self._pos]
        self._pos += 1
        return line.split()[:n]

    def _skip_blanks(self) -> None:
        while self._pos < len(self._lines) and not self._lines[self._pos].strip():
            self._pos += 1
        if self._pos >= len(self._lines):
            raise EOFError("Unexpected end of input file")

    def peek_token(self) -> str:
        """Return first token of next non-blank line without advancing position."""
        pos = self._pos
        while pos < len(self._lines) and not self._lines[pos].strip():
            pos += 1
        if pos >= len(self._lines):
            return ''
        tokens = self._lines[pos].split()
        return tokens[0] if tokens else ''


def _find_comment_start(line: str) -> int:
    """Return index of the first ! not inside a quoted string, or -1."""
    in_sq = False
    in_dq = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif ch == '!' and not in_sq and not in_dq:
            return i
    return -1


# ---------------------------------------------------------------------------
# Value converters
# ---------------------------------------------------------------------------

def _parse_bool(token: str) -> bool:
    t = token.strip().lower().strip("'\"")
    if t in ('t', 'true'):
        return True
    if t in ('f', 'false'):
        return False
    raise ValueError(f"Cannot parse boolean from: {token!r}")


def _parse_float(token: str) -> float:
    return float(token.strip().strip("'\"").replace('d', 'e').replace('D', 'E'))


def _parse_int(token: str) -> int:
    return int(token.strip().strip("'\""))


def _parse_str(token: str) -> str:
    return token.strip().strip("'\"")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_bmi(path: str | pathlib.Path) -> BMIFile:
    """Parse a .bmi main input file and return a :class:`BMIFile`."""
    path = pathlib.Path(path)
    lines = path.read_text(encoding='latin-1').splitlines()
    r = _LineReader(lines)
    return _parse(r, source_file=path)


def _parse(r: _LineReader, source_file: pathlib.Path | None = None) -> BMIFile:
    # ---- header + title ----
    r.read_com()                           # file header line
    title = r.read_str()                   # user title

    # ---- general parameters ----
    r.read_com()                           # blank line
    r.read_com()                           # "--------- General parameters"
    echo       = _parse_bool(r.read_var())
    beam_type  = _parse_int(r.read_var())
    rot_rpm    = _parse_float(r.read_var())
    rpm_mult   = _parse_float(r.read_var())
    radius     = _parse_float(r.read_var())
    hub_rad    = _parse_float(r.read_var())
    precone    = _parse_float(r.read_var())
    bl_thp     = _parse_float(r.read_var())
    hub_conn   = _parse_int(r.read_var())
    n_modes_print = _parse_int(r.read_var())
    tab_delim  = _parse_bool(r.read_var())
    mid_node_tw = _parse_bool(r.read_var())

    # ---- tip / tower-top mass ----
    r.read_com()                           # blank
    r.read_com()                           # "--------- Blade-tip or tower-top mass"
    tip_mass_kg = _parse_float(r.read_var())
    cm_loc      = _parse_float(r.read_var())
    cm_axial    = _parse_float(r.read_var())
    ixx_tip     = _parse_float(r.read_var())
    iyy_tip     = _parse_float(r.read_var())
    izz_tip     = _parse_float(r.read_var())
    ixy_tip     = _parse_float(r.read_var())
    izx_tip     = _parse_float(r.read_var())
    iyz_tip     = _parse_float(r.read_var())
    tip_mass = TipMassProps(
        mass=tip_mass_kg, cm_offset=cm_loc, cm_axial=cm_axial,
        ixx=ixx_tip, iyy=iyy_tip, izz=izz_tip,
        ixy=ixy_tip, izx=izx_tip, iyz=iyz_tip,
    )

    # ---- distributed-property identifiers ----
    r.read_com()                           # blank
    r.read_com()                           # "--------- Distributed-property identifiers"
    id_mat         = _parse_int(r.read_var())
    sec_props_file = _parse_str(r.read_var())

    # ---- property scaling factors ----
    r.read_com()                           # blank
    r.read_com()                           # "Property scaling factors..."
    sec_mass_mult   = _parse_float(r.read_var())
    flp_iner_mult   = _parse_float(r.read_var())
    lag_iner_mult   = _parse_float(r.read_var())
    flp_stff_mult   = _parse_float(r.read_var())
    edge_stff_mult  = _parse_float(r.read_var())
    tor_stff_mult   = _parse_float(r.read_var())
    axial_stff_mult = _parse_float(r.read_var())
    cg_offst_mult   = _parse_float(r.read_var())
    sc_offst_mult   = _parse_float(r.read_var())
    tc_offst_mult   = _parse_float(r.read_var())
    scaling = ScalingFactors(
        sec_mass=sec_mass_mult, flp_iner=flp_iner_mult, lag_iner=lag_iner_mult,
        flp_stff=flp_stff_mult, edge_stff=edge_stff_mult, tor_stff=tor_stff_mult,
        axial_stff=axial_stff_mult, cg_offst=cg_offst_mult,
        sc_offst=sc_offst_mult, tc_offst=tc_offst_mult,
    )

    # ---- FE discretization ----
    r.read_com()                           # blank
    r.read_com()                           # "--------- Finite element discretization"
    n_elements = _parse_int(r.read_var())
    r.read_com()                           # "Distance of element boundary nodes..."
    el_loc = np.array([_parse_float(t) for t in r.read_ary(n_elements + 1)])

    # ---- tower support (only when beam_type == 2) ----
    tow_support = 0
    support: TensionWireSupport | PlatformSupport | None = None

    if beam_type == 2:
        r.read_com()                       # blank
        r.read_com()                       # "--------- Properties of tower support subsystem"
        tow_support = _parse_int(r.read_var())

        if tow_support == 1:
            # Detect BModes_JJ format (platform) vs original (tension wires) by peeking:
            # BModes_JJ: next token is a float (the 'draft' value)
            # Original:  next token is a non-numeric label line ("Tension-wires data")
            if _is_float(r.peek_token()):
                tow_support = 2            # remap to internal platform code
                support = _parse_platform_jj(r)
            else:
                support = _parse_tension_wires(r)
        elif tow_support == 2:             # offshore platform (original format)
            support = _parse_platform(r)

    return BMIFile(
        title=title,
        echo=echo,
        beam_type=beam_type,
        rot_rpm=rot_rpm,
        rpm_mult=rpm_mult,
        radius=radius,
        hub_rad=hub_rad,
        precone=precone,
        bl_thp=bl_thp,
        hub_conn=hub_conn,
        n_modes_print=n_modes_print,
        tab_delim=tab_delim,
        mid_node_tw=mid_node_tw,
        tip_mass=tip_mass,
        id_mat=id_mat,
        sec_props_file=sec_props_file,
        scaling=scaling,
        n_elements=n_elements,
        el_loc=el_loc,
        tow_support=tow_support,
        support=support,
        source_file=source_file,
    )


def _is_float(s: str) -> bool:
    """Return True if the string can be parsed as a float."""
    try:
        float(s.strip().strip("'\"").replace('d', 'e').replace('D', 'E'))
        return True
    except (ValueError, AttributeError):
        return False


def _parse_tension_wires(r: _LineReader) -> TensionWireSupport:
    r.read_com()                           # "Tension wires data"
    n_att = _parse_int(r.read_var())
    n_wires      = [_parse_int(t)   for t in r.read_ary(n_att)]
    node_attach  = [_parse_int(t)   for t in r.read_ary(n_att)]
    wire_stiff   = [_parse_float(t) for t in r.read_ary(n_att)]
    th_wire      = [_parse_float(t) for t in r.read_ary(n_att)]
    return TensionWireSupport(
        n_attachments=n_att,
        n_wires=n_wires,
        node_attach=node_attach,
        wire_stiffness=wire_stiff,
        th_wire=th_wire,
    )


def _parse_platform(r: _LineReader) -> PlatformSupport:
    draft      = _parse_float(r.read_var())
    cm_pform   = _parse_float(r.read_var())
    mass_pform = _parse_float(r.read_var())

    # inertia matrix: diagonal set to mass_pform externally, read 3×3 rotational block
    i_mat = np.zeros((6, 6))
    for i in range(3):
        i_mat[i, i] = mass_pform
    r.read_com()                           # "Platform inertia matrix comment line"
    for row in range(3, 6):
        i_mat[row, 3:6] = [_parse_float(t) for t in r.read_ary(3)]

    ref_msl = _parse_float(r.read_var())

    r.read_com()                           # "hydrodynamic added-mass matrix comment"
    hydro_M = np.zeros((6, 6))
    for row in range(6):
        hydro_M[row, :] = [_parse_float(t) for t in r.read_ary(6)]

    r.read_com()                           # "hydrodynamic stiffness matrix comment"
    hydro_K = np.zeros((6, 6))
    for row in range(6):
        hydro_K[row, :] = [_parse_float(t) for t in r.read_ary(6)]

    r.read_com()                           # "mooring stiffness matrix comment"
    mooring_K = np.zeros((6, 6))
    for row in range(6):
        mooring_K[row, :] = [_parse_float(t) for t in r.read_ary(6)]

    # distributed added-mass
    r.read_com()                           # blank
    r.read_com()                           # "Distributed added-mass..."
    n_m = _parse_int(r.read_var())
    if n_m > 0:
        z_distr_m = np.array([_parse_float(t) for t in r.read_ary(n_m)])
        distr_m   = np.array([_parse_float(t) for t in r.read_ary(n_m)])
    else:
        z_distr_m = np.array([])
        distr_m   = np.array([])
        if _is_float(r.peek_token()):
            r.read_com()
            if _is_float(r.peek_token()):
                r.read_com()

    # distributed stiffness
    r.read_com()                           # blank
    r.read_com()                           # "Distributed elastic stiffness..."
    n_k = _parse_int(r.read_var())
    if n_k > 0:
        z_distr_k = np.array([_parse_float(t) for t in r.read_ary(n_k)])
        distr_k   = np.array([_parse_float(t) for t in r.read_ary(n_k)])
    else:
        z_distr_k = np.array([])
        distr_k   = np.array([])
        if _is_float(r.peek_token()):
            r.read_com()
            if _is_float(r.peek_token()):
                r.read_com()

    return PlatformSupport(
        draft=draft, cm_pform=cm_pform, mass_pform=mass_pform,
        i_matrix=i_mat, ref_msl=ref_msl,
        hydro_M=hydro_M, hydro_K=hydro_K, mooring_K=mooring_K,
        distr_m_z=z_distr_m, distr_m=distr_m,
        distr_k_z=z_distr_k, distr_k=distr_k,
    )


def _parse_platform_jj(r: _LineReader) -> PlatformSupport:
    """Parse offshore platform block in BModes_JJ format (tow_support=1 in file).

    Reads: draft, cm/mass/inertia, ref_msl, hydro_M, hydro_K, mooring_K,
    distributed mass/stiffness sections (honoured when n>0), and trailing
    tension-wire section.
    """
    draft      = _parse_float(r.read_var())
    cm_pform   = _parse_float(r.read_var())
    mass_pform = _parse_float(r.read_var())

    r.read_com()                           # "Platform mass inertia 3X3 matrix" label
    i_mat_3x3 = np.zeros((3, 3))
    for row in range(3):
        i_mat_3x3[row, :] = [_parse_float(t) for t in r.read_ary(3)]

    ref_msl = _parse_float(r.read_var())

    r.read_com()                           # "hydrodynamic 6X6 matrix" label
    hydro_M = np.zeros((6, 6))
    for row in range(6):
        hydro_M[row, :] = [_parse_float(t) for t in r.read_ary(6)]

    r.read_com()                           # "hydrodynamic 6X6 stiffness" label
    hydro_K = np.zeros((6, 6))
    for row in range(6):
        hydro_K[row, :] = [_parse_float(t) for t in r.read_ary(6)]

    r.read_com()                           # "mooring stiffness" label
    mooring_K = np.zeros((6, 6))
    for row in range(6):
        mooring_K[row, :] = [_parse_float(t) for t in r.read_ary(6)]

    # Distributed hydrodynamic added mass
    r.read_com()                           # blank line
    r.read_com()                           # "Distributed added-mass..." label
    n_m = _parse_int(r.read_var())
    if n_m > 0:
        z_m_raw = [_parse_float(t) for t in r.read_ary(n_m)]
        d_m_raw = [_parse_float(t) for t in r.read_ary(n_m)]
        z_distr_m = np.array(z_m_raw)
        distr_m   = np.array(d_m_raw)
    else:
        z_distr_m = np.array([])
        distr_m   = np.array([])
        if _is_float(r.peek_token()):
            r.read_com()
            if _is_float(r.peek_token()):
                r.read_com()

    # Distributed elastic stiffness
    r.read_com()                           # blank line
    r.read_com()                           # "Distributed elastic stiffness..." label
    n_k = _parse_int(r.read_var())
    if n_k > 0:
        z_k_raw = [_parse_float(t) for t in r.read_ary(n_k)]
        d_k_raw = [_parse_float(t) for t in r.read_ary(n_k)]
        z_distr_k = np.array(z_k_raw)
        distr_k   = np.array(d_k_raw)
    else:
        z_distr_k = np.array([])
        distr_k   = np.array([])
        if _is_float(r.peek_token()):
            r.read_com()
            if _is_float(r.peek_token()):
                r.read_com()

    # Trailing tension-wire section (always present, n_attachments may be 0)
    r.read_com()                           # blank line before "Tension wires data"
    wires = _parse_tension_wires(r)

    # Build 6×6 structural mass matrix: translational block = m*I, rotational = i_mat_3x3
    i_mat = np.zeros((6, 6))
    for k in range(3):
        i_mat[k, k] = mass_pform
    i_mat[3:6, 3:6] = i_mat_3x3

    return PlatformSupport(
        draft=draft, cm_pform=cm_pform, mass_pform=mass_pform,
        i_matrix=i_mat, ref_msl=ref_msl,
        hydro_M=hydro_M, hydro_K=hydro_K, mooring_K=mooring_K,
        distr_m_z=z_distr_m, distr_m=distr_m,
        distr_k_z=z_distr_k, distr_k=distr_k,
        wires=wires,
    )
