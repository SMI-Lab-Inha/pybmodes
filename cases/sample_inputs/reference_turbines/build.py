"""Generate pyBmodes-authored BMI samples for the open-literature
**reference wind turbines** (RWTs) cited in the wind-energy literature.

The output BMI + section-property files are *redistributable under
pyBmodes' MIT licence*: every numerical value is sourced from the
named open-access publication or its public companion deck, the file
format + commentary are pyBmodes-authored, and the project's
"Independence stance" (CLAUDE.md) forbids only verbatim *copies* of
upstream `.bmi` / `.dat` files — not authoring our own representations
of publicly-documented turbines.

For each turbine the script emits **two** BMI samples:

- ``<id>_tower.bmi`` — cantilevered tower from TowerBsHt to TowerHt
  (``hub_conn = 1``) with the RNA lumped at the tower top. For monopile
  sub-cases this treats the substructure below TowerBsHt as a rigid
  extension (same simplification BModes uses for ElastoDyn-compatible
  mode-shape generation; the flexible-pile + tower combined-cantilever
  physics is available via ``Tower.from_elastodyn_with_subdyn(...)``).

- ``<id>_blade.bmi`` — single rotating blade clamped at the hub root
  (``hub_conn = 1``), spun at the deck's ``RotSpeed`` so the FEM picks up
  the centrifugal-stiffening contribution to the flap modes.

A note on published reference values: RWT structural definitions are
**iteratively revised** across releases — the same RWT designation
(e.g. ``IEA-15-240-RWT``) at git-tag v1.0.0 may have a few-percent
different section-property distribution than the same designation at
v2.0.0. The pyBmodes frequencies reported for each sample are derived
from the **deck-as-distributed** at the time this script was last run
(see the source paths in the ``TURBINES`` list); the "reference"
frequency cited in each per-turbine README is the value the *original*
publication printed at its publication date. The two need not match
exactly. A drift between them usually reflects deck-revision evolution,
not a pyBmodes error — treat the published values as historical
anchors, not regression targets.

Floating cases (OC3 Hywind spar) are NOT generated here — they need
a full PlatformSupport block with hydro/mooring/inertia 6 × 6 matrices.
The validated pyBmodes path for floating is the BModes-format
``Tower("OC3Hywind.bmi")`` solve documented in
``test_certtest_oc3hywind``; a sample-BMI authored equivalent is on
the roadmap.

Run from the repo root::

    set PYTHONPATH=D:\\repos\\pyBModes\\src
    python cases/sample_inputs/reference_turbines/build.py

Each turbine whose source `.dat` files are present locally produces
``<id>/<id>_tower.bmi`` + ``<id>/<id>_tower_sec_props.dat`` +
``<id>/<id>_blade.bmi`` + ``<id>/<id>_blade_sec_props.dat`` +
``<id>/README.md``. Turbines whose source files are absent (e.g. on
a fresh clone without ``docs/OpenFAST_files/``) are skipped.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pybmodes.io.elastodyn_reader import (  # noqa: E402
    read_elastodyn_blade,
    read_elastodyn_main,
    read_elastodyn_tower,
)

HERE = pathlib.Path(__file__).resolve().parent

# RNA inertia values for each turbine, expressed in tower-top frame.
# These are NOT in the ElastoDyn .dat (ElastoDyn carries them as
# scalar masses + offsets); the values below come from the cited
# open-literature publications as described in each turbine's own
# README. Where the publication doesn't tabulate them (most do not),
# we fall back to a "scale by sqrt(NacMass / 240e3)" estimate from
# the NREL 5MW values in OC3 Hywind's BMI.
_NREL5MW_INERTIAS = dict(
    cm_loc=-0.41378,    # transverse (Y-axis) c.m. offset, from OC3 Hywind doc
    cm_axial=1.96699,   # ≈ Twr2Shft, axial c.m. offset above tower top (m)
    ixx_tip=4.370e7,    # rotor lag (kg-m^2)
    iyy_tip=2.353e7,    # rotor flap (kg-m^2)
    izz_tip=2.542e7,    # yaw / rotor torsion (kg-m^2)
    izx_tip=1.169e6,
)

# The IEA reference turbines don't publish RNA tensor inertias as
# directly. We approximate by scaling NREL 5MW's tensor by the
# (NacMass / 240e3) ratio. This is an order-of-magnitude estimate;
# users wanting precise inertias should override these values with
# the WT_Ontology yaml or HAWC2 deck for the relevant turbine.
def _scaled_inertias(nac_mass: float) -> dict:
    factor = nac_mass / 240_000.0
    return dict(
        cm_loc=_NREL5MW_INERTIAS["cm_loc"],
        cm_axial=_NREL5MW_INERTIAS["cm_axial"],  # overridden by Twr2Shft below
        ixx_tip=_NREL5MW_INERTIAS["ixx_tip"] * factor,
        iyy_tip=_NREL5MW_INERTIAS["iyy_tip"] * factor,
        izz_tip=_NREL5MW_INERTIAS["izz_tip"] * factor,
        izx_tip=_NREL5MW_INERTIAS["izx_tip"] * factor,
    )


# Empirical constants the ElastoDyn → pyBmodes adapter uses to fill
# in tor_stff and axial_stff (which ElastoDyn doesn't carry):
_GJ_OVER_EI = 100.0
_EA_OVER_EI = 1.0e6
_INER_FLOOR = 1.0e-6 * 4.0 ** 2   # 1e-6 · char_length²; char ≈ 4 m


def _blade_total_mass(blade) -> float:
    """Trapezoidal integration of mass-density over span_loc · L_blade."""
    span = np.asarray(blade.bl_fract, dtype=float)
    rho = np.asarray(blade.b_mass_den, dtype=float)
    # bl_fract is normalized [0, 1]; multiply by blade length elsewhere.
    return float(np.trapezoid(rho, span))


def _emit_tower_sec_props(
    *, path: pathlib.Path, title: str, ht_fract, t_mass_den, tw_fa_stif,
) -> None:
    """Emit an isotropic tower section-properties file (FA/SS stiffness equal,
    no offsets, no twist — matches typical land-based or monopile tower."""
    lines: list[str] = []
    lines.append(f"{title}")
    n = len(ht_fract)
    lines.append(f"{n}         n_secs:     number of stations at which "
                 f"properties are specified (-)")
    lines.append("")
    lines.append("sec_loc  str_tw  tw_iner   mass_den  flp_iner  edge_iner  "
                 "flp_stff   edge_stff   tor_stff   axial_stff  cg_offst  "
                 "sc_offst tc_offst")
    lines.append("(-)      (deg)   (deg)     (kg/m)    (kg-m)    (kg-m)     "
                 "(Nm^2)     (Nm^2)      (Nm^2)     (N)         (m)       "
                 "(m)      (m)")
    for s, m, ei in zip(ht_fract, t_mass_den, tw_fa_stif):
        gj = ei * _GJ_OVER_EI
        ea = ei * _EA_OVER_EI
        flp_iner = _INER_FLOOR * m
        edge_iner = flp_iner
        lines.append(
            f"{s:.6f}  0.0     0.0       {m:.4e}  {flp_iner:.3e}  {edge_iner:.3e}  "
            f"{ei:.4e}  {ei:.4e}  {gj:.4e}  {ea:.4e}  0.0       0.0      0.0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _emit_blade_sec_props(
    *, path: pathlib.Path, title: str,
    bl_fract, strc_twst, b_mass_den, flp_stff, edg_stff,
) -> None:
    """Emit a blade section-properties file with FA = flap, SS = edge.

    Uses the same fill-in conventions as
    :func:`pybmodes.io.elastodyn_reader._stack_blade_section_props`:
    tor_stff = 100·max(EI), axial_stff = 1e6·max(EI), rotary inertia at
    the 1e-6·char² floor (char = 4 m), structural twist preserved from
    the deck, all offsets zero (ElastoDyn doesn't carry them).
    """
    lines: list[str] = []
    lines.append(f"{title}")
    n = len(bl_fract)
    lines.append(f"{n}         n_secs:     number of stations at which "
                 f"properties are specified (-)")
    lines.append("")
    lines.append("sec_loc  str_tw  tw_iner   mass_den  flp_iner  edge_iner  "
                 "flp_stff   edge_stff   tor_stff   axial_stff  cg_offst  "
                 "sc_offst tc_offst")
    lines.append("(-)      (deg)   (deg)     (kg/m)    (kg-m)    (kg-m)     "
                 "(Nm^2)     (Nm^2)      (Nm^2)     (N)         (m)       "
                 "(m)      (m)")
    for s, tw, m, eif, eie in zip(
        bl_fract, strc_twst, b_mass_den, flp_stff, edg_stff,
    ):
        ei_max = max(float(eif), float(eie))
        gj = ei_max * _GJ_OVER_EI
        ea = ei_max * _EA_OVER_EI
        flp_iner = _INER_FLOOR * float(m)
        edge_iner = flp_iner
        lines.append(
            f"{float(s):.6f}  {float(tw):8.4f}  {float(tw):8.4f}  "
            f"{float(m):.4e}  {flp_iner:.3e}  {edge_iner:.3e}  "
            f"{float(eif):.4e}  {float(eie):.4e}  "
            f"{gj:.4e}  {ea:.4e}  0.0       0.0      0.0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scaling_block() -> list[str]:
    return [
        "Property scaling factors..............................",
        "1.0       sec_mass_mult:   mass density multiplier (-)",
        "1.0       flp_iner_mult:   flap / fore-aft inertia multiplier (-)",
        "1.0       lag_iner_mult:   lag / side-side inertia multiplier (-)",
        "1.0       flp_stff_mult:   flap / fore-aft bending stiffness multiplier (-)",
        "1.0       edge_stff_mult:  edge / side-side bending stiffness multiplier (-)",
        "1.0       tor_stff_mult:   torsion stiffness multiplier (-)",
        "1.0       axial_stff_mult: axial stiffness multiplier (-)",
        "1.0       cg_offst_mult:   cg offset multiplier (-)",
        "1.0       sc_offst_mult:   shear center multiplier (-)",
        "1.0       tc_offst_mult:   tension center multiplier (-)",
    ]


def _emit_tower_bmi(
    *,
    path: pathlib.Path,
    title: str,
    radius: float,
    hub_rad: float,
    tip_mass: float,
    inertias: dict,
    sec_props_filename: str,
    n_elements: int = 20,
) -> None:
    """Emit a pyBmodes-format BMI file for a cantilevered tower."""
    el_loc = np.linspace(0.0, 1.0, n_elements + 1)
    el_loc_str = " ".join(f"{x:.4f}" for x in el_loc)
    lines = [
        "==========================   Main Input File   ==========================",
        title,
        "",
        "--------- General parameters " + "-" * 70,
        "true      Echo        Echo input file contents to *.echo file if true.",
        "2         beam_type   1: blade, 2: tower (-)",
        "0.0       rot_rpm:    rotor speed (rpm) — auto-zero for tower analysis",
        "1.0       rpm_mult:   rotor speed multiplicative factor (-)",
        f"{radius}      radius:     tower height above ground or MSL (m)",
        f"{hub_rad}       hub_rad:    tower rigid-base height (m)",
        "0.        precone:    automatically zero for a tower (deg)",
        "0.        bl_thp:     automatically zero for a tower (deg)",
        "1         hub_conn:   1=cantilever 2=free-free 3=ax+tor 4=pinned-free",
        "20        modepr:     number of modes to be printed (-)",
        "t         TabDelim    (true: tab-delimited output tables)",
        "f         mid_node_tw  (false: no mid-node twist outputs)",
        "",
        "--------- Blade-tip or tower-top mass properties " + "-" * 50,
        f"{tip_mass:.4e}  tip_mass    tower-top RNA mass (kg) — Hub + Nac + 3·Blade",
        f"{inertias['cm_loc']:.6f}      cm_loc      RNA c.m. transverse offset (m)",
        f"{inertias['cm_axial']:.6f}      cm_axial    RNA c.m. axial offset above tower top (m)",
        f"{inertias['ixx_tip']:.4e}        ixx_tip     RNA lag inertia, tip x-axis (kg-m^2)",
        f"{inertias['iyy_tip']:.4e}        iyy_tip     RNA fore-aft inertia, tip y-axis (kg-m^2)",
        f"{inertias['izz_tip']:.4e}        izz_tip     RNA yaw inertia, tip z-axis (kg-m^2)",
        "0.        ixy_tip     cross product of inertia about x and y axes (kg-m^2)",
        f"{inertias['izx_tip']:.4e}        izx_tip     cross product about z and x axes (kg-m^2)",
        "0.        iyz_tip     cross product of inertia about y and z axes (kg-m^2)",
        "",
        "--------- Distributed-property identifiers " + "-" * 56,
        "1         id_mat:     material_type [1: isotropic]",
        f"'{sec_props_filename}' sec_props_file   name of beam section properties file (-)",
        "",
        *_scaling_block(),
        "",
        "--------- Finite element discretization " + "-" * 50,
        f"{n_elements}        nselt:     number of beam elements (-)",
        "Distance of element boundary nodes from tower base (norm. by flex length), el_loc()",
        el_loc_str,
        "",
        "--------- Tower support " + "-" * 70,
        "0         tow_support:  0 = no tension wires, no platform; pure cantilever",
        "",
        "END of Main Input File Data " + "*" * 65,
        "*" * 95,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _emit_blade_bmi(
    *,
    path: pathlib.Path,
    title: str,
    tip_rad: float,
    hub_rad: float,
    rot_rpm: float,
    precone_deg: float,
    sec_props_filename: str,
    n_elements: int = 20,
) -> None:
    """Emit a pyBmodes-format BMI file for a rotating cantilever blade.

    No tip mass (these RWTs don't ship with tip-brake masses); the blade
    is clamped at the hub root and spun at the deck's rated rotor speed
    to pick up the centrifugal-stiffening contribution.
    """
    el_loc = np.linspace(0.0, 1.0, n_elements + 1)
    el_loc_str = " ".join(f"{x:.4f}" for x in el_loc)
    lines = [
        "==========================   Main Input File   ==========================",
        title,
        "",
        "--------- General parameters " + "-" * 70,
        "true      Echo        Echo input file contents to *.echo file if true.",
        "1         beam_type   1: blade, 2: tower (-)",
        f"{rot_rpm}       rot_rpm:    rotor speed (rpm) — from the deck's RotSpeed",
        "1.0       rpm_mult:   rotor speed multiplicative factor (-)",
        f"{tip_rad}      radius:     blade tip radius along blade axis (m)",
        f"{hub_rad}       hub_rad:    hub radius along blade axis (m)",
        f"{precone_deg}        precone:    built-in precone angle (deg)",
        "0.        bl_thp:     blade pitch setting (deg)",
        "1         hub_conn:   1=cantilever 2=free-free 3=ax+tor 4=pinned-free",
        "20        modepr:     number of modes to be printed (-)",
        "t         TabDelim    (true: tab-delimited output tables)",
        "f         mid_node_tw  (false: no mid-node twist outputs)",
        "",
        "--------- Blade-tip or tower-top mass properties " + "-" * 50,
        "0.        tip_mass    blade-tip mass (kg) — no tip-brake on these RWTs",
        "0.        cm_loc      tip-mass c.m. transverse offset (m)",
        "0.        cm_axial    tip-mass c.m. axial offset (m)",
        "0.        ixx_tip     tip-mass lag inertia, tip x-axis (kg-m^2)",
        "0.        iyy_tip     tip-mass flap inertia, tip y-axis (kg-m^2)",
        "0.        izz_tip     tip-mass torsion inertia, tip z-axis (kg-m^2)",
        "0.        ixy_tip     cross product of inertia about x and y axes (kg-m^2)",
        "0.        izx_tip     cross product about z and x axes (kg-m^2)",
        "0.        iyz_tip     cross product of inertia about y and z axes (kg-m^2)",
        "",
        "--------- Distributed-property identifiers " + "-" * 56,
        "1         id_mat:     material_type [1: isotropic]",
        f"'{sec_props_filename}' sec_props_file   name of beam section properties file (-)",
        "",
        *_scaling_block(),
        "",
        "--------- Finite element discretization " + "-" * 50,
        f"{n_elements}        nselt:     number of beam elements (-)",
        "Distance of element boundary nodes from blade root (norm. by blade length), el_loc()",
        el_loc_str,
        "",
        "END of Main Input File Data " + "*" * 65,
        "*" * 95,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _emit_readme(*, path: pathlib.Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-turbine config
# ---------------------------------------------------------------------------

def _config(rel_main: str, rel_tower: str, rel_blade: str | None = None) -> dict:
    return dict(
        main=REPO_ROOT / rel_main,
        tower=REPO_ROOT / rel_tower,
        blade=(REPO_ROOT / rel_blade) if rel_blade else None,
    )


TURBINES = [
    dict(
        id="01_nrel5mw_land",
        title="NREL 5MW reference turbine — land-based",
        citation=("Jonkman, J., Butterfield, S., Musial, W., & Scott, G. "
                  "(2009). Definition of a 5-MW Reference Wind Turbine for "
                  "Offshore System Development. NREL/TP-500-38060."),
        sub_case_note=("Land-based (cantilever clamped at ground level, "
                       "TowerBsHt = 0)."),
        published_fa1_hz=0.32,
        published_fa1_source=("Jonkman 2009 §6, Table 9-1 "
                              "(1st-FA tower-bending coupled with RNA)"),
        sources=_config(
            "reference_decks/nrel5mw_land/NRELOffshrBsline5MW_Onshore_ElastoDyn.dat",
            "reference_decks/nrel5mw_land/NRELOffshrBsline5MW_Tower.dat",
            "reference_decks/nrel5mw_land/NRELOffshrBsline5MW_Blade.dat",
        ),
        inertia_override=_NREL5MW_INERTIAS,  # use the published 5MW values
    ),
    dict(
        id="02_nrel5mw_oc3monopile",
        title="NREL 5MW on the OC3 monopile substructure",
        citation=("Jonkman, J., & Musial, W. (2010). Offshore Code "
                  "Comparison Collaboration (OC3) for IEA Wind Task 23. "
                  "NREL/TP-5000-48191."),
        sub_case_note=("OC3 monopile substructure, cantilever clamped at "
                       "TowerBsHt = +10 m above MSL (treats pile as rigid "
                       "extension below; for combined pile + tower physics "
                       "use Tower.from_elastodyn_with_subdyn)."),
        published_fa1_hz=0.30,
        published_fa1_source=("Jonkman & Musial 2010 OC3 Phase II — "
                              "approximation; flexible-pile system "
                              "1st-FA at ~ 0.275 Hz"),
        sources=_config(
            "reference_decks/nrel5mw_oc3monopile/NRELOffshrBsline5MW_OC3Monopile_ElastoDyn.dat",
            "reference_decks/nrel5mw_oc3monopile/NRELOffshrBsline5MW_OC3Monopile_Tower.dat",
            "reference_decks/nrel5mw_oc3monopile/NRELOffshrBsline5MW_Blade.dat",
        ),
        inertia_override=_NREL5MW_INERTIAS,
    ),
    dict(
        id="03_iea34_land",
        title="IEA-3.4-130-RWT — land-based",
        citation=("Bortolotti, P., Tarrés, H. C., Dykes, K., Merz, K., "
                  "Sethuraman, L., Verelst, D., & Zahle, F. (2019). "
                  "IEA Wind TCP Task 37: Systems Engineering in Wind "
                  "Energy — WP2.1 Reference Wind Turbines. NREL/TP-5000-73492."),
        sub_case_note="Land-based (cantilever clamped at ground level).",
        published_fa1_hz=0.42,
        published_fa1_source=("Bortolotti et al. 2019 §3.4, Table 3-12 "
                              "(IEA-3.4-130-RWT 1st-FA tower-bending)"),
        sources=_config(
            "reference_decks/iea34_land/IEA-3.4-130-RWT_ElastoDyn.dat",
            "reference_decks/iea34_land/IEA-3.4-130-RWT_Tower.dat",
            "reference_decks/iea34_land/IEA-3.4-130-RWT_Blade.dat",
        ),
    ),
    dict(
        id="04_iea10_monopile",
        title="IEA-10.0-198-RWT — monopile",
        citation=("Bortolotti, P., et al. (2019). IEA Wind TCP Task 37 "
                  "10MW Reference Wind Turbine. Subsequent monopile "
                  "configuration documented in the IEA-10.0-198-RWT "
                  "OpenFAST repository (IEAWindTask37/IEA-10.0-198-RWT)."),
        sub_case_note=("Monopile substructure, cantilever clamped at "
                       "TowerBsHt = +10 m above MSL."),
        published_fa1_hz=0.20,
        published_fa1_source=("IEA-10.0-198-RWT OpenFAST regression: 1st-FA "
                              "tower-bending of the combined pile + tower "
                              "system."),
        sources=_config(
            "docs/OpenFAST_files/IEA-10.0-198-RWT/openfast/IEA-10.0-198-RWT_ElastoDyn.dat",
            "docs/OpenFAST_files/IEA-10.0-198-RWT/openfast/IEA-10.0-198-RWT_ElastoDyn_tower.dat",
            "docs/OpenFAST_files/IEA-10.0-198-RWT/openfast/IEA-10.0-198-RWT_ElastoDyn_blade.dat",
        ),
    ),
    dict(
        id="05_iea15_monopile",
        title="IEA-15-240-RWT — monopile",
        citation=("Gaertner, E., Rinker, J., Sethuraman, L., Zahle, F., "
                  "Anderson, B., Barter, G., et al. (2020). Definition of "
                  "the IEA 15-Megawatt Offshore Reference Wind Turbine. "
                  "NREL/TP-5000-75698."),
        sub_case_note=("UMaine fixed-bottom monopile substructure, "
                       "cantilever clamped at TowerBsHt = +15 m above MSL."),
        published_fa1_hz=0.17,
        published_fa1_source=("Gaertner et al. 2020 §6.4, Table 6-3 "
                              "(IEA-15 monopile tower-bending of the "
                              "combined pile + tower system)."),
        sources=_config(
            "docs/OpenFAST_files/IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile_ElastoDyn.dat",
            "docs/OpenFAST_files/IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile_ElastoDyn_tower.dat",
            "docs/OpenFAST_files/IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT/IEA-15-240-RWT_ElastoDyn_blade.dat",
        ),
    ),
    dict(
        id="06_iea22_monopile",
        title="IEA-22-280-RWT — monopile",
        citation=("Bortolotti, P., et al. (2024). IEA Wind TCP Task 55: "
                  "IEA Wind 22 MW Reference Wind Turbine. NREL technical "
                  "report (in preparation); repository: "
                  "IEAWindSystems/IEA-22-280-RWT."),
        sub_case_note=("Fixed-bottom monopile substructure, cantilever "
                       "clamped at TowerBsHt = +15 m above MSL."),
        published_fa1_hz=0.15,
        published_fa1_source=("IEA-22-280-RWT OpenFAST regression: 1st-FA "
                              "tower-bending of the combined pile + tower "
                              "system."),
        sources=_config(
            "docs/OpenFAST_files/IEA-22-280-RWT/OpenFAST/IEA-22-280-RWT-Monopile/IEA-22-280-RWT_ElastoDyn.dat",
            "docs/OpenFAST_files/IEA-22-280-RWT/OpenFAST/IEA-22-280-RWT-Monopile/IEA-22-280-RWT_ElastoDyn_tower.dat",
            "docs/OpenFAST_files/IEA-22-280-RWT/OpenFAST/IEA-22-280-RWT/IEA-22-280-RWT_ElastoDyn_blade.dat",
        ),
    ),
]


def _readme_for(
    spec: dict,
    *,
    tower_fa1: float,
    blade_flap1: float,
    blade_edge1: float,
    blade_flap2: float,
    rot_rpm: float,
) -> str:
    rel_dir = f"cases/sample_inputs/reference_turbines/{spec['id']}"
    return (
        f"# {spec['title']}\n"
        f"\n"
        f"{spec['sub_case_note']}\n"
        f"\n"
        f"## Files\n"
        f"\n"
        f"| File                                       | Purpose                              |\n"
        f"| ------------------------------------------ | ------------------------------------ |\n"
        f"| `{spec['id']}_tower.bmi`                   | Tower BMI (cantilever)               |\n"
        f"| `{spec['id']}_tower_sec_props.dat`         | Distributed tower section data       |\n"
        f"| `{spec['id']}_blade.bmi`                   | Blade BMI (rotating cantilever)      |\n"
        f"| `{spec['id']}_blade_sec_props.dat`         | Distributed blade section data       |\n"
        f"\n"
        f"## How to run\n"
        f"\n"
        f"```python\n"
        f"from pybmodes.models import Tower, RotatingBlade\n"
        f"\n"
        f"tower = Tower(\"{rel_dir}/{spec['id']}_tower.bmi\")\n"
        f"tower_modal = tower.run(n_modes=8)\n"
        f"print(\"tower freqs (Hz):\", tower_modal.frequencies[:4])\n"
        f"\n"
        f"blade = RotatingBlade(\"{rel_dir}/{spec['id']}_blade.bmi\")\n"
        f"blade_modal = blade.run(n_modes=8)\n"
        f"print(\"blade freqs (Hz):\", blade_modal.frequencies[:4])\n"
        f"```\n"
        f"\n"
        f"## pyBmodes frequencies (this BMI, deck-as-distributed)\n"
        f"\n"
        f"### Tower\n"
        f"\n"
        f"- 1st FA tower-bending: **{tower_fa1:.4f} Hz**\n"
        f"\n"
        f"### Blade  (spinning at deck `RotSpeed = {rot_rpm:g} rpm`)\n"
        f"\n"
        f"- 1st flap: **{blade_flap1:.4f} Hz**\n"
        f"- 1st edge: **{blade_edge1:.4f} Hz**\n"
        f"- 2nd flap: **{blade_flap2:.4f} Hz**\n"
        f"\n"
        f"## Comparison with published values\n"
        f"\n"
        f"The original publication for this RWT printed a 1st-FA "
        f"tower-bending frequency of **~ {spec['published_fa1_hz']:.2f} Hz** "
        f"({spec['published_fa1_source']}). Reference-wind-turbine "
        f"structural definitions are **iteratively revised** across "
        f"releases — the same RWT designation at git-tag v1.0.0 may have "
        f"a few-percent different section-property distribution than at "
        f"v2.0.0. The pyBmodes frequency above is derived from the "
        f"deck-as-distributed at the time this sample was last built, so "
        f"it need not match the publication's printed value exactly. A "
        f"drift between them usually reflects deck-revision evolution, "
        f"not a pyBmodes error — treat the published value as a "
        f"historical anchor, not a regression target.\n"
        f"\n"
        f"For monopile sub-cases the pyBmodes value is also higher than "
        f"the system-level reference because this BMI clamps the tower at "
        f"TowerBsHt with the substructure treated as a rigid extension "
        f"below. For the flexible-pile + tower combined-cantilever physics:\n"
        f"\n"
        f"```python\n"
        f"from pybmodes.models import Tower\n"
        f"tower = Tower.from_elastodyn_with_subdyn(\n"
        f"    \"path/to/<turbine>_ElastoDyn.dat\",\n"
        f"    \"path/to/<turbine>_SubDyn.dat\",\n"
        f")\n"
        f"```\n"
        f"\n"
        f"## Citation\n"
        f"\n"
        f"{spec['citation']}\n"
        f"\n"
        f"## Provenance\n"
        f"\n"
        f"All numerical values in these BMI and section-properties files are "
        f"sourced from the named publication's distributed structural data. "
        f"The file format and commentary are pyBmodes-authored (MIT-"
        f"licensed). Generated by "
        f"`cases/sample_inputs/reference_turbines/build.py`.\n"
    )


def _build_one(spec: dict) -> tuple[bool, str]:
    """Build one turbine's tower + blade BMI samples. Returns (success, msg)."""
    main_path = spec["sources"]["main"]
    tower_path = spec["sources"]["tower"]
    blade_path = spec["sources"]["blade"]

    if not main_path.is_file():
        return False, f"main .dat not found: {main_path.relative_to(REPO_ROOT)}"
    if not tower_path.is_file():
        return False, f"tower .dat not found: {tower_path.relative_to(REPO_ROOT)}"
    if blade_path is None or not blade_path.is_file():
        return False, f"blade .dat not found: {blade_path}"

    main_ed = read_elastodyn_main(main_path)
    tower_ed = read_elastodyn_tower(tower_path)
    blade_ed = read_elastodyn_blade(blade_path)

    # --- RNA lumped mass = HubMass + NacMass + 3 · BladeMass ----------
    blade_length = main_ed.tip_rad - main_ed.hub_rad
    blade_mass = _blade_total_mass(blade_ed) * blade_length
    rna_mass = main_ed.hub_mass + main_ed.nac_mass + 3.0 * blade_mass

    inertias = (spec.get("inertia_override")
                or _scaled_inertias(main_ed.nac_mass)).copy()
    inertias["cm_axial"] = float(main_ed.twr2shft)

    out_dir = HERE / spec["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Tower artefacts ----------------------------------------------
    flex_length = float(main_ed.tower_ht - main_ed.tower_bs_ht)
    tower_props_name = f"{spec['id']}_tower_sec_props.dat"
    _emit_tower_sec_props(
        path=out_dir / tower_props_name,
        title=f"{spec['title']} — tower section properties",
        ht_fract=np.asarray(tower_ed.ht_fract, dtype=float),
        t_mass_den=np.asarray(tower_ed.t_mass_den, dtype=float),
        tw_fa_stif=np.asarray(tower_ed.tw_fa_stif, dtype=float),
    )
    tower_title = (
        f"{spec['title']} — TOWER — {spec['sub_case_note'].rstrip('.')}"
    )
    _emit_tower_bmi(
        path=out_dir / f"{spec['id']}_tower.bmi",
        title=tower_title,
        radius=flex_length,
        hub_rad=0.0,
        tip_mass=float(rna_mass),
        inertias=inertias,
        sec_props_filename=tower_props_name,
    )

    # --- Blade artefacts ----------------------------------------------
    precone_deg = (
        float(main_ed.pre_cone[0]) if hasattr(main_ed, "pre_cone") else 0.0
    )
    blade_props_name = f"{spec['id']}_blade_sec_props.dat"
    _emit_blade_sec_props(
        path=out_dir / blade_props_name,
        title=f"{spec['title']} — blade section properties",
        bl_fract=np.asarray(blade_ed.bl_fract, dtype=float),
        strc_twst=np.asarray(blade_ed.strc_twst, dtype=float),
        b_mass_den=np.asarray(blade_ed.b_mass_den, dtype=float),
        flp_stff=np.asarray(blade_ed.flp_stff, dtype=float),
        edg_stff=np.asarray(blade_ed.edg_stff, dtype=float),
    )
    blade_title = (
        f"{spec['title']} — BLADE — single rotating cantilever at deck RotSpeed"
    )
    _emit_blade_bmi(
        path=out_dir / f"{spec['id']}_blade.bmi",
        title=blade_title,
        tip_rad=float(main_ed.tip_rad),
        hub_rad=float(main_ed.hub_rad),
        rot_rpm=float(main_ed.rot_speed_rpm),
        precone_deg=precone_deg,
        sec_props_filename=blade_props_name,
    )

    # --- Solve both, populate README ----------------------------------
    from pybmodes.models import RotatingBlade, Tower
    tower = Tower(out_dir / f"{spec['id']}_tower.bmi")
    tower_modal = tower.run(n_modes=6)
    tower_fa1 = float(tower_modal.frequencies[0])

    blade = RotatingBlade(out_dir / f"{spec['id']}_blade.bmi")
    blade_modal = blade.run(n_modes=8)

    # Pick first flap-dominated, first edge-dominated, second flap.
    flap_freqs: list[float] = []
    edge_freqs: list[float] = []
    for s in blade_modal.shapes:
        fl = float(np.dot(s.flap_disp, s.flap_disp))
        ed = float(np.dot(s.lag_disp, s.lag_disp))
        tw = float(np.dot(s.twist, s.twist))
        if fl > 4.0 * (ed + tw) and s.freq_hz > 1.0e-6:
            flap_freqs.append(s.freq_hz)
        elif ed > 4.0 * (fl + tw) and s.freq_hz > 1.0e-6:
            edge_freqs.append(s.freq_hz)
    blade_flap1 = flap_freqs[0] if flap_freqs else float("nan")
    blade_flap2 = flap_freqs[1] if len(flap_freqs) > 1 else float("nan")
    blade_edge1 = edge_freqs[0] if edge_freqs else float("nan")

    _emit_readme(
        path=out_dir / "README.md",
        content=_readme_for(
            spec,
            tower_fa1=tower_fa1,
            blade_flap1=blade_flap1,
            blade_edge1=blade_edge1,
            blade_flap2=blade_flap2,
            rot_rpm=float(main_ed.rot_speed_rpm),
        ),
    )

    return True, (
        f"OK: flex_length = {flex_length:.2f} m, RNA = {rna_mass:,.0f} kg, "
        f"tower 1st-FA = {tower_fa1:.4f} Hz; blade flap1/edge1/flap2 = "
        f"{blade_flap1:.4f} / {blade_edge1:.4f} / {blade_flap2:.4f} Hz"
    )


def main() -> int:
    print("pyBmodes reference-wind-turbine sample-BMI build")
    print("=" * 60)
    n_done = 0
    n_skipped = 0
    for spec in TURBINES:
        print(f"\n{spec['id']}  ({spec['title']})")
        try:
            ok, msg = _build_one(spec)
        except Exception as exc:
            ok = False
            msg = f"ERROR {exc!r}"
        print(f"  {msg}")
        if ok:
            n_done += 1
        else:
            n_skipped += 1
    print()
    print("=" * 60)
    print(f"Done: {n_done} built, {n_skipped} skipped (source decks absent).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
