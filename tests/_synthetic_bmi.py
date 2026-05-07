"""Helpers for building synthetic .bmi and .dat files in tmp_path.

The pyBModes parser expects every section to be preceded by exactly two
separator lines (comment or section-header text). Stations that need
data values must be preceded by exactly two such separators. These
helpers wrap that structural rule so individual tests do not need to
get the spacing right by hand.

Nothing here references any third-party data; every numeric value below
is freely chosen by this module's author.
"""

from __future__ import annotations

import pathlib

import numpy as np

_GEN_LABELS = (
    "echo", "beam_type", "rot_rpm", "rpm_mult", "radius", "hub_rad",
    "precone", "bl_thp", "hub_conn", "n_modes_print", "tab_delim",
    "mid_node_tw",
)
_TIP_LABELS = (
    "tip_mass", "cm_offset", "cm_axial",
    "ixx", "iyy", "izz", "ixy", "izx", "iyz",
)
_SCALE_LABELS = (
    "sec_mass", "flp_iner", "lag_iner", "flp_stff", "edge_stff",
    "tor_stff", "axial_stff", "cg_offst", "sc_offst", "tc_offst",
)


def write_bmi(
    path: pathlib.Path,
    *,
    title: str = "synthetic case",
    beam_type: int = 1,
    radius: float = 50.0,
    hub_rad: float = 0.0,
    rot_rpm: float = 0.0,
    rpm_mult: float = 1.0,
    precone: float = 0.0,
    bl_thp: float = 0.0,
    hub_conn: int = 1,
    n_modes_print: int = 10,
    tip_mass: float = 0.0,
    sec_props_file: str = "secs.dat",
    n_elements: int = 4,
    el_loc: list[float] | None = None,
    tow_support: int = 0,
    wire_data: tuple[list[int], list[int], list[float], list[float]] | None = None,
) -> pathlib.Path:
    """Write a structurally-valid .bmi file to *path* and return it.

    The format mirrors what the pybmodes parser expects: 2-line section
    headers, value-then-label data lines.
    """
    if el_loc is None:
        el_loc = list(np.linspace(0.0, 1.0, n_elements + 1))

    def gen_block() -> list[str]:
        gen_values = (
            "f", str(beam_type), f"{rot_rpm}", f"{rpm_mult}",
            f"{radius}", f"{hub_rad}", f"{precone}", f"{bl_thp}",
            str(hub_conn), str(n_modes_print), "f", "f",
        )
        return [f"{v}    ! {lbl}" for v, lbl in zip(gen_values, _GEN_LABELS)]

    def tip_block() -> list[str]:
        tip_values = (str(tip_mass),) + ("0.0",) * 8
        return [f"{v}    ! {lbl}" for v, lbl in zip(tip_values, _TIP_LABELS)]

    def secref_block() -> list[str]:
        return [
            "1    ! id_mat",
            f"'{sec_props_file}'    ! sec_props_file",
        ]

    def scale_block() -> list[str]:
        return [f"1.0    ! {lbl}" for lbl in _SCALE_LABELS]

    def disc_block() -> list[str]:
        line_disc = [f"{n_elements}    ! n_elements"]
        line_disc.append("--- el_loc ---")
        line_disc.append("  ".join(f"{v}" for v in el_loc))
        return line_disc

    def support_block() -> list[str]:
        if beam_type != 2:
            return []
        head = [
            "================= tower support =================",
            "------- tow_support -------",
            f"{tow_support}    ! tow_support",
        ]
        if tow_support == 0:
            return head
        if tow_support == 1 and wire_data is not None:
            n_wires, node_attach, k_wire, th_wire = wire_data
            head.extend([
                "----- tension wire data -----",
                f"{len(n_wires)}    ! n_attachments",
                "  ".join(str(n) for n in n_wires),
                "  ".join(str(n) for n in node_attach),
                "  ".join(f"{k:.3e}" for k in k_wire),
                "  ".join(f"{t}" for t in th_wire),
            ])
            return head
        raise ValueError(f"Unsupported wire_data={wire_data!r} for tow_support={tow_support}")

    lines: list[str] = []
    lines.append("================= synthetic .bmi =================")
    lines.append(f"'{title}'")

    # General parameters
    lines.append("--------------- general parameters ---------------")
    lines.append("--------------- (echo through mid_node_tw) -------")
    lines.extend(gen_block())

    # Tip mass
    lines.append("--------------- tip mass -------------------------")
    lines.append("--------------- (mass + 9 inertia/offsets) -------")
    lines.extend(tip_block())

    # Section-properties reference
    lines.append("--------------- distributed properties -----------")
    lines.append("--------------- (id_mat + filename) --------------")
    lines.extend(secref_block())

    # Scaling
    lines.append("--------------- scaling factors ------------------")
    lines.append("--------------- (10 unity multipliers) -----------")
    lines.extend(scale_block())

    # Discretisation
    lines.append("--------------- fe discretisation ----------------")
    lines.append("--------------- (nselt + el_loc) -----------------")
    lines.extend(disc_block())

    # Tower support (only emitted for towers)
    lines.extend(support_block())

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_uniform_sec_props(
    path: pathlib.Path,
    *,
    n_secs: int = 5,
    mass_den: float = 100.0,
    flp_stff: float = 1.0e8,
    edge_stff: float = 1.0e9,
    tor_stff: float = 1.0e7,
    axial_stff: float = 1.0e10,
) -> pathlib.Path:
    """Write a synthetic uniform section-properties .dat file."""
    span = np.linspace(0.0, 1.0, n_secs)
    rows = "\n".join(
        f"{s:.4f}  0.0  0.0  {mass_den}  {mass_den * 0.1}  {mass_den * 0.1}  "
        f"{flp_stff}  {edge_stff}  {tor_stff}  {axial_stff}  0.0  0.0  0.0"
        for s in span
    )
    body = (
        "synthetic uniform section properties\n"
        f"{n_secs}  n_secs\n"
        "\n"
        "span_loc str_tw tw_iner mass_den flp_iner edge_iner "
        "flp_stff edge_stff tor_stff axial_stff cg_offst sc_offst tc_offst\n"
        "  -      deg    deg     kg/m     kg.m     kg.m     "
        "N.m^2   N.m^2     N.m^2   N        m        m        m\n"
        + rows + "\n"
    )
    path.write_text(body, encoding="utf-8")
    return path
