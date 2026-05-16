"""Read a WindIO ontology ``.yaml`` blade and reduce it to the FEM
section-property table (issue #35, Phase 2, SP-5).

This is the public glue that ties Phase-2 together, mirroring
:mod:`pybmodes.io.windio` (tower) / :func:`pybmodes.io.geometry.
tubular_section_props`:

* :func:`read_windio_blade` — parse the blade component
  (dialect-robust, reusing the duplicate-anchor-tolerant loader from
  :mod:`pybmodes.io.windio`): span axis, chord, twist, reference-axis
  chordwise location, the spanwise airfoil set, the resolved web /
  layer ``nd_arc`` bands (:mod:`pybmodes.io._precomp.arc_resolver`),
  and the material table.
* :func:`windio_blade_section_props` — walk the span, blend the
  airfoil, build each station's shell-layer / web stacks, run the
  thin-wall reduction (:mod:`pybmodes.io._precomp.reduction`), and
  assemble a :class:`pybmodes.io.sec_props.SectionProperties` ready
  for :class:`pybmodes.models.RotatingBlade` (SP-6).

Both WindIO key dialects are handled — modern ``outer_shape`` /
``structure`` (IEA-15 WT_Ontology, every WISDEM example incl. the
floating ones) and older ``outer_shape_bem`` /
``internal_structure_2d_fem`` (IEA-3.4 / 10 / 22). Needs the optional
``[windio]`` extra (PyYAML); the runtime core stays numpy+scipy.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

from pybmodes.io._precomp.arc_resolver import (
    ResolvedBladeStructure,
    resolve_blade_structure,
)
from pybmodes.io._precomp.laminate import material_plane_stress
from pybmodes.io._precomp.profile import Profile
from pybmodes.io._precomp.reduction import (
    LayerStation,
    WebStation,
    reduce_section,
)
from pybmodes.io.sec_props import SectionProperties
from pybmodes.io.windio import _dup_anchor_loader, _require_yaml


@dataclass
class WindIOBlade:
    """Geometry + layup of a WindIO blade, resolved onto a span grid."""

    span_grid: np.ndarray          # normalised [0, 1], root → tip
    flexible_length: float         # m, |z_tip − z_root|
    chord: np.ndarray              # m, per station
    twist_deg: np.ndarray          # deg, per station (structural twist)
    ref_axis_xc: np.ndarray        # reference-axis chord fraction
    profiles: list[Profile]        # blended airfoil per station
    resolved: ResolvedBladeStructure
    materials: dict


def _blade_shape_and_structure(comp: dict, component: str):
    """``(outer_shape, structure)`` across both WindIO dialects
    (mirrors ``pybmodes.io.windio._shape_and_structure``)."""
    shape = comp.get("outer_shape", comp.get("outer_shape_bem"))
    structure = comp.get("structure",
                          comp.get("internal_structure_2d_fem"))
    if shape is None or structure is None:
        raise KeyError(
            f"components.{component} has neither modern "
            f"'outer_shape'/'structure' nor older "
            f"'outer_shape_bem'/'internal_structure_2d_fem'."
        )
    return shape, structure


def _reference_axis(comp: dict, shape: dict, structure: dict,
                    component: str) -> dict:
    for holder in (comp, shape, structure):
        ra = holder.get("reference_axis")
        if ra is not None and "z" in ra:
            return ra
    raise KeyError(f"components.{component} has no reference_axis.z")


def _curve(spec: dict, at: np.ndarray) -> np.ndarray:
    """Linear-interpolate a WindIO ``{grid, values}`` onto ``at``
    (WindIO-native interpolation; mirrors the tower reader)."""
    g = np.asarray(spec["grid"], dtype=float)
    v = np.asarray(spec["values"], dtype=float)
    return np.interp(at, g, v)


def read_windio_blade(
    yaml_path: str | pathlib.Path,
    *,
    component: str = "blade",
    n_span: int = 30,
) -> WindIOBlade:
    """Parse the structural subset of a WindIO blade component."""
    yaml = _require_yaml()
    yaml_path = pathlib.Path(yaml_path)
    with yaml_path.open("r", encoding="utf-8") as fh:
        doc = yaml.load(fh, Loader=_dup_anchor_loader(yaml))

    try:
        comp = doc["components"][component]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"WindIO file {yaml_path} has no components.{component!r}."
        ) from exc

    shape, structure = _blade_shape_and_structure(comp, component)
    ra = _reference_axis(comp, shape, structure, component)
    z = ra["z"]
    z_grid = np.asarray(z["grid"], dtype=float)
    z_vals = np.asarray(z["values"], dtype=float)
    flexible_length = float(abs(z_vals[-1] - z_vals[0]))

    # Output span stations: a uniform grid over the defined span
    # (offsets/twist/chord interpolated linearly onto it).
    span = np.linspace(float(z_grid[0]), float(z_grid[-1]), n_span)

    chord = _curve(shape["chord"], span)
    twist_deg = np.degrees(_curve(shape["twist"], span))
    if "pitch_axis" in shape:                    # older: chord fraction
        ref_xc = _curve(shape["pitch_axis"], span)
    elif "section_offset_y" in shape:            # modern: metres / chord
        ref_xc = _curve(shape["section_offset_y"], span) / chord
    else:
        ref_xc = np.full(n_span, 0.5)            # fallback: mid-chord

    # Airfoil set: name → Profile, and the spanwise schedule.
    af_coords = {a["name"]: a["coordinates"] for a in doc.get("airfoils", [])}

    def _profile(name: str) -> Profile:
        c = af_coords[name]
        return Profile.from_windio_coords(c["x"], c["y"])

    if "airfoil_position" in shape:              # older dialect
        af_grid = np.asarray(shape["airfoil_position"]["grid"], float)
        af_labels = list(shape["airfoil_position"]["labels"])
    else:                                        # modern dialect
        afs = sorted(shape["airfoils"],
                     key=lambda a: a["spanwise_position"])
        af_grid = np.asarray([a["spanwise_position"] for a in afs], float)
        af_labels = [a["name"] for a in afs]

    cache: dict[str, Profile] = {}

    def _blended(s: float) -> Profile:
        j = int(np.clip(np.searchsorted(af_grid, s) - 1, 0,
                        len(af_grid) - 2))
        nlo, nhi = af_labels[j], af_labels[j + 1]
        plo = cache.setdefault(nlo, _profile(nlo))
        if nhi == nlo:
            return plo
        phi = cache.setdefault(nhi, _profile(nhi))
        span_lo, span_hi = af_grid[j], af_grid[j + 1]
        w = 0.0 if span_hi <= span_lo else (s - span_lo) / (span_hi -
                                                            span_lo)
        return plo.blend(phi, float(np.clip(w, 0.0, 1.0)))

    profiles = [_blended(float(s)) for s in span]
    resolved = resolve_blade_structure(structure, span)
    materials = {m["name"]: m for m in doc.get("materials", [])
                 if "name" in m}

    return WindIOBlade(
        span_grid=span, flexible_length=flexible_length, chord=chord,
        twist_deg=twist_deg, ref_axis_xc=ref_xc, profiles=profiles,
        resolved=resolved, materials=materials,
    )


def windio_blade_section_props(
    blade: WindIOBlade,
    *,
    n_perim: int = 300,
    title: str = "WindIO composite-blade section properties",
) -> SectionProperties:
    """Reduce every span station to the FEM section-property table."""
    n = len(blade.span_grid)
    cols = {k: np.zeros(n) for k in (
        "mass_den", "flp_iner", "edge_iner", "flp_stff", "edge_stff",
        "tor_stff", "axial_stff", "cg_offst", "sc_offst", "tc_offst",
    )}

    for i in range(n):
        web_plies: dict[str, list] = {}
        shell: list[LayerStation] = []
        for ly in blade.resolved.layers:
            if ly.material not in blade.materials:
                raise KeyError(
                    f"WindIO blade layer {ly.name!r} references material "
                    f"{ly.material!r} not in the top-level materials list"
                )
            pe = material_plane_stress(blade.materials[ly.material])
            t = float(ly.thickness[i])
            if t <= 0.0:
                continue
            th = float(ly.fiber_orientation[i])
            if ly.web is not None:
                web_plies.setdefault(ly.web, []).append((pe, t, th))
            else:
                shell.append(LayerStation(pe, t, th,
                                          float(ly.start_nd[i]),
                                          float(ly.end_nd[i])))
        webs = [
            WebStation(float(w.start_nd[i]), float(w.end_nd[i]),
                       web_plies.get(w.name, []))
            for w in blade.resolved.webs
        ]
        res = reduce_section(
            blade.profiles[i], float(blade.chord[i]),
            float(blade.ref_axis_xc[i]), shell, webs, n_perim=n_perim,
        )
        cols["mass_den"][i] = res.mass
        cols["flp_iner"][i] = res.flap_iner
        cols["edge_iner"][i] = res.edge_iner
        cols["flp_stff"][i] = res.EI_flap
        cols["edge_stff"][i] = res.EI_edge
        cols["tor_stff"][i] = res.GJ
        cols["axial_stff"][i] = res.EA
        cols["cg_offst"][i] = res.x_cg
        cols["sc_offst"][i] = res.x_sc
        cols["tc_offst"][i] = res.x_tc

    z = np.asarray(blade.span_grid, dtype=float)
    zeros = np.zeros(n)
    return SectionProperties(
        title=title,
        n_secs=n,
        span_loc=z,
        str_tw=np.asarray(blade.twist_deg, dtype=float),
        tw_iner=zeros.copy(),
        mass_den=cols["mass_den"],
        flp_iner=cols["flp_iner"],
        edge_iner=cols["edge_iner"],
        flp_stff=cols["flp_stff"],
        edge_stff=cols["edge_stff"],
        tor_stff=cols["tor_stff"],
        axial_stff=cols["axial_stff"],
        cg_offst=cols["cg_offst"],
        sc_offst=cols["sc_offst"],
        tc_offst=cols["tc_offst"],
    )
