"""Resolve WindIO blade web / layer ``nd_arc`` bands onto the span grid
(issue #35, Phase 2, SP-2b).

A WindIO blade locates every web and every shell/web layer by a
*normalised arc* band ``[start_nd_arc, end_nd_arc]`` along the airfoil
perimeter (the :mod:`pybmodes.io._precomp.profile` ``nd_arc`` spine),
varying along span. The band is given in one of two dialects:

* **older** (IEA-3.4 / 10 / 22 ``internal_structure_2d_fem``): each
  web / layer carries explicit ``start_nd_arc``/``end_nd_arc``
  ``{grid, values}`` curves directly.
* **modern** (IEA-15 ``WT_Ontology``, every WISDEM example incl. the
  floating ones): each web / layer's ``start_nd_arc``/``end_nd_arc``
  is ``{anchor: {name, handle}}`` — an *indirection* into the
  blade-level ``structure.anchors[]`` registry. In the published
  IEA / WISDEM RWT files every registry entry ships the **fully
  resolved** explicit ``{grid, values}`` for both handles (verified:
  IEA-15 WT_Ontology + VolturnUS-S FOWT carry 50-point resolved
  curves for layers *and* webs, beside the parametric
  ``plane_intersection`` recipe).

So resolution is a **dereference + dialect-normalisation**: follow the
``anchor`` to its registry entry, take the explicit ``{grid, values}``
for the requested ``handle``, and interpolate onto the blade span
grid. The purely-parametric geometric fallback (``plane_intersection``
offset/rotation, ``midpoint_nd_arc`` + ``width`` — WISDEM's
``build_layer`` 1–6 in ``gc_WT_DataStruc.py``) is only needed for
hand-authored yamls that omit the resolved values; that case raises a
clear, actionable error rather than silently guessing.

Interpolation mirrors WISDEM (``PchipInterpolator(extrapolate=False)``
then ``nan_to_num`` → 0 outside the defined grid) so the bands match
the WISDEM / BeamDyn validation oracle with minimal numerical noise.
Clean-room reimplementation; the WISDEM source is studied as the
reference, not vendored (independence stance, ``CLAUDE.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class ResolvedWeb:
    """A shear web's perimeter attachment band along span."""

    name: str
    start_nd: np.ndarray   # (n_span,) nd_arc of the suction-side foot
    end_nd: np.ndarray     # (n_span,) nd_arc of the pressure-side foot


@dataclass
class ResolvedLayer:
    """A composite layer: material, thickness/orientation, and the
    perimeter band it covers (or the web it sits on)."""

    name: str
    material: str
    thickness: np.ndarray         # (n_span,) m
    fiber_orientation: np.ndarray  # (n_span,) rad
    start_nd: np.ndarray          # (n_span,) shell-arc band start
    end_nd: np.ndarray            # (n_span,) shell-arc band end
    web: str | None               # web name if this is an on-web layer


@dataclass
class ResolvedBladeStructure:
    """Every web + layer resolved onto a common blade span grid."""

    nd_span: np.ndarray
    webs: list[ResolvedWeb]
    layers: list[ResolvedLayer]


def _interp(grid, values, nd_span: np.ndarray) -> np.ndarray:
    """WISDEM-faithful curve interpolation onto the span grid.

    ``PchipInterpolator`` with ``extrapolate=False`` (NaN outside the
    defined grid) then ``nan_to_num`` → 0, exactly as
    ``gc_WT_InitModel.py`` does, so a region defined only over part of
    the span (e.g. a web starting at r/R = 0.1) is zero elsewhere and
    matches the WISDEM / BeamDyn oracle. A 2-point grid degenerates to
    linear, which is the intended behaviour for ``[0, 1]`` constants.
    """
    g = np.asarray(grid, dtype=float)
    v = np.asarray(values, dtype=float)
    if g.size < 2:
        # A single point: constant over the whole span.
        return np.full_like(np.asarray(nd_span, dtype=float), float(v.flat[0]))
    order = np.argsort(g)
    g, v = g[order], v[order]
    # PCHIP needs strictly increasing x; collapse exact dups (keep last).
    keep = np.concatenate([np.diff(g) > 0, [True]])
    g, v = g[keep], v[keep]
    out = PchipInterpolator(g, v, extrapolate=False)(np.asarray(nd_span,
                                                                dtype=float))
    return np.nan_to_num(out, nan=0.0)


def _resolve_handle(
    spec: dict,
    anchors_by_name: dict,
    nd_span: np.ndarray,
    *,
    what: str,
) -> np.ndarray:
    """Resolve one ``start_nd_arc`` / ``end_nd_arc`` spec to a span curve.

    Explicit ``{grid, values}`` → interpolate. ``{anchor: {name,
    handle}}`` → dereference the ``structure.anchors[]`` registry entry
    ``name`` and take its explicit ``{grid, values}`` for ``handle``.
    A registry entry that carries only a parametric recipe
    (``plane_intersection`` / ``midpoint_nd_arc`` + ``width``) with no
    resolved values raises a clear, actionable error.
    """
    if "grid" in spec and "values" in spec:
        return _interp(spec["grid"], spec["values"], nd_span)

    if "anchor" in spec:
        a_name = spec["anchor"]["name"]
        a_handle = spec["anchor"]["handle"]
        anchor = anchors_by_name.get(a_name)
        if anchor is None:
            raise KeyError(
                f"WindIO blade {what} references anchor {a_name!r}, which "
                f"is not in the blade structure.anchors[] registry "
                f"(have: {sorted(anchors_by_name)})."
            )
        sub = anchor.get(a_handle)
        if isinstance(sub, dict) and "grid" in sub and "values" in sub:
            return _interp(sub["grid"], sub["values"], nd_span)
        recipe = next((k for k in ("plane_intersection", "midpoint_nd_arc",
                                   "width") if k in anchor), None)
        raise NotImplementedError(
            f"WindIO blade anchor {a_name!r} has no resolved "
            f"{a_handle}.{{grid,values}} — only a parametric "
            f"{recipe!r} recipe. pyBmodes resolves blade structure by "
            f"dereferencing resolved arc curves; the published IEA / "
            f"WISDEM RWT yamls include them. Geometric parametric "
            f"resolution (offset/rotation/width → nd_arc) is not yet "
            f"implemented — supply a WISDEM-resolved blade yaml."
        )

    raise ValueError(
        f"WindIO blade {what} arc spec is neither explicit "
        f"{{grid, values}} nor an {{anchor}}: keys {sorted(spec)}"
    )


def resolve_blade_structure(
    structure: dict, nd_span: np.ndarray
) -> ResolvedBladeStructure:
    """Resolve a WindIO blade ``structure`` block onto ``nd_span``.

    ``structure`` is the blade structure mapping — modern
    ``components.blade.structure`` or older
    ``components.blade.internal_structure_2d_fem`` (the caller selects
    the right block; this function is dialect-agnostic from here on).
    It carries ``webs`` (list), ``layers`` (list) and, for the modern
    dialect, a blade-level ``anchors`` registry (list).

    ``nd_span`` is the normalised blade span grid (0 → 1, root → tip),
    typically the blade ``reference_axis.z`` grid.
    """
    nd_span = np.asarray(nd_span, dtype=float)
    anchors_by_name = {
        a["name"]: a for a in structure.get("anchors", []) if "name" in a
    }
    web_names = {w.get("name") for w in structure.get("webs", [])}

    webs: list[ResolvedWeb] = []
    for w in structure.get("webs", []):
        name = w.get("name", f"web{len(webs)}")
        webs.append(ResolvedWeb(
            name=name,
            start_nd=_resolve_handle(w["start_nd_arc"], anchors_by_name,
                                     nd_span, what=f"web {name!r}"),
            end_nd=_resolve_handle(w["end_nd_arc"], anchors_by_name,
                                   nd_span, what=f"web {name!r}"),
        ))

    layers: list[ResolvedLayer] = []
    for ly in structure.get("layers", []):
        name = ly.get("name", f"layer{len(layers)}")
        if "material" not in ly:
            raise KeyError(f"WindIO blade layer {name!r} has no 'material'")
        if "thickness" not in ly:
            raise KeyError(f"WindIO blade layer {name!r} has no 'thickness'")
        th = ly["thickness"]
        thickness = _interp(th["grid"], th["values"], nd_span)
        fo = ly.get("fiber_orientation")
        if isinstance(fo, dict) and "grid" in fo:
            fiber = _interp(fo["grid"], fo["values"], nd_span)
        else:
            fiber = np.zeros_like(nd_span)

        # An on-web layer: the explicit `web:` key (WindIO standard) or
        # an arc anchor that points at a web's name.
        web = ly.get("web")
        if web is None:
            for side in ("start_nd_arc", "end_nd_arc"):
                spec = ly.get(side, {})
                a = spec.get("anchor", {}) if isinstance(spec, dict) else {}
                if a.get("name") in web_names:
                    web = a["name"]
                    break

        if web is not None:
            # On-web layer: it spans the web line, not a shell arc.
            zeros = np.zeros_like(nd_span)
            layers.append(ResolvedLayer(
                name=name, material=ly["material"], thickness=thickness,
                fiber_orientation=fiber, start_nd=zeros, end_nd=zeros.copy(),
                web=web,
            ))
            continue

        layers.append(ResolvedLayer(
            name=name, material=ly["material"], thickness=thickness,
            fiber_orientation=fiber,
            start_nd=_resolve_handle(ly["start_nd_arc"], anchors_by_name,
                                     nd_span, what=f"layer {name!r}"),
            end_nd=_resolve_handle(ly["end_nd_arc"], anchors_by_name,
                                   nd_span, what=f"layer {name!r}"),
            web=None,
        ))

    return ResolvedBladeStructure(nd_span=nd_span, webs=webs, layers=layers)
