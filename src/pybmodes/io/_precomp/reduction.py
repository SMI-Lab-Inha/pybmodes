"""Thin-wall composite cross-section reduction — single-cell
(issue #35, Phase 2, SP-3).

Given an airfoil perimeter (:class:`pybmodes.io._precomp.profile.Profile`),
the chord, the reference-axis chordwise location, and the resolved
shell layers at one span station (each a material + thickness + fibre
angle over an ``nd_arc`` band), reduce the section to the *uncoupled
diagonal* 1-D beam properties pyBmodes' FEM consumes: axial ``EA``,
principal/chordline bending ``EI_flap`` / ``EI_edge``, single-cell
torsion ``GJ``, mass per length, flap/edge mass moments, and the
tension- / centre-of-mass / shear-centre chordwise offsets.

Method (textbook thin-wall + classical lamination theory; Bir 2006,
*User's Guide to PreComp*, NREL/TP-500-38929, §3 — diagonal subset):

* Per perimeter segment, stack the covering layers' plies through the
  wall and form the laminate membrane stiffness condensed of wall
  bending (``Atilde = A − B Dinv B``, :mod:`.laminate`). The effective
  *longitudinal* modulus·thickness is the transverse-force-free
  reduction ``Ex = Atilde11 − Atilde12²/Atilde22`` (the wall is free to
  strain circumferentially, ``N_y = 0`` — this is what makes an
  isotropic tube reduce to the exact ``E·A`` rather than
  ``E/(1−ν²)·A``). The shear stiffness·thickness is ``Gxy = Atilde66``.
* ``EA = Σ Ex·ds``; tension centre = ``Ex``-weighted centroid;
  ``EI`` = parallel-axis ``Σ Ex·d²`` about it (chordline frame), with
  the cross term giving the principal-axis angle.
* Single-cell Bredt–Batho ``GJ = 4 A_cell² / ∮(ds / Gxy)``
  (multi-cell webs are SP-4).
* Mass: ``Σ ρt·ds``; c.g. = mass-weighted centroid; flap/edge mass
  moments about it. Shear centre is approximated by the tension
  centre at SP-3 (exact SC needs the shear-flow warping solve — SP-4+;
  for the symmetric closed-form tube/box gate SC ≡ centroid ≡ TC, so
  the approximation is exact there).

Clean-room; the WISDEM PreComp port is the studied reference, not
vendored (independence stance, ``CLAUDE.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pybmodes.io._precomp.laminate import (
    PlyElastic,
    abd_matrices,
    reduced_stiffness,
    transform_reduced_stiffness,
)
from pybmodes.io._precomp.profile import Profile


@dataclass
class LayerStation:
    """One WindIO shell layer evaluated at a single span station: its
    material, ply thickness (m), fibre angle (rad, from the blade-span
    axis), and the ``nd_arc`` band ``[s0, s1]`` it covers."""

    ply: PlyElastic
    thickness: float
    theta: float
    s0: float
    s1: float


@dataclass
class SectionResult:
    """Uncoupled diagonal beam properties at one station (SI)."""

    EA: float              # N
    EI_flap: float         # N·m²  (out-of-plane / flap, chordline frame)
    EI_edge: float         # N·m²  (in-plane / edge)
    GJ: float              # N·m²  (single-cell)
    mass: float            # kg/m
    flap_iner: float       # kg·m
    edge_iner: float       # kg·m
    x_tc: float            # chordwise tension-centre offset re ref axis (m)
    y_tc: float            # flap-normal tension-centre offset (m)
    x_cg: float            # chordwise c.g. offset (m)
    y_cg: float            # flap-normal c.g. offset (m)
    x_sc: float            # chordwise shear-centre offset (m) [≈ tc, SP-3]
    y_sc: float            # flap-normal shear-centre offset (m) [≈ tc, SP-3]
    principal_angle: float  # rad, chordline → principal bending axes


def _segment_atilde(
    plies: list[tuple[PlyElastic, float, float]],
) -> tuple[float, float, float]:
    """Return ``(Ex, Gxy, rho_t)`` for a through-wall ply stack.

    ``Ex`` longitudinal modulus·thickness with ``N_y = 0`` (N/m),
    ``Gxy`` shear modulus·thickness (N/m), ``rho_t`` mass per unit
    wall area (kg/m²). Empty stack → all zero (an uncovered segment).
    """
    if not plies:
        return 0.0, 0.0, 0.0
    stack = []
    rho_t = 0.0
    for ply, t, theta in plies:
        Q = reduced_stiffness(ply)
        stack.append((transform_reduced_stiffness(Q, theta), t))
        rho_t += ply.rho * t
    # Thin-wall beam reduction uses the *membrane* laminate stiffness A
    # directly. The wall's own bending-extension coupling (B, D) is NOT
    # condensed into the axial term: the section-level bending is
    # already carried by the parallel-axis Σ Ex·d² assembly, and the
    # wall is thin so its self-bending is second order (this is the
    # PreComp thin-wall assumption). Effective longitudinal modulus·
    # thickness with the laminate free to strain transversely
    # (N_y = 0): Ex = A11 − A12²/A22. Shear: Gxy = A66.
    A, _B, _D = abd_matrices(stack)
    a11, a12, a22, a66 = A[0, 0], A[0, 1], A[1, 1], A[2, 2]
    ex = a11 - (a12 * a12 / a22 if a22 > 0.0 else 0.0)
    return float(ex), float(a66), float(rho_t)


def reduce_section(
    profile: Profile,
    chord: float,
    ref_axis_xc: float,
    shell_layers: list[LayerStation],
    *,
    n_perim: int = 400,
) -> SectionResult:
    """Reduce one station to its uncoupled diagonal beam properties.

    ``profile`` — airfoil perimeter (unit-chord ``nd_arc`` spine).
    ``chord`` — section chord (m). ``ref_axis_xc`` — chordwise
    location of the blade reference (pitch) axis as a fraction of
    chord measured from the airfoil's own x-origin (the offsets are
    reported relative to it, on the chord line). ``shell_layers`` —
    the resolved shell layers at this station, **outermost first**
    (WindIO lists the outer skin / gelcoat first). Webs are SP-4.
    """
    # Physical perimeter length of the (unit-chord) profile × chord.
    seg = np.hypot(np.diff(profile.xc), np.diff(profile.yc))
    perim_phys = float(seg.sum()) * float(chord)
    if perim_phys <= 0.0:
        raise ValueError("degenerate section: zero physical perimeter")

    # Uniform sampling in nd_arc == uniform physical arc (the spine is
    # cumulative-arc normalised). Segment midpoints + equal ds.
    s_edges = np.linspace(0.0, 1.0, n_perim + 1)
    s_mid = 0.5 * (s_edges[:-1] + s_edges[1:])
    ds = perim_phys / n_perim

    xc_m, yc_m = profile.arc_to_xy(s_mid)
    # Position relative to the reference axis (origin on the chord line
    # at ref_axis_xc·chord): X chordwise (+ toward TE), Y flap-normal.
    X = (xc_m - ref_axis_xc) * chord
    Y = yc_m * chord

    # Enclosed area for Bredt–Batho (shoelace on the closed perimeter).
    xv, yv = profile.arc_to_xy(s_edges)
    xv = (xv - ref_axis_xc) * chord
    yv = yv * chord
    a_cell = 0.5 * abs(np.sum(xv[:-1] * yv[1:] - xv[1:] * yv[:-1]))

    ex = np.zeros(n_perim)
    gxy = np.zeros(n_perim)
    rho_t = np.zeros(n_perim)
    for k in range(n_perim):
        sk = s_mid[k]
        plies = [
            (ly.ply, ly.thickness, ly.theta)
            for ly in shell_layers
            if ly.s0 - 1e-12 <= sk <= ly.s1 + 1e-12
        ]
        ex[k], gxy[k], rho_t[k] = _segment_atilde(plies)

    ea_seg = ex * ds                       # N per segment
    m_seg = rho_t * ds                     # kg/m per segment

    EA = float(ea_seg.sum())
    mass = float(m_seg.sum())
    if EA <= 0.0 or mass <= 0.0:
        raise ValueError(
            "section has no load-bearing material (EA or mass is zero) — "
            "check the resolved layer bands cover the perimeter"
        )

    x_tc = float((ea_seg * X).sum() / EA)
    y_tc = float((ea_seg * Y).sum() / EA)
    x_cg = float((m_seg * X).sum() / mass)
    y_cg = float((m_seg * Y).sum() / mass)

    dxt, dyt = X - x_tc, Y - y_tc
    ei_edge = float((ea_seg * dxt * dxt).sum())   # in-plane (edgewise)
    ei_flap = float((ea_seg * dyt * dyt).sum())   # out-of-plane (flap)
    ei_xy = float((ea_seg * dxt * dyt).sum())
    principal = 0.5 * float(np.arctan2(2.0 * ei_xy, ei_edge - ei_flap))

    dxc, dyc = X - x_cg, Y - y_cg
    edge_iner = float((m_seg * dxc * dxc).sum())
    flap_iner = float((m_seg * dyc * dyc).sum())

    # Single-cell Bredt–Batho. ∮ ds/(Gt) over load-bearing segments.
    mask = gxy > 0.0
    if a_cell > 0.0 and np.any(mask):
        line_integral = float((ds / gxy[mask]).sum())
        GJ = 4.0 * a_cell * a_cell / line_integral
    else:
        GJ = 0.0

    return SectionResult(
        EA=EA, EI_flap=ei_flap, EI_edge=ei_edge, GJ=GJ, mass=mass,
        flap_iner=flap_iner, edge_iner=edge_iner,
        x_tc=x_tc, y_tc=y_tc, x_cg=x_cg, y_cg=y_cg,
        x_sc=x_tc, y_sc=y_tc,            # SC ≈ TC at SP-3 (exact for sym.)
        principal_angle=principal,
    )
