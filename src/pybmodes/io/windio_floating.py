"""Read the WindIO ``components.floating_platform`` + ``mooring``
blocks (issue #35, Phase 3, P3-1 geometry).

A WindIO floating substructure is a set of named **joints** (3-D
points, MSL datum, ``z`` up) connected by slender circular
**members** (each a wall layup + optional bulkhead + ballast, plus
Morison ``Ca``/``Cd``). The downstream physics — hydrostatic
restoring (P3-2), Morison added mass + buoyancy + rigid-body inertia
(P3-3), and the catenary mooring stiffness (P3-4,
:mod:`pybmodes.mooring`) — all build on the geometry parsed here.

This module is *only* the parser + geometry primitives. Conventions:

* ``cylindrical: true`` joints give ``location = [r, theta_deg, z]``
  (WISDEM windIO convention) → ``x = r cos θ``, ``y = r sin θ``.
* ``axial_joints`` (named fractions along a member, e.g. a fairlead)
  are resolved into the joint table so ``mooring`` can reference them.
* the joint flagged ``transition: true`` is where the tower foots.

Needs the optional ``[windio]`` extra (PyYAML); runtime core stays
numpy+scipy. Dialect-robust via the duplicate-anchor-tolerant loader
shared with :mod:`pybmodes.io.windio`.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np

from pybmodes.io.windio import _dup_anchor_loader, _require_yaml


@dataclass
class FloatingMember:
    """One circular member: end points (m, MSL datum), the spanwise
    outer-diameter curve, the wall + bulkhead layup, ballast, and the
    Morison coefficients."""

    name: str
    end1: np.ndarray                      # (3,) m
    end2: np.ndarray                      # (3,) m
    od_grid: np.ndarray                   # outer-diameter grid [0, 1]
    od_values: np.ndarray                 # outer diameter (m)
    wall_material: str
    wall_t_grid: np.ndarray
    wall_t_values: np.ndarray
    ca: float = 1.0                       # transverse added-mass coeff
    bulkhead_material: str | None = None
    bulkhead_t: float = 0.0
    ballast: list = field(default_factory=list)   # raw WindIO entries

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end2 - self.end1))

    @property
    def axis(self) -> np.ndarray:
        """Unit vector end1 → end2."""
        v = self.end2 - self.end1
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def diameter_at(self, frac: np.ndarray) -> np.ndarray:
        """Outer diameter at member fraction(s) ``frac`` ∈ [0, 1]."""
        return np.interp(frac, self.od_grid, self.od_values)

    def wall_t_at(self, frac: np.ndarray) -> np.ndarray:
        return np.interp(frac, self.wall_t_grid, self.wall_t_values)

    def point_at(self, frac: float) -> np.ndarray:
        """3-D location at member fraction ``frac`` ∈ [0, 1]."""
        return self.end1 + float(frac) * (self.end2 - self.end1)


@dataclass
class WindIOFloating:
    """Parsed floating substructure + raw mooring block."""

    members: list[FloatingMember]
    joints: dict                          # name -> (3,) np.ndarray (m)
    transition_joint: str | None          # tower-foot joint name
    transition_piece_mass: float
    mooring: dict                         # raw WindIO mooring mapping
    materials: dict                       # name -> material mapping


def _joint_xyz(loc, cylindrical: bool) -> np.ndarray:
    a, b, c = float(loc[0]), float(loc[1]), float(loc[2])
    if cylindrical:
        th = np.deg2rad(b)
        return np.array([a * np.cos(th), a * np.sin(th), c], dtype=float)
    return np.array([a, b, c], dtype=float)


def read_windio_floating(
    yaml_path: str | pathlib.Path,
    *,
    component: str = "floating_platform",
) -> WindIOFloating:
    """Parse the floating substructure + mooring from a WindIO file."""
    yaml = _require_yaml()
    yaml_path = pathlib.Path(yaml_path)
    with yaml_path.open("r", encoding="utf-8") as fh:
        doc = yaml.load(fh, Loader=_dup_anchor_loader(yaml))

    try:
        fp = doc["components"][component]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"WindIO file {yaml_path} has no components.{component!r} "
            f"block (this is not a floating turbine ontology)."
        ) from exc

    # Base joints.
    joints: dict = {}
    transition_joint: str | None = None
    for j in fp.get("joints", []):
        name = j["name"]
        joints[name] = _joint_xyz(j["location"],
                                  bool(j.get("cylindrical", False)))
        if j.get("transition", False):
            transition_joint = name

    materials = {m["name"]: m for m in doc.get("materials", [])
                 if "name" in m}

    members: list[FloatingMember] = []
    for m in fp.get("members", []):
        nm = m["name"]
        try:
            e1 = joints[m["joint1"]]
            e2 = joints[m["joint2"]]
        except KeyError as exc:
            raise KeyError(
                f"floating member {nm!r} references joint {exc} not in "
                f"the joints list {sorted(joints)}"
            ) from exc

        od = m["outer_shape"]["outer_diameter"]
        layers = m.get("structure", {}).get("layers", [])
        if not layers:
            raise ValueError(
                f"floating member {nm!r} has no structure.layers (wall)"
            )
        wall = layers[0]
        wt = wall["thickness"]
        ca_spec = m.get("Ca", 1.0)
        ca = float(ca_spec[0] if isinstance(ca_spec, (list, tuple))
                   else ca_spec)
        bh = m.get("structure", {}).get("bulkhead")
        bh_mat = bh["material"] if isinstance(bh, dict) else None
        bh_t = (float(np.mean(bh["thickness"]["values"]))
                if isinstance(bh, dict) else 0.0)

        mem = FloatingMember(
            name=nm,
            end1=e1, end2=e2,
            od_grid=np.asarray(od["grid"], dtype=float),
            od_values=np.asarray(od["values"], dtype=float),
            wall_material=wall["material"],
            wall_t_grid=np.asarray(wt["grid"], dtype=float),
            wall_t_values=np.asarray(wt["values"], dtype=float),
            ca=ca,
            bulkhead_material=bh_mat,
            bulkhead_t=bh_t,
            ballast=list(m.get("structure", {}).get("ballast", [])),
        )
        members.append(mem)

        # Resolve named axial joints (e.g. fairleads) into the table so
        # the mooring block can reference them.
        for aj in m.get("axial_joints", []):
            joints[aj["name"]] = mem.point_at(float(aj["grid"]))

    return WindIOFloating(
        members=members,
        joints=joints,
        transition_joint=transition_joint,
        transition_piece_mass=float(fp.get("transition_piece_mass", 0.0)),
        mooring=doc.get("components", {}).get("mooring", {}),
        materials=materials,
    )


# --------------------------------------------------------------------------
# P3-2  Hydrostatic restoring (buoyancy + waterplane), WAMIT/.hst
#       convention (the gravitational −m g z_g terms are added by the
#       equations of motion via the body mass, not here — this matches
#       what HydroDynReader reads from the companion WAMIT `.hst`).
# --------------------------------------------------------------------------

RHO_SW = 1025.0       # seawater density, kg/m³
G = 9.80665           # gravity, m/s²


def _member_submerged(mem: FloatingMember, n: int, z_msl: float):
    """Trapezoidal displaced volume + centre of buoyancy of the
    below-water part of a circular member, plus its waterplane
    contribution (area + global first/second moments) if it pierces
    the free surface roughly vertically."""
    fr = np.linspace(0.0, 1.0, n + 1)
    pts = mem.end1[None, :] + fr[:, None] * (mem.end2 - mem.end1)[None, :]
    dia = mem.diameter_at(fr)
    area = 0.25 * np.pi * dia**2
    L = mem.length
    vol = 0.0
    cob = np.zeros(3)
    for k in range(n):
        z0, z1 = pts[k, 2], pts[k + 1, 2]
        if z0 >= z_msl and z1 >= z_msl:
            continue                                  # fully above MSL
        dl = L / n
        a0, a1 = area[k], area[k + 1]
        if z0 < z_msl and z1 < z_msl:                 # fully submerged
            frac = 1.0
        else:                                         # straddles MSL
            frac = abs(z_msl - min(z0, z1)) / abs(z1 - z0)
        dv = 0.5 * (a0 + a1) * dl * frac
        c = 0.5 * (pts[k] + pts[k + 1])
        vol += dv
        cob += dv * c

    awp = sx = sy = sxx = syy = sxy = 0.0
    nz = mem.axis[2]
    if abs(nz) > 0.3 and (pts[0, 2] - z_msl) * (pts[-1, 2] - z_msl) < 0.0:
        # Vertical-ish member crossing the free surface: an elliptical
        # cut of a circular section, area = πr²/|n_z|.
        t = (z_msl - mem.end1[2]) / (mem.end2[2] - mem.end1[2])
        xw, yw = (mem.end1 + t * (mem.end2 - mem.end1))[:2]
        r = 0.5 * float(mem.diameter_at(t))
        awp = np.pi * r * r / abs(nz)
        i_own = 0.25 * np.pi * r**4 / abs(nz)
        sx, sy = awp * xw, awp * yw
        sxx = awp * xw * xw + i_own
        syy = awp * yw * yw + i_own
        sxy = awp * xw * yw
    if vol > 0.0:
        cob = cob / vol
    return vol, cob, (awp, sx, sy, sxx, syy, sxy)


def hydrostatic_restoring(
    floating: WindIOFloating,
    *,
    rho: float = RHO_SW,
    g: float = G,
    z_msl: float = 0.0,
    n_seg: int = 200,
) -> np.ndarray:
    """6×6 hydrostatic restoring (DOF order surge, sway, heave, roll,
    pitch, yaw) from member geometry — the WAMIT/`.hst` buoyancy +
    waterplane convention (no gravitational term; that enters via the
    body mass elsewhere). For a freely-floating semi the heave /
    roll / pitch entries are geometry-exact, so this matches a
    potential-flow `.hst` closely (P3-2 integration anchor)."""
    S = Sx = Sy = Sxx = Syy = Sxy = 0.0
    vol = 0.0
    cob = np.zeros(3)
    for mem in floating.members:
        v, c, (a, sx, sy, sxx, syy, sxy) = _member_submerged(
            mem, n_seg, z_msl
        )
        vol += v
        cob += v * c
        S += a
        Sx += sx
        Sy += sy
        Sxx += sxx
        Syy += syy
        Sxy += sxy
    if vol > 0.0:
        cob = cob / vol
    zb = cob[2]

    C = np.zeros((6, 6))
    rg = rho * g
    C[2, 2] = rg * S
    C[2, 3] = C[3, 2] = rg * Sy
    C[2, 4] = C[4, 2] = -rg * Sx
    C[3, 3] = rg * Syy + rg * vol * zb
    C[3, 4] = C[4, 3] = -rg * Sxy
    C[3, 5] = C[5, 3] = -rg * vol * cob[0]
    C[4, 4] = rg * Sxx + rg * vol * zb
    C[4, 5] = C[5, 4] = -rg * vol * cob[1]
    return C
