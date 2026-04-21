"""15-DOF Bernoulli-Euler beam element mass and stiffness matrices.

Covers free-vibration about the undeformed frame; straight elements only
(no sweep, droop, or elastic coupling terms eb1/eb2/ec1/ec2).

Local DOF layout (0-based):
  0-3  : axial u  — cubic Lagrange  hu(0..3)
  4-7  : v (lag/edge) — Hermite cubic h(0..3)   [uses eiz, edge stiffness]
  8-11 : w (flap)     — Hermite cubic h(0..3)   [uses eiy, flap stiffness]
  12-14: torsion phi  — quadratic     hf(0..2)

Note on axis labelling: the 'v' DOFs (4-7) couple to eiz (edge/lag stiffness)
and the 'w' DOFs (8-11) couple to eiy (flap stiffness).  This is correct for
the coordinate system adopted here, even though the names look transposed.
"""

from __future__ import annotations

import numpy as np

from .gauss import gauss_6pt

# 6-point Gauss quadrature on [0,1] (cached at module load)
_GQP, _GQW = gauss_6pt()
_NGAUSS = 6


def element_matrices(
    eli: float,
    xbi: float,
    eiy: float,
    eiz: float,
    gj: float,
    eac: float,
    rmas: float,
    skm1: float,
    skm2: float,
    eg: float,
    ea: float,
    axfi: float,
    omega2: float,
    sec_loc: np.ndarray,
    str_tw: np.ndarray,
    distr_k: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 15×15 element stiffness and mass matrices.

    Parameters
    ----------
    eli   : element length (non-dimensional)
    xbi   : inboard (root-side) end position (non-dimensional)
    eiy   : flap bending stiffness EI_y (non-dimensional)
    eiz   : edge bending stiffness EI_z (non-dimensional)
    gj    : torsion stiffness GJ (non-dimensional)
    eac   : axial stiffness EA (non-dimensional)
    rmas  : mass per unit length (non-dimensional)
    skm1  : mass moment of inertia/length about flap axis (non-dim, = I_flap/m)
    skm2  : mass moment of inertia/length about lag axis  (non-dim, = I_lag/m)
    eg    : CG offset × mass  (= cg_offst * rmas, non-dimensional)
    ea    : tension-centre offset (tc_offst, non-dimensional) — used in aeithp
    axfi  : centrifugal tension at outboard end (non-dimensional)
    omega2: (omega/romg)² — non-dimensional rotational speed squared
    sec_loc: spanwise station locations (non-dimensional, for twist interpolation)
    str_tw : structural twist at each station (radians, negated for id_form=1)

    Returns
    -------
    ek : (15, 15) element stiffness matrix
    em : (15, 15) element mass matrix
    """
    ek = np.zeros((15, 15))
    em = np.zeros((15, 15))

    xbils = (xbi + eli) ** 2  # (outboard end)²

    # Derived elastic constants (zero because eb1=eb2=ec1=ec2=0, th0p=0)
    dei = eiz - eiy
    dsk = skm2 - skm1
    skm = skm1 + skm2
    # aeithp = aei * th0p = 0

    for n in range(_NGAUSS):
        gqwl  = _GQW[n] * eli
        gqwml = gqwl * rmas
        ci    = _GQP[n]
        cis   = ci * ci
        cic   = ci * cis
        xi    = ci * eli
        x     = xbi + xi   # absolute non-dimensional position

        # ----------------------------------------------------------------
        # Shape functions  (ci = normalised coordinate ∈ [0,1])
        # ----------------------------------------------------------------

        # Cubic Lagrange for axial (4-node: at 0, 1/3, 2/3, 1)
        hu = np.array([
            -4.5*cic +  9.0*cis - 5.5*ci + 1.0,
             13.5*cic - 22.5*cis + 9.0*ci,
            -13.5*cic + 18.0*cis - 4.5*ci,
              4.5*cic -  4.5*cis + ci,
        ])
        hup = np.array([
            (-13.5*cis + 18.0*ci - 5.5) / eli,
            ( 40.5*cis - 45.0*ci + 9.0) / eli,
            (-40.5*cis + 36.0*ci - 4.5) / eli,
            ( 13.5*cis -  9.0*ci + 1.0) / eli,
        ])

        # Hermite cubic for transverse displacement (4-node: disp+slope at each end)
        h = np.array([
             2.0*cic - 3.0*cis + 1.0,
             eli*(cic - 2.0*cis + ci),
            -2.0*cic + 3.0*cis,
             eli*(cic - cis),
        ])
        hp = np.array([
             6.0*(cis - ci) / eli,
             3.0*cis - 4.0*ci + 1.0,
            -6.0*(cis - ci) / eli,
             3.0*cis - 2.0*ci,
        ])
        hs = np.array([
            ( 12.0*ci - 6.0) / eli / eli,
            (  6.0*ci - 4.0) / eli,
            (-12.0*ci + 6.0) / eli / eli,
            (  6.0*ci - 2.0) / eli,
        ])

        # Quadratic for torsion (3-node: at 0, 1/2, 1)
        hf  = np.array([2.0*cis - 3.0*ci + 1.0,
                        -4.0*cis + 4.0*ci,
                         2.0*cis - ci])
        hfp = np.array([(4.0*ci - 3.0) / eli,
                        (-8.0*ci + 4.0) / eli,
                        ( 4.0*ci - 1.0) / eli])

        # ----------------------------------------------------------------
        # Structural twist at this Gauss point (interpolate)
        # ----------------------------------------------------------------
        twx = float(np.interp(x, sec_loc, str_tw))
        th0 = twx   # th75 = 0 for free-vibration (blade pitch = 0 in analysis)

        ct = np.cos(th0)
        st = np.sin(th0)
        cst = ct * st
        cts = ct * ct
        sts = st * st
        c2t = cts - sts
        dskc2t = dsk * c2t

        # Composite bending stiffness terms
        ceicss = eiz * cts + eiy * sts   # flap group
        ceiscs = eiz * sts + eiy * cts   # lag group
        deicst = dei * cst

        # Coupling with CG offset
        egct  = eg * ct
        egst  = eg * st
        egxct = egct * x
        egxst = egst * x

        # Axial-transverse coupling: eacea = EA × tc_offst
        eacea  = eac * ea
        eaeact = eacea * ct
        eaeast = eacea * st

        # Centrifugal tension at Gauss point
        fi = (axfi + 0.5 * rmas * (xbils - x * x)) * omega2

        # ----------------------------------------------------------------
        # Accumulate stiffness and mass matrices
        # ----------------------------------------------------------------
        for i in range(4):
            i4  = i + 4
            i8  = i + 8

            for j in range(4):
                j4  = j + 4
                j8  = j + 8

                # Products of shape functions
                hupj_hupj = hup[i] * hup[j]
                hh        = h[i]   * h[j]
                hphp      = hp[i]  * hp[j]
                hshs      = hs[i]  * hs[j]

                # Axial-axial stiffness
                ek[i, j]   += gqwl * hupj_hupj * eac

                # Axial–flap (v) coupling
                ek[i, j4]  -= gqwl * hup[i] * hs[j] * eaeact
                ek[j4, i]   = ek[i, j4]

                # Axial–lag (w) coupling
                ek[i, j8]  -= gqwl * hup[i] * hs[j] * eaeast
                ek[j8, i]   = ek[i, j8]

                # Flap (v) stiffness: centrifugal + bending − centrifugal softening + foundation
                ek[i4, j4] += (gqwl * (hphp * fi + hshs * ceicss + hh * distr_k)
                               - gqwml * omega2 * hh)

                # Flap–lag (v–w) cross-bending coupling
                ek[i4, j8] += gqwl * hshs * deicst
                ek[j8, i4]  = ek[i4, j8]

                # Lag (w) stiffness: centrifugal + bending + foundation
                ek[i8, j8] += gqwl * (hphp * fi + hshs * ceiscs + hh * distr_k)

                # Mass: axial
                em[i, j]   += gqwml * hu[i] * hu[j]

                # Mass: flap (v) and lag (w)
                em[i4, j4] += gqwml * hh
                em[i8, j8] += gqwml * hh

            # Torsion coupling (j = 0..2)
            for j in range(3):
                j12 = j + 12

                hhf  = h[i]  * hf[j]
                hphf = hp[i] * hf[j]

                # Flap(v)–torsion stiffness (eb2=ec2=0, aeithp=0)
                ek[i4, j12] += gqwml * omega2 * (hhf * egst - hphf * egxst)
                ek[j12, i4]  = ek[i4, j12]

                # Lag(w)–torsion stiffness (eb2=ec2=0)
                ek[i8, j12] += gqwml * omega2 * hphf * egxct
                ek[j12, i8]  = ek[i8, j12]

                # Mass: flap(v)–torsion and lag(w)–torsion coupling
                em[i4, j12] -= gqwml * hhf * egst
                em[j12, i4]  = em[i4, j12]

                em[i8, j12] += gqwml * hhf * egct
                em[j12, i8]  = em[i8, j12]

        # Torsion–torsion
        for i in range(3):
            i12 = i + 12
            for j in range(3):
                j12 = j + 12
                hfhf = hf[i] * hf[j]
                # Centrifugal torsion softening + GJ bending
                ek[i12, j12] += gqwml * hfhf * dskc2t * omega2 + gqwl * hfp[i] * hfp[j] * gj
                em[i12, j12] += gqwml * hfhf * skm

    return ek, em
