"""Non-dimensionalization helpers for the FEM system.

Reference parameters:
  rm   = 10.0   (reference mass/length, kg/m)
  romg = 10.0   (reference angular velocity, rad/s)
  radius        (rotor tip radius or tower height, m)

Derived:
  om2    = romg²
  ref1   = rm * om2 * radius      = rm * romg² * radius
  ref2   = ref1 * radius          = rm * romg² * radius²
  ref4   = ref2 * radius²         = rm * romg² * radius⁴
  ref_mr = rm * radius
  ref_mr3= rm * radius³
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RM   = 10.0
ROMG = 10.0


@dataclass
class NondimParams:
    """Pre-computed non-dimensionalisation factors."""
    rm:      float
    romg:    float
    radius:  float
    hub_rad: float
    bl_len:  float    # = radius - hub_rad
    omega:   float    # non-dim rotational speed = omega_SI / romg
    omega2:  float    # omega²
    ref1:    float
    ref2:    float
    ref4:    float
    ref_mr:  float
    ref_mr3: float


def make_params(radius: float, hub_rad: float, rot_rpm: float,
                draft: float = 0.0) -> NondimParams:
    """Create NondimParams from physical inputs.

    Parameters
    ----------
    radius  : tower height above MSL [m] (or rotor tip radius for blades)
    hub_rad : hub radius / tower rigid-base height [m]
    rot_rpm : rotor speed [rpm]
    draft   : depth of tower base below MSL [m]; negative = base above MSL.
              For offshore towers the total flexible-beam length is
              radius + draft - hub_rad instead of radius - hub_rad.
    """
    omega_si = rot_rpm * np.pi / 30.0    # rad/s
    omega    = omega_si / ROMG
    om2      = ROMG ** 2
    # Fortran BModes.f90 line 616: radius = radius + draft (global replacement).
    # All non-dim factors use the total beam reference length, not just tower height.
    radius_nd = radius + draft
    ref1     = RM * om2 * radius_nd
    ref2     = ref1 * radius_nd
    ref4     = ref2 * radius_nd * radius_nd
    return NondimParams(
        rm      = RM,
        romg    = ROMG,
        radius  = radius_nd,
        hub_rad = hub_rad,
        bl_len  = radius_nd - hub_rad,
        omega   = omega,
        omega2  = omega ** 2,
        ref1    = ref1,
        ref2    = ref2,
        ref4    = ref4,
        ref_mr  = RM * radius_nd,
        ref_mr3 = RM * radius_nd ** 3,
    )


def nondim_section_props(sp: object, nd: NondimParams, id_form: int = 1,
                         beam_type: int = 1) -> dict:
    """Non-dimensionalise section properties array-by-array.

    Parameters
    ----------
    sp        : SectionProperties (from io.sec_props) in SI units
    nd        : NondimParams
    id_form   : 1 = wind-turbine (blade/tower), sign conventions apply
    beam_type : 1 = blade, 2 = tower

    Returns
    -------
    dict with keys matching SectionProperties field names, values as np.ndarray.
    Also adds 'sq_km1' and 'sq_km2' (mass-normalised moments of inertia).
    All arrays remain in the same span_loc ordering as the input.
    """
    # Sign conventions for id_form=1 (wind-turbine convention)
    # WT blade: str_tw, cg_offst, tc_offst, sc_offst are all negated.
    wt_blade = (id_form == 1 and beam_type == 1)
    sign_tw  = -1.0 if id_form == 1 else 1.0
    sign_off = -1.0 if wt_blade else 1.0
    d2r = np.pi / 180.0

    mass_den   = sp.mass_den   / nd.rm
    flp_stff   = sp.flp_stff  / nd.ref4
    edge_stff  = sp.edge_stff / nd.ref4
    tor_stff   = sp.tor_stff  / nd.ref4
    axial_stff = sp.axial_stff / nd.ref2
    str_tw     = sign_tw * sp.str_tw * d2r

    # cg_offst: negated for WT blade (id_form=1, beam_type=1)
    cg_offst = sign_off * sp.cg_offst / nd.radius

    # tc_offst: for WT blade, sign_off is applied before subtracting sc_offst so both are flipped
    tc_offst = sign_off * (sp.tc_offst - sp.sc_offst) / nd.radius

    # sq_km1/2: (mass_moment_per_length / mass_per_length) / radius²  (dimensionless)
    safe_m_si = np.where(sp.mass_den > 0.0, sp.mass_den, 1.0)
    sq_km1 = (sp.flp_iner  / safe_m_si) / nd.radius ** 2
    sq_km2 = (sp.edge_iner / safe_m_si) / nd.radius ** 2

    # sec_loc in file is [0,1] of blade length; shift to radius-normalised [hub_r..1]
    sec_loc_nd = (sp.span_loc * nd.bl_len + nd.hub_rad) / nd.radius

    return dict(
        sec_loc   = sec_loc_nd,
        str_tw    = str_tw,
        mass_den  = mass_den,
        flp_stff  = flp_stff,
        edge_stff = edge_stff,
        tor_stff  = tor_stff,
        axial_stff= axial_stff,
        cg_offst  = cg_offst,
        tc_offst  = tc_offst,
        sq_km1    = sq_km1,
        sq_km2    = sq_km2,
    )


@dataclass
class PlatformND:
    """Non-dimensionalized offshore platform support, referred to the tower base."""
    stiffness: 'np.ndarray'   # (6,6) combined (hydro_K + mooring_K), non-dim, at tower base
    mass: 'np.ndarray'        # (6,6) combined (i_matrix + hydro_M), non-dim, at tower base


def nondim_platform(plat: object, nd: 'NondimParams') -> 'PlatformND':
    """Non-dimensionalise platform 6×6 matrices and transform to tower-base FEM DOFs.

    Replicates the Fortran T_pform transformation from BModes.f90 lines 747-782.

    The .bmi file stores matrices with DOF order [sway, surge, heave, -pitch, roll, yaw].
    This function transforms each matrix to FEM DOF ordering at the tower base:
      [axial(0), v_disp(1), v_slope(2), w_disp(3), w_slope(4), phi(5)]

    The structural (i_matrix) and hydrodynamic/mooring matrices use different reference
    points: cm_pform for i_matrix, ref_msl for hydro/mooring.

    Parameters
    ----------
    plat : PlatformSupport  (from io.bmi)
    nd   : NondimParams
    """
    rroot = nd.hub_rad   # rigid-base height (= 0 for most towers)

    def _make_T(p_base: float) -> np.ndarray:
        # T maps FEM base DOFs → file DOFs:  u_file = T @ u_FEM
        # File DOF order: [sway(0), surge(1), heave(2), -pitch(3), roll(4), yaw(5)]
        # FEM DOF order:  [axial(0), v_disp(1), v_slope(2), w_disp(3), w_slope(4), phi(5)]
        T = np.zeros((6, 6))
        T[0, 1] =  1.0       # sway  ← v_disp
        T[0, 2] =  -p_base    # sway  ← v_slope  (rigid-arm coupling)
        T[1, 3] =  1.0       # surge ← w_disp
        T[1, 4] =  -p_base    # surge ← w_slope  (rigid-arm coupling)
        T[2, 0] =  1.0       # heave ← axial
        T[3, 4] = -1.0       # -pitch ← w_slope (sign flip)
        T[4, 2] =  1.0       # roll  ← v_slope
        T[5, 5] =  1.0       # yaw   ← phi
        return T

    # Structural mass: reference at the platform CM (cm_pform below MSL)
    p_base_i = plat.cm_pform - plat.draft + rroot
    T_i = _make_T(p_base_i)
    M_i_fem = T_i.T @ plat.i_matrix @ T_i

    # Hydrodynamic and mooring: reference at ref_msl below MSL
    p_base_h = plat.ref_msl - plat.draft + rroot
    T_h = _make_T(p_base_h)
    K_fem = T_h.T @ (plat.hydro_K + plat.mooring_K) @ T_h
    M_h_fem = T_h.T @ plat.hydro_M @ T_h

    K_base = K_fem
    M_base = M_i_fem + M_h_fem

    # Non-dimensionalise in FEM DOF ordering.
    # FEM translations: [0=axial, 1=v_disp, 3=w_disp]
    # FEM rotations:    [2=v_slope, 4=w_slope, 5=phi]
    tr = np.array([0, 1, 3])
    ro = np.array([2, 4, 5])

    ref1  = nd.ref1
    ref2  = nd.ref2
    ref3  = ref1 * nd.radius ** 2   # rm * romg^2 * radius^3

    K_nd = K_base.copy()
    K_nd[np.ix_(tr, tr)] /= ref1
    K_nd[np.ix_(tr, ro)] /= ref2
    K_nd[np.ix_(ro, tr)] /= ref2
    K_nd[np.ix_(ro, ro)] /= ref3

    ref_mr  = nd.ref_mr              # rm * radius
    ref_mr2 = nd.rm * nd.radius ** 2
    ref_mr3 = nd.ref_mr3             # rm * radius^3

    M_nd = M_base.copy()
    M_nd[np.ix_(tr, tr)] /= ref_mr
    M_nd[np.ix_(tr, ro)] /= ref_mr2
    M_nd[np.ix_(ro, tr)] /= ref_mr2
    M_nd[np.ix_(ro, ro)] /= ref_mr3

    return PlatformND(stiffness=K_nd, mass=M_nd)


@dataclass
class TipMassND:
    """Non-dimensionalized tip-mass / tower-top-mass contributions."""
    mass:     float   # tip_mass / ref_mr
    cm_loc:   float   # cm_loc_SI / radius  (sign-convention already applied)
    cm_axial: float   # cm_axial_SI / radius
    ixx: float        # after axis-remapping and / ref_mr3
    iyy: float
    izz: float
    ixy: float
    iyz: float
    izx: float


def nondim_tip_mass(tip, nd: NondimParams, beam_type: int = 1,
                    id_form: int = 1, hub_conn: int = 1) -> TipMassND:
    """Non-dimensionalise tip / tower-top mass and remap axes.

    Parameters
    ----------
    tip       : TipMassProps  (SI, from bmi.tip_mass)
    nd        : NondimParams
    beam_type : 1 = blade, 2 = tower
    id_form   : 1 = wind-turbine sign convention
    """
    # BModes' legacy overwrite quirk is needed to match the blade and most tower
    # reference cases. The bottom-fixed monopile path (hub_conn=3) uses the
    # literal tower-top offsets from the .bmi file instead.
    if beam_type == 2 and hub_conn == 3:
        cm_loc_SI   = tip.cm_offset
        cm_axial_SI = tip.cm_axial
    else:
        cm_loc_SI   = tip.cm_axial
        cm_axial_SI = 0.0

    # --- sign convention on cm_loc (blade only, id_form=1) ---
    if id_form == 1 and beam_type == 1:
        cm_loc_SI = -cm_loc_SI

    # --- axis remapping for inertia tensor ---
    # .bmi inertia components use a different axis convention per beam type; remap here
    if id_form == 1:
        if beam_type == 1:   # wt blade
            ixx_tp = tip.izz
            iyy_tp = tip.iyy
            izz_tp = tip.ixx
            ixy_tp = -tip.iyz
            iyz_tp = -tip.ixy
            izx_tp =  tip.izx
        else:                # wt tower
            ixx_tp =  tip.izz
            iyy_tp =  tip.ixx
            izz_tp =  tip.iyy
            ixy_tp =  tip.izx
            iyz_tp =  tip.ixy
            izx_tp =  tip.iyz
    else:                    # h/c blade
        ixx_tp = tip.ixx
        iyy_tp = tip.iyy
        izz_tp = tip.izz
        ixy_tp = tip.ixy
        iyz_tp = tip.iyz
        izx_tp = tip.izx

    # --- non-dimensionalise ---
    ref_mr  = nd.ref_mr          # rm * radius
    ref_mr3 = nd.ref_mr3         # rm * radius^3

    return TipMassND(
        mass    = tip.mass      / ref_mr,
        cm_loc  = cm_loc_SI    / nd.radius,
        cm_axial= cm_axial_SI  / nd.radius,
        ixx     = ixx_tp      / ref_mr3,
        iyy     = iyy_tp      / ref_mr3,
        izz     = izz_tp      / ref_mr3,
        ixy     = ixy_tp      / ref_mr3,
        iyz     = iyz_tp      / ref_mr3,
        izx     = izx_tp      / ref_mr3,
    )
