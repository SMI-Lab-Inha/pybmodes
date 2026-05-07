"""Unit tests for non-dimensionalisation helpers in :mod:`pybmodes.fem.nondim`.

Cover:
  * ``make_params`` reference factors and offshore draft handling
  * ``nondim_section_props`` sign conventions and section-location remapping
  * ``nondim_tip_mass`` axis remapping for blade vs. tower vs. monopile
  * ``nondim_platform`` shape / symmetry
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from pybmodes.fem.nondim import (
    RM,
    ROMG,
    NondimParams,
    PlatformND,
    TipMassND,
    make_params,
    nondim_platform,
    nondim_section_props,
    nondim_tip_mass,
)

# ===========================================================================
# make_params
# ===========================================================================

class TestMakeParamsOnshore:

    def test_radius_passthrough_when_no_draft(self):
        nd = make_params(radius=50.0, hub_rad=2.0, rot_rpm=0.0)
        assert nd.radius == pytest.approx(50.0)
        assert nd.hub_rad == pytest.approx(2.0)
        assert nd.bl_len == pytest.approx(48.0)

    def test_zero_rpm_implies_zero_omega(self):
        nd = make_params(radius=50.0, hub_rad=0.0, rot_rpm=0.0)
        assert nd.omega == pytest.approx(0.0)
        assert nd.omega2 == pytest.approx(0.0)

    def test_omega_normalisation(self):
        # rot_rpm=60 -> omega_SI=2pi rad/s, omega = 2pi/10
        nd = make_params(radius=10.0, hub_rad=0.0, rot_rpm=60.0)
        assert nd.omega == pytest.approx(2.0 * np.pi / 10.0)

    def test_reference_factors(self):
        nd = make_params(radius=4.0, hub_rad=0.0, rot_rpm=0.0)
        # ref1 = rm * romg^2 * r,  ref2 = rm * romg^2 * r^2,  ref4 = rm * romg^2 * r^4
        assert nd.ref1 == pytest.approx(RM * ROMG**2 * 4.0)
        assert nd.ref2 == pytest.approx(RM * ROMG**2 * 16.0)
        assert nd.ref4 == pytest.approx(RM * ROMG**2 * 256.0)
        assert nd.ref_mr == pytest.approx(RM * 4.0)
        assert nd.ref_mr3 == pytest.approx(RM * 64.0)

    def test_returns_dataclass(self):
        nd = make_params(radius=1.0, hub_rad=0.0, rot_rpm=0.0)
        assert isinstance(nd, NondimParams)


class TestMakeParamsOffshore:

    def test_draft_extends_radius(self):
        # Total beam length for offshore towers is radius + draft
        nd = make_params(radius=80.0, hub_rad=0.0, rot_rpm=0.0, draft=20.0)
        assert nd.radius == pytest.approx(100.0)
        assert nd.bl_len == pytest.approx(100.0)

    def test_draft_propagates_into_reference_factors(self):
        nd = make_params(radius=10.0, hub_rad=0.0, rot_rpm=0.0, draft=10.0)
        # radius_nd = 20 -> ref4 = rm * romg^2 * 20^4
        assert nd.ref4 == pytest.approx(RM * ROMG**2 * 20.0**4)

    def test_negative_draft_shrinks_radius(self):
        # Per the docstring, a negative draft means the base sits above MSL
        # (rare, but the helper should accept it).
        nd = make_params(radius=80.0, hub_rad=0.0, rot_rpm=0.0, draft=-10.0)
        assert nd.radius == pytest.approx(70.0)


# ===========================================================================
# nondim_section_props
# ===========================================================================

def _make_sec_props(n: int = 5) -> SimpleNamespace:
    """Build a minimal duck-typed SectionProperties for nondim tests."""
    span = np.linspace(0.0, 1.0, n)
    one = np.ones(n)
    return SimpleNamespace(
        span_loc   = span,
        str_tw     = 10.0 * one,             # 10 deg
        tw_iner    = 0.0 * one,
        mass_den   = 100.0 * one,            # kg/m
        flp_iner   = 5.0 * one,
        edge_iner  = 5.0 * one,
        flp_stff   = 1.0e8 * one,
        edge_stff  = 2.0e8 * one,
        tor_stff   = 5.0e7 * one,
        axial_stff = 1.0e9 * one,
        cg_offst   = 0.05 * one,
        sc_offst   = 0.01 * one,
        tc_offst   = 0.02 * one,
    )


class TestNondimSectionProps:

    def test_returns_dict_with_expected_keys(self):
        sp = _make_sec_props()
        nd = make_params(radius=10.0, hub_rad=0.0, rot_rpm=0.0)
        out = nondim_section_props(sp, nd, id_form=1, beam_type=1)
        for key in ("sec_loc", "str_tw", "mass_den", "flp_stff",
                    "edge_stff", "tor_stff", "axial_stff",
                    "cg_offst", "tc_offst", "sq_km1", "sq_km2"):
            assert key in out

    def test_blade_sign_convention(self):
        # WT blade (id_form=1, beam_type=1): str_tw, cg_offst, tc_offst negated
        sp = _make_sec_props()
        nd = make_params(radius=10.0, hub_rad=0.0, rot_rpm=0.0)
        out = nondim_section_props(sp, nd, id_form=1, beam_type=1)
        assert np.all(out["str_tw"] < 0.0)
        assert np.all(out["cg_offst"] < 0.0)
        # tc_offst depends on (tc - sc) sign
        assert np.all(out["tc_offst"] <= 0.0)

    def test_tower_does_not_negate_offsets(self):
        # Tower (beam_type=2): only twist gets a sign flip, not the offsets
        sp = _make_sec_props()
        nd = make_params(radius=10.0, hub_rad=0.0, rot_rpm=0.0)
        out = nondim_section_props(sp, nd, id_form=1, beam_type=2)
        # cg_offst is now positive (no extra negation)
        assert np.all(out["cg_offst"] > 0.0)

    def test_safe_mass_division(self):
        # If mass_den has a zero, sq_km1/2 should not yield NaN/Inf.
        sp = _make_sec_props()
        sp.mass_den = np.array([0.0, 100.0, 100.0, 100.0, 100.0])
        nd = make_params(radius=10.0, hub_rad=0.0, rot_rpm=0.0)
        out = nondim_section_props(sp, nd, id_form=1, beam_type=2)
        assert np.all(np.isfinite(out["sq_km1"]))
        assert np.all(np.isfinite(out["sq_km2"]))

    def test_sec_loc_remap_no_hub_offset(self):
        # With hub_rad=0, sec_loc should remain in [0, 1].
        sp = _make_sec_props()
        nd = make_params(radius=10.0, hub_rad=0.0, rot_rpm=0.0)
        out = nondim_section_props(sp, nd, id_form=1, beam_type=2)
        assert out["sec_loc"][0] == pytest.approx(0.0)
        assert out["sec_loc"][-1] == pytest.approx(1.0)

    def test_sec_loc_remap_with_hub_offset(self):
        # With hub_rad > 0, sec_loc starts at hub_rad / radius.
        sp = _make_sec_props()
        nd = make_params(radius=10.0, hub_rad=2.0, rot_rpm=0.0)
        out = nondim_section_props(sp, nd, id_form=1, beam_type=2)
        assert out["sec_loc"][0] == pytest.approx(0.2)
        assert out["sec_loc"][-1] == pytest.approx(1.0)


# ===========================================================================
# nondim_tip_mass
# ===========================================================================

@dataclass
class _TipStub:
    mass: float
    cm_offset: float
    cm_axial: float
    ixx: float
    iyy: float
    izz: float
    ixy: float
    izx: float
    iyz: float


def _tip(**overrides) -> _TipStub:
    base = dict(mass=10.0, cm_offset=0.1, cm_axial=0.05,
                ixx=1.0, iyy=2.0, izz=3.0,
                ixy=0.1, izx=0.2, iyz=0.3)
    base.update(overrides)
    return _TipStub(**base)


class TestNondimTipMass:

    def test_blade_axis_remap(self):
        tip = _tip()
        nd = make_params(radius=10.0, hub_rad=0.0, rot_rpm=0.0)
        out = nondim_tip_mass(tip, nd, beam_type=1, id_form=1, hub_conn=1)
        # blade: ixx_tp <- izz_SI, izz_tp <- ixx_SI
        assert out.ixx == pytest.approx(3.0 / nd.ref_mr3)
        assert out.izz == pytest.approx(1.0 / nd.ref_mr3)
        assert out.iyy == pytest.approx(2.0 / nd.ref_mr3)
        # ixy_tp = -iyz, iyz_tp = -ixy
        assert out.ixy == pytest.approx(-0.3 / nd.ref_mr3)
        assert out.iyz == pytest.approx(-0.1 / nd.ref_mr3)
        assert out.izx == pytest.approx(0.2 / nd.ref_mr3)

    def test_tower_axis_remap(self):
        tip = _tip()
        nd = make_params(radius=10.0, hub_rad=0.0, rot_rpm=0.0)
        out = nondim_tip_mass(tip, nd, beam_type=2, id_form=1, hub_conn=1)
        # tower: ixx_tp <- izz, iyy_tp <- ixx, izz_tp <- iyy
        assert out.ixx == pytest.approx(3.0 / nd.ref_mr3)
        assert out.iyy == pytest.approx(1.0 / nd.ref_mr3)
        assert out.izz == pytest.approx(2.0 / nd.ref_mr3)

    def test_blade_negates_cm_loc(self):
        tip = _tip()
        nd = make_params(radius=10.0, hub_rad=0.0, rot_rpm=0.0)
        out = nondim_tip_mass(tip, nd, beam_type=1, id_form=1, hub_conn=1)
        # For blade, cm_loc = -cm_axial / radius
        assert out.cm_loc == pytest.approx(-0.05 / 10.0)
        assert out.cm_axial == pytest.approx(0.0)

    def test_monopile_uses_literal_offsets(self):
        # For tower with hub_conn=3 (monopile), cm_loc = cm_offset directly.
        tip = _tip(cm_offset=0.7, cm_axial=0.05)
        nd = make_params(radius=10.0, hub_rad=0.0, rot_rpm=0.0)
        out = nondim_tip_mass(tip, nd, beam_type=2, id_form=1, hub_conn=3)
        assert out.cm_loc == pytest.approx(0.7 / 10.0)
        assert out.cm_axial == pytest.approx(0.05 / 10.0)

    def test_mass_normalisation(self):
        tip = _tip(mass=2500.0)
        nd = make_params(radius=10.0, hub_rad=0.0, rot_rpm=0.0)
        out = nondim_tip_mass(tip, nd, beam_type=2, id_form=1, hub_conn=1)
        assert out.mass == pytest.approx(2500.0 / nd.ref_mr)

    def test_returns_dataclass(self):
        tip = _tip()
        nd = make_params(radius=10.0, hub_rad=0.0, rot_rpm=0.0)
        out = nondim_tip_mass(tip, nd)
        assert isinstance(out, TipMassND)


# ===========================================================================
# nondim_platform
# ===========================================================================

def _platform_stub(**kwargs) -> SimpleNamespace:
    """Minimal duck-typed PlatformSupport for nondim_platform."""
    base = dict(
        draft=20.0,
        cm_pform=0.0,
        ref_msl=20.0,
        i_matrix=np.eye(6) * 1.0e6,
        hydro_M=np.eye(6) * 1.0e5,
        hydro_K=np.eye(6) * 1.0e7,
        mooring_K=np.eye(6) * 1.0e6,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestNondimPlatform:

    def test_returns_platform_nd(self):
        plat = _platform_stub()
        nd = make_params(radius=80.0, hub_rad=0.0, rot_rpm=0.0, draft=20.0)
        out = nondim_platform(plat, nd)
        assert isinstance(out, PlatformND)

    def test_shapes_are_6x6(self):
        plat = _platform_stub()
        nd = make_params(radius=80.0, hub_rad=0.0, rot_rpm=0.0, draft=20.0)
        out = nondim_platform(plat, nd)
        assert out.stiffness.shape == (6, 6)
        assert out.mass.shape == (6, 6)

    def test_outputs_finite(self):
        plat = _platform_stub()
        nd = make_params(radius=80.0, hub_rad=0.0, rot_rpm=0.0, draft=20.0)
        out = nondim_platform(plat, nd)
        assert np.all(np.isfinite(out.stiffness))
        assert np.all(np.isfinite(out.mass))

    def test_zero_input_gives_zero_output(self):
        plat = _platform_stub(
            i_matrix=np.zeros((6, 6)),
            hydro_M=np.zeros((6, 6)),
            hydro_K=np.zeros((6, 6)),
            mooring_K=np.zeros((6, 6)),
        )
        nd = make_params(radius=80.0, hub_rad=0.0, rot_rpm=0.0, draft=20.0)
        out = nondim_platform(plat, nd)
        assert np.allclose(out.stiffness, 0.0)
        assert np.allclose(out.mass, 0.0)
