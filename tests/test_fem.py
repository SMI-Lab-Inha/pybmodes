"""FEM core tests — Gauss quadrature, element matrices, and CertTest integration."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pybmodes.fem.gauss import gauss_5pt, gauss_6pt
from pybmodes.fem.boundary import build_connectivity, n_free_dof, NEDOF, NESH
from pybmodes.fem.nondim import make_params, nondim_section_props, nondim_tip_mass
from pybmodes.fem.assembly import assemble, compute_element_props
from pybmodes.fem.solver import solve_modes, eigvals_to_hz
from pybmodes.io.bmi import read_bmi
from pybmodes.io.sec_props import read_sec_props
from pybmodes.io.out_parser import read_out


# ============================================================================
# Gauss quadrature
# ============================================================================

class TestGauss6pt:
    def setup_method(self):
        self.gqp, self.gqw = gauss_6pt()

    def test_n_points(self):
        assert len(self.gqp) == 6
        assert len(self.gqw) == 6

    def test_weights_sum_to_one(self):
        assert np.sum(self.gqw) == pytest.approx(1.0, rel=1e-12)

    def test_points_in_unit_interval(self):
        assert np.all(self.gqp > 0.0)
        assert np.all(self.gqp < 1.0)

    def test_symmetric_about_half(self):
        # Points and weights should be symmetric about 0.5
        gqp_s = np.sort(self.gqp)
        assert gqp_s[0] + gqp_s[5] == pytest.approx(1.0, abs=1e-12)
        assert gqp_s[1] + gqp_s[4] == pytest.approx(1.0, abs=1e-12)
        assert gqp_s[2] + gqp_s[3] == pytest.approx(1.0, abs=1e-12)

    def test_integrates_quadratic_exactly(self):
        # ∫₀¹ x² dx = 1/3
        gqp, gqw = gauss_6pt()
        result = np.sum(gqw * gqp ** 2)
        assert result == pytest.approx(1.0 / 3.0, rel=1e-10)

    def test_integrates_degree11_exactly(self):
        # 6-pt rule is exact for degree ≤ 11; ∫₀¹ x¹¹ dx = 1/12
        gqp, gqw = gauss_6pt()
        result = np.sum(gqw * gqp ** 11)
        assert result == pytest.approx(1.0 / 12.0, rel=1e-8)


class TestGauss5pt:
    def test_weights_sum_to_one(self):
        _, gqw = gauss_5pt()
        assert np.sum(gqw) == pytest.approx(1.0, rel=1e-12)

    def test_n_points(self):
        gqp, gqw = gauss_5pt()
        assert len(gqp) == len(gqw) == 5


# ============================================================================
# Connectivity
# ============================================================================

class TestConnectivity:
    def test_ngd_formula(self):
        for n in (1, 5, 12, 20):
            assert n_free_dof(n) == 9 * n

    def test_shape(self):
        indeg = build_connectivity(5)
        assert indeg.shape == (NEDOF, 5)

    def test_tip_element_axial_tip(self):
        # Local DOF 4 (0-based: 3) of element 0 (tip) → global DOF 1 (Fortran) = 0 (Python)
        indeg = build_connectivity(3)
        assert indeg[3, 0] == 1   # 1-based

    def test_root_dofs_zeroed(self):
        nselt = 4
        indeg = build_connectivity(nselt)
        # Root local DOFs (0-based): 0, 4, 5, 8, 9, 12 of last element
        for j in [0, 4, 5, 8, 9, 12]:
            assert indeg[j, nselt - 1] == 0, f"root DOF {j} not zeroed"

    def test_shared_node_flap_disp(self):
        # For element 0 (tip), local DOF 7 (0-based: 6) = flap disp at outboard = global 2
        # For element 1, local DOF 5 (0-based: 4) = flap disp at inboard  = should also be global 2
        # Wait: for inboard end of element 1, local DOF 5 → ivecbe[4]+9 = 11+9 = 20 ≠ 2
        # Actually the SHARED DOF for junction 0-1 is outboard of element 1 = local DOF 6 → 2+9=11
        # and inboard of element 0 = local DOF 4 → 11
        nselt = 3
        indeg = build_connectivity(nselt)
        # inboard end of element 0 (tip): local 4 (v disp) → 11+0=11
        assert indeg[4, 0] == 11
        # outboard end of element 1: local 6 (v disp) → 2+9=11
        assert indeg[6, 1] == 11


# ============================================================================
# CertTest integration tests
# ============================================================================

CERT_DIR = pathlib.Path(__file__).parent / "data" / "certtest"
REF_DIR  = CERT_DIR / "expected"


def _run_fem(bmi_path: pathlib.Path, n_modes: int = 20):
    """Full FEM pipeline: read → non-dim → assemble → solve → extract frequencies."""
    bmi = read_bmi(bmi_path)
    sp  = read_sec_props(bmi.resolve_sec_props_path())

    # Non-dimensionalisation
    nd = make_params(bmi.radius, bmi.hub_rad, bmi.rot_rpm)
    props_nd = nondim_section_props(sp, nd, id_form=1, beam_type=bmi.beam_type)

    # Apply scaling factors from bmi
    sc = bmi.scaling
    props_nd['mass_den']   *= sc.sec_mass
    props_nd['flp_stff']   *= sc.flp_stff
    props_nd['edge_stff']  *= sc.edge_stff
    props_nd['tor_stff']   *= sc.tor_stff
    props_nd['axial_stff'] *= sc.axial_stff
    props_nd['cg_offst']   *= sc.cg_offst
    props_nd['tc_offst']   *= sc.tc_offst
    props_nd['sq_km1']     *= sc.flp_iner
    props_nd['sq_km2']     *= sc.lag_iner

    nselt = bmi.n_elements

    hub_r = nd.hub_rad / nd.radius

    el, xb, cfe, eiy, eiz, gj, eac, rmas, skm1, skm2, eg, ea = compute_element_props(
        nselt   = nselt,
        el_loc  = bmi.el_loc,
        sp      = type('SP', (), {   # simple namespace
            'span_loc'   : props_nd['sec_loc'],
            'flp_stff'   : props_nd['flp_stff'],
            'edge_stff'  : props_nd['edge_stff'],
            'tor_stff'   : props_nd['tor_stff'],
            'axial_stff' : props_nd['axial_stff'],
            'mass_den'   : props_nd['mass_den'],
            'cg_offst'   : props_nd['cg_offst'],
            'tc_offst'   : props_nd['tc_offst'],
            'flp_iner'   : sp.flp_iner,   # still SI here; skm computed separately
            'edge_iner'  : sp.edge_iner,
        })(),
        hub_r   = hub_r,
    )

    # Overwrite skm1/skm2 with non-dimensionalised values interpolated at midpoints
    xmid = xb + 0.5 * el
    skm1 = np.interp(xmid, props_nd['sec_loc'], props_nd['sq_km1'])
    skm2 = np.interp(xmid, props_nd['sec_loc'], props_nd['sq_km2'])

    tip_mass_nd = None
    if bmi.tip_mass is not None and bmi.tip_mass.mass > 0.0:
        tip_mass_nd = nondim_tip_mass(bmi.tip_mass, nd,
                                      beam_type=bmi.beam_type, id_form=1)

    wire_k_nd = None
    wire_node_attach = None
    if bmi.tow_support == 1 and bmi.support is not None:
        sup = bmi.support
        wire_k_nd = np.array([
            sup.n_wires[i] * sup.wire_stiffness[i]
            * np.cos(np.radians(sup.th_wire[i])) ** 2
            / nd.ref1
            for i in range(sup.n_attachments)
        ])
        wire_node_attach = sup.node_attach

    gk, gm, _ = assemble(
        nselt            = nselt,
        el               = el,
        xb               = xb,
        cfe              = cfe,
        eiy              = eiy,
        eiz              = eiz,
        gj               = gj,
        eac              = eac,
        rmas             = rmas,
        skm1             = skm1,
        skm2             = skm2,
        eg               = eg,
        ea               = ea,
        omega2           = nd.omega2,
        sec_loc          = props_nd['sec_loc'],
        str_tw           = props_nd['str_tw'],
        tip_mass         = tip_mass_nd,
        wire_k_nd        = wire_k_nd,
        wire_node_attach = wire_node_attach,
    )

    eigvals, _ = solve_modes(gk, gm, n_modes=n_modes)
    freqs_hz   = eigvals_to_hz(eigvals, nd.romg)
    return freqs_hz


@pytest.mark.integration
class TestCertTest01Blade:
    """Test01: non-uniform rotating blade, no tip mass."""

    @pytest.fixture(autouse=True)
    def compute(self):
        bmi_path = CERT_DIR / "Test01_nonunif_blade.bmi"
        ref_path = REF_DIR / "Test01_nonunif_blade.out"
        self.freqs = _run_fem(bmi_path, n_modes=20)
        self.ref   = read_out(ref_path)

    def test_mode1_freq(self):
        assert self.freqs[0] == pytest.approx(self.ref[0].frequency, rel=5e-3)

    def test_mode2_freq(self):
        assert self.freqs[1] == pytest.approx(self.ref[1].frequency, rel=5e-3)

    def test_mode3_freq(self):
        assert self.freqs[2] == pytest.approx(self.ref[2].frequency, rel=5e-3)

    def test_first_5_modes_within_05pct(self):
        ref_freqs = self.ref.frequencies()[:5]
        for k in range(5):
            assert self.freqs[k] == pytest.approx(ref_freqs[k], rel=5e-3), \
                f"mode {k+1}: got {self.freqs[k]:.4f} Hz, ref {ref_freqs[k]:.4f} Hz"


@pytest.mark.integration
class TestCertTest02BladeWithTipMass:
    """Test02: non-uniform rotating blade with 40 kg tip mass."""

    @pytest.fixture(autouse=True)
    def compute(self):
        bmi_path = CERT_DIR / "Test02_blade_with_tip_mass.bmi"
        ref_path = REF_DIR / "Test02_blade_with_tip_mass.out"
        self.freqs = _run_fem(bmi_path, n_modes=20)
        self.ref   = read_out(ref_path)

    def test_mode1_freq(self):
        assert self.freqs[0] == pytest.approx(self.ref[0].frequency, rel=5e-3)

    def test_mode2_freq(self):
        assert self.freqs[1] == pytest.approx(self.ref[1].frequency, rel=5e-3)

    def test_mode3_freq(self):
        assert self.freqs[2] == pytest.approx(self.ref[2].frequency, rel=5e-3)

    def test_first_5_modes_within_05pct(self):
        ref_freqs = self.ref.frequencies()[:5]
        for k in range(5):
            assert self.freqs[k] == pytest.approx(ref_freqs[k], rel=5e-3), \
                f"mode {k+1}: got {self.freqs[k]:.4f} Hz, ref {ref_freqs[k]:.4f} Hz"


@pytest.mark.integration
class TestCertTest03Tower:
    """Test03: onshore tower, no rotation."""

    @pytest.fixture(autouse=True)
    def compute(self):
        bmi_path = CERT_DIR / "Test03_tower.bmi"
        ref_path = REF_DIR / "Test03_tower.out"
        self.freqs = _run_fem(bmi_path, n_modes=20)
        self.ref   = read_out(ref_path)

    def test_mode1_freq(self):
        assert self.freqs[0] == pytest.approx(self.ref[0].frequency, rel=5e-3)

    def test_mode2_freq(self):
        assert self.freqs[1] == pytest.approx(self.ref[1].frequency, rel=5e-3)

    def test_first_4_modes_within_05pct(self):
        ref_freqs = self.ref.frequencies()[:4]
        for k in range(4):
            assert self.freqs[k] == pytest.approx(ref_freqs[k], rel=5e-3), \
                f"mode {k+1}: got {self.freqs[k]:.4f} Hz, ref {ref_freqs[k]:.4f} Hz"


@pytest.mark.integration
class TestCertTest04WireTower:
    """Test04: tower with two tension-wire attachment sets."""

    @pytest.fixture(autouse=True)
    def compute(self):
        bmi_path = CERT_DIR / "Test04_wires_supported_tower.bmi"
        ref_path = REF_DIR / "Test04_wires_supported_tower.out"
        self.freqs = _run_fem(bmi_path, n_modes=20)
        self.ref   = read_out(ref_path)

    def test_mode1_freq(self):
        assert self.freqs[0] == pytest.approx(self.ref[0].frequency, rel=5e-3)

    def test_mode2_freq(self):
        assert self.freqs[1] == pytest.approx(self.ref[1].frequency, rel=5e-3)

    def test_first_4_modes_within_05pct(self):
        ref_freqs = self.ref.frequencies()[:4]
        for k in range(4):
            assert self.freqs[k] == pytest.approx(ref_freqs[k], rel=5e-3), \
                f"mode {k+1}: got {self.freqs[k]:.4f} Hz, ref {ref_freqs[k]:.4f} Hz"
