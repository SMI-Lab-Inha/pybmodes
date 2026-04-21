"""Tests for pybmodes.io against the four CertTest input/output files."""

import numpy as np
import pytest

from pybmodes.io.bmi import read_bmi, TensionWireSupport
from pybmodes.io.sec_props import read_sec_props
from pybmodes.io.out_parser import read_out


# ===========================================================================
# bmi.py — main input file parser
# ===========================================================================

class TestBmiBladeSimple:
    """Test01: non-uniform blade, no tip mass, cantilevered."""

    @pytest.fixture(autouse=True)
    def load(self, blade_bmi):
        self.bmi = read_bmi(blade_bmi)

    def test_beam_type(self):
        assert self.bmi.beam_type == 1

    def test_general_params(self):
        assert self.bmi.rot_rpm == pytest.approx(60.0)
        assert self.bmi.rpm_mult == pytest.approx(1.0)
        assert self.bmi.radius == pytest.approx(35.0)
        assert self.bmi.hub_rad == pytest.approx(1.75)
        assert self.bmi.precone == pytest.approx(0.0)
        assert self.bmi.hub_conn == 1
        assert self.bmi.n_modes_print == 20
        assert self.bmi.tab_delim is True
        assert self.bmi.mid_node_tw is False

    def test_tip_mass_zero(self):
        tm = self.bmi.tip_mass
        assert tm.mass == pytest.approx(0.0)
        assert tm.ixx == pytest.approx(0.0)

    def test_sec_props_file(self):
        assert 'blade_sec_props' in self.bmi.sec_props_file

    def test_scaling_all_unity(self):
        s = self.bmi.scaling
        for attr in ('sec_mass', 'flp_iner', 'lag_iner', 'flp_stff',
                     'edge_stff', 'tor_stff', 'axial_stff',
                     'cg_offst', 'sc_offst', 'tc_offst'):
            assert getattr(s, attr) == pytest.approx(1.0), f"scaling.{attr}"

    def test_n_elements(self):
        assert self.bmi.n_elements == 12

    def test_el_loc_length(self):
        assert len(self.bmi.el_loc) == 13  # nselt + 1

    def test_el_loc_bounds(self):
        assert self.bmi.el_loc[0] == pytest.approx(0.0)
        assert self.bmi.el_loc[-1] == pytest.approx(1.0)

    def test_el_loc_monotone(self):
        assert np.all(np.diff(self.bmi.el_loc) > 0)

    def test_tow_support_absent(self):
        assert self.bmi.tow_support == 0
        assert self.bmi.support is None

    def test_sec_props_path_resolvable(self):
        p = self.bmi.resolve_sec_props_path()
        assert p.exists(), f"Section props not found at {p}"


class TestBmiBladeTipMass:
    """Test02: blade with non-zero tip mass."""

    @pytest.fixture(autouse=True)
    def load(self, blade_tip_bmi):
        self.bmi = read_bmi(blade_tip_bmi)

    def test_tip_mass_value(self):
        assert self.bmi.tip_mass.mass == pytest.approx(40.0)

    def test_tip_mass_cm_offset(self):
        assert self.bmi.tip_mass.cm_offset == pytest.approx(0.05)

    def test_tip_mass_inertias(self):
        tm = self.bmi.tip_mass
        assert tm.ixx == pytest.approx(10.0)
        assert tm.iyy == pytest.approx(1.75)
        assert tm.izz == pytest.approx(2.5)
        assert tm.ixy == pytest.approx(0.0)


class TestBmiTower:
    """Test03: tower, tow_support = 0."""

    @pytest.fixture(autouse=True)
    def load(self, tower_bmi):
        self.bmi = read_bmi(tower_bmi)

    def test_beam_type(self):
        assert self.bmi.beam_type == 2

    def test_rot_rpm_zero(self):
        # rot_rpm is read from file; for towers it may be non-zero in the file
        # but Fortran zeros it internally.  Here we only test what the parser reads.
        assert self.bmi.rot_rpm == pytest.approx(0.0)

    def test_tow_support_none(self):
        assert self.bmi.tow_support == 0
        assert self.bmi.support is None


class TestBmiWireTower:
    """Test04: wire-supported tower, tow_support = 1."""

    @pytest.fixture(autouse=True)
    def load(self, wire_tower_bmi):
        self.bmi = read_bmi(wire_tower_bmi)

    def test_beam_type(self):
        assert self.bmi.beam_type == 2

    def test_tow_support_wires(self):
        assert self.bmi.tow_support == 1
        assert isinstance(self.bmi.support, TensionWireSupport)

    def test_n_attachments(self):
        assert self.bmi.support.n_attachments == 2

    def test_n_wires(self):
        assert self.bmi.support.n_wires == [3, 3]

    def test_node_attach(self):
        assert self.bmi.support.node_attach == [6, 10]

    def test_wire_stiffness(self):
        ws = self.bmi.support.wire_stiffness
        assert ws[0] == pytest.approx(9.0e9)
        assert ws[1] == pytest.approx(1.6e9)

    def test_th_wire(self):
        assert self.bmi.support.th_wire == pytest.approx([45.0, 30.0])


# ===========================================================================
# sec_props.py — section properties parser
# ===========================================================================

class TestSectionPropsBlade:
    """Blade section properties file (21 stations)."""

    @pytest.fixture(autouse=True)
    def load(self, blade_sec_props):
        self.sp = read_sec_props(blade_sec_props)

    def test_n_secs(self):
        assert self.sp.n_secs == 21

    def test_array_lengths(self):
        for attr in ('span_loc', 'str_tw', 'mass_den', 'flp_stff',
                     'edge_stff', 'tor_stff', 'axial_stff',
                     'cg_offst', 'sc_offst', 'tc_offst'):
            arr = getattr(self.sp, attr)
            assert len(arr) == 21, f"{attr} length"

    def test_span_loc_bounds(self):
        assert self.sp.span_loc[0] == pytest.approx(0.0)
        assert self.sp.span_loc[-1] == pytest.approx(1.0)

    def test_span_loc_monotone(self):
        assert np.all(np.diff(self.sp.span_loc) > 0)

    def test_root_mass_density(self):
        # from the file: 1447.607 kg/m at span_loc=0
        assert self.sp.mass_den[0] == pytest.approx(1447.607)

    def test_tip_mass_density(self):
        # from the file: 11.353 kg/m at span_loc=1
        assert self.sp.mass_den[-1] == pytest.approx(11.353)

    def test_root_flp_stff(self):
        # 7681.46E+06 N·m²
        assert self.sp.flp_stff[0] == pytest.approx(7681.46e6, rel=1e-4)

    def test_str_tw_root(self):
        assert self.sp.str_tw[0] == pytest.approx(11.1)

    def test_str_tw_tip(self):
        assert self.sp.str_tw[-1] == pytest.approx(0.0)


class TestSectionPropsTower:
    """Tower section properties file (10 stations)."""

    @pytest.fixture(autouse=True)
    def load(self, tower_sec_props):
        self.sp = read_sec_props(tower_sec_props)

    def test_n_secs(self):
        assert self.sp.n_secs == 10

    def test_span_loc_bounds(self):
        assert self.sp.span_loc[0] == pytest.approx(0.0)
        assert self.sp.span_loc[-1] == pytest.approx(1.0)

    def test_str_tw_all_zero(self):
        # Tower str_tw values are in the file but zeroed by BModes internally.
        # The parser reads the raw file values, which may be 0.0.
        assert np.all(self.sp.str_tw == pytest.approx(0.0))


# ===========================================================================
# out_parser.py — reference output parser
# ===========================================================================

class TestOutParserBlade:
    """Parse Test01 reference output (20 modes, rotating blade)."""

    @pytest.fixture(autouse=True)
    def load(self, blade_ref_out):
        self.out = read_out(blade_ref_out)

    def test_beam_type(self):
        assert self.out.beam_type == 'blade'

    def test_n_modes(self):
        assert len(self.out) == 20

    def test_mode1_freq(self):
        assert self.out[0].frequency == pytest.approx(1.7517, rel=1e-3)

    def test_mode2_freq(self):
        assert self.out[1].frequency == pytest.approx(1.9444, rel=1e-3)

    def test_mode3_freq(self):
        assert self.out[2].frequency == pytest.approx(4.1912, rel=1e-3)

    def test_mode1_n_stations(self):
        assert len(self.out[0].span_loc) == 13  # root + 12 elements

    def test_mode1_root_zero(self):
        m = self.out[0]
        assert m.flap_disp[0] == pytest.approx(0.0)
        assert m.lag_disp[0]  == pytest.approx(0.0)
        assert m.twist[0]     == pytest.approx(0.0)

    def test_mode1_tip_flap(self):
        # from the .out file: span_loc=1.0, flap_disp=0.201750
        assert self.out[0].flap_disp[-1] == pytest.approx(0.201750, rel=1e-4)

    def test_mode1_tip_lag(self):
        assert self.out[0].lag_disp[-1] == pytest.approx(0.053894, rel=1e-4)

    def test_frequencies_ascending(self):
        freqs = self.out.frequencies()
        assert np.all(np.diff(freqs) > 0)


class TestOutParserTower:
    """Parse Test03 reference output (20 modes, tower)."""

    @pytest.fixture(autouse=True)
    def load(self, tower_ref_out):
        self.out = read_out(tower_ref_out)

    def test_beam_type(self):
        assert self.out.beam_type == 'tower'

    def test_n_modes(self):
        assert len(self.out) == 20

    def test_mode1_freq(self):
        # from .out: freq = 0.42074E+00 Hz
        assert self.out[0].frequency == pytest.approx(0.42074, rel=1e-3)

    def test_mode1_fa_disp_tip(self):
        # span_loc=1.0, fa_disp=0.223645
        assert self.out[0].fa_disp[-1] == pytest.approx(0.223645, rel=1e-4)

    def test_mode1_ss_disp_zero(self):
        # first mode is pure FA; ss_disp should be all zeros
        assert np.all(np.abs(self.out[0].ss_disp) < 1e-8)
