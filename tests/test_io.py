"""Tests for the pybmodes input parsers using inline synthetic fixtures.

Every fixture used here is constructed in-test from numbers chosen by the
author of this file (uniform stiffness, simple geometry).  No bundled
reference data is used.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.io.bmi import (
    PlatformSupport,
    TensionWireSupport,
    read_bmi,
)
from pybmodes.io.sec_props import read_sec_props

from ._synthetic_bmi import write_bmi, write_uniform_sec_props

# ===========================================================================
# Top-level read_bmi: blade
# ===========================================================================

class TestBmiBlade:

    @pytest.fixture
    def bmi(self, tmp_path):
        bmi_path = write_bmi(
            tmp_path / "blade.bmi",
            beam_type=1, radius=50.0, hub_rad=0.0,
            sec_props_file="blade_secs.dat",
        )
        write_uniform_sec_props(tmp_path / "blade_secs.dat")
        return read_bmi(bmi_path)

    def test_beam_type(self, bmi):
        assert bmi.beam_type == 1

    def test_basic_fields(self, bmi):
        assert bmi.radius == pytest.approx(50.0)
        assert bmi.hub_rad == pytest.approx(0.0)
        assert bmi.rot_rpm == pytest.approx(0.0)
        assert bmi.hub_conn == 1
        assert bmi.tab_delim is False

    def test_zero_tip_mass(self, bmi):
        assert bmi.tip_mass.mass == pytest.approx(0.0)
        assert bmi.tip_mass.ixx == pytest.approx(0.0)

    def test_n_elements_and_el_loc(self, bmi):
        assert bmi.n_elements == 4
        assert len(bmi.el_loc) == 5
        assert bmi.el_loc[0] == pytest.approx(0.0)
        assert bmi.el_loc[-1] == pytest.approx(1.0)
        assert np.all(np.diff(bmi.el_loc) > 0)

    def test_scaling_factors_unity(self, bmi):
        for attr in ("sec_mass", "flp_iner", "lag_iner", "flp_stff",
                     "edge_stff", "tor_stff", "axial_stff",
                     "cg_offst", "sc_offst", "tc_offst"):
            assert getattr(bmi.scaling, attr) == pytest.approx(1.0)

    def test_no_tower_support(self, bmi):
        assert bmi.tow_support == 0
        assert bmi.support is None

    def test_bang_inside_quoted_filename_is_not_comment(self, tmp_path):
        """The BMI reader strips ``!`` comments, but not inside quotes."""
        bmi_path = write_bmi(
            tmp_path / "blade.bmi",
            beam_type=1,
            sec_props_file="blade!with-bang.dat",
        )
        write_uniform_sec_props(tmp_path / "blade!with-bang.dat")
        bmi = read_bmi(bmi_path)
        assert bmi.sec_props_file == "blade!with-bang.dat"
        assert bmi.resolve_sec_props_path() == (tmp_path / "blade!with-bang.dat").resolve()

    def test_fortran_d_exponents_in_bmi_values(self, tmp_path):
        """Fortran-style ``D`` exponents remain accepted in scalar fields."""
        bmi_path = write_bmi(
            tmp_path / "blade.bmi",
            beam_type=1,
            radius=50.0,
            rot_rpm=12.1,
            sec_props_file="blade_secs.dat",
        )
        write_uniform_sec_props(tmp_path / "blade_secs.dat")
        text = bmi_path.read_text(encoding="utf-8")
        text = text.replace("12.1    ! rot_rpm", "1.21D+1    ! rot_rpm")
        text = text.replace("50.0    ! radius", "5.0D+1    ! radius")
        bmi_path.write_text(text, encoding="utf-8")

        bmi = read_bmi(bmi_path)
        assert bmi.rot_rpm == pytest.approx(12.1)
        assert bmi.radius == pytest.approx(50.0)


# ===========================================================================
# Top-level read_bmi: tower (cantilevered, no support)
# ===========================================================================

class TestBmiTower:

    @pytest.fixture
    def bmi(self, tmp_path):
        bmi_path = write_bmi(
            tmp_path / "tower.bmi",
            beam_type=2, radius=80.0, hub_rad=0.0,
            sec_props_file="tower_secs.dat",
            tow_support=0,
        )
        write_uniform_sec_props(tmp_path / "tower_secs.dat",
                                 mass_den=5000.0, flp_stff=5.0e10,
                                 edge_stff=5.0e10)
        return read_bmi(bmi_path)

    def test_beam_type(self, bmi):
        assert bmi.beam_type == 2

    def test_no_support_default(self, bmi):
        assert bmi.tow_support == 0
        assert bmi.support is None


# ===========================================================================
# Top-level read_bmi: wire-supported tower
# ===========================================================================

class TestBmiWireTower:

    @pytest.fixture
    def bmi(self, tmp_path):
        bmi_path = write_bmi(
            tmp_path / "wire_tower.bmi",
            beam_type=2, radius=80.0, hub_rad=0.0,
            sec_props_file="tower_secs.dat",
            tow_support=1,
            wire_data=([3, 3], [2, 3], [5.0e8, 3.0e8], [45.0, 30.0]),
        )
        write_uniform_sec_props(tmp_path / "tower_secs.dat",
                                 mass_den=5000.0, flp_stff=5.0e10,
                                 edge_stff=5.0e10)
        return read_bmi(bmi_path)

    def test_tow_support_wires(self, bmi):
        assert bmi.tow_support == 1
        assert isinstance(bmi.support, TensionWireSupport)

    def test_attachment_data(self, bmi):
        sup = bmi.support
        assert sup.n_attachments == 2
        assert sup.n_wires == [3, 3]
        assert sup.node_attach == [2, 3]
        assert sup.wire_stiffness == pytest.approx([5.0e8, 3.0e8])
        assert sup.th_wire == pytest.approx([45.0, 30.0])


# ===========================================================================
# read_sec_props
# ===========================================================================

class TestReadSecProps:

    def test_round_trip(self, tmp_path):
        write_uniform_sec_props(tmp_path / "secs.dat", n_secs=11)
        sp = read_sec_props(tmp_path / "secs.dat")
        assert sp.n_secs == 11
        assert sp.span_loc[0] == pytest.approx(0.0)
        assert sp.span_loc[-1] == pytest.approx(1.0)
        assert np.all(np.diff(sp.span_loc) > 0)

    def test_uniform_values(self, tmp_path):
        write_uniform_sec_props(tmp_path / "secs.dat")
        sp = read_sec_props(tmp_path / "secs.dat")
        assert np.all(sp.mass_den == pytest.approx(100.0))
        assert np.all(sp.flp_stff == pytest.approx(1.0e8))
        assert np.all(sp.edge_stff == pytest.approx(1.0e9))

    def test_array_lengths(self, tmp_path):
        write_uniform_sec_props(tmp_path / "secs.dat", n_secs=7)
        sp = read_sec_props(tmp_path / "secs.dat")
        for attr in ("span_loc", "str_tw", "mass_den", "flp_stff",
                     "edge_stff", "tor_stff", "axial_stff"):
            assert len(getattr(sp, attr)) == 7

    def test_fortran_d_exponents_and_trailing_notes(self, tmp_path):
        path = tmp_path / "secs.dat"
        path.write_text(
            "section props with d exponents\n"
            "2 n_secs\n"
            "\n"
            "span_loc str_tw tw_iner mass_den flp_iner edge_iner "
            "flp_stff edge_stff tor_stff axial_stff cg_offst sc_offst tc_offst\n"
            "- - - - - - - - - - - - -\n"
            "0.0D+0 0 0 1.0D+2 1.0D+1 1.0D+1 1.0D+8 1.0D+9 1.0D+7 1.0D+10 0 0 0\n"
            "1.0D+0 0 0 2.0D+2 2.0D+1 2.0D+1 2.0D+8 2.0D+9 2.0D+7 2.0D+10 0 0 0\n"
            "this trailing note is ignored\n",
            encoding="utf-8",
        )
        sp = read_sec_props(path)
        assert sp.n_secs == 2
        np.testing.assert_allclose(sp.span_loc, [0.0, 1.0])
        np.testing.assert_allclose(sp.mass_den, [100.0, 200.0])
        np.testing.assert_allclose(sp.axial_stff, [1.0e10, 2.0e10])


# ===========================================================================
# resolve_sec_props_path
# ===========================================================================

class TestSecPropsResolution:

    def test_relative_path_resolves_against_bmi(self, tmp_path):
        bmi_path = write_bmi(
            tmp_path / "blade.bmi",
            sec_props_file="blade_secs.dat",
        )
        write_uniform_sec_props(tmp_path / "blade_secs.dat")
        bmi = read_bmi(bmi_path)
        resolved = bmi.resolve_sec_props_path()
        assert resolved.exists()
        assert resolved == (tmp_path / "blade_secs.dat").resolve()


def test_support_types_importable():
    """The two support dataclasses must remain in the public surface."""
    assert TensionWireSupport is not None
    assert PlatformSupport is not None


# ===========================================================================
# Friendly diagnostics for malformed section-properties files
# ===========================================================================

class TestSecPropsDiagnostics:

    def test_empty_file_raises_pathaware_valueerror(self, tmp_path):
        bad = tmp_path / "empty.dat"
        bad.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty or truncated"):
            read_sec_props(bad)

    def test_truncated_file_raises_pathaware_valueerror(self, tmp_path):
        bad = tmp_path / "truncated.dat"
        bad.write_text("only one line\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty or truncated"):
            read_sec_props(bad)

    def test_unparseable_n_secs_raises_pathaware_valueerror(self, tmp_path):
        bad = tmp_path / "no_n_secs.dat"
        bad.write_text(
            "title line\n"
            "not-an-integer  n_secs\n"
            "header\n"
            "units\n"
            "0.0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="cannot parse n_secs"):
            read_sec_props(bad)

    def test_early_trailing_note_counts_as_missing_data(self, tmp_path):
        bad = tmp_path / "too_few_rows.dat"
        bad.write_text(
            "title line\n"
            "2 n_secs\n"
            "header\n"
            "units\n"
            "0.0 0 0 1 1 1 1 1 1 1 0 0 0\n"
            "note before second row\n"
            "1.0 0 0 1 1 1 1 1 1 1 0 0 0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="expected 2 data rows, found 1"):
            read_sec_props(bad)
