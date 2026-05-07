"""Unit tests for the lower-level helpers inside :mod:`pybmodes.io`.

These complement ``test_io.py`` (which exercises full reference files) by
focusing on the parser primitives and edge cases:
  * ``_LineReader`` line iteration / blank skipping / EOF handling
  * ``_find_comment_start`` quote-aware comment stripping
  * ``_parse_bool`` / ``_parse_float`` / ``_parse_int`` / ``_parse_str``
  * ``_is_float`` token-classification helper
  * ``_parse_fortran_float`` from ``sec_props.py``
  * ``BMIFile.resolve_sec_props_path`` (relative/absolute, with/without source)
  * ``read_bmi`` / ``read_sec_props`` against malformed inputs
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pybmodes.io.bmi import (
    BMIFile,
    ScalingFactors,
    TipMassProps,
    _find_comment_start,
    _is_float,
    _LineReader,
    _parse_bool,
    _parse_float,
    _parse_int,
    _parse_str,
    read_bmi,
)
from pybmodes.io.sec_props import _parse_fortran_float, read_sec_props

# ===========================================================================
# _find_comment_start
# ===========================================================================

class TestFindCommentStart:
    """Quote-aware Fortran-style ``!`` comment detection."""

    def test_no_comment(self):
        assert _find_comment_start("just data here") == -1

    def test_simple_trailing_comment(self):
        line = "0.123  ! a comment"
        assert _find_comment_start(line) == line.index("!")

    def test_bang_inside_single_quotes(self):
        # ! inside '' must be ignored as a string literal
        line = "'has ! inside'   ! real comment"
        idx = _find_comment_start(line)
        assert idx == line.rindex("!")
        assert line[:idx].count("!") == 1  # the embedded ! is preserved

    def test_bang_inside_double_quotes(self):
        line = '"foo ! bar" ! real'
        idx = _find_comment_start(line)
        assert idx == line.rindex("!")

    def test_bang_at_position_zero(self):
        assert _find_comment_start("!whole line is comment") == 0

    def test_only_quotes_no_comment(self):
        assert _find_comment_start("'a' \"b\"") == -1

    def test_unterminated_quote_swallows_bang(self):
        # An unterminated quote leaves the ! "inside" the string
        assert _find_comment_start("'open string ! still in") == -1


# ===========================================================================
# Token parsers
# ===========================================================================

class TestParseBool:
    @pytest.mark.parametrize("token", ["t", "T", "true", "TRUE", "True", "'t'", '"t"'])
    def test_truthy(self, token):
        assert _parse_bool(token) is True

    @pytest.mark.parametrize("token", ["f", "F", "false", "FALSE", "False", "'f'"])
    def test_falsy(self, token):
        assert _parse_bool(token) is False

    @pytest.mark.parametrize("token", ["yes", "no", "1", "0", "", "tr"])
    def test_invalid_raises(self, token):
        with pytest.raises(ValueError, match="Cannot parse boolean"):
            _parse_bool(token)


class TestParseFloat:
    def test_plain_float(self):
        assert _parse_float("1.23") == pytest.approx(1.23)

    def test_negative(self):
        assert _parse_float("-2.5e3") == pytest.approx(-2500.0)

    def test_fortran_d_lower(self):
        # Fortran double-precision literal "1.5d3" -> 1500.0
        assert _parse_float("1.5d3") == pytest.approx(1500.0)

    def test_fortran_d_upper(self):
        assert _parse_float("2.0D-2") == pytest.approx(0.02)

    def test_quoted_value(self):
        assert _parse_float("'3.14'") == pytest.approx(3.14)
        assert _parse_float('"42.0"') == pytest.approx(42.0)

    def test_whitespace_padding(self):
        assert _parse_float("   7.5   ") == pytest.approx(7.5)


class TestParseInt:
    def test_plain(self):
        assert _parse_int("42") == 42

    def test_negative(self):
        assert _parse_int("-7") == -7

    def test_quoted(self):
        assert _parse_int("'12'") == 12


class TestParseStr:
    def test_strips_quotes(self):
        assert _parse_str("'hello'") == "hello"
        assert _parse_str('"world"') == "world"

    def test_strips_whitespace(self):
        assert _parse_str("  bare  ") == "bare"


class TestIsFloat:
    @pytest.mark.parametrize("token", ["1.0", "-2", "3e5", "4D-2", ""])
    def test_classification(self, token):
        # Empty string is not a valid float
        if token == "":
            assert _is_float(token) is False
        else:
            assert _is_float(token) is True

    @pytest.mark.parametrize("token", ["abc", "12abc", "tower_props.dat"])
    def test_non_float(self, token):
        assert _is_float(token) is False


# ===========================================================================
# _LineReader
# ===========================================================================

class TestLineReader:

    def test_strips_inline_comments(self):
        r = _LineReader(["1.0   ! this is ignored"])
        assert r.read_var() == "1.0"

    def test_skips_blank_lines(self):
        r = _LineReader(["", "   ", "\t", "42"])
        assert r.read_var() == "42"

    def test_eof_raises(self):
        r = _LineReader(["", "  "])
        with pytest.raises(EOFError, match="Unexpected end"):
            r.read_var()

    def test_read_str_returns_full_line(self):
        r = _LineReader(["  hello world  "])
        assert r.read_str() == "hello world"

    def test_read_com_does_not_skip_blanks(self):
        # read_com always advances exactly one line, even if it is blank
        r = _LineReader(["", "marker"])
        r.read_com()
        assert r.read_var() == "marker"

    def test_read_ary_returns_first_n(self):
        r = _LineReader(["1.0 2.0 3.0 4.0 5.0"])
        assert r.read_ary(3) == ["1.0", "2.0", "3.0"]

    def test_peek_does_not_advance(self):
        r = _LineReader(["", "  ", "first", "second"])
        assert r.peek_token() == "first"
        # subsequent peeks return the same value
        assert r.peek_token() == "first"
        assert r.read_var() == "first"
        assert r.peek_token() == "second"

    def test_peek_at_eof(self):
        r = _LineReader(["", "   "])
        # peek must not raise — only return ""
        assert r.peek_token() == ""

    def test_empty_data_line_raises(self):
        # _skip_blanks ignores fully blank lines, but read_var on a remaining
        # whitespace-only line is impossible — the only path to ValueError is
        # an explicitly empty (already-consumed) line.  Construct one.
        r = _LineReader(["   "])  # only whitespace
        with pytest.raises(EOFError):
            r.read_var()


# ===========================================================================
# _parse_fortran_float (sec_props)
# ===========================================================================

class TestParseFortranFloat:

    def test_plain_float(self):
        assert _parse_fortran_float("1.5") == pytest.approx(1.5)

    def test_d_notation(self):
        assert _parse_fortran_float("7.681460D+09") == pytest.approx(7.68146e9)

    def test_e_notation_passthrough(self):
        assert _parse_fortran_float("1.0e-3") == pytest.approx(0.001)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_fortran_float("not-a-number")


# ===========================================================================
# BMIFile.resolve_sec_props_path
# ===========================================================================

def _stub_bmi(sec_props_file: str, source: pathlib.Path | None) -> BMIFile:
    """Construct a minimal BMIFile to exercise path resolution."""
    return BMIFile(
        title="stub",
        echo=False,
        beam_type=1,
        rot_rpm=0.0,
        rpm_mult=1.0,
        radius=1.0,
        hub_rad=0.0,
        precone=0.0,
        bl_thp=0.0,
        hub_conn=1,
        n_modes_print=1,
        tab_delim=True,
        mid_node_tw=False,
        tip_mass=TipMassProps(0, 0, 0, 0, 0, 0, 0, 0, 0),
        id_mat=1,
        sec_props_file=sec_props_file,
        scaling=ScalingFactors(),
        n_elements=1,
        el_loc=np.array([0.0, 1.0]),
        source_file=source,
    )


class TestResolveSecPropsPath:

    def test_absolute_path_returned_as_is(self, tmp_path):
        absolute = tmp_path / "elsewhere" / "props.dat"
        bmi = _stub_bmi(str(absolute), source=tmp_path / "input.bmi")
        assert bmi.resolve_sec_props_path() == absolute

    def test_relative_to_source_file(self, tmp_path):
        source = tmp_path / "case" / "input.bmi"
        bmi = _stub_bmi("blade.dat", source=source)
        resolved = bmi.resolve_sec_props_path()
        assert resolved == (tmp_path / "case" / "blade.dat").resolve()

    def test_relative_without_source_resolves_to_cwd(self, tmp_path, monkeypatch):
        bmi = _stub_bmi("blade.dat", source=None)
        monkeypatch.chdir(tmp_path)
        resolved = bmi.resolve_sec_props_path()
        assert resolved == (tmp_path / "blade.dat").resolve()


# ===========================================================================
# read_bmi against the certtest fixtures (high-level integration)
# ===========================================================================

class TestReadBmiSourceFile:
    """``read_bmi`` should always set ``source_file`` to the input path."""

    def test_blade(self, blade_bmi):
        bmi = read_bmi(blade_bmi)
        assert bmi.source_file == blade_bmi
        assert bmi.resolve_sec_props_path().exists()

    def test_str_path_accepted(self, blade_bmi):
        # The function accepts str or Path.
        bmi = read_bmi(str(blade_bmi))
        assert bmi.source_file == pathlib.Path(str(blade_bmi))


# ===========================================================================
# read_sec_props edge cases
# ===========================================================================

class TestReadSecPropsErrors:

    def test_row_count_mismatch_raises(self, tmp_path):
        bad = tmp_path / "bad.dat"
        # Declare 3 rows but provide only 2.  Layout must match the parser:
        # title / "n_secs label desc" / blank / header / units / data...
        bad.write_text(
            "stub title\n"
            "3   n_secs   number of stations\n"
            "\n"
            "span_loc str_tw tw_iner mass_den flp_iner edge_iner "
            "flp_stff edge_stff tor_stff axial_stff cg_offst sc_offst tc_offst\n"
            "  -      deg    deg     kg/m     kg.m     kg.m     "
            "N.m^2   N.m^2     N.m^2   N        m        m        m\n"
            "0.0 0 0 1.0 0 0 1e6 1e6 1e5 1e7 0 0 0\n"
            "1.0 0 0 1.0 0 0 1e6 1e6 1e5 1e7 0 0 0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="expected 3 data rows"):
            read_sec_props(bad)

    def test_trailing_notes_are_ignored(self, tmp_path):
        good = tmp_path / "ok.dat"
        good.write_text(
            "stub title\n"
            "2   n_secs   number of stations\n"
            "\n"
            "span_loc str_tw tw_iner mass_den flp_iner edge_iner "
            "flp_stff edge_stff tor_stff axial_stff cg_offst sc_offst tc_offst\n"
            "  -      deg    deg     kg/m     kg.m     kg.m     "
            "N.m^2   N.m^2     N.m^2   N        m        m        m\n"
            "0.0 0 0 1.0 0 0 1e6 1e6 1e5 1e7 0 0 0\n"
            "1.0 0 0 1.0 0 0 1e6 1e6 1e5 1e7 0 0 0\n"
            "\n"
            "this is a footer note that should be ignored\n",
            encoding="utf-8",
        )
        sp = read_sec_props(good)
        assert sp.n_secs == 2
        assert sp.span_loc[-1] == pytest.approx(1.0)

    def test_source_file_is_set(self, blade_sec_props):
        sp = read_sec_props(blade_sec_props)
        assert sp.source_file == blade_sec_props
