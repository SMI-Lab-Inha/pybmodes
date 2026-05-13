"""Regression tests for the pass-4 static-review findings.

Seven classes of bug + one false-positive confirmation:

- **#2 (HIGH)** ``check_model`` silently passed models with NaN / Inf
  in section properties because every downstream comparison
  (``<=``, ``>``, ratio) returns False on NaN. New
  ``_check_section_properties_finite`` fires an ERROR before the
  per-field checks run.

- **#3 (MED)** ``bmi.py:_parse_float``, ``sec_props.py:_parse_fortran_float``,
  and the MoorDyn LINE TYPES / POINTS strict-parse paths all
  accepted ``nan`` / ``inf`` as valid floats. All three now reject
  non-finite results.

- **#4 (MED)** MoorDyn OPTIONS silently swallowed malformed
  ``WtrDpth`` / ``rhoW`` / ``g`` values via ``try / except: pass``,
  silently shifting wet weight (and hence mooring stiffness). Now
  routed through ``_parse_finite_option`` which raises.

- **#5 (MED)** ElastoDyn tower / blade distributed-property table
  parsers broke the loop on short rows or non-numeric tokens
  without checking the resulting row count against the file's
  declared ``NTwInpSt`` / ``NBlInpSt``. Silent truncation became a
  ``ValueError`` naming the gap.

- **#6 (MED/LOW)** ``fit_mode_shape`` accepted any input shape /
  length / finiteness; bad inputs produced IndexError, broadcasting
  errors, or NaN coefficients. Now validates 1-D shape, length ≥ 2
  (enough to define an interpolation; lstsq still tolerates 2- and
  3-station underdetermined fits via its min-norm fallback), finite
  values, and strictly-increasing span_loc up front.

- **#7 (LOW)** ``ElastoDynMain.compute_rot_mass`` integrated raw
  ``b_mass_den`` and ignored ``adj_bl_ms`` (the blade adapter
  ``to_pybmodes_blade`` already applies it). Method now multiplies
  by ``adj_bl_ms`` for consistency.

- **#8 (LOW)** ``_metadata_to_npz_value`` stored metadata as
  ``dtype=object`` (pickle-backed) even though the module docstring
  promised pickle-free loading. Switched to ``dtype=np.str_`` so
  the round-trip works under ``np.load(..., allow_pickle=False)``.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared synthetic-deck helpers (mirror the pass-2 fixtures)
# ---------------------------------------------------------------------------

_MOORDYN_HEADER = "----------- MoorDyn v2 Input File -------------------\n"


def _write_moordyn(path: pathlib.Path, *sections: str) -> pathlib.Path:
    path.write_text(_MOORDYN_HEADER + "".join(sections), encoding="utf-8")
    return path


_VALID_LINE_TYPES = """\
---------------------- LINE TYPES ----------------------------------------
Name    Diam    MassPerLength   EA      diff
(-)     (m)     (kg/m)          (N)     (-)
chain   0.10    50.0            1.0e9   0.0
"""
_VALID_POINTS = """\
---------------------- POINTS --------------------------------------------
ID      Attachment      X       Y       Z
(-)     (-)             (m)     (m)     (m)
1       Fixed           100.0   0.0     -50.0
2       Vessel          5.0     0.0     -10.0
"""
_VALID_LINES = """\
---------------------- LINES ---------------------------------------------
ID      LineType        AttachA AttachB UnstrLen        NumSegs Outputs
(-)     (-)             (-)     (-)     (m)             (-)     (-)
1       chain           1       2       102.0           20      -
"""


# ---------------------------------------------------------------------------
# #2 — check_model flags non-finite section properties
# ---------------------------------------------------------------------------

class TestCheckModelFiniteSectionProperties:
    """Non-finite (NaN / ±Inf) entries in any numeric section-property
    field produce an ERROR-severity ``ModelWarning`` *before* the per-
    field checks run. Pre-1.0 review pass 4 surfaced that NaN / Inf
    passed silently because every downstream comparison returned False
    on NaN.
    """

    def _build_tower_with_section_props(self, sp_overrides: dict):
        """Build an in-memory Tower with a synthetic SectionProperties.
        ``sp_overrides`` mutates specific fields after the clean build
        so each test can install exactly one offending value."""
        from pybmodes.io.bmi import BMIFile, ScalingFactors, TipMassProps
        from pybmodes.io.sec_props import SectionProperties
        from pybmodes.models import Tower

        n = 5
        span = np.linspace(0.0, 1.0, n)
        sp = SectionProperties(
            title="t", n_secs=n,
            span_loc=span, str_tw=np.zeros(n), tw_iner=np.zeros(n),
            mass_den=np.full(n, 100.0),
            flp_iner=np.full(n, 10.0), edge_iner=np.full(n, 10.0),
            flp_stff=np.full(n, 1.0e9), edge_stff=np.full(n, 1.0e9),
            tor_stff=np.full(n, 1.0e8), axial_stff=np.full(n, 1.0e10),
            cg_offst=np.zeros(n), sc_offst=np.zeros(n), tc_offst=np.zeros(n),
        )
        for fname, value in sp_overrides.items():
            setattr(sp, fname, value)
        bmi = BMIFile(
            title="t", echo=False, beam_type=2, rot_rpm=0.0, rpm_mult=1.0,
            radius=90.0, hub_rad=0.0, precone=0.0, bl_thp=0.0, hub_conn=1,
            n_modes_print=20, tab_delim=True, mid_node_tw=False,
            tip_mass=TipMassProps(
                mass=100_000.0, cm_offset=0.0, cm_axial=0.0,
                ixx=0.0, iyy=0.0, izz=0.0, ixy=0.0, izx=0.0, iyz=0.0,
            ),
            id_mat=1, sec_props_file="", scaling=ScalingFactors(),
            n_elements=10, el_loc=np.linspace(0.0, 1.0, 11),
            tow_support=0, support=None, source_file=None,
        )
        tower = Tower.__new__(Tower)
        tower._bmi = bmi
        tower._sp = sp
        return tower

    def test_nan_in_mass_den_raises_error(self) -> None:
        from pybmodes.checks import check_model

        m = np.full(5, 100.0)
        m[2] = np.nan
        tower = self._build_tower_with_section_props({"mass_den": m})
        out = check_model(tower)
        hits = [w for w in out if "section_properties.mass_den" in w.location]
        assert len(hits) == 1 and hits[0].severity == "ERROR"
        assert "non-finite" in hits[0].message.lower()

    def test_inf_in_flp_stff_raises_error(self) -> None:
        from pybmodes.checks import check_model

        ei = np.full(5, 1.0e9)
        ei[3] = np.inf
        tower = self._build_tower_with_section_props({"flp_stff": ei})
        out = check_model(tower)
        hits = [w for w in out if "section_properties.flp_stff" in w.location]
        assert len(hits) == 1 and hits[0].severity == "ERROR"

    def test_nan_in_span_loc_raises_error(self) -> None:
        from pybmodes.checks import check_model

        span = np.linspace(0.0, 1.0, 5)
        span[1] = np.nan
        tower = self._build_tower_with_section_props({"span_loc": span})
        out = check_model(tower)
        hits = [w for w in out if "section_properties.span_loc" in w.location]
        assert len(hits) >= 1
        assert any(w.severity == "ERROR" for w in hits)


# ---------------------------------------------------------------------------
# #3 — file parsers reject non-finite
# ---------------------------------------------------------------------------

class TestParsersRejectNonFinite:
    """``bmi._parse_float`` and ``sec_props._parse_fortran_float`` now
    raise on ``nan`` / ``inf``. The MoorDyn LINE TYPES / POINTS
    strict-parse paths fold the finite check into the existing
    "malformed row" raise."""

    def test_bmi_parse_float_rejects_nan(self) -> None:
        from pybmodes.io.bmi import _parse_float

        with pytest.raises(ValueError, match="Non-finite float"):
            _parse_float("nan")

    def test_bmi_parse_float_rejects_inf(self) -> None:
        from pybmodes.io.bmi import _parse_float

        with pytest.raises(ValueError, match="Non-finite float"):
            _parse_float("inf")

    def test_bmi_parse_float_accepts_normal_value(self) -> None:
        from pybmodes.io.bmi import _parse_float

        assert _parse_float("1.5e3") == 1500.0
        assert _parse_float("7.466D+06") == 7.466e6  # Fortran D-exponent

    def test_sec_props_parse_fortran_float_rejects_nan(self) -> None:
        from pybmodes.io.sec_props import _parse_fortran_float

        with pytest.raises(ValueError, match="Non-finite"):
            _parse_fortran_float("nan")

    def test_sec_props_row_with_nan_raises_with_path(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """A nan in any column of a section-properties row raises a
        ``ValueError`` naming the path, row, and column."""
        from pybmodes.io.sec_props import read_sec_props

        path = tmp_path / "bad_sec.dat"
        path.write_text(
            "synthetic\n"
            "5  n_secs\n"
            "\n"
            "header_row_intentionally_skipped\n"
            "units_row_intentionally_skipped\n"
            "0.00  0  0  100  10  10  1e9  1e9  1e8  1e10  0  0  0\n"
            "0.25  0  0  100  10  nan  1e9  1e9  1e8  1e10  0  0  0\n"  # bad row 2 col 6
            "0.50  0  0  100  10  10  1e9  1e9  1e8  1e10  0  0  0\n"
            "0.75  0  0  100  10  10  1e9  1e9  1e8  1e10  0  0  0\n"
            "1.00  0  0  100  10  10  1e9  1e9  1e8  1e10  0  0  0\n",
            encoding="latin-1",
        )
        with pytest.raises(
            ValueError, match="non-finite value.*row 2, column 6",
        ):
            read_sec_props(path)

    def test_moordyn_line_types_with_inf_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        bad_section = """\
---------------------- LINE TYPES ----------------------------------------
Name    Diam    MassPerLength   EA      diff
(-)     (m)     (kg/m)          (N)     (-)
chain   0.10    inf             1.0e9   0.0
"""
        path = _write_moordyn(
            tmp_path / "bad.dat", bad_section, _VALID_POINTS, _VALID_LINES,
        )
        from pybmodes.mooring import MooringSystem
        with pytest.raises(ValueError, match="Malformed LINE TYPES"):
            MooringSystem.from_moordyn(path)

    def test_moordyn_points_with_nan_coord_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        bad_section = """\
---------------------- POINTS --------------------------------------------
ID      Attachment      X       Y       Z
(-)     (-)             (m)     (m)     (m)
1       Fixed           nan     0.0     -50.0
2       Vessel          5.0     0.0     -10.0
"""
        path = _write_moordyn(
            tmp_path / "bad.dat", _VALID_LINE_TYPES, bad_section, _VALID_LINES,
        )
        from pybmodes.mooring import MooringSystem
        with pytest.raises(ValueError, match="finite X / Y / Z"):
            MooringSystem.from_moordyn(path)


# ---------------------------------------------------------------------------
# #4 — MoorDyn OPTIONS strict parse for recognized keys
# ---------------------------------------------------------------------------

class TestMoorDynOptionsStrictParse:
    """The three recognised OPTIONS keys (``WtrDpth`` / ``rhoW`` /
    ``g``) raise on a malformed value rather than silently falling
    back to constructor defaults. Unknown keys remain permissive."""

    def test_rhow_with_garbage_value_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        options = """\
---------------------- OPTIONS -------------------------------------------
not_a_number    rhoW
200.0           WtrDpth
"""
        path = _write_moordyn(
            tmp_path / "bad.dat",
            _VALID_LINE_TYPES, _VALID_POINTS, _VALID_LINES, options,
        )
        from pybmodes.mooring import MooringSystem
        with pytest.raises(ValueError, match="Malformed OPTIONS.*rhoW"):
            MooringSystem.from_moordyn(path)

    def test_wtrdpth_with_inf_raises(self, tmp_path: pathlib.Path) -> None:
        options = """\
---------------------- OPTIONS -------------------------------------------
inf             WtrDpth
"""
        path = _write_moordyn(
            tmp_path / "bad.dat",
            _VALID_LINE_TYPES, _VALID_POINTS, _VALID_LINES, options,
        )
        from pybmodes.mooring import MooringSystem
        with pytest.raises(ValueError, match="not finite"):
            MooringSystem.from_moordyn(path)

    def test_g_with_typo_raises(self, tmp_path: pathlib.Path) -> None:
        options = """\
---------------------- OPTIONS -------------------------------------------
9.8point1       g
"""
        path = _write_moordyn(
            tmp_path / "bad.dat",
            _VALID_LINE_TYPES, _VALID_POINTS, _VALID_LINES, options,
        )
        from pybmodes.mooring import MooringSystem
        with pytest.raises(ValueError, match="Malformed OPTIONS.*['\"]g['\"]"):
            MooringSystem.from_moordyn(path)

    def test_unknown_option_still_permissive(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """An unknown key with a malformed value doesn't raise — the
        OPTIONS block can legitimately carry MoorDyn-version-specific
        informational lines we don't recognise."""
        options = """\
---------------------- OPTIONS -------------------------------------------
not_a_number    SomeNewKeyword
200.0           WtrDpth
"""
        path = _write_moordyn(
            tmp_path / "ok.dat",
            _VALID_LINE_TYPES, _VALID_POINTS, _VALID_LINES, options,
        )
        from pybmodes.mooring import MooringSystem
        ms = MooringSystem.from_moordyn(path)
        assert ms.depth == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# #5 — ElastoDyn tower / blade table row-count validation
# ---------------------------------------------------------------------------

class TestElastoDynRowCountMismatch:
    """If the parsed-row count doesn't match ``NTwInpSt`` /
    ``NBlInpSt`` declared in the same file, the parser raises a
    ``ValueError`` rather than silently truncating the table."""

    def test_tower_short_table_raises(self, tmp_path: pathlib.Path) -> None:
        # NTwInpSt = 5 but only 3 data rows — truncated table.
        tower = tmp_path / "short.dat"
        tower.write_text(
            "------- ELASTODYN V1.00.* TOWER INPUT FILE -------------------------------\n"
            "Synthetic short-table tower for the pyBmodes pass-4 ratchet.\n"
            "---------------------- TOWER PARAMETERS ----------------------------------\n"
            "         5    NTwInpSt    - Number of input stations\n"
            "          1   TwrFADmp(1) - Tower 1st FA damping (%)\n"
            "          1   TwrFADmp(2) - Tower 2nd FA damping (%)\n"
            "          1   TwrSSDmp(1) - Tower 1st SS damping (%)\n"
            "          1   TwrSSDmp(2) - Tower 2nd SS damping (%)\n"
            "---------------------- TOWER ADJUSTMUNT FACTORS --------------------------\n"
            "          1   FAStTunr(1)\n"
            "          1   FAStTunr(2)\n"
            "          1   SSStTunr(1)\n"
            "          1   SSStTunr(2)\n"
            "          1   AdjTwMa\n"
            "          1   AdjFASt\n"
            "          1   AdjSSSt\n"
            "---------------------- DISTRIBUTED TOWER PROPERTIES ----------------------\n"
            "  HtFract       TMassDen         TwFAStif       TwSSStif\n"
            "   (-)           (kg/m)           (Nm^2)         (Nm^2)\n"
            "0.0      5000.0    5.0e10    5.0e10\n"
            "0.25     5000.0    5.0e10    5.0e10\n"
            "0.5      5000.0    5.0e10    5.0e10\n"  # only 3 rows of 5 declared
            "---------------------- TOWER FORE-AFT MODE SHAPES ------------------------\n"
            "1.0   TwFAM1Sh(2)\n"
            "0.0   TwFAM1Sh(3)\n"
            "0.0   TwFAM1Sh(4)\n"
            "0.0   TwFAM1Sh(5)\n"
            "0.0   TwFAM1Sh(6)\n",
            encoding="latin-1",
        )
        from pybmodes.io.elastodyn_reader import read_elastodyn_tower

        with pytest.raises(
            ValueError, match="tower distributed-properties.*NTwInpSt = 5",
        ):
            read_elastodyn_tower(tower)


# ---------------------------------------------------------------------------
# #6 — fit_mode_shape input validation
# ---------------------------------------------------------------------------

class TestFitModeShapeValidation:

    def test_empty_arrays_raise(self) -> None:
        from pybmodes.fitting.poly_fit import fit_mode_shape

        with pytest.raises(ValueError, match="at least 2 stations"):
            fit_mode_shape(np.array([]), np.array([]))

    def test_single_station_raises(self) -> None:
        from pybmodes.fitting.poly_fit import fit_mode_shape

        with pytest.raises(ValueError, match="at least 2 stations"):
            fit_mode_shape(np.array([1.0]), np.array([1.0]))

    def test_shape_mismatch_raises(self) -> None:
        from pybmodes.fitting.poly_fit import fit_mode_shape

        with pytest.raises(ValueError, match="must have the same length"):
            fit_mode_shape(np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0]))

    def test_two_d_input_raises(self) -> None:
        from pybmodes.fitting.poly_fit import fit_mode_shape

        with pytest.raises(ValueError, match="must be 1-D"):
            fit_mode_shape(np.array([[0.0, 0.5]]), np.array([[0.0, 1.0]]))

    def test_nan_displacement_raises(self) -> None:
        from pybmodes.fitting.poly_fit import fit_mode_shape

        with pytest.raises(ValueError, match="must be finite"):
            fit_mode_shape(
                np.linspace(0.0, 1.0, 6),
                np.array([0.0, 0.1, np.nan, 0.4, 0.7, 1.0]),
            )

    def test_non_monotonic_span_raises(self) -> None:
        from pybmodes.fitting.poly_fit import fit_mode_shape

        with pytest.raises(ValueError, match="strictly increasing"):
            fit_mode_shape(
                np.array([0.0, 0.3, 0.2, 0.6, 0.8, 1.0]),  # 0.2 < 0.3
                np.linspace(0.0, 1.0, 6),
            )

    def test_well_formed_input_still_fits(self) -> None:
        """Sanity: a well-formed input still produces a clean fit."""
        from pybmodes.fitting.poly_fit import fit_mode_shape

        x = np.linspace(0.0, 1.0, 11)
        y = x ** 2  # exact polynomial → fits to machine precision
        fit = fit_mode_shape(x, y)
        assert fit.rms_residual < 1e-10
        assert fit.c2 == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# #7 — compute_rot_mass applies adj_bl_ms
# ---------------------------------------------------------------------------

def test_compute_rot_mass_applies_adj_bl_ms() -> None:
    """``AdjBlMs`` should multiply through the per-blade mass
    integral. Pre-1.0 review pass 4 surfaced that the
    ``compute_rot_mass`` method ignored it, even though the blade
    adapter ``to_pybmodes_blade`` already applied the same scalar."""
    from pybmodes.io._elastodyn.types import ElastoDynBlade, ElastoDynMain

    main = ElastoDynMain(
        header="h", title="t",
        num_bl=3, tip_rad=10.0, hub_rad=0.0,
        hub_mass=1_000.0,
    )
    # Uniform blade mass density: total mass per blade = mass_den * length
    # = 100 * 10 = 1000 kg. With AdjBlMs = 2.0 it should double to 2000 kg.
    blade_base = ElastoDynBlade(
        header="h", title="t",
        bl_fract=np.linspace(0.0, 1.0, 11),
        b_mass_den=np.full(11, 100.0),
    )
    # Baseline: adj_bl_ms = 1.0 (default).
    rot_mass_1 = main.compute_rot_mass(blade_base)
    expected_1 = main.hub_mass + main.num_bl * (100.0 * 10.0) * 1.0
    assert rot_mass_1 == pytest.approx(expected_1)

    # AdjBlMs = 2.0 must scale the per-blade integral.
    blade_scaled = dataclasses.replace(blade_base, adj_bl_ms=2.0)
    rot_mass_2 = main.compute_rot_mass(blade_scaled)
    expected_2 = main.hub_mass + main.num_bl * (100.0 * 10.0) * 2.0
    assert rot_mass_2 == pytest.approx(expected_2)
    assert rot_mass_2 > rot_mass_1


# ---------------------------------------------------------------------------
# #8 — NPZ metadata round-trips without allow_pickle
# ---------------------------------------------------------------------------

def test_npz_metadata_loads_without_pickle() -> None:
    """The new ``dtype=np.str_`` form for ``__meta__`` loads cleanly
    under ``np.load(..., allow_pickle=False)`` — closes the
    docstring-vs-implementation drift the previous ``dtype=object``
    introduced."""
    from pybmodes.fem.normalize import NodeModeShape
    from pybmodes.models.result import ModalResult

    shapes = [NodeModeShape(
        mode_number=1, freq_hz=0.5,
        span_loc=np.linspace(0, 1, 5),
        flap_disp=np.linspace(0, 1, 5), flap_slope=np.zeros(5),
        lag_disp=np.zeros(5), lag_slope=np.zeros(5), twist=np.zeros(5),
    )]
    r = ModalResult(
        frequencies=np.array([0.5]),
        shapes=shapes,
    )
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "r.npz"
        r.save(p, source_file="dummy.bmi")
        # The load WITHOUT allow_pickle proves the metadata isn't
        # pickle-backed. Just opening the archive used to require
        # allow_pickle=True under the old dtype=object regime.
        with np.load(p, allow_pickle=False) as npz:
            meta_arr = npz["__meta__"]
            # ``kind == "U"`` is unicode fixed-length, NOT object.
            assert meta_arr.dtype.kind == "U", (
                f"__meta__ should be a unicode string array; got "
                f"dtype={meta_arr.dtype}"
            )
            meta = json.loads(str(meta_arr))
            assert "pybmodes_version" in meta
            assert "timestamp" in meta
            assert meta["source_file"] == "dummy.bmi"
