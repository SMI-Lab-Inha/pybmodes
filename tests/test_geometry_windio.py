"""Geometry / WindIO tower input (1.4.0, issue #35).

`Tower.from_geometry` derives structural properties from a tube's
outer diameter + wall thickness + material; `Tower.from_windio` reads
the structural subset of a WindIO ontology .yaml and feeds it through.

Default-suite coverage (self-contained):
1. Closed-form: a uniform steel tube cantilever reproduces the
   Euler-Bernoulli analytic 1st bending frequency.
2. `tubular_section_props` unit values vs hand-computed tube formulae,
   and the outfitting_factor mass-only / stiffness-untouched split.
3. WindIO parser on a hand-written minimal yaml fixture: layer-sum,
   material lookup, span from reference axis, and the
   linear-vs-piecewise-constant thickness interpretation actually
   differing on a tapered profile (the open question from the #35
   thread with Hisham Tariq — quantified here).

Integration (upstream WindIO + ElastoDyn for the SAME turbine):
4. The IEA-15 base WindIO tower, run through the geometry pipeline,
   reproduces the IEA-15 *Monopile* ElastoDyn deck's tabulated
   distributed mass / EI to ~0 % (the ElastoDyn deck was itself
   generated from this WindIO geometry) — an RNA-independent, exact
   like-for-like anchor — plus a `from_windio` modal smoke.
"""

from __future__ import annotations

import math
import pathlib
import textwrap

import numpy as np
import pytest

from pybmodes.io.geometry import tubular_section_props
from pybmodes.models import Tower

# ---------------------------------------------------------------------------
# 1. Closed-form: uniform steel tube cantilever vs Euler-Bernoulli
# ---------------------------------------------------------------------------

def test_uniform_tube_cantilever_matches_euler_bernoulli() -> None:
    D, t, L = 6.0, 0.03, 80.0
    E, rho = 2.0e11, 7850.0
    ro, ri = D / 2, D / 2 - t
    i_area = 0.25 * math.pi * (ro**4 - ri**4)
    area = math.pi * (ro**2 - ri**2)
    f1 = (1.875104**2) / (2 * math.pi * L**2) * math.sqrt(
        E * i_area / (rho * area)
    )
    n = 41
    grid = np.linspace(0.0, 1.0, n)
    res = Tower.from_geometry(
        grid, np.full(n, D), np.full(n, t),
        flexible_length=L, E=E, rho=rho,
    ).run(n_modes=4, check_model=False)
    assert res.frequencies[0] == pytest.approx(f1, rel=1e-3), (
        res.frequencies[0], f1,
    )


# ---------------------------------------------------------------------------
# 2. tubular_section_props unit values + outfitting split
# ---------------------------------------------------------------------------

def test_tubular_section_props_closed_form_values() -> None:
    D, t, E, rho, nu = 8.0, 0.05, 2.0e11, 7800.0, 0.3
    sp = tubular_section_props(
        np.array([0.0, 1.0]), np.full(2, D), np.full(2, t),
        E=E, rho=rho, nu=nu,
    )
    ro, ri = D / 2, D / 2 - t
    area = math.pi * (ro**2 - ri**2)
    i_area = 0.25 * math.pi * (ro**4 - ri**4)
    G = E / (2 * (1 + nu))
    assert sp.mass_den[0] == pytest.approx(rho * area)
    assert sp.flp_stff[0] == pytest.approx(E * i_area)
    assert sp.edge_stff[0] == pytest.approx(E * i_area)        # FA == SS
    assert sp.tor_stff[0] == pytest.approx(G * 2 * i_area)     # J = 2I
    assert sp.axial_stff[0] == pytest.approx(E * area)
    assert sp.flp_iner[0] == pytest.approx(rho * i_area)


def test_outfitting_factor_scales_mass_not_stiffness() -> None:
    args = dict(E=2.0e11, rho=7800.0, nu=0.3)
    base = tubular_section_props(
        np.array([0.0, 1.0]), np.full(2, 8.0), np.full(2, 0.05), **args
    )
    fat = tubular_section_props(
        np.array([0.0, 1.0]), np.full(2, 8.0), np.full(2, 0.05),
        outfitting_factor=1.07, **args
    )
    assert fat.mass_den[0] == pytest.approx(1.07 * base.mass_den[0])
    assert fat.flp_iner[0] == base.flp_iner[0]                 # structural
    assert fat.flp_stff[0] == base.flp_stff[0]                 # untouched
    assert fat.axial_stff[0] == base.axial_stff[0]
    assert fat.tor_stff[0] == base.tor_stff[0]


def test_tubular_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        tubular_section_props(np.array([0.0, 1.0]), np.array([6.0, 6.0]),
                              np.array([0.0, 0.03]), E=2e11, rho=7800.0)
    with pytest.raises(ValueError, match="2.t < outer_diameter|outer radius"):
        tubular_section_props(np.array([0.0, 1.0]), np.array([6.0, 6.0]),
                              np.array([3.5, 3.5]), E=2e11, rho=7800.0)


# ---------------------------------------------------------------------------
# 3. WindIO parser on a minimal hand-written fixture
# ---------------------------------------------------------------------------

_MIN_WINDIO = textwrap.dedent("""\
    components:
      tower:
        outer_shape:
          outer_diameter:
            grid: [0.0, 0.5, 1.0]
            values: [8.0, 7.0, 6.0]
        structure:
          outfitting_factor: 1.1
          layers:
            - name: tower_wall
              material: steel
              thickness:
                grid: [0.0, 1.0]
                values: [0.05, 0.02]
        reference_axis:
          z:
            grid: [0.0, 1.0]
            values: [20.0, 120.0]
    materials:
      - name: steel
        E: 2.0e11
        rho: 7800.0
        nu: 0.3
        G: 7.7e10
    """)


def test_windio_parser_minimal(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_tubular

    p = tmp_path / "min.yaml"
    p.write_text(_MIN_WINDIO, encoding="utf-8")
    g = read_windio_tubular(p, component="tower")

    assert g.E == 2.0e11 and g.rho == 7800.0 and g.nu == 0.3
    assert g.outfitting_factor == 1.1
    assert g.flexible_length == pytest.approx(100.0)        # |120 - 20|
    np.testing.assert_allclose(g.station_grid, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(g.outer_diameter, [8.0, 7.0, 6.0])
    # linear thickness interp onto the [0,0.5,1] grid: 0.05,0.035,0.02
    np.testing.assert_allclose(g.wall_thickness, [0.05, 0.035, 0.02])


def test_windio_thickness_interp_differs_on_taper(tmp_path: pathlib.Path) -> None:
    """The linear-vs-piecewise-constant choice (the open question
    raised with Hisham Tariq in #35) must measurably move the wall
    thickness — hence the 2nd-mode coefficients — on a *smoothly
    tapered* profile (it is ~0 on IEA-15's step profile, which is why
    this uses a taper)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_tubular

    p = tmp_path / "taper.yaml"
    p.write_text(_MIN_WINDIO, encoding="utf-8")
    lin = read_windio_tubular(p, component="tower", thickness_interp="linear")
    pc = read_windio_tubular(
        p, component="tower", thickness_interp="piecewise_constant"
    )
    # mid station: linear -> 0.035, piecewise-constant -> 0.05 (lower
    # grid point governs). Materially different.
    assert lin.wall_thickness[1] == pytest.approx(0.035)
    assert pc.wall_thickness[1] == pytest.approx(0.05)
    assert abs(lin.wall_thickness[1] - pc.wall_thickness[1]) > 0.01


def test_from_windio_friendly_error_without_pyyaml(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent PyYAML, the error names the [windio] extra (mirrors the
    matplotlib-gated plots)."""
    import builtins

    real_import = builtins.__import__

    def _no_yaml(name, *a, **k):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_yaml)
    from pybmodes.io.windio import _require_yaml

    with pytest.raises(ModuleNotFoundError, match=r"pybmodes\[windio\]"):
        _require_yaml()


# ---------------------------------------------------------------------------
# 3b. WindIO schema-dialect robustness (synthetic; no external data)
#
# Upstream RWT ontology files come in two key dialects and one strict-
# PyYAML-hostile anchor habit; gate all three in the DEFAULT suite via
# synthetic fixtures so a fresh clone proves the parser without needing
# the gitignored docs/ corpus (independence stance). The integration
# block below then exercises the *real* IEA-3.4/10/15/22 files.
# ---------------------------------------------------------------------------

# Older dialect: outer_shape_bem / internal_structure_2d_fem, with
# reference_axis nested in the shape block and aliased into the
# structure block (exactly IEA-3.4/10/22's layout). Numerically
# identical to _MIN_WINDIO so the two must parse to the same struct.
_OLD_WINDIO = textwrap.dedent("""\
    components:
      tower:
        outer_shape_bem:
          reference_axis: &ref_axis_tower
            x: {grid: [0.0, 1.0], values: [0.0, 0.0]}
            y: {grid: [0.0, 1.0], values: [0.0, 0.0]}
            z:
              grid: [0.0, 1.0]
              values: [20.0, 120.0]
          outer_diameter:
            grid: [0.0, 0.5, 1.0]
            values: [8.0, 7.0, 6.0]
        internal_structure_2d_fem:
          outfitting_factor: 1.1
          reference_axis: *ref_axis_tower
          layers:
            - name: tower_wall
              material: steel
              thickness:
                grid: [0.0, 1.0]
                values: [0.05, 0.02]
    materials:
      - name: steel
        E: 2.0e11
        rho: 7800.0
        nu: 0.3
        G: 7.7e10
    """)


def test_windio_older_dialect_matches_modern(tmp_path: pathlib.Path) -> None:
    """The older `outer_shape_bem` / `internal_structure_2d_fem` dialect
    (IEA-3.4/10/22) parses to the *same* WindIOTubular as the modern
    `outer_shape` / `structure` form for numerically identical input."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_tubular

    pm = tmp_path / "modern.yaml"
    po = tmp_path / "older.yaml"
    pm.write_text(_MIN_WINDIO, encoding="utf-8")
    po.write_text(_OLD_WINDIO, encoding="utf-8")
    m = read_windio_tubular(pm, component="tower")
    o = read_windio_tubular(po, component="tower")

    assert (o.E, o.rho, o.nu, o.outfitting_factor) == (m.E, m.rho, m.nu,
                                                       m.outfitting_factor)
    assert o.flexible_length == pytest.approx(m.flexible_length)
    np.testing.assert_allclose(o.station_grid, m.station_grid)
    np.testing.assert_allclose(o.outer_diameter, m.outer_diameter)
    np.testing.assert_allclose(o.wall_thickness, m.wall_thickness)


# Duplicate anchor `&g` (no alias) — strict PyYAML raises ComposerError;
# ruamel / YAML-1.2 (and WISDEM's IEA-10 file) accept it. The shape-block
# reference_axis is resolved before the structure-block one, so the span
# is deterministically the shape value (|50 - 0| = 50).
_DUP_ANCHOR_WINDIO = textwrap.dedent("""\
    components:
      tower:
        outer_shape_bem:
          reference_axis:
            z: &g
              grid: [0.0, 1.0]
              values: [0.0, 50.0]
          outer_diameter: {grid: [0.0, 1.0], values: [6.0, 6.0]}
        internal_structure_2d_fem:
          outfitting_factor: 1.0
          reference_axis:
            z: &g
              grid: [0.0, 1.0]
              values: [10.0, 110.0]
          layers:
            - name: w
              material: steel
              thickness: {grid: [0.0, 1.0], values: [0.03, 0.03]}
    materials:
      - {name: steel, E: 2.0e11, rho: 7800.0, nu: 0.3}
    """)


def test_windio_tolerates_duplicate_anchors(tmp_path: pathlib.Path) -> None:
    """Strict PyYAML rejects a redefined anchor with ComposerError;
    WindIO files from the WISDEM toolchain (IEA-10 reuses `&id004`)
    routinely do this. The duplicate-anchor-tolerant loader must accept
    it (last-wins) and resolve the shape-block reference axis."""
    yaml = pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_tubular

    p = tmp_path / "dup.yaml"
    p.write_text(_DUP_ANCHOR_WINDIO, encoding="utf-8")

    # Sanity: the stock SafeLoader genuinely chokes on this fixture.
    with pytest.raises(yaml.composer.ComposerError):
        yaml.safe_load(_DUP_ANCHOR_WINDIO)

    g = read_windio_tubular(p, component="tower")
    assert g.flexible_length == pytest.approx(50.0)   # shape block wins
    assert g.E == 2.0e11 and g.rho == 7800.0


_ORTHO_WINDIO = textwrap.dedent("""\
    components:
      tower:
        outer_shape:
          outer_diameter: {grid: [0.0, 1.0], values: [6.0, 6.0]}
        structure:
          layers:
            - name: w
              material: triax
              thickness: {grid: [0.0, 1.0], values: [0.03, 0.03]}
        reference_axis:
          z: {grid: [0.0, 1.0], values: [0.0, 100.0]}
    materials:
      - name: triax
        E: [2.0e10, 1.4e10, 1.4e10]
        G: [9.4e9, 4.5e9, 4.5e9]
        rho: 1845.0
        nu: [0.48, 0.48, 0.48]
    """)


def test_windio_rejects_orthotropic_wall_material(
    tmp_path: pathlib.Path,
) -> None:
    """A list-valued (orthotropic composite) wall material — the
    triax/biax entries that sit beside `steel` in every RWT
    materials[] list — must raise a clear out-of-scope error, not a
    bare `float(list)` TypeError."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_tubular

    p = tmp_path / "ortho.yaml"
    p.write_text(_ORTHO_WINDIO, encoding="utf-8")
    with pytest.raises(ValueError, match="orthotropic"):
        read_windio_tubular(p, component="tower")


def test_windio_missing_shape_and_structure_raises(
    tmp_path: pathlib.Path,
) -> None:
    """A component with neither dialect's blocks names both spellings."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_tubular

    p = tmp_path / "bad.yaml"
    p.write_text("components:\n  tower:\n    foo: 1\n", encoding="utf-8")
    with pytest.raises(KeyError, match="outer_shape_bem"):
        read_windio_tubular(p, component="tower")


# ---------------------------------------------------------------------------
# 4. Integration — real upstream WindIO corpus (IEA-3.4 / 10 / 15 / 22)
# ---------------------------------------------------------------------------

_DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "OpenFAST_files"
_IEA15_YAML = _DOCS / "IEA-15-240-RWT/WT_Ontology/IEA-15-240-RWT.yaml"
_IEA15_MONO_ED = (
    _DOCS / "IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-Monopile/"
    "IEA-15-240-RWT-Monopile_ElastoDyn.dat"
)


@pytest.mark.integration
@pytest.mark.skipif(
    not (_IEA15_YAML.is_file() and _IEA15_MONO_ED.is_file()),
    reason="IEA-15 WindIO yaml / Monopile ElastoDyn deck not present",
)
def test_windio_iea15_matches_elastodyn_section_props() -> None:
    """RNA-independent like-for-like anchor: the IEA-15 base WindIO
    tower, through the geometry pipeline, must reproduce the IEA-15
    Monopile ElastoDyn deck's tabulated distributed mass / EI for the
    same physical tower (that deck was generated from this geometry)."""
    pytest.importorskip("yaml")
    from pybmodes.io._elastodyn.parser import (
        read_elastodyn_main,
        read_elastodyn_tower,
    )
    from pybmodes.io.windio import read_windio_tubular

    g = read_windio_tubular(_IEA15_YAML, component="tower")
    sp = tubular_section_props(
        g.station_grid, g.outer_diameter, g.wall_thickness,
        E=g.E, rho=g.rho, nu=g.nu, outfitting_factor=g.outfitting_factor,
    )
    m = read_elastodyn_main(_IEA15_MONO_ED)
    ted = read_elastodyn_tower(_IEA15_MONO_ED.parent / m.twr_file)
    hf = np.asarray(ted.ht_fract, float)
    ed_mass = np.asarray(ted.t_mass_den, float)
    ed_ei = np.asarray(ted.tw_fa_stif, float)
    w_mass = np.interp(hf, sp.span_loc, sp.mass_den)
    w_ei = np.interp(hf, sp.span_loc, sp.flp_stff)

    assert np.max(np.abs(w_mass - ed_mass) / np.maximum(ed_mass, 1.0)) < 5e-3
    assert np.max(np.abs(w_ei - ed_ei) / np.maximum(ed_ei, 1.0)) < 5e-3


@pytest.mark.integration
@pytest.mark.skipif(
    not _IEA15_YAML.is_file(),
    reason="IEA-15 WindIO yaml not present",
)
def test_from_windio_modal_smoke() -> None:
    """`Tower.from_windio` solves to a physical bare-tower spectrum
    (positive, finite, ascending) for tower and monopile."""
    pytest.importorskip("yaml")
    for comp in ("tower", "monopile"):
        f = Tower.from_windio(_IEA15_YAML, component=comp).run(
            n_modes=4, check_model=False
        ).frequencies
        assert np.all(np.isfinite(f)) and np.all(f > 0.0)
        assert np.all(np.diff(f) >= -1e-9)


# Full upstream corpus: every RWT ontology .yaml we ship-test against,
# spanning both key dialects and IEA-10's duplicate-anchor habit.
# (id, yaml relative to _DOCS, component). The id encodes the dialect so
# a failure pinpoints which parser path broke.
_WINDIO_CORPUS = [
    ("iea3.4-older-tower",
     "IEA-3.4-130-RWT/yaml/IEA-3.4-130-RWT.yaml", "tower"),
    ("iea10-older-dupanchor-tower",
     "IEA-10.0-198-RWT/yaml/IEA-10-198-RWT.yaml", "tower"),
    ("iea15-modern-tower",
     "IEA-15-240-RWT/WT_Ontology/IEA-15-240-RWT.yaml", "tower"),
    ("iea15-modern-monopile",
     "IEA-15-240-RWT/WT_Ontology/IEA-15-240-RWT.yaml", "monopile"),
    ("iea22-older-tower",
     "IEA-22-280-RWT/windIO/IEA-22-280-RWT.yaml", "tower"),
    ("iea22-older-monopile",
     "IEA-22-280-RWT/windIO/IEA-22-280-RWT.yaml", "monopile"),
    ("wisdem-nrel5mw-tower",
     "WISDEM/examples/05_tower_monopile/nrel5mw_tower.yaml", "tower"),
    ("wisdem-nrel5mw-monopile",
     "WISDEM/examples/05_tower_monopile/nrel5mw_monopile.yaml", "monopile"),
    ("wisdem-iea3p4-tower",
     "WISDEM/examples/02_reference_turbines/IEA-3p4-130-RWT.yaml", "tower"),
    ("wisdem-iea10-tower",
     "WISDEM/examples/02_reference_turbines/IEA-10-198-RWT.yaml", "tower"),
    ("wisdem-iea15-tower",
     "WISDEM/examples/02_reference_turbines/IEA-15-240-RWT.yaml", "tower"),
    ("wisdem-iea22-tower",
     "WISDEM/examples/02_reference_turbines/IEA-22-280-RWT.yaml", "tower"),
]


@pytest.mark.integration
@pytest.mark.parametrize(
    "rel,component",
    [(rel, comp) for _id, rel, comp in _WINDIO_CORPUS],
    ids=[i for i, _r, _c in _WINDIO_CORPUS],
)
def test_windio_corpus_parses_to_sane_tubular(rel: str, component: str) -> None:
    """Every shipped RWT ontology .yaml — both key dialects plus IEA-10's
    redefined-anchor file — parses to a physically sane steel tube:
    monotone-ish grid in [0,1], positive D / t with 2t < D at every
    station, a plausible steel modulus / density, and a positive span."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_tubular

    yaml_path = _DOCS / rel
    if not yaml_path.is_file():
        pytest.skip(f"{rel} not present")
    g = read_windio_tubular(yaml_path, component=component)

    assert g.station_grid[0] == pytest.approx(0.0)
    assert g.station_grid[-1] == pytest.approx(1.0)
    assert np.all(np.diff(g.station_grid) >= -1e-12)
    assert np.all(g.outer_diameter > 0.0)
    assert np.all(g.wall_thickness > 0.0)
    assert np.all(2.0 * g.wall_thickness < g.outer_diameter)
    assert 1.5e11 <= g.E <= 2.2e11          # structural steel, Pa
    assert 7000.0 <= g.rho <= 9000.0        # steel (+ outfit baked in some)
    assert 0.0 < g.nu < 0.5
    assert 0.9 <= g.outfitting_factor <= 1.2
    assert g.flexible_length > 1.0
    # The whole point: geometry -> exact section props, no exception.
    sp = tubular_section_props(
        g.station_grid, g.outer_diameter, g.wall_thickness,
        E=g.E, rho=g.rho, nu=g.nu, outfitting_factor=g.outfitting_factor,
    )
    assert np.all(sp.mass_den > 0.0)
    assert np.all(sp.flp_stff > 0.0)


@pytest.mark.integration
@pytest.mark.parametrize(
    "rel,component",
    [
        ("IEA-3.4-130-RWT/yaml/IEA-3.4-130-RWT.yaml", "tower"),       # older
        ("IEA-10.0-198-RWT/yaml/IEA-10-198-RWT.yaml", "tower"),       # dup-anc
        ("IEA-22-280-RWT/windIO/IEA-22-280-RWT.yaml", "tower"),       # older
        ("IEA-22-280-RWT/windIO/IEA-22-280-RWT.yaml", "monopile"),
    ],
    ids=["iea3.4", "iea10", "iea22-tower", "iea22-monopile"],
)
def test_from_windio_corpus_modal_smoke(rel: str, component: str) -> None:
    """`Tower.from_windio` drives the full FEM pipeline to a physical
    bare-member spectrum on the *older* dialect (and IEA-10's
    duplicate-anchor file) — not just IEA-15's modern form."""
    pytest.importorskip("yaml")
    yaml_path = _DOCS / rel
    if not yaml_path.is_file():
        pytest.skip(f"{rel} not present")
    f = Tower.from_windio(yaml_path, component=component).run(
        n_modes=4, check_model=False
    ).frequencies
    assert np.all(np.isfinite(f)) and np.all(f > 0.0)
    assert np.all(np.diff(f) >= -1e-9)


# IEA-3.4 / IEA-10 / IEA-22 ship an ElastoDyn `_tower.dat` for the *same*
# turbine, but — unlike IEA-15's Monopile deck — those tables were NOT
# regenerated by a 1:1 geometry round-trip from the ontology yaml, so
# the match is same-turbine *ballpark* (a few % to ~20 %), not the
# machine-exact anchor IEA-15 gives. Asserting a generous envelope still
# catches gross unit / layer-sum / dialect regressions against a wholly
# independent reference. The exact anchor stays IEA-15-only (above).
_BALLPARK = [
    ("IEA-3.4-130-RWT/yaml/IEA-3.4-130-RWT.yaml",
     "IEA-3.4-130-RWT/openfast/IEA-3.4-130-RWT_ElastoDyn_tower.dat"),
    ("IEA-10.0-198-RWT/yaml/IEA-10-198-RWT.yaml",
     "IEA-10.0-198-RWT/openfast/IEA-10.0-198-RWT_ElastoDyn_tower.dat"),
    ("IEA-22-280-RWT/windIO/IEA-22-280-RWT.yaml",
     "IEA-22-280-RWT/OpenFAST/IEA-22-280-RWT-Monopile/"
     "IEA-22-280-RWT_ElastoDyn_tower.dat"),
]


@pytest.mark.integration
@pytest.mark.parametrize(
    "yrel,erel", _BALLPARK, ids=["iea3.4", "iea10", "iea22"]
)
def test_windio_older_dialect_same_turbine_ballpark(
    yrel: str, erel: str
) -> None:
    """Older-dialect yaml-derived distributed mass / EI lands within a
    same-turbine envelope of that turbine's own ElastoDyn tower table —
    an independent cross-check that the older parser path produces
    physically right-sized properties (NOT the exact IEA-15 anchor)."""
    pytest.importorskip("yaml")
    from pybmodes.io._elastodyn.parser import read_elastodyn_tower
    from pybmodes.io.windio import read_windio_tubular

    yaml_path, ed_path = _DOCS / yrel, _DOCS / erel
    if not (yaml_path.is_file() and ed_path.is_file()):
        pytest.skip("yaml / ElastoDyn tower deck not present")

    g = read_windio_tubular(yaml_path, component="tower")
    sp = tubular_section_props(
        g.station_grid, g.outer_diameter, g.wall_thickness,
        E=g.E, rho=g.rho, nu=g.nu, outfitting_factor=g.outfitting_factor,
    )
    ted = read_elastodyn_tower(ed_path)
    hf = np.asarray(ted.ht_fract, float)
    w_mass = np.interp(hf, sp.span_loc, sp.mass_den)
    w_ei = np.interp(hf, sp.span_loc, sp.flp_stff)
    e_m = np.max(np.abs(w_mass - ted.t_mass_den) / ted.t_mass_den)
    e_e = np.max(np.abs(w_ei - ted.tw_fa_stif) / ted.tw_fa_stif)
    assert e_m < 0.25, f"mass off by {e_m:.1%}"
    assert e_e < 0.30, f"EI off by {e_e:.1%}"
