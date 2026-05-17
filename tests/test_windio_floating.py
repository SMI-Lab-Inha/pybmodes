"""WindIO floating-platform reader + hydro/mooring assembly
(1.4.0, issue #35, Phase 3).

Validation ladder (default rungs self-contained; integration rungs
anchor on the IEA-15 UMaine VolturnUS-S yaml vs its companion
WAMIT / MoorDyn / ElastoDyn decks):

* P3-1 — floating geometry: joint parsing (cartesian + cylindrical),
  member geometry, axial-joint resolution, transition joint.
  **(this file, below)**
* P3-2 — hydrostatic restoring C_hst vs closed-form cylinder + WAMIT.
* P3-3 — Morison added mass + buoyancy + rigid-body inertia.
* P3-4 — mooring catenary stiffness vs companion MoorDyn.
* P3-5 — PlatformSupport assembly → floating rigid-body + tower modes.
"""

from __future__ import annotations

import pathlib
import textwrap

import numpy as np
import pytest

_DOCS = (pathlib.Path(__file__).resolve().parents[1]
         / "docs" / "OpenFAST_files")
_IEA15_FLOAT_Y = (_DOCS / "IEA-15-240-RWT/WT_Ontology/"
                  "IEA-15-240-RWT_VolturnUS-S.yaml")
_IEA15_HD = (_DOCS / "IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-UMaineSemi/"
             "IEA-15-240-RWT-UMaineSemi_HydroDyn.dat")

# ---------------------------------------------------------------------------
# P3-1. Floating-platform geometry reader (default; no external data)
# ---------------------------------------------------------------------------

_MIN_FLOAT = textwrap.dedent("""\
    components:
      floating_platform:
        transition_piece_mass: 100000.0
        joints:
          - {name: keel, location: [0.0, 0.0, -20.0]}
          - {name: top, location: [0.0, 0.0, 10.0], transition: true}
          - {name: c1k, location: [50.0, 120.0, -20.0], cylindrical: true}
        members:
          - name: main
            joint1: keel
            joint2: top
            Ca: [1.0, 1.0]
            Cd: [0.8, 0.8]
            outer_shape:
              shape: circular
              outer_diameter: {grid: [0.0, 1.0], values: [10.0, 10.0]}
            axial_joints:
              - {name: fairlead, grid: 0.25}
            structure:
              layers:
                - name: wall
                  material: steel
                  thickness: {grid: [0.0, 1.0], values: [0.05, 0.05]}
              bulkhead:
                material: steel
                thickness: {grid: [0.0, 1.0], values: [0.04, 0.04]}
      mooring:
        nodes: []
        lines: []
    materials:
      - {name: steel, E: 2.0e11, rho: 7800.0, nu: 0.3}
    """)


def _read(tmp_path, text=_MIN_FLOAT):
    from pybmodes.io.windio_floating import read_windio_floating
    p = tmp_path / "float.yaml"
    p.write_text(text, encoding="utf-8")
    return read_windio_floating(p)


def test_floating_joints_cartesian_and_cylindrical(tmp_path) -> None:
    pytest.importorskip("yaml")
    f = _read(tmp_path)
    np.testing.assert_allclose(f.joints["keel"], [0.0, 0.0, -20.0])
    np.testing.assert_allclose(f.joints["top"], [0.0, 0.0, 10.0])
    # cylindrical [r=50, θ=120°, z=-20] → (50cos120, 50sin120, -20)
    np.testing.assert_allclose(
        f.joints["c1k"],
        [50.0 * np.cos(np.deg2rad(120.0)),
         50.0 * np.sin(np.deg2rad(120.0)), -20.0],
        atol=1e-9,
    )
    assert f.transition_joint == "top"
    assert f.transition_piece_mass == 100000.0
    assert f.materials["steel"]["rho"] == 7800.0


def test_floating_member_geometry(tmp_path) -> None:
    pytest.importorskip("yaml")
    f = _read(tmp_path)
    assert [m.name for m in f.members] == ["main"]
    m = f.members[0]
    assert m.length == pytest.approx(30.0)
    np.testing.assert_allclose(m.axis, [0.0, 0.0, 1.0])
    assert m.diameter_at(0.5) == pytest.approx(10.0)
    assert m.wall_t_at(0.7) == pytest.approx(0.05)
    assert m.ca == 1.0
    assert m.bulkhead_material == "steel"
    assert m.bulkhead_t == pytest.approx(0.04)
    np.testing.assert_allclose(m.point_at(0.0), [0.0, 0.0, -20.0])
    np.testing.assert_allclose(m.point_at(1.0), [0.0, 0.0, 10.0])


def test_floating_axial_joint_resolved_into_table(tmp_path) -> None:
    """A named axial joint (e.g. a fairlead) becomes a referenceable
    joint at its member fraction — mooring needs this."""
    pytest.importorskip("yaml")
    f = _read(tmp_path)
    assert "fairlead" in f.joints
    # keel + 0.25·(top−keel) = (0,0,−20 + 0.25·30) = (0,0,−12.5)
    np.testing.assert_allclose(f.joints["fairlead"], [0.0, 0.0, -12.5])


def test_floating_missing_joint_reference_raises(tmp_path) -> None:
    pytest.importorskip("yaml")
    bad = _MIN_FLOAT.replace("joint1: keel", "joint1: ghost")
    with pytest.raises(KeyError, match="ghost|joints list"):
        _read(tmp_path, bad)


def test_floating_member_without_wall_raises(tmp_path) -> None:
    pytest.importorskip("yaml")
    bad = textwrap.dedent("""\
        components:
          floating_platform:
            joints:
              - {name: a, location: [0.0, 0.0, -10.0]}
              - {name: b, location: [0.0, 0.0, 5.0]}
            members:
              - name: m
                joint1: a
                joint2: b
                outer_shape:
                  shape: circular
                  outer_diameter: {grid: [0.0, 1.0], values: [6.0, 6.0]}
                structure:
                  layers: []
        materials:
          - {name: steel, E: 2.0e11, rho: 7800.0}
        """)
    with pytest.raises(ValueError, match="structure.layers"):
        _read(tmp_path, bad)


def test_floating_missing_component_raises(tmp_path) -> None:
    pytest.importorskip("yaml")
    with pytest.raises(KeyError, match="floating_platform"):
        _read(tmp_path, "components:\n  tower: {}\n")


# ---------------------------------------------------------------------------
# P3-2. Hydrostatic restoring vs closed form (default; no external data)
# ---------------------------------------------------------------------------


def _cyl_yaml(r: float, keel_z: float, top_z: float) -> str:
    return textwrap.dedent(f"""\
        components:
          floating_platform:
            joints:
              - {{name: k, location: [0.0, 0.0, {keel_z}]}}
              - {{name: t, location: [0.0, 0.0, {top_z}], transition: true}}
            members:
              - name: col
                joint1: k
                joint2: t
                Ca: [1.0, 1.0]
                outer_shape:
                  shape: circular
                  outer_diameter:
                    grid: [0.0, 1.0]
                    values: [{2 * r}, {2 * r}]
                structure:
                  layers:
                    - name: w
                      material: steel
                      thickness: {{grid: [0.0, 1.0], values: [0.05, 0.05]}}
          mooring: {{nodes: [], lines: []}}
        materials:
          - {{name: steel, E: 2.0e11, rho: 7800.0}}
        """)


def test_hydrostatic_single_cylinder_closed_form(tmp_path) -> None:
    """A vertical surface-piercing cylinder reduces to the exact
    waterplane + buoyancy restoring (WAMIT/.hst convention, no
    gravity term): C₃₃ = ρg·πr², C₄₄ = C₅₅ = ρg(πr⁴/4 − πr²·d²/2);
    a single deep cylinder is hydrostatically unstable (C₄₄ < 0)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_floating import (
        RHO_SW,
        G,
        hydrostatic_restoring,
        read_windio_floating,
    )

    r, d, h = 5.0, 20.0, 10.0          # radius, draft, freeboard
    p = tmp_path / "cyl.yaml"
    p.write_text(_cyl_yaml(r, -d, h), encoding="utf-8")
    C = hydrostatic_restoring(read_windio_floating(p), n_seg=400)

    rg = RHO_SW * G
    S = np.pi * r**2
    Iwp = np.pi * r**4 / 4.0
    vol = np.pi * r**2 * d
    zb = -d / 2.0
    assert C[2, 2] == pytest.approx(rg * S, rel=1e-4)
    assert C[3, 3] == pytest.approx(rg * Iwp + rg * vol * zb, rel=5e-3)
    assert C[4, 4] == pytest.approx(rg * Iwp + rg * vol * zb, rel=5e-3)
    assert C[3, 3] < 0.0                       # unstable bare cylinder
    # Axisymmetric & on-axis: no surge/sway/yaw, no cross terms.
    for i, j in [(2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]:
        assert abs(C[i, j]) < 1e-6 * abs(C[2, 2])
    assert np.allclose(C, C.T)
    assert C[0, 0] == 0.0 and C[1, 1] == 0.0 and C[5, 5] == 0.0


def test_hydrostatic_offset_column_parallel_axis(tmp_path) -> None:
    """An off-axis column adds the parallel-axis ρg·A·x² to the
    pitch restoring (the semi-submersible stabilising mechanism)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_floating import (
        RHO_SW,
        G,
        hydrostatic_restoring,
        read_windio_floating,
    )

    r, d = 4.0, 15.0
    y_off = 30.0
    txt = textwrap.dedent(f"""\
        components:
          floating_platform:
            joints:
              - {{name: k, location: [0.0, {y_off}, {-d}]}}
              - {{name: t, location: [0.0, {y_off}, 8.0]}}
            members:
              - name: c
                joint1: k
                joint2: t
                outer_shape:
                  shape: circular
                  outer_diameter: {{grid: [0.0, 1.0], values: [{2*r}, {2*r}]}}
                structure:
                  layers:
                    - {{name: w, material: steel,
                        thickness: {{grid: [0.0, 1.0], values: [0.05, 0.05]}}}}
          mooring: {{nodes: [], lines: []}}
        materials:
          - {{name: steel, rho: 7800.0, E: 2.0e11}}
        """)
    p = tmp_path / "off.yaml"
    p.write_text(txt, encoding="utf-8")
    C = hydrostatic_restoring(read_windio_floating(p), n_seg=400)

    rg = RHO_SW * G
    S = np.pi * r**2
    # Roll restoring picks up ρg·A·y² (parallel axis) about global x.
    expect_C44 = rg * (np.pi * r**4 / 4.0 + S * y_off**2) \
        + rg * (S * d) * (-d / 2.0)
    assert C[3, 3] == pytest.approx(expect_C44, rel=1e-2)
    assert C[3, 3] > 0.0           # offset column → roll-stable


@pytest.mark.integration
@pytest.mark.skipif(
    not (_IEA15_FLOAT_Y.is_file() and _IEA15_HD.is_file()),
    reason="IEA-15 VolturnUS-S yaml / UMaineSemi HydroDyn deck absent",
)
def test_hydrostatic_iea15_volturnus_vs_wamit_hst() -> None:
    """Geometry-exact anchor: the IEA-15 UMaine VolturnUS-S WindIO
    floating geometry, through the member waterplane reduction,
    reproduces the companion **potential-flow WAMIT `.hst`**
    (HydroDynReader) heave / roll / pitch restoring. Hydrostatics are
    geometry-only, so panel-method and member-integration agree
    closely — measured heave 0.8 %, roll/pitch 1.6 %."""
    pytest.importorskip("yaml")
    from pybmodes.io.wamit_reader import HydroDynReader
    from pybmodes.io.windio_floating import (
        hydrostatic_restoring,
        read_windio_floating,
    )

    C = hydrostatic_restoring(read_windio_floating(_IEA15_FLOAT_Y))
    Cw = np.asarray(
        HydroDynReader(_IEA15_HD).read_platform_matrices().C_hst, float
    )

    def rel(i, j):
        return abs(C[i, j] - Cw[i, j]) / abs(Cw[i, j])

    assert rel(2, 2) < 0.03        # heave  (measured 0.8 %)
    assert rel(3, 3) < 0.04        # roll   (measured 1.6 %)
    assert rel(4, 4) < 0.04        # pitch  (measured 1.6 %)
    # Same physical sign structure as the panel method.
    assert C[2, 2] > 0.0 and C[3, 3] > 0.0 and C[4, 4] > 0.0
