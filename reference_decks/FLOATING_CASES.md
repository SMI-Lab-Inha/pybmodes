<!-- markdownlint-disable MD013 -->
# Why floating cases aren't in `reference_decks/`

The three turbine configurations included in this directory
(NREL 5MW land, NREL 5MW on the rigid OC3 monopile, IEA-3.4-130-RWT
land) all use a **clamped tower base**: every reaction the tower
needs comes from the rigid foundation, so the structural problem is
fully described by the `.dat` files in the same case directory and
pyBmodes can regenerate the polynomial coefficients from those alone.

Floating-platform turbines (spar, semi-submersible, barge) do not have
this property, and the gap is not pyBmodes' to fix — it lives in the
ElastoDyn file format itself.

## What ElastoDyn `.dat` files do *not* contain

For a floating tower the modal problem cannot be solved without:

- **Hydrodynamic added-mass 6×6 matrix** at the platform reference
  point (the structurally-relevant inertia at hydrodynamic frequencies
  differs from the dry mass by tens of percent on a spar-buoy and a
  factor of ~2 on a barge in heave).
- **Hydrostatic restoring 6×6 matrix** including the negative
  pitch / roll restoring of a hydrostatically-unstable spar.
- **Mooring stiffness 6×6 matrix** linearised about the static
  equilibrium offset.
- **Platform inertia 6×6 matrix** referred to the platform centre of
  mass (translations and rotations coupled by c.m. offsets).

ElastoDyn `.dat` files carry **none** of these — they live in
`HydroDyn.dat`, `MoorDyn.dat`, and the platform-property block of
ElastoDyn (which only has the diagonal physical mass + inertia of the
platform body, not the hydrodynamic / mooring matrices). The OpenFAST
glue code assembles the full system at runtime by combining all four
modules; ElastoDyn alone cannot produce a floating tower's modal
frequencies, and therefore cannot produce a self-consistent
polynomial-coefficient block for one.

## What BModes-format `.bmi` files *do* contain

The 6×6 hydro / mooring / platform-inertia matrices are exactly what a
BModes `.bmi` deck carries in its tower-support block (`tow_support=1`
or `tow_support=2`). pyBmodes' `Tower.from_bmi(...)` constructor parses
these and feeds them into the FEM via the `PlatformSupport` matrix
assembly path (free-free root, `hub_conn=2`, full 6×6 transformations
across rigid-arm offsets). Two such decks are supported and
validated when available locally:

- `docs/BModes/docs/examples/CS_Monopile.bmi` — *NREL 5MW Reference
  Turbine* on the *OC3 Monopile* configuration (Jonkman & Musial 2010);
  soft monopile, mooring-equivalent foundation springs.
- `docs/BModes/docs/examples/OC3Hywind.bmi` — *NREL 5MW* on the
  *OC3 Hywind* floating spar (Jonkman 2010); full hydro + mooring +
  6×6 platform inertia.

Both decks are upstream NREL BModes example data — **not bundled in
this repository** (`docs/BModes/` is gitignored as local-only third-
party reference data; see CLAUDE.md *Independence stance*). When
present locally, they pass the cert-test suite at < 0.01 % per mode
against BModes JJ (see
[`tests/test_certtest.py`](../tests/test_certtest.py),
`test_certtest_cs_monopile` and `test_certtest_oc3hywind`); when
absent, those tests skip cleanly under the `integration` marker
without affecting the default-run pass status.

## Currently supported via `Tower.from_bmi()`

If you have a BModes `.bmi` file for your floating platform, pyBmodes
can solve and fit polynomials directly:

```python
from pybmodes.models import Tower
from pybmodes.elastodyn import compute_tower_params, patch_dat

tower = Tower.from_bmi("my_floating_tower.bmi")
result = tower.run(n_modes=10)

params = compute_tower_params(result)
patch_dat("my_ElastoDyn.dat", params)
```

The OC3Hywind and CS_Monopile BModes example decks are supported
and validated when present locally at
[`docs/BModes/docs/examples/`](../docs/BModes/docs/examples/) — the
project's cert-test suite reproduces them to < 0.01 % against BModes
output across the platform rigid-body modes plus the first three
tower-bending pairs. The decks themselves are upstream NREL data and
are **not bundled in this repository**; clone them locally if you want
to exercise the floating-tower code path against the same reference
fixtures.

## What pyBmodes will *not* do automatically

There is no `Tower.from_floating_elastodyn(elastodyn.dat, hydrodyn.dat,
moordyn.dat)` constructor and one is not on the roadmap. The 6×6 matrix
assembly from a HydroDyn `.dat` requires either WAMIT-output ingestion
(reading `.1` / `.3` / `.hst` files and selecting the infinite-frequency
added-mass + hydrostatic-restoring rows) or a frequency-domain
linearisation pass — both substantial pieces that duplicate what
`HydroDyn`/`MoorDyn` already do at OpenFAST runtime, with no obvious
correctness advantage. The supported workflow for floating decks is the
BModes-format path above; if your design lives only in OpenFAST format,
running BModes once to produce the `.bmi` is a one-time cost that we
recommend over reimplementing the WAMIT/Linfeasible path inside
pyBmodes.
