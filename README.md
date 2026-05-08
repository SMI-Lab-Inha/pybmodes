# pyBModes

[![CI](https://github.com/SMI-Lab-Inha/pyBModes/actions/workflows/ci.yml/badge.svg)](https://github.com/SMI-Lab-Inha/pyBModes/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`pybmodes` is a pure-Python finite-element library for wind-turbine blade and tower modal analysis.

## Overview

`pybmodes` solves coupled flap–lag–torsion–axial vibration modes of slender beams using a 15-DOF Bernoulli-Euler beam element formulation and the standard generalised eigenvalue solver from SciPy. It can:

- read line-ordered `.bmi` main-input files and tabulated section-property `.dat` files;
- read OpenFAST input decks directly (`Tower.from_elastodyn(...)`, `Tower.from_elastodyn_with_subdyn(...)`, `RotatingBlade.from_elastodyn(...)`) — parses ElastoDyn main + tower + blade files, with optional SubDyn pile geometry spliced below the tower for monopile decks;
- solve rotating blade modal problems with centrifugal stiffening, tip masses, and pre-twist;
- solve onshore and offshore tower modal problems;
- fit ElastoDyn-compatible 6th-order blade and tower mode-shape polynomials, with design-matrix condition-number reporting and automatic resolution of degenerate FA/SS eigenpairs on symmetric structures;
- patch OpenFAST ElastoDyn input files in place with fitted coefficients;
- assemble a Campbell diagram from a single OpenFAST deck — blade modes swept across rotor speed with MAC-based tracking, tower modes overlaid as horizontal lines, plus the per-rev (1P, 3P, 6P, …) excitation family — for resonance checks like NREL 5MW's *3P × 1st-tower-FA at ~6–7 rpm*;
- plot FEM mode shapes, polynomial-fit quality, and Campbell diagrams (MATLAB-styled defaults via `apply_style()`).

Supported tower configurations (all cross-verified against the BModes Fortran reference solver):

- cantilevered onshore towers (with optional concentrated tip mass)
- tension-wire-supported towers
- floating-platform-supported towers (free-free root, 6×6 platform mass / hydro / mooring matrices, including hydrostatically-unstable spar configurations)
- monopile-supported towers (axial + torsion-fixed root, with mooring stiffness or distributed soil stiffness along the embedded section)

## Installation

Requires Python `>= 3.11`.

### Recommended: Windows + conda (newbie-friendly)

If you don't already have a Python environment set up, this is the path of least resistance on Windows. It uses [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create an isolated environment, then installs `pybmodes` with `pip` from inside it.

1. **Install Miniconda.** Download and run the Miniconda installer for Windows from the link above. Accept the defaults.
2. **Open the Anaconda Prompt** from the Start menu (not regular CMD or PowerShell — the Anaconda Prompt has `conda` already on `PATH`).
3. **Create and activate a dedicated environment.** Naming it `pybmodes` keeps your base environment clean:

   ```cmd
   conda create -n pybmodes python=3.11 -y
   conda activate pybmodes
   ```

4. **Clone and install in editable mode** with the development and plotting extras:

   ```cmd
   git clone https://github.com/SMI-Lab-Inha/pyBModes.git
   cd pyBModes
   pip install -e ".[dev,plots]"
   ```

   `pip install -e ".[dev,plots]"` pulls runtime dependencies (`numpy`, `scipy`), development tools (`pytest`, `ruff`, `mypy`), and `matplotlib` for the plotting helpers.

5. **Verify the install** by running the test suite:

   ```cmd
   pytest
   ```

   You should see all tests pass in a few seconds.

After the first install, you only need `conda activate pybmodes` in a new shell to start working.

### Quick install (existing Python environment)

If you already manage your own virtualenv / conda env / Poetry / uv setup, install straight from GitHub:

```bash
pip install git+https://github.com/SMI-Lab-Inha/pyBModes.git
```

### Development install

```bash
git clone https://github.com/SMI-Lab-Inha/pyBModes.git
cd pyBModes
pip install -e ".[dev]"
```

Add the `plots` extra if you want `matplotlib`-based plotting helpers:

```bash
pip install -e ".[dev,plots]"
```

## Quick Start

### Blade analysis

```python
from pybmodes.models import RotatingBlade
from pybmodes.elastodyn import compute_blade_params

blade = RotatingBlade("my_blade.bmi")
result = blade.run(n_modes=10)
params = compute_blade_params(result)

print(result.frequencies[:3])
print(params.BldFl1Sh)
```

### Tower analysis

```python
from pybmodes.models import Tower
from pybmodes.elastodyn import compute_tower_params, compute_tower_params_report

tower = Tower("my_tower.bmi")          # or Tower.from_bmi("my_tower.bmi")
result = tower.run(n_modes=10)

params = compute_tower_params(result)
params2, report = compute_tower_params_report(result)

print(result.frequencies[:4])
print(report.selected_fa_modes)
print(report.selected_ss_modes)
```

### Reading OpenFAST decks directly

```python
from pybmodes.models import Tower, RotatingBlade

# Onshore tower from ElastoDyn — parses main + tower + blade files,
# lumps the rotor mass into a tower-top tip-mass assembly automatically.
tower = Tower.from_elastodyn("MyTurbine_ElastoDyn.dat")

# Monopile tower with SubDyn pile geometry spliced below the ElastoDyn
# tower into a single combined cantilever (clamped at the seabed).
tower_mp = Tower.from_elastodyn_with_subdyn(
    "MyTurbine_ElastoDyn.dat",
    "MyTurbine_SubDyn.dat",
)

# Rotating blade from the same ElastoDyn deck. RotSpeed is taken from
# the deck; override on the BMI for off-rated analyses.
blade = RotatingBlade.from_elastodyn("MyTurbine_ElastoDyn.dat")
result = blade.run(n_modes=8)
```

### Patch ElastoDyn files

```python
from pybmodes.models import Tower
from pybmodes.elastodyn import compute_tower_params, patch_dat

result = Tower("my_tower.bmi").run(n_modes=10)
params = compute_tower_params(result)

patch_dat("ElastoDyn_tower.dat", params)
```

### Campbell diagram

```python
import numpy as np
from pybmodes.campbell import campbell_sweep, plot_campbell
from pybmodes.plots import apply_style

apply_style()  # MATLAB-styled defaults

rpm = np.linspace(0.0, 15.0, 16)
# An ElastoDyn .dat input auto-loads the blade *and* the tower from
# the same deck; the tower frequencies overlay as horizontal lines
# (rotor-speed-independent, since the tower lives in an Earth-fixed
# frame). For a .bmi input the sweep covers whichever component the
# file describes; pair a blade .bmi with a tower .bmi via tower_input.
result = campbell_sweep(
    "MyTurbine_ElastoDyn.dat",
    rpm,
    n_blade_modes=4,    # 1st/2nd flap + 1st/2nd edge (default)
    n_tower_modes=4,    # 1st/2nd FA + 1st/2nd SS (default)
)

# Blade modes are MAC-tracked across rotor speeds, so
# result.frequencies[:, k] is the same physical mode at every speed.
print(result.labels)
# ['1st flap', '1st edge', '2nd flap', '2nd edge',
#  '1st tower SS', '1st tower FA', '2nd tower SS', '2nd tower FA']
print(result.frequencies.shape)  # (16, 8)
print(result.n_blade_modes, result.n_tower_modes)  # 4 4

fig = plot_campbell(
    result,
    excitation_orders=[1, 2, 3, 6, 9],
    rated_rpm=12.1,
)
fig.savefig("campbell.png")
```

The plot draws blade modes as solid coloured lines with markers, tower
modes as horizontal dashed dark-grey lines, and the per-rev family
(1P, 2P, …) as red dotted rays from the origin (shaded medium-to-dark
red so the order of crossings is readable at a glance) — exactly the
layout needed for the canonical *3P × 1st-tower-FA* resonance check at
~6–7 rpm on the NREL 5MW.

The same sweep is also available as a CLI subcommand:

```bash
pybmodes campbell MyTurbine_ElastoDyn.dat \
    --rated-rpm 12.1 --max-rpm 15 \
    --n-blade-modes 4 --n-tower-modes 4 \
    --orders 1,2,3,6,9 --out campbell.png
```

## Walkthrough notebook

[`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) is a self-contained end-to-end tour of the public API.  It builds two synthetic cases inline (a uniform Euler-Bernoulli blade and a uniform tower with a concentrated top mass), runs the FEM solver, fits ElastoDyn polynomials, and validates the FEM frequencies against published closed-form formulas — all without bundling any external data.

## Inputs and Outputs

### Inputs

- `.bmi` main input files (line-ordered text, values precede labels) plus the section-property `.dat` they reference
- OpenFAST ElastoDyn main `.dat` plus the tower and blade files referenced via `TwrFile` / `BldFile(1)`
- OpenFAST SubDyn `.dat` (minimal subset: joints, members, circular cross-section properties, base reaction joint, interface joint — sufficient for OC3-style fixed-base monopiles)
- BModes `.out` reference output (parsed for cross-solver verification, not used as a primary input)

### Outputs

The high-level model APIs return:

- natural frequencies
- nodal mode shapes
- ElastoDyn-ready polynomial coefficients for blades and towers (with per-fit RMS-residual and design-matrix condition-number diagnostics)

For tower fitting, the companion reporting API exposes which modal candidates were considered and why specific FA/SS family members were selected, and warns when the polynomial-basis condition number exceeds a configurable threshold.

## Validation

The codebase is validated against two complementary sources of truth.

### Closed-form analytical references

| Case | Reference | Tolerance |
| --- | --- | --- |
| Uniform Euler-Bernoulli cantilever (first 5 flap modes) | Analytical: $\beta_n L = [1.875, 4.694, 7.855, 10.996, 14.137]$ | < 0.5 % |
| Uniform cantilever with concentrated tip mass | Frequency equation in Blevins (1979), *Formulas for Natural Frequency and Mode Shape*; Karnovsky & Lebed (2001), *Formulas for Structural Dynamics* | < 0.5 % |
| Hermite-cubic mesh-convergence | $h^4$ convergence rate for first five frequencies | confirmed |

All analytical test cases are constructed in-test from numbers that come from peer-reviewed textbooks or analytical formulas. Section properties for the synthetic validation cases are generated programmatically by the test suite.

### Cross-solver certification against BModes

Six certification cases compare the pyBmodes pipeline against the BModes Fortran reference solver (Bir 2010), with paired `.bmi` inputs and `.out` reference frequencies. All six pass at strict tolerances on the local development machine:

| Case (BModes deck) | Reference turbine / configuration | Boundary condition | Tolerance | Worst-mode error |
| --- | --- | --- | --- | --- |
| Test01, Test02 | non-uniform rotating blade (with and without tip mass) — BModes v3.00 CertTest | cantilever | < 1 % on modes 1-6, < 3 % on modes 7+ | < 0.005 % across 20 modes |
| Test03 | 82.4 m tower with top mass and offsets — BModes v3.00 CertTest | cantilever | < 1 % / < 3 % | < 0.005 % across 20 modes |
| Test04 | Test03 + tension-wire support — BModes v3.00 CertTest | cantilever + wires | < 1 % / < 3 % | < 0.005 % across 20 modes |
| `CS_Monopile.bmi` | *NREL 5MW Reference Turbine* (Jonkman et al. 2009) on the *OC3 Monopile* configuration (Jonkman & Musial 2010) | `hub_conn = 3` (soft monopile, mooring stiffness, lateral + rocking free) | 0.01 % | < 0.005 % across 10 modes |
| `OC3Hywind.bmi` | *NREL 5MW* on the *OC3 Hywind* floating spar (Jonkman 2010) — full hydro + mooring + 6×6 platform inertia | `hub_conn = 2` (free-free) | 0.01 % | ≤ 0.0003 % across 9 modes (surge, sway, pitch, roll, heave, yaw, 1st-2nd tower bending) |

Citations:

- Jonkman, J., Butterfield, S., Musial, W., & Scott, G. (2009). *Definition of a 5-MW Reference Wind Turbine for Offshore System Development*. NREL/TP-500-38060.
- Jonkman, J., & Musial, W. (2010). *Offshore Code Comparison Collaboration (OC3) for IEA Wind Task 23 Offshore Wind Technology and Deployment*. NREL/TP-5000-48191.
- Jonkman, J. (2010). *Definition of the Floating System for Phase IV of OC3*. NREL/TP-500-47535.

These certification inputs and reference outputs are local-only — the test module skips at module level when they are absent, so contributors who don't have them can still run the rest of the suite.

#### What "BModes reference" means in the table above

For all six rows the **reference is the BModes Fortran solver `.out`
output run locally on the same `.bmi` input** that pyBmodes consumes.
This is **not** the same as the published frequency tables in
Bir (2010, NREL/CP-500-47953):

- Bir 2010 Tables 1 (land tower), 2 (barge), 3 (monopile DS) and
  4 (spar-buoy) report frequencies from a specific BModes version
  with a specific gravity convention; comparing pyBmodes against
  *those published numbers* on the bundled decks shows 4–9 %
  deviation on platform rigid-body modes.
- Comparing pyBmodes against **BModes JJ `.out` on the same deck and
  the same physics** is a direct solver comparison with no version or
  convention ambiguity, and is reproducibly within 0.01 %.

The one Bir 2010 table where pyBmodes *is* validated directly against
the published numbers is **Table 5** (rotating uniform blade with tip
mass) — those values come from the Wright et al. (1982) analytical
solution and are version-independent. That direct comparison lives at
[`tests/fem/test_rotating_blade_with_tip_mass.py`](tests/fem/test_rotating_blade_with_tip_mass.py)
and passes at ≤ 0.1 %. Wright et al.'s rotating-uniform-blade values
(transcribed from Bir 2009 / AIAA 2009-1035 Table 3a) are similarly
validated at ≤ 0.5 % by
[`tests/fem/test_rotating_uniform_blade.py`](tests/fem/test_rotating_uniform_blade.py).

### Test suite

Every validation case — what's compared against what, at what tolerance, with what worst observed margin, and which test enforces it — is enumerated in [`VALIDATION.md`](VALIDATION.md). That file is the source of truth and stays in sync with the test suite; the CI badge at the top of this README shows the current public pass status.

The tests cover:

- input parsing and path resolution (parser primitives + inline-fixture round-trips)
- FEM building blocks (boundary conditions, generalised eigensolver — both symmetric `eigh` and asymmetric `eig` paths — non-dimensionalisation, mode-shape extraction)
- model pipelines for blades and towers, including OpenFAST deck adapters
- degenerate FA/SS eigenpair resolution for symmetric tower models
- polynomial fitting and tower FA/SS family classification, with conditioning diagnostics
- ElastoDyn parameter generation and file patching
- closed-form / analytical validation of representative blade and tower configurations
- cross-solver certification against BModes on the six cases above

## Case studies

The [`cases/`](cases/) directory contains exploratory studies that exercise the full pipeline (parse → solve → fit → diagnose) against publicly available reference turbines:

- [`cases/nrel5mw_land/`](cases/nrel5mw_land/) — *NREL 5MW Reference Turbine* (Jonkman et al. 2009) on the OpenFAST land-based deck from the OpenFAST regression-test corpus.
- [`cases/iea3mw_land/`](cases/iea3mw_land/) — *IEA-3.4-130-RWT* (Bortolotti et al. 2019, IEA Wind Task 37) land-based deck.
- [`cases/nrel5mw_monopile/`](cases/nrel5mw_monopile/) — *NREL 5MW* on a rigid monopile substructure (the OC3 Monopile configuration without soil flexibility) using the SubDyn pile geometry spliced below the ElastoDyn tower.

Each case has a `run.py` that prints a coefficient comparison table and writes outputs (mode-shape PNGs, full diagnostic text). The cross-deck summary in [`cases/ECOSYSTEM_FINDING.md`](cases/ECOSYSTEM_FINDING.md) documents a recurring pattern: the polynomial-coefficient blocks shipped in industry `_ElastoDyn.dat` files are not always reproducible from the structural-property blocks in the same files — the coefficients in many decks were generated against an earlier revision of the structural model and have not been regenerated.

References for the case-study turbines:

- Jonkman, Butterfield, Musial, & Scott (2009). *Definition of a 5-MW Reference Wind Turbine for Offshore System Development*. NREL/TP-500-38060.
- Bortolotti, Tarrés, Dykes, Merz, Sethuraman, Verelst, & Zahle (2019). *IEA Wind TCP Task 37: Systems Engineering in Wind Energy — WP2.1 Reference Wind Turbines*. NREL/TP-5000-73492.

## Project Layout

```text
src/pybmodes/
  io/         input/output parsers (.bmi, section-properties .dat, .out)
  fem/        finite-element core
  models/     high-level blade and tower APIs
  fitting/    mode-shape polynomial fitting
  elastodyn/  ElastoDyn parameter generation and file patching
  campbell.py rotor-speed sweep + MAC-tracked Campbell diagram
  plots/      plotting helpers + MATLAB-styled matplotlib defaults
  cli.py      `pybmodes` CLI (validate / patch / campbell)
notebooks/    walkthrough.ipynb — end-to-end usage tour
scripts/      one-off project scripts (build_reference_decks, campbell)
tests/        unit + closed-form-analytical validation
```

## Development

```bash
# Default test run — self-contained, no external decks
pytest

# Integration tests — requires upstream OpenFAST / BModes data under docs/
pytest -m integration

# Both
pytest -m ""

# Lint
ruff check src/ tests/

# Type check
mypy src/pybmodes
```

The default `pytest` run is **self-contained** and works on a fresh
clone with no external data. The `integration` marker gates the
subset of tests that need locally-checked-out OpenFAST `r-test`
decks, BModes CertTest data, or BModes `.bmi` / `.out` reference
outputs under `docs/` (none of which are bundled — see the
*Compatibility policy* below for why). CI runs both steps on every
commit; the integration step is allowed to skip when data isn't
present. See [`VALIDATION.md`](VALIDATION.md) for the per-case
breakdown of which tests need external data and which don't.

## Public API

These imports are stable across 0.x patch releases (subject to the
*Compatibility policy* below):

```python
from pybmodes.models    import RotatingBlade, Tower, ModalResult
from pybmodes.elastodyn import (
    compute_blade_params,
    compute_tower_params,
    compute_tower_params_report,
    patch_dat,
    validate_dat_coefficients,
    BladeElastoDynParams,
    TowerElastoDynParams,
    ValidationResult,
    CoeffBlockResult,
)
from pybmodes.fitting   import PolyFitResult, fit_mode_shape
from pybmodes.campbell  import (
    CampbellResult,
    campbell_sweep,
    plot_campbell,
)
from pybmodes.plots     import (
    apply_style,
    plot_mode_shapes,
    plot_fit_quality,
    bir_mode_shape_plot,
    bir_mode_shape_subplot,
)
```

The CLI is exposed as `pybmodes` (see `pybmodes --help`).

Internal modules — `pybmodes.fem.*`, `pybmodes.io.*`, the
underscore-prefixed `pybmodes.models._pipeline` — carry the
implementation. Their signatures may change between 0.x patch
releases; user code should not import from them directly.

## Compatibility policy

Until the 1.0 release:

- **Numerical outputs may change** when validation tightens or a
  modelling correction lands. Each release that moves a published
  number is called out in `CHANGELOG.md` under *Fixed* / *Changed*
  with the magnitude of the shift and the affected case (e.g. the
  May 2026 OC3 Hywind 1st-tower-bending fix that closed a 3.7 % gap).
- **Public constructors and result dataclasses are kept
  source-compatible where possible.** Adding new keyword arguments
  with defaults is non-breaking; renaming or removing existing fields
  goes through one release of `DeprecationWarning`.
- **Parser behaviour is best-effort across OpenFAST versions.** The
  ElastoDyn / SubDyn readers handle FAST v8 and OpenFAST v3+ format
  drift via label-based scanning, but new file-format changes
  upstream may need tracking patches; if a deck parses on one
  pyBmodes release and not on the next, that's a bug worth reporting.
- **The `pybmodes` CLI** (`validate`, `patch`, `campbell`) is stable;
  new subcommands may be added but existing ones don't change exit
  codes or output schema in patch releases.

After 1.0, source-compatibility on the public API tier becomes a
hard guarantee; numerical outputs continue to follow the changelog
discipline above.

## License

Released under the [MIT License](LICENSE).
