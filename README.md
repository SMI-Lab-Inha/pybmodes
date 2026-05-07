# pyBModes

[![CI](https://github.com/SMI-Lab-Inha/pyBModes/actions/workflows/ci.yml/badge.svg)](https://github.com/SMI-Lab-Inha/pyBModes/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`pybmodes` is a pure-Python finite-element library for wind-turbine blade and tower modal analysis.

## Overview

`pybmodes` solves coupled flap–lag–torsion–axial vibration modes of slender beams using a 15-DOF Bernoulli-Euler beam element formulation and the standard generalised eigenvalue solver from SciPy. It can:

- read line-ordered `.bmi` main-input files and tabulated section-property `.dat` files;
- solve rotating blade modal problems with centrifugal stiffening, tip masses, and pre-twist;
- solve onshore and offshore tower modal problems;
- fit ElastoDyn-compatible 6th-order blade and tower mode-shape polynomials;
- patch OpenFAST ElastoDyn input files in place with fitted coefficients;
- plot FEM mode shapes and polynomial-fit quality.

Supported tower configurations:

- cantilevered onshore towers (with optional concentrated tip mass)
- tension-wire-supported towers
- floating-platform-supported towers (free-free root, 6×6 platform mass / hydro / mooring matrices)
- monopile-supported towers (axial + torsion-fixed root, distributed soil stiffness along the embedded section)

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

tower = Tower("my_tower.bmi")
result = tower.run(n_modes=10)

params = compute_tower_params(result)
params2, report = compute_tower_params_report(result)

print(result.frequencies[:4])
print(report.selected_fa_modes)
print(report.selected_ss_modes)
```

### Patch ElastoDyn files

```python
from pybmodes.models import Tower
from pybmodes.elastodyn import compute_tower_params, patch_dat

result = Tower("my_tower.bmi").run(n_modes=10)
params = compute_tower_params(result)

patch_dat("ElastoDyn_tower.dat", params)
```

## Walkthrough notebook

[`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) is a self-contained end-to-end tour of the public API.  It builds two synthetic cases inline (a uniform Euler-Bernoulli blade and a uniform tower with a concentrated top mass), runs the FEM solver, fits ElastoDyn polynomials, and validates the FEM frequencies against published closed-form formulas — all without bundling any external data.

## Inputs and Outputs

### Inputs

- `.bmi` main input files (line-ordered text, values precede labels)
- tabulated section-property `.dat` files

### Outputs

The high-level model APIs return:

- natural frequencies
- nodal mode shapes
- ElastoDyn-ready polynomial coefficients for blades and towers

For tower fitting, the companion reporting API exposes which modal candidates were considered and why specific FA/SS family members were selected.

## Validation

The codebase is validated against published closed-form results from beam-vibration theory:

| Case | Reference | Tolerance |
| --- | --- | --- |
| Uniform Euler-Bernoulli cantilever (first 5 flap modes) | Analytical: $\beta_n L = [1.875, 4.694, 7.855, 10.996, 14.137]$ | < 0.5 % |
| Uniform cantilever with concentrated tip mass | Frequency equation in Blevins (1979), *Formulas for Natural Frequency and Mode Shape*; Karnovsky & Lebed (2001), *Formulas for Structural Dynamics* | < 0.5 % |
| Hermite-cubic mesh-convergence | $h^4$ convergence rate for first five frequencies | confirmed |

All test cases are constructed in-test from numbers that come from peer-reviewed textbooks or analytical formulas. No third-party reference data is bundled with the repository. Section properties for the synthetic validation cases are generated programmatically by the test suite.

The full local suite passes with:

```bash
195 passed
```

The tests cover:

- input parsing and path resolution (parser primitives + inline-fixture round-trips)
- FEM building blocks (boundary conditions, generalised eigensolver, non-dimensionalisation, mode-shape extraction)
- model pipelines for blades and towers
- polynomial fitting and tower FA/SS family classification
- ElastoDyn parameter generation and file patching
- closed-form / analytical validation of representative blade and tower configurations

## Project Layout

```text
src/pybmodes/
  io/         input/output parsers (.bmi, section-properties .dat, .out)
  fem/        finite-element core
  models/     high-level blade and tower APIs
  fitting/    mode-shape polynomial fitting
  elastodyn/  ElastoDyn parameter generation and file patching
  plots/      plotting helpers for mode shapes and fit quality
notebooks/    walkthrough.ipynb — end-to-end usage tour
tests/        unit + closed-form-analytical validation
```

## Development

```bash
# Run the full test suite
pytest

# Skip integration tests
pytest -m "not integration"

# Lint
ruff check src/ tests/

# Type check
mypy src/pybmodes
```

## License

Released under the [MIT License](LICENSE).
