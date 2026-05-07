# pyBModes

[![CI](https://github.com/SMI-Lab-Inha/pybmodes/actions/workflows/ci.yml/badge.svg)](https://github.com/SMI-Lab-Inha/pybmodes/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`pybmodes` is a pure-Python finite-element library for wind-turbine blade and tower modal analysis.

It is best understood as a modern interpretation of the legacy **BModes** Fortran workflow developed by **NREL**: the project keeps the familiar engineering inputs and reference behavior of BModes, while providing a cleaner Python API, automated regression tests, and direct integration with modern OpenFAST/ElastoDyn workflows.

## Overview

`pybmodes` can:

- read BModes-style `.bmi` main input files and tabulated section-property files
- solve rotating blade modal problems
- solve onshore and offshore tower modal problems
- fit ElastoDyn-compatible 6th-order blade and tower mode-shape polynomials
- patch OpenFAST ElastoDyn input files with fitted coefficients
- plot FEM mode shapes and polynomial-fit quality

Supported tower workflows currently include:

- cantilevered onshore towers
- tension-wire-supported towers
- floating spar-type offshore towers
- bottom-fixed monopile towers

## Why This Exists

BModes is still a valuable engineering reference, but its legacy Fortran form can make it harder to inspect, test, automate, and extend in modern Python-heavy workflows.

`pybmodes` does not try to replace the BModes lineage or hide it. Instead, it re-expresses that workflow in Python so it is easier to:

- validate against reference cases
- integrate into reproducible analysis pipelines
- generate ElastoDyn-ready mode-shape coefficients
- inspect intermediate results, fits, and classification decisions

This repository is therefore both:

- a practical Python modal-analysis library
- a validation-driven reinterpretation of legacy BModes behavior

## Installation

Requires Python `>= 3.11`.

Install directly from GitHub:

```bash
pip install git+https://github.com/SMI-Lab-Inha/pybmodes.git
```

For development:

```bash
git clone https://github.com/SMI-Lab-Inha/pybmodes.git
cd pybmodes
pip install -e ".[dev]"
```

If you want plotting support as well:

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

## Examples

The [`examples/`](examples/) directory contains compact end-to-end scripts:

- [`01_blade_analysis.py`](examples/01_blade_analysis.py): rotating blade modal analysis and ElastoDyn blade fits
- [`02_tower_analysis.py`](examples/02_tower_analysis.py): tower modal analysis and ElastoDyn tower fits
- [`03_patch_elastodyn.py`](examples/03_patch_elastodyn.py): patch existing ElastoDyn blade/tower files in place
- [`04_plot_results.py`](examples/04_plot_results.py): plot FEM mode shapes and fit quality

Run them from the repository root, for example:

```bash
python examples/01_blade_analysis.py
python examples/02_tower_analysis.py
```

## Inputs and Outputs

### Inputs

`pybmodes` reads BModes-style inputs without forcing a new file format:

- `.bmi` main input files
- tabulated section-property `.dat` files

### Outputs

The high-level model APIs return:

- natural frequencies
- nodal mode shapes
- ElastoDyn-ready polynomial coefficients for blades and towers

For tower fitting, the companion reporting API can also expose which modal candidates were considered and why specific FA/SS family members were selected.

## Validation

The codebase is validated against BModes reference outputs and offshore benchmark-style cases.

Reference cases currently exercised in the test suite:

| Case | Configuration | Status |
| --- | --- | --- |
| CertTest01 | Non-uniform rotating blade | Passed |
| CertTest02 | Blade with tip mass | Passed |
| CertTest03 | Onshore cantilevered tower | Passed |
| CertTest04 | Tension-wire-supported tower | Passed |
| OC3Hywind | Floating offshore spar | Passed |
| CS_Monopile | Bottom-fixed monopile | Passed |

At the time of this README update, the full local suite passes with:

```bash
372 passed
```

The tests cover:

- input parsing and path resolution, including BMI parser primitives and `.out` reference parsing
- FEM building blocks (boundary conditions, generalised eigensolver, non-dimensionalisation, mode-shape extraction)
- model pipelines for blades and towers
- polynomial fitting and tower FA/SS family classification
- ElastoDyn parameter generation and file patching
- regression checks against validated blade, tower, and offshore cases

## Project Layout

```text
src/pybmodes/
  io/         BModes-style input/output parsers
  fem/        Finite-element core
  models/     High-level blade and tower APIs
  fitting/    Mode-shape polynomial fitting
  elastodyn/  ElastoDyn parameter generation and file patching
  plots/      Plotting helpers for mode shapes and fit quality
examples/     End-to-end usage scripts
tests/        Validation and regression coverage
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

## Relationship to BModes

This project is a modern Python reinterpretation of the legacy **BModes** program developed by **NREL**.

BModes remains the numerical and workflow reference point for much of the validation philosophy used here. `pybmodes` is an independent Python reimplementation and is not an official NREL release.

## License

Released under the [MIT License](LICENSE).
