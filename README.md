# pybmodes

[![CI](https://github.com/SMI-Lab-Inha/pybmodes/actions/workflows/ci.yml/badge.svg)](https://github.com/SMI-Lab-Inha/pybmodes/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`pybmodes` is a pure-Python finite-element library for wind-turbine blade and tower modal analysis.

It is a modern interpretation of the legacy **BModes** code developed by **NREL**: the goal is to preserve the practical engineering workflow of BModes while providing a cleaner Python API, automated tests, and direct integration with modern OpenFAST/ElastoDyn tooling.

## What It Does

- Compute natural frequencies and mode shapes for rotating blades
- Compute onshore and offshore tower modes
- Fit ElastoDyn-compatible 6th-order mode-shape polynomials
- Patch OpenFAST ElastoDyn input files with fitted coefficients

Supported tower configurations currently include:

- Cantilevered onshore towers
- Tension-wire-supported towers
- Floating spar-type offshore towers
- Bottom-fixed monopile towers

## Why This Project Exists

BModes remains a valuable reference tool, but it is distributed as legacy Fortran and is not especially convenient to inspect, extend, test, or integrate into modern Python-based workflows.

`pybmodes` is not trying to erase that lineage. It builds on it. The project treats NREL BModes as the numerical and workflow reference, then re-expresses that logic in a more accessible Python codebase.

That means the repository is best understood as:

- a Python library for modal analysis of wind-turbine structural components
- a validation-driven reinterpretation of legacy BModes behavior
- a bridge from BModes-style inputs to OpenFAST ElastoDyn-ready coefficients

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

## Quick Start

```python
from pybmodes.models import RotatingBlade, Tower
from pybmodes.elastodyn import compute_blade_params, compute_tower_params, patch_dat

# Blade analysis
blade_result = RotatingBlade("my_blade.bmi").run(n_modes=10)
blade_params = compute_blade_params(blade_result)
print(blade_params.BldFl1Sh)

# Tower analysis
tower_result = Tower("my_tower.bmi").run(n_modes=10)
tower_params = compute_tower_params(tower_result)

# Patch an ElastoDyn tower file in place
patch_dat("ElastoDyn_tower.dat", tower_params)
```

See the [`examples/`](examples/) directory for complete scripts covering:

- rotating blade modal analysis
- tower modal analysis
- ElastoDyn patching
- plotting mode shapes and fit quality

## Input Files

`pybmodes` reads the standard BModes-style input formats:

- `.bmi` main input files
- tabulated section-property `.dat` files

No input-format redesign is required for the validated workflows in this repository.

## Validation Status

The codebase is validated against reference BModes outputs and example offshore cases.

At the time of writing, the local test suite covers:

- IO/parsing
- FEM building blocks
- blade and tower model pipelines
- polynomial fitting
- ElastoDyn parameter generation and file patching

Reference cases currently exercised in tests:

| Case | Configuration | Status |
| --- | --- | --- |
| CertTest01 | Non-uniform rotating blade | Passed |
| CertTest02 | Blade with tip mass | Passed |
| CertTest03 | Onshore cantilevered tower | Passed |
| CertTest04 | Tension-wire-supported tower | Passed |
| OC3Hywind | Floating offshore spar | Passed |
| CS_Monopile | Bottom-fixed monopile | Passed |

The full local test suite currently passes:

```bash
131 passed
```

## Project Layout

```text
src/pybmodes/
  io/         Input/output parsers
  fem/        Finite-element core
  models/     High-level blade/tower APIs
  fitting/    Mode-shape polynomial fitting
  elastodyn/  ElastoDyn parameter generation and patching
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

## Acknowledgement

This project is a modern Python interpretation of the legacy **BModes** program developed by **NREL**. BModes is the reference point for much of the workflow and validation philosophy used here.

`pybmodes` is an independent reimplementation and is not an official NREL release.

## License

Released under the [MIT License](LICENSE).
