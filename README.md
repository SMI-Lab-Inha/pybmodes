# pybmodes


[![CI](https://github.com/SMI-Lab-Inha/pybmodes/actions/workflows/ci.yml/badge.svg)](https://github.com/SMI-Lab-Inha/pybmodes/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A pure-Python finite-element library for computing wind turbine blade and tower natural frequencies and mode shapes, with direct output of polynomial mode shape coefficients for [OpenFAST](https://github.com/OpenFAST/openfast) ElastoDyn input files.

---

## Features

- **Rotating blade** modal analysis — flap, edge, and torsion modes
- **Onshore tower** analysis — cantilevered and tension-wire supported
- **Offshore tower** analysis — floating spar and bottom-fixed monopile
- **Mode shape fitting** — constrained 6th-order polynomial fit (C₂ + C₃ + C₄ + C₅ + C₆ = 1)
- **ElastoDyn integration** — read, compute, and patch `.dat` files in place

## Installation

Requires Python ≥ 3.11, NumPy, and SciPy.

**From GitHub (current):**

```bash
pip install git+https://github.com/SMI-Lab-Inha/pybmodes.git
```

**For development:**

```bash
git clone https://github.com/SMI-Lab-Inha/pybmodes.git
cd pybmodes
pip install -e ".[dev]"
```

## Quick Start

```python
from pybmodes.models import RotatingBlade, Tower
from pybmodes.elastodyn import compute_blade_params, compute_tower_params, patch_dat

# --- Blade ---
result = RotatingBlade("my_blade.bmi").run(n_modes=10)
params = compute_blade_params(result)
print(params.BldFl1Sh)  # PolyFitResult(c2, c3, c4, c5, c6, rms_residual)

# --- Tower ---
result = Tower("my_tower.bmi").run(n_modes=10)
params = compute_tower_params(result)
patch_dat("ElastoDyn_tower.dat", params)  # updates coefficients in place
```

See [`examples/`](examples/) for complete worked examples covering blade analysis, tower analysis, and end-to-end ElastoDyn patching.

## Input Format

pybmodes reads the standard `.bmi` main input files and tab-delimited section-property `.dat` files. No format changes are required compared to the original tool.

## Validation

All reference cases pass within **0.5% frequency tolerance**. Mode shape nodal displacements match within **2%**.

| Test case | Configuration | Result |
| --------- | ------------- | ------ |
| CertTest01 — non-uniform rotating blade | Blade, 60 RPM | ✅ pass |
| CertTest02 — blade with tip mass | Blade + tip mass | ✅ pass |
| CertTest03 — onshore tower | Cantilevered tower | ✅ pass |
| CertTest04 — tension-wire supported tower | Tower + wires | ✅ pass |
| OC3Hywind | Offshore, floating spar | ✅ pass |
| CS Monopile | Offshore, bottom-fixed | ✅ pass |

## Development

```bash
# Run all tests
pytest

# Run only unit tests (skip full FE solves)
pytest -m "not integration"

# Lint
ruff check src/ tests/

# Type check
mypy src/pybmodes
```

## License

Released under the [MIT License](LICENSE).
