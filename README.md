# pybmodes

A pure-Python finite-element library for computing wind turbine blade and tower natural
frequencies and mode shapes. Designed as a drop-in replacement for the NREL BModes tool,
with direct output of polynomial mode shape coefficients ready for OpenFAST ElastoDyn input files.

## Features

- Rotating blade modal analysis (flap, edge, torsion)
- Onshore tower analysis (cantilevered, tension-wire supported)
- Offshore tower analysis (floating spar, bottom-fixed monopile)
- 6th-order polynomial mode shape fitting (ElastoDyn constraint: C2+…+C6 = 1)
- In-place patching of ElastoDyn `.dat` files

## Installation

Requires Python ≥ 3.11 and NumPy/SciPy.

```bash
pip install pybmodes
```

For development:

```bash
git clone https://github.com/SMI-Lab-Inha/pybmodes.git
cd pybmodes
pip install -e ".[dev]"
```

## Quick start

```python
from pybmodes.models import RotatingBlade, Tower
from pybmodes.elastodyn import compute_blade_params, compute_tower_params, patch_dat

# Blade
result = RotatingBlade("my_blade.bmi").run(n_modes=10)
blade_params = compute_blade_params(result)
print(blade_params.BldFl1Sh)   # c2..c6 + rms_residual

# Tower
result = Tower("my_tower.bmi").run(n_modes=10)
tower_params = compute_tower_params(result)
patch_dat("ElastoDyn_tower.dat", tower_params)   # patch in place
```

See [`examples/`](examples/) for complete worked examples.

## Input files

pybmodes reads the same `.bmi` main input files and tab-delimited section-property
`.dat` files used by the original BModes tool. No format changes are required.

## Validation

All reference test cases from the BModes CertTest suite pass within 0.5% frequency
tolerance. Offshore cases (OC3Hywind floating spar, CS Monopile) are also included.

| Test case | Type | Status |
|-----------|------|--------|
| CertTest01 — non-uniform rotating blade | Blade | ✅ |
| CertTest02 — blade with tip mass | Blade + tip mass | ✅ |
| CertTest03 — onshore tower | Tower | ✅ |
| CertTest04 — tension-wire supported tower | Tower + wires | ✅ |
| OC3Hywind — floating spar | Offshore (free-free) | ✅ |
| CS Monopile — bottom-fixed | Offshore (monopile) | ✅ |

## Running tests

```bash
pytest                          # all tests
pytest -m "not integration"    # unit tests only
```

## License

MIT — see [LICENSE](LICENSE).
