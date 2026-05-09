# Case 02 — Uniform tower with concentrated top mass (Blevins reference)

A uniform-section cantilevered tower with a concentrated mass lumped
at the top — the textbook offshore-tower-with-RNA problem. The
frequency equation is the classical Blevins / Karnovsky form

```
1 + cos(βL) cosh(βL) − μ · βL · (sin(βL) cosh(βL) − cos(βL) sinh(βL)) = 0
```

where `μ = m_tip / (ρA · L)` is the tip-mass-to-tower-mass ratio.
Solving for `μ = 1.0` (tip mass equal to total tower mass) gives
`β₁L ≈ 1.24793`. Frequencies follow `ωₙ = (βₙL)² · √(EI / (ρA · L⁴))`.

## Files

| File                            | Purpose                          |
| ------------------------------- | -------------------------------- |
| `tower_topmass.bmi`             | Main BModes input                |
| `tower_topmass_sec_props.dat`   | Distributed section properties   |

## Physical parameters

| Parameter   | Value          | Units    | Notes                           |
| ----------- | -------------: | -------- | ------------------------------- |
| `radius`    | 80.0           | m        | tower height (no rigid base)    |
| `hub_rad`   | 0.0            | m        | full 80 m flexible              |
| `beam_type` | 2              | —        | tower                           |
| `hub_conn`  | 1              | —        | cantilevered base               |
| `tow_support` | 0            | —        | no wires, no platform           |
| `mass_den`  | 5000.0         | kg/m     | heavy steel section             |
| `EI`        | 5.0 × 10¹⁰     | N·m²     | both fore-aft and side-side     |
| `tip_mass`  | 4.0 × 10⁵      | kg       | μ = m/(ρAL) = 1.0 exactly       |

`EI_FA = EI_SS` makes the tower axisymmetric, so the bending modes
appear as **degenerate fore-aft / side-side pairs** — modes 1 and 2
share the same frequency, modes 3 and 4 share the same frequency, etc.
This is the canonical case the project's degenerate-eigenpair resolver
([`_rotate_degenerate_pairs`](../../../src/pybmodes/elastodyn/params.py))
is designed to handle without ambiguity.

## Expected frequencies (pyBmodes vs analytical)

With `√(EI / (ρA · L⁴)) = 0.4941 rad/s` and `μ = 1.0`:

| Mode | Type           | ω_FEM (rad/s) | f_FEM (Hz) | βₙL_FEM | βₙL ref       |
| ---: | -------------- | ------------: | ---------: | ------: | ------------: |
|    1 | 1st FA         |        0.7695 |     0.1225 |  1.2479 |  1.24793      |
|    2 | 1st SS         |        0.7695 |     0.1225 |  1.2479 |  1.24793 (≡)  |
|    3 | 2nd FA         |        8.0293 |     1.2779 |  4.0311 |  4.0312       |
|    4 | 2nd SS         |        8.0293 |     1.2779 |  4.0311 |  4.0312 (≡)   |
|    5 | 3rd FA         |       25.1482 |     4.0025 |  7.1342 |  ~ 7.13       |

The 1st-mode frequency drops to `≈ 0.1225 Hz` from the no-tip-mass
value of `0.4941 · 1.875² / (2π) = 0.2767 Hz` — a 56 % reduction
that's purely the mass-loading effect (lower βL because the tip mass
shifts the modal energy distribution toward the cantilever's free
end).

## How to run

```python
from pybmodes.models import Tower
tower = Tower("cases/sample_inputs/02_tower_topmass/tower_topmass.bmi")
modal = tower.run(n_modes=8)
print(modal.frequencies[:4])   # [0.1225, 0.1225, 1.2779, 1.2779]
```

## Validation reference

The `μ = 1.0` row of the Blevins / Karnovsky cantilever-with-tip-mass
solution. Equivalent parameters are exercised in
[`tests/fem/test_uniform_tower_analytical.py`](../../../tests/fem/test_uniform_tower_analytical.py)
which sweeps μ ∈ {0.1, 0.5, 1.0, 2.0} and asserts each FEM frequency
matches the Blevins root-finding solution to within 0.5 %.

Citations:

- Blevins, R. D. (1979). *Formulas for Natural Frequency and Mode
  Shape*. Krieger Publishing. Table 8-1.
- Karnovsky, I. A.; Lebed, O. I. (2001). *Formulas for Structural
  Dynamics: Tables, Graphs and Solutions*. McGraw-Hill.
