# Case 04 — Rotating pinned-free cable (Bir 2009 Eq. 8)

A rotating "cable" — a uniform beam with vanishingly small bending
stiffness, held by a pinned-free root (deflection = 0 at the root,
slope = free), spun at angular velocity Ω. With `EI → 0` the
flap-bending stiffness is negligible compared to the centrifugal
restoring force `T(r) · w''`, and the modal eigenvalue problem reduces
to the rotating-string equation. Bir (2009) AIAA-2009-1035 §III.B
solves that limit analytically using Legendre polynomials and reports
the closed-form frequency relation as Eq. (8):

```
ωₖ = Ω · √(k(2k − 1))     for k = 1, 2, 3, …
```

So the natural frequencies are integer-multiple-like ratios of Ω
that depend only on the mode number — no material, length, or mass
appears in the analytical limit (the FEM result confirms this
remarkable property).

## Files

| File                     | Purpose                          |
| ------------------------ | -------------------------------- |
| `cable.bmi`              | Main BModes input                |
| `cable_sec_props.dat`    | Distributed section properties   |

## Physical parameters

| Parameter   | Value      | Units    | Notes                                |
| ----------- | ---------: | -------- | ------------------------------------ |
| `radius`    | 31.623     | m        | cable length                         |
| `hub_rad`   | 0.0        | m        | flexible from r = 0                  |
| `rot_rpm`   | 95.49297   | rpm      | exactly Ω = 10 rad/s                 |
| `hub_conn`  | 4          | —        | **pinned-free** (slope free at root) |
| `mass_den`  | 100.0      | kg/m     | uniform mass per length              |
| `EI_flap`   | 1.0 × 10³  | N·m²     | small enough for cable limit         |
| `EI_edge`   | 1.0 × 10⁹  | N·m²     | unused for this analysis             |
| `GJ`        | 1.0 × 10⁵  | N·m²     | unused                               |

The pinned-free root condition (`hub_conn = 4`) is the project's
addition to BModes' three classical BCs — it constrains transverse
deflection but lets the slope rotate freely, matching Bir's
Legendre-polynomial setup. The standard cantilever `hub_conn = 1`
clamps the slope and would *not* match Bir's solution — at Ω = 10
rad/s the cantilever 1st mode lands ~ 30 % higher than the Bir
analytical answer.

## Expected frequencies (pyBmodes vs Bir 2009 Eq. 8)

At Ω = 10 rad/s:

| Mode | Type      | ω_FEM (rad/s) | ωₖ/Ω_FEM | ωₖ/Ω ref `√(k(2k−1))` |
| ---: | --------- | ------------: | -------: | --------------------: |
|    1 | flap-1    |       10.0000 |   1.0000 |             1.00000   |
|    2 | flap-2    |       24.4950 |   2.4495 |             2.44949   |
|    3 | flap-3    |       38.7314 |   3.8731 |             3.87298   |

Each ratio matches the analytical Legendre solution to four-plus
digits. The lag-1 mode appears between flap-1 and flap-2 in the
eigensolver order (because `EI_edge = 10⁶ · EI_flap` makes lag still
genuinely stiff while flap is cable-soft); the table above filters
to flap-dominated modes by participation, the same way
[`tests/fem/test_rotating_cable.py`](../../../tests/fem/test_rotating_cable.py)
does.

## How to run

```python
from pybmodes.models import RotatingBlade
import numpy as np
cable = RotatingBlade(
    "cases/sample_inputs/04_pinned_free_cable/cable.bmi"
)
modal = cable.run(n_modes=12)
omega = 2 * np.pi * np.asarray(modal.frequencies)
print(omega[:4])  # [10.0000, ..., 24.4950, 38.7314] (lag mode interleaved)
```

## Validation reference

Bir (2009) AIAA-2009-1035 / NREL/CP-500-44749, §III.B and Eq. (8) —
the analytical Legendre-polynomial solution for the rotating pinned-
free cable. The full Ω sweep `Ω ∈ {2, 6, 10, 15, 20, 25, 30}` rad/s is
exercised by
[`tests/fem/test_rotating_cable.py`](../../../tests/fem/test_rotating_cable.py),
which gates each mode at < 0.5 % relative error against the
analytical reference.

Citation:

- Bir, G. S. (2009). *Blades and Towers Modal Analysis Code (BModes):
  Verification of Blade Modal Analysis Capability*. AIAA-2009-1035 /
  NREL/CP-500-44749. Section III.B and Equation 8.
