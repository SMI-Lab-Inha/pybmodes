# Case 03 — Rotating uniform blade (Wright 1982 / Bir 2009 Table 3a)

A rotating cantilevered blade — same uniform-section physics as
Case 01 but spun at `Ω = 6 rad/s` to exercise the centrifugal-
stiffening contribution to the FEM stiffness matrix. The lowest three
flapwise frequencies are tabulated in Bir, G. S. (2009),
AIAA 2009-1035 ("Blades and Towers Modal Analysis Code (BModes):
Verification of Blade Modal Analysis Capability"), Table 3a, which
itself transcribes Wright et al. (1982), *Vibration Modes of
Centrifugally Stiffened Beams*. The Bir 2009 row at Ω = 6 rad/s
gives:

| Flap mode | ω (rad/s)  |
| --------: | ---------: |
|         1 |     7.360  |
|         2 |    26.809  |
|         3 |    66.684  |

## Files

| File                              | Purpose                          |
| --------------------------------- | -------------------------------- |
| `rotating_blade.bmi`              | Main BModes input                |
| `rotating_blade_sec_props.dat`    | Distributed section properties   |

## Physical parameters

| Parameter   | Value      | Units    | Notes                              |
| ----------- | ---------: | -------- | ---------------------------------- |
| `radius`    | 31.623     | m        | blade length                       |
| `hub_rad`   | 0.0        | m        | flexible from r = 0                |
| `rot_rpm`   | 57.29578   | rpm      | exactly Ω = 6 rad/s                |
| `hub_conn`  | 1          | —        | cantilevered root                  |
| `mass_den`  | 100.0      | kg/m     | uniform mass per length            |
| `EI_flap`   | 1.0 × 10⁸  | N·m²     | flapwise bending stiffness         |
| `EI_edge`   | 1.0 × 10⁹  | N·m²     | edgewise bending stiffness         |
| `GJ`        | 1.0 × 10⁵  | N·m²     | torsional stiffness                |

`rot_rpm = 6 · 30 / π = 57.29578 rpm` is exactly Ω = 6 rad/s, so the
FEM output can be compared directly to Bir's table without unit
conversion.

The blade is the same as Case 01 — only `rot_rpm` changes. Comparing
the two cases lets you read off the centrifugal-stiffening lift on
each flap mode directly:

| Mode    | ω at Ω = 0 (Case 01) | ω at Ω = 6 (this case) | lift   |
| ------: | -------------------: | ---------------------: | -----: |
| flap-1  |                3.516 |                 7.360  | + 109 % |
| flap-2  |               22.034 |                26.809  | +  22 % |
| flap-3  |               61.697 |                66.684  | +   8 % |

The 1st flap mode picks up far more centrifugal stiffening than the
higher modes (Wright's classical Southwell-coefficient observation:
`ω² = ω₀² + Kₙ · Ω²`, with `K₁ ≈ 1.193` and `Kₙ` decreasing rapidly
for higher `n`).

## Expected frequencies (pyBmodes vs Bir 2009 Table 3a)

| Mode | Type           | ω_FEM (rad/s) | ω_ref (rad/s) | f_FEM (Hz) |
| ---: | -------------- | ------------: | ------------: | ---------: |
|    1 | flap-1         |        7.3604 |         7.360 |     1.1714 |
|    2 | lag-1          |       11.4207 |             — |     1.8177 |
|    3 | flap-2         |       26.8089 |        26.809 |     4.2668 |
|    4 | flap-3         |       66.6841 |        66.684 |    10.6131 |
|    5 | lag-2          |       71.0796 |             — |    11.3127 |

The lag modes (2 and 5) are not in Bir's flap-only table; they sit at
roughly `√10 ≈ 3.16` × the corresponding flap frequency because
`EI_edge = 10 · EI_flap`.

## How to run

```python
from pybmodes.models import RotatingBlade
blade = RotatingBlade(
    "cases/sample_inputs/03_rotating_uniform_blade/rotating_blade.bmi"
)
modal = blade.run(n_modes=8)
print(modal.frequencies[:5])
# [1.1714, 1.8177, 4.2668, 10.6131, 11.3127]
```

The same blade can be swept across rotor speeds with
:func:`pybmodes.campbell.campbell_sweep` to produce a Campbell
diagram showing how each flap mode lifts with Ω.

## Validation reference

Bir (2009) AIAA 2009-1035 Table 3a (analytical column), which
transcribes Wright et al. (1982). The full row sweep `Ω ∈ {0, 1, 2,
…, 12}` rad/s is exercised by the project's whitebox test
[`tests/fem/test_rotating_uniform_blade.py`](../../../tests/fem/test_rotating_uniform_blade.py)
which gates each mode at < 0.5 % relative error.

Citations:

- Wright, A. D.; Smith, C. E.; Thresher, R. W.; Wang, J. C. (1982).
  *Vibration Modes of Centrifugally Stiffened Beams*. ASME *Journal of
  Applied Mechanics*, Vol. 49, March 1982.
- Bir, G. S. (2009). *Blades and Towers Modal Analysis Code (BModes):
  Verification of Blade Modal Analysis Capability*. AIAA-2009-1035 /
  NREL/CP-500-44749. Table 3a.
