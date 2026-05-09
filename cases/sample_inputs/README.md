# pyBmodes sample input files

Public-facing, pyBmodes-authored `.bmi` and section-properties `.dat`
files corresponding to validated analytical reference cases. Every
file in this tree was hand-written for the project — no third-party
data is bundled, and every numeric value is reproducible from the
cited textbook formula or peer-reviewed paper named in the case's
own README.

These are the input files to copy / adapt as a starting point when
you want to:

- understand what a minimal, well-formed `.bmi` deck looks like;
- spot-check that pyBmodes parses your local install correctly;
- benchmark a structural FEM change against an analytical reference;
- carve out a small reproducer when filing a bug or a feature
  request.

The four cases together exercise the four boundary conditions
pyBmodes supports (`hub_conn ∈ {1, 4}` here; `hub_conn ∈ {2, 3}` are
covered by the bundled offshore certtest decks under
`docs/BModes/docs/examples/` when you have those locally), the tower
+ blade beam-type split, the rotating + non-rotating split, and the
tip-mass + no-tip-mass split.

## Index — analytical-reference cases (this directory)

| Case | Title                                          | Beam   | Ω (rad/s) | Tip mass | BC          | Ref              |
| ---- | ---------------------------------------------- | ------ | --------: | -------- | ----------- | ---------------- |
| [01](01_uniform_blade/) | Uniform isotropic cantilever blade  | blade  |       0   | none     | cantilever  | Euler-Bernoulli  |
| [02](02_tower_topmass/) | Uniform tower with concentrated top mass | tower  |       0   | μ = 1.0  | cantilever  | Blevins (1979)   |
| [03](03_rotating_uniform_blade/) | Rotating uniform blade           | blade  |       6   | none     | cantilever  | Wright 1982 / Bir 2009 Table 3a |
| [04](04_pinned_free_cable/) | Rotating pinned-free cable       | blade  |      10   | none     | pinned-free | Bir 2009 Eq. 8   |

## Index — industry-turbine cases

The [`industry_turbines/`](industry_turbines/) sub-directory ships
BMI samples for six widely-cited industry reference turbines
(NREL 5MW land + OC3 monopile, IEA-3.4 / IEA-10 / IEA-15 / IEA-22),
generated from the open-literature structural data via
[`industry_turbines/build.py`](industry_turbines/build.py). See
[`industry_turbines/README.md`](industry_turbines/README.md) for the
full list, modelling assumption (cantilever from TowerBsHt with RNA
lumped at the top), per-case 1st-FA frequency vs published reference,
and the path to the flexible-pile-+-tower combined-cantilever solve
when monopile foundation flexibility matters.

Each case directory contains:

- a `.bmi` main input;
- a companion section-properties `.dat`;
- a `README.md` documenting the physical parameters, the analytical
  reference value(s) the FEM is expected to match, and a one-line
  copy-paste Python invocation that runs pyBmodes on the case.

## Verification

`verify.py` (in this directory) runs pyBmodes on every sample case
and asserts that the lowest few computed frequencies match the
analytical reference to within 1 % relative error. From the repo
root::

    set PYTHONPATH=D:\repos\pyBModes\src
    python cases/sample_inputs/verify.py

Output is one PASS / FAIL line per case plus a one-line summary at
the end. No external data is required — the verification uses only
the closed-form formulas cited in each case's README.

## Conventions

All sample `.bmi` files follow the canonical line-ordered format —
section headers, value-then-label data lines, two-line block
separators — that pyBmodes' parser shares with BModes JJ (Fortran),
so the same files can also be fed to BModes JJ for cross-solver
verification. They differ from BModes' own example decks under
`docs/BModes/docs/examples/` only in that the parameter values here
come from peer-reviewed analytical references rather than industry
reference turbines, so they're safe to redistribute under pyBmodes'
MIT licence.

## See also

- [`tests/fem/`](../../tests/fem/) — the project's whitebox FEM
  validation tests, which build the FEM matrices directly without
  going through the BMI parser. Each case here mirrors one of the
  whitebox tests with the same physical parameters.
- [`docs/BModes/docs/examples/`](../../docs/BModes/docs/examples/) —
  BModes JJ's own example decks (`OC3Hywind.bmi`, `CS_Monopile.bmi`)
  for offshore + floating + rigid-monopile configurations. These are
  external-data: gitignored under the project's "Independence
  stance"; clone the upstream BModes repo locally to see them.
- [`tests/_synthetic_bmi.py`](../../tests/_synthetic_bmi.py) — the
  programmatic `.bmi` writer used by tests for in-tmp_path round-trip
  fixtures. Useful as a starting point if you need to generate sample
  decks with parameter sweeps rather than by hand.
