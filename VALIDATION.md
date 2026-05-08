<!-- markdownlint-disable MD013 -->
# pyBmodes validation matrix

This document is the **single structured source of truth** for what
pyBmodes is validated against, at what tolerance, with what worst
observed error, and which test file enforces it. Prose-heavy reports
elsewhere in the repo (`cases/ECOSYSTEM_FINDING.md`,
`reference_decks/VALIDATION_SUMMARY.md`, the README's *Validation*
section) refer back to this matrix.

The validation work is split into **two tracks** with different
metrics:

- **Track A — frequency accuracy.** Eigenvalue agreement against an
  external reference (closed-form formula, published table, or the
  BModes Fortran solver). Tolerance is a relative frequency error.
- **Track B — coefficient consistency.** The mode-shape polynomial
  blocks shipped in OpenFAST ElastoDyn `.dat` files. Tolerance is
  the RMS residual of the polynomial against the FEM mode shape.

## Track A — frequency-accuracy cases

| Case | Source / reference | Quantity | Tolerance | Worst observed | Test file | Needs external data |
| --- | --- | --- | ---: | ---: | --- | :---: |
| Uniform Euler-Bernoulli cantilever, modes 1-3 | $\beta_n L = \{1.875,\,4.694,\,7.855,\ldots\}$ (textbook) | flap frequency | < 0.5 % | < 0.005 % | [`tests/fem/test_cantilever.py`](tests/fem/test_cantilever.py) | no |
| Same, mode 4 | textbook | flap frequency | < 1 % | (within tol) | [`tests/fem/test_cantilever.py`](tests/fem/test_cantilever.py) | no |
| Same, mode 5 | textbook | flap frequency | < 2 % | (within tol) | [`tests/fem/test_cantilever.py`](tests/fem/test_cantilever.py) | no |
| Hermite cubic mesh-convergence | analytical $h^4$ rate | error ratio at h × 2 | > 5 (i.e. ≥ 5×) | confirmed | [`tests/fem/test_cantilever.py`](tests/fem/test_cantilever.py) | no |
| Cantilever + tip mass, $\mu \in [0, 5]$ | Blevins (1979) / Karnovsky & Lebed (2001) frequency equation | 1st bending frequency | < 0.5 % | (within tol) | [`tests/fem/test_uniform_tower_analytical.py`](tests/fem/test_uniform_tower_analytical.py) | no |
| Rotating uniform blade, flap modes 1-3, Ω ∈ [0, 12] rad/s | Wright et al. (1982) — *Vibration Modes of Centrifugally Stiffened Beams*, J. Appl. Mech. (transcribed from Bir 2009 Table 3a) | flap frequency | < 0.5 % | (within tol) | [`tests/fem/test_rotating_uniform_blade.py`](tests/fem/test_rotating_uniform_blade.py) | no |
| Rotating uniform blade + tip mass, flap modes 1-2, Ω ∈ [0, 12] rad/s | Wright et al. (1982) (Bir 2010 Table 5) | flap frequency | < 0.1 % | (within tol) | [`tests/fem/test_rotating_blade_with_tip_mass.py`](tests/fem/test_rotating_blade_with_tip_mass.py) | no |
| Inextensible spinning cable, flap modes 1-3, Ω ∈ [2, 30] rad/s | Bir 2009 §III.B / Eq. 8: $\omega_k = \Omega\sqrt{k(2k-1)}$ | flap frequency | < 0.5 % | (within tol) | [`tests/fem/test_rotating_cable.py`](tests/fem/test_rotating_cable.py) | no |
| BModes v3.00 CertTest **Test01** — non-uniform rotating blade, 60 rpm | BModes Fortran solver `.out` | per-mode frequency, modes 1-6 | < 1 % | < 0.005 % across 20 modes | [`tests/test_certtest.py`](tests/test_certtest.py) | yes |
| BModes v3.00 CertTest **Test01**, modes 7+ | BModes Fortran solver `.out` | per-mode frequency | < 3 % | (within tol) | [`tests/test_certtest.py`](tests/test_certtest.py) | yes |
| BModes v3.00 CertTest **Test02** — rotating blade + tip mass + offsets | BModes Fortran solver `.out` | per-mode frequency, modes 1-6 / 7+ | < 1 % / < 3 % | < 0.005 % across 20 modes | [`tests/test_certtest.py`](tests/test_certtest.py) | yes |
| BModes v3.00 CertTest **Test03** — 82.4 m tower with top mass + c.m. offsets | BModes Fortran solver `.out` | per-mode frequency, modes 1-6 / 7+ | < 1 % / < 3 % | < 0.005 % across 20 modes | [`tests/test_certtest.py`](tests/test_certtest.py) | yes |
| BModes v3.00 CertTest **Test04** — Test03 + tension-wire support | BModes Fortran solver `.out` | per-mode frequency, modes 1-6 / 7+ | < 1 % / < 3 % | < 0.005 % across 20 modes | [`tests/test_certtest.py`](tests/test_certtest.py) | yes |
| **CS_Monopile** — *NREL 5MW Reference Turbine* on the *OC3 Monopile* (`hub_conn = 3`, soft monopile, mooring stiffness) | BModes JJ (v1.03.01) `.out` | per-mode frequency, first 10 modes | < 0.01 % | < 0.005 % | [`tests/test_certtest.py`](tests/test_certtest.py) | yes |
| **OC3Hywind** — *NREL 5MW* on the *OC3 Hywind* floating spar (`hub_conn = 2`, full hydro + mooring + 6×6 platform inertia) | BModes JJ (v1.03.01) `.out` | per-mode frequency, first 9 modes | < 0.01 % | ≤ 0.0003 % across surge / sway / yaw / roll / pitch / heave + 1st-2nd tower bending | [`tests/test_certtest.py`](tests/test_certtest.py) | yes |
| Degenerate-pair resolver, symmetric tower | construction (`EI_FA == EI_SS`) | post-rotation FA / SS purity | $p_{\text{FA}}, p_{\text{SS}} > 0.99$ | (within tol) | [`tests/test_classifier.py`](tests/test_classifier.py) | no |
| IEA-3.4 modes 1-2 — degenerate-pair resolver fires no warning | IEA Wind Task 37 RWT deck | classifier verdict | no `RuntimeWarning` | clean | [`tests/test_classifier.py`](tests/test_classifier.py) | yes |

**Citations** (full author / year forms used in the table above):

- Blevins (1979). *Formulas for Natural Frequency and Mode Shape*. Krieger Publishing.
- Karnovsky & Lebed (2001). *Formulas for Structural Dynamics*. McGraw-Hill.
- Wright, Smith, Thresher & Wang (1982). *Vibration Modes of Centrifugally Stiffened Beams*. *Journal of Applied Mechanics*, Vol. 104, March 1982.
- Bir (2009). *Blades and Towers Modal Analysis Code (BModes): Verification of Blade Modal Analysis Capability*. AIAA 2009-1035.
- Bir (2010). *Verification of BModes: Rotary Beam and Tower Modal Analysis Code*. NREL/CP-500-47953.
- Jonkman, Butterfield, Musial & Scott (2009). *Definition of a 5-MW Reference Wind Turbine for Offshore System Development*. NREL/TP-500-38060.
- Jonkman & Musial (2010). *Offshore Code Comparison Collaboration (OC3) for IEA Wind Task 23*. NREL/TP-5000-48191.
- Jonkman (2010). *Definition of the Floating System for Phase IV of OC3*. NREL/TP-500-47535.
- Bortolotti, Tarrés, Dykes, Merz, Sethuraman, Verelst & Zahle (2019). *IEA Wind TCP Task 37: Systems Engineering in Wind Energy — WP2.1 Reference Wind Turbines*. NREL/TP-5000-73492.

## Track B — coefficient-consistency cases

The metric here is the RMS residual of an ElastoDyn polynomial block
evaluated against the FEM-computed mode shape produced by the deck's
own structural inputs. See `pybmodes.elastodyn.validate` for the
implementation. Per-block verdicts: PASS < 0.01, WARN < 0.10, FAIL ≥ 0.10.

| Case | Source / reference | Quantity | Tolerance | Worst observed (file_rms) | Test file | Needs external data |
| --- | --- | --- | ---: | ---: | --- | :---: |
| Validator on stock NREL 5MW r-test deck — TwFAM2Sh / TwSSM2Sh | OpenFAST r-test (commit `dd5feaaa`) | file polynomial RMS vs pyBmodes mode shape (detection target) | verdict = FAIL | 5.08 (TwFAM2Sh) / 5.90 (TwSSM2Sh) | [`tests/test_validate.py`](tests/test_validate.py) | yes |
| Validator on same deck — 1st tower modes + blade modes | OpenFAST r-test | file polynomial RMS | verdict = PASS | 0.0081 / 0.0075 / 0.0020-0.0090 | [`tests/test_validate.py`](tests/test_validate.py) | yes |
| Patch round-trip on staged NREL 5MW copy | self (post-`patch_dat`) | file polynomial RMS after patching | verdict = PASS, ratio ≈ 1.0 | ratio drift < 1 % (text-precision artefact) | [`tests/test_validate.py`](tests/test_validate.py) | yes |
| `reference_decks/nrel5mw_land/` patched deck | committed deliverable | per-block verdict | all PASS | all PASS | [`tests/test_reference_decks.py`](tests/test_reference_decks.py) | no (artefact tracked) |
| `reference_decks/nrel5mw_oc3monopile/` patched deck | committed deliverable | per-block verdict | all PASS | all PASS | [`tests/test_reference_decks.py`](tests/test_reference_decks.py) | no |
| `reference_decks/iea34_land/` patched deck | committed deliverable | per-block verdict | all PASS | all PASS | [`tests/test_reference_decks.py`](tests/test_reference_decks.py) | no |
| Pre-patch sanity — at least one `before_patch.txt` shows FAIL or WARN | committed before-patch reports | per-deck overall verdict | ≥ 1 FAIL/WARN | 3/3 FAIL | [`tests/test_reference_decks.py`](tests/test_reference_decks.py) | no |

## What "needs external data" means

`yes` rows skip in the default `pytest` run (they carry the
`integration` marker). Run them locally with `pytest -m integration`
once you have the upstream sources:

- `docs/OpenFAST_files/r-test/` — clone of the OpenFAST regression-test
  corpus (any recent commit; pyBmodes was last validated against
  `dd5feaaa`).
- `docs/OpenFAST_files/IEA-3.4-130-RWT/` — clone of the IEA Wind Task 37
  reference-turbines repository.
- `docs/BModes/CertTest/` — BModes v3.00 CertTest reference outputs.
- `docs/BModes/docs/examples/` — the bundled `CS_Monopile.bmi` and
  `OC3Hywind.bmi` example decks plus their BModes JJ `.out` files.

These directories are gitignored (see `.gitignore`); the contributor
adds them locally. CI runs the integration step with
`continue-on-error: true` so a missing-data skip doesn't fail the
build.

## Reproducing every row

Track A (no external data):

```bash
pytest tests/fem/test_cantilever.py
pytest tests/fem/test_uniform_tower_analytical.py
pytest tests/fem/test_rotating_uniform_blade.py
pytest tests/fem/test_rotating_blade_with_tip_mass.py
pytest tests/fem/test_rotating_cable.py
pytest tests/test_classifier.py        # 3 of 4 pass without external data
```

Track A (external data needed):

```bash
pytest tests/test_certtest.py -m integration
```

Track B:

```bash
pytest tests/test_validate.py -m integration   # validator round-trip
pytest tests/test_reference_decks.py            # always self-contained
```

Or run the full matrix at once:

```bash
pytest -m ""        # default + integration combined
```
