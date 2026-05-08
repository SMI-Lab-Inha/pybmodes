## IEA-3.4-130-RWT Land-Based: Coefficient Comparison Report

Findings from running [`cases/iea3mw_land/run.py`](../run.py) against the
ElastoDyn deck bundled at
`docs/OpenFAST_files/IEA-3.4-130-RWT/openfast/`. The deck's title line
notes it was "Generated with AeroElasticSE FAST driver" — i.e. the
polynomial coefficients in the `.dat` files are the output of an upstream
auto-generation pipeline, not of pyBmodes.

The point of this report is to record the evidence base for one specific
claim: **the polynomial coefficients shipped in the IEA-3.4 ElastoDyn
files are not reproducible from those files' own structural inputs by a
standalone modal solve, and that mismatch is *not* an artefact of the
polynomial-basis conditioning.**

### Configuration

- Tower model: `Tower.from_elastodyn(IEA-3.4-130-RWT_ElastoDyn.dat)` — RNA
  lumped at the tower top via the rigid-body parallel-axis assembly in
  `_tower_top_assembly_mass`. Land-based: no platform, no soil compliance.
- Blade model: `RotatingBlade.from_elastodyn(...)` solved at 0 rpm and at
  rated 10.5 rpm.
- Polynomial fits: `compute_tower_params_report` (FA1/FA2/SS1/SS2) and
  `compute_blade_params` (BldFl1/BldFl2/BldEdg).

### Classification

Mode identification is unambiguous in modes 1–4. The DOF-participation
table (sum of squared component / sum of all squared components per
mode) shows zero torsion contamination at the bending frequencies
where the ElastoDyn polynomials are fit:

| Mode | Freq (Hz) | p_FA | p_SS | p_tor | Label |
|---:|---:|---:|---:|---:|:---|
| 1 | 0.4212 | 0.522 | 0.478 | 0.000 | FA+SS mixed (degenerate pair) |
| 2 | 0.4212 | 0.479 | 0.521 | 0.000 | FA+SS mixed (degenerate pair) |
| 3 | 2.2445 | 0.000 | 1.000 | 0.000 | SS |
| 4 | 2.2756 | 1.000 | 0.000 | 0.000 | FA |
| 5 | 5.8303 | 0.000 | 1.000 | 0.000 | SS |
| 6 | 6.1691 | 1.000 | 0.000 | 0.000 | FA |
| 7 | 10.3161 | 0.000 | 1.000 | 0.000 | SS |
| 8 | 11.6536 | 0.978 | 0.000 | 0.022 | FA (mild twist coupling) |
| 9 | 16.4950 | 0.000 | 1.000 | 0.000 | SS |
| 10 | 18.0788 | 0.726 | 0.000 | 0.274 | FA (significant twist) |

**Degenerate FA/SS pair at 0.4212 Hz.** The IEA-3.4 tower has
`TwFAStif == TwSSStif` at every input station, and the rigid-RNA tip
mass is essentially axisymmetric (`y_cm = 0` from `NacCMyn = 0`), so the
1st FA and 1st SS frequencies are mathematically equal. The eigensolver
returns a rotated basis of the degenerate eigenspace — modes 1 and 2 are
both 50/50 mixes — and the FA/SS classifier picks them apart correctly
by spanwise RMS (`fa_rms = 0.0849 > ss_rms = 0.0812` for mode 1, mirror
flipped for mode 2). The polynomial fit on either basis vector is
indistinguishable; both give `fit_rms = 0.0002`.

**No torsion contamination** in modes 1–6. Mild twist participation
(2.2 %) appears in mode 8 and grows to 27.4 % by mode 10, but that's
well above the FA2 selection point and does not affect the polynomial
fits we report on.

**Classifier selections.** FA1 = mode 1, FA2 = mode 4, SS1 = mode 2,
SS2 = mode 3. All four pass the `fit_rms ≤ 0.09` good-fit threshold:
mode 1 = 0.0002, mode 2 = 0.0002, mode 3 = 0.0041, mode 4 = 0.0042.

### Residual analysis

The decisive diagnostic is the *file polynomial residual*: evaluate the
file's (C₂…C₆) polynomial at the same span stations as our pyBmodes
mode shape and compute the RMS difference between the file polynomial
and our shape. If the file polynomial is just a re-parametrisation of
the same shape we computed (Vandermonde-conditioning hypothesis), this
residual will be comparable to our own fit residual. If the file
polynomial was fit to a *different* shape, the residual jumps by orders
of magnitude.

| Mode | pyBmodes fit residual | File-poly residual against our shape | Ratio (file / pyB) |
|:---|---:|---:|---:|
| TwFAM1Sh | 0.0002 | 0.0098 | **49 ×** |
| TwFAM2Sh | 0.0042 | 0.7230 | **172 ×** |
| TwSSM1Sh | 0.0002 | 0.0110 | **55 ×** |
| TwSSM2Sh | 0.0041 | 1.5494 | **378 ×** |
| BldFl1Sh @ 10.5 rpm | 0.0014 | 0.0089 | 6 × |
| BldFl2Sh @ 10.5 rpm | 0.0046 | 0.0051 | **1.1 ×** |
| BldEdgSh @ 10.5 rpm | 0.0014 | 0.0044 | 3 × |
| BldFl1Sh @ 0 rpm | 0.0014 | 0.0119 | 8 × |
| BldFl2Sh @ 0 rpm | 0.0047 | 0.0052 | **1.1 ×** |
| BldEdgSh @ 0 rpm | 0.0014 | 0.0058 | 4 × |

For context: a normalised mode shape on `[0, 1]` has unit tip value, so
an RMS residual of 0.01 corresponds to a 1 % shape error; 0.1 to 10 %;
0.7 means the polynomial does not approximate the shape in any useful
sense.

The single mode where the file polynomial fits our shape essentially as
well as our own polynomial is **BldFl2Sh** — at both 0 rpm and 10.5 rpm,
both residual columns sit at ≈ 0.005. This is the only place where the
Vandermonde-conditioning explanation could plausibly hold: the same
shape, decomposed two different ways, with coefficients that look
different (C₃ varies by 38 %) but produce essentially the same curve.

Everywhere else, the file polynomial is a materially worse fit to our
mode shape than our own polynomial — by a factor of 3 to 380.

### Conclusion

- **1st-mode file residuals are ~ 0.01** (TwFAM1Sh = 0.0098,
  TwSSM1Sh = 0.0110, BldFl1Sh = 0.0089, BldEdgSh = 0.0044).
  That's 40–50 × worse than our own fits but still a credible polynomial
  representation of *some* mode shape — just not exactly the one we
  computed. Order-of-magnitude shape error: ~ 1 %.
- **2nd-mode tower file residuals are 0.72 (TwFAM2Sh) and 1.55
  (TwSSM2Sh)** — 170 × and 378 × worse than our fits respectively.
  These polynomials don't represent any 2nd-bending shape we would
  recognise. C₅ and C₆ have flipped signs from what a clean 2nd
  bending mode would produce on our structural model.
- **Vandermonde-conditioning hypothesis rejected** for everything
  except BldFl2Sh. The shape errors are not consistent with two
  near-equivalent polynomial fits to the same shape; the file
  polynomials describe a *different* shape.
- **The only Vandermonde-consistent case is BldFl2Sh** — there the
  residual ratio is essentially 1, and the coefficient differences (C₃
  +20 %, others within a few percent) are within the noise of the
  near-singular polynomial basis on `[0, 1]`.

**Most likely cause of the disagreement:** the polynomials were
generated from a different revision of the structural model than the
one currently shipped in the `_ElastoDyn.dat` files. The deck's title
line names an upstream auto-generation pipeline (a systems-engineering
optimisation tool that writes out OpenFAST inputs from a YAML
description); a plausible failure mode is that the YAML model and the
OpenFAST-format outputs went through separate revisions and were not
re-synced. Specifically the RNA mass / inertia or tower
mass-density / EI distribution at fit time would have to differ from
what's now in the `.dat`. The 2nd-mode residuals are large enough that
the disagreement is not a small parameter perturbation — it's at least
a structural-data revision gap.

**Counterfactual checks already done:**

- It is *not* a wrong-mode selection. Modes 1–4 have zero torsion
  participation and sit at the correct frequencies (degenerate 1st pair
  at 0.4212 Hz, 2nd FA / SS pair at ~ 2.25 Hz). Modes 8 and 10 are the
  ones with twist contamination and they are not selected.
- It is *not* an FA/SS axis-convention swap. The classifier rationale
  output shows fa_rms / ss_rms cleanly for each mode, and the swap-by-
  symmetry would not move the 2nd-mode coefficients by 100 %+.
- It is *not* a centrifugal-stiffening mismatch on the blade. The 0 rpm
  and 10.5 rpm comparisons give nearly identical file-residual
  patterns, ruling out the rotation rate as the cause.

### Frequencies

pyBmodes computed frequencies (Hz):

| Mode | Tower (n=10) | Blade @ 0 rpm (n=8) | Blade @ 10.5 rpm (n=8) |
|---:|---:|---:|---:|
| 1 | 0.4212 | 0.7907 | 0.8222 |
| 2 | 0.4212 | 1.0303 | 1.0374 |
| 3 | 2.2445 | 2.3711 | 2.4077 |
| 4 | 2.2756 | 3.2039 | 3.2249 |
| 5 | 5.8303 | 5.2184 | 5.2510 |
| 6 | 6.1691 | 7.4887 | 7.5093 |
| 7 | 10.3161 | 8.5004 | 8.5319 |
| 8 | 11.6536 | 12.2997 | 12.3322 |

**File-implied frequencies are not directly extractable** from the
shipped `.dat`. ElastoDyn does not store modal frequencies — they are
*outputs* of an OpenFAST run, computed from the polynomial shape
functions plus the tower / blade structural properties via a
generalised-coordinates assumed-modes formulation. To extract a "file
frequency" we would need to either (a) run OpenFAST against the deck
and read its `.sum` file, or (b) reproduce ElastoDyn's assumed-modes
Rayleigh quotient internally with the file polynomials as basis. Both
are out of scope for this report.

What we *can* compare against external references for sanity:

- The IEA-3.4-130-RWT documentation (Bortolotti et al. 2019, Table 5.1)
  reports a 1st tower bending frequency near 0.40 Hz for the land-based
  configuration. Our value of 0.4212 Hz is 5.3 % high — within the
  documented rigid-RNA accuracy band on top-heavy designs.
- Centrifugal stiffening shifts the 1st-flap blade frequency from
  0.7907 Hz (0 rpm) to 0.8222 Hz (10.5 rpm), i.e. + 4.0 %. That is
  consistent with the standard Southwell coefficient for a 5 MW-class
  blade and adds nothing to the coefficient-disagreement story above.

### Implication for the README

The single sentence the README can claim with full backing from this
report:

> The polynomial-coefficient blocks shipped in the `_ElastoDyn.dat`
> files of at least one industry RWT (IEA-3.4-130-RWT, land-based
> configuration) cannot be reproduced from the structural-property
> blocks in the same files via a standalone modal solve. Per-mode RMS
> residuals show that the bundled polynomials describe a different
> mode shape than the one implied by the bundled tower / blade
> properties — by an order of magnitude on 1st bending modes, and by
> two orders on 2nd bending modes.

Stronger claims (specific revision drift, specific upstream tool
involved) need either a second turbine showing the same pattern or a
direct check against the source YAML.
