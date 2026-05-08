<!-- markdownlint-disable MD024 -->

# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Bir (2010) NREL/CP-500-47953 reproduction suite.** Four new things reproduce the canonical BModes verification paper:
  - **Closed-form regression tests against Wright et al. 1982 / Bir 2009 Tables 2a + 3a and Bir 2010 Table 5.** `tests/fem/test_rotating_uniform_blade.py` gates flap modes 1-3 of a uniform rotating cantilever blade (L = 31.623 m, m = 100 kg/m, EI_flap = 1e8, EI_lag = 1e9, GJ = 1e5) at ≤ 0.5 % across Ω ∈ {0..12} rad/s. `tests/fem/test_rotating_blade_with_tip_mass.py` gates flap modes 1-2 of the same blade plus a μ = 1 tip mass against Bir 2010 Table 5 at ≤ 0.1 %. The latter wires up the previously-missing tip-mass centrifugal-tension contribution to `cfe`; without it the rotating-tip-mass frequencies are 14-50 % low at moderate Ω. `tests/fem/test_rotating_cable.py` gates the inextensible spinning cable (Bir 2009 §III.B / Eq. 8: ω = Ω·√(k(2k−1))) on the new `hub_conn=4` BC at ≤ 0.5 %. Closes the "Centrifugal-stiffening validation" roadmap item.
  - **`hub_conn=4` (pinned-free) tower-base BC.** Locks axial, lag/flap deflections, and twist at the root while leaving the bending slopes FREE. Matches the implicit BC of Bir 2009's Legendre-polynomial cable solution. Implemented in `pybmodes.fem.boundary` (`build_connectivity`, `n_free_dof`, `active_dof_indices`).
  - **`pybmodes.plots.bir_mode_shape_plot` and `bir_mode_shape_subplot`.** Plot mode shapes with modal displacement on the x-axis (mass-normalised, *not* unit-tip) and normalised height $z/H$ on the y-axis, matching Bir 2010 Figs 4, 5a, 5b, 6a-c, 8. Optional horizontal annotation lines (Mean Sea Level, Mud Line) for offshore configurations and dashed coupling overlays for hybrid modes.
  - **Three case-study scripts** (`cases/bir_2010_land_tower/`, `cases/bir_2010_monopile/`, `cases/bir_2010_floating/`) reproduce Bir's figures using the cert-test decks. The scripts render Fig 4 (synthetic uniform cantilever, no head mass), Fig 5a / 5b (Test03 = land tower with head mass), Fig 8 (CS_Monopile with MSL marker), and Fig 6a / 6b / 6c (OC3Hywind floating spar). The monopile case classifies hybrid modes (e.g. CS_Monopile mode 4 is a 2nd-FA + twist coupled hybrid) with explicit "(+ F-A part)" labels. Frequencies on the cert-test decks are already validated against BModes JJ at ≤ 0.01 %; these PNGs are the visual companion.
- Self-contained walkthrough notebook (`notebooks/walkthrough.ipynb`) demonstrating the full public API on synthetic uniform blade and tower cases.
- Inline synthetic-fixture helpers (`tests/_synthetic_bmi.py`) that build `.bmi` and section-property files at test time, with numbers freely chosen by the project author.
- Closed-form analytical regression suite for the cantilever-with-tip-mass configuration (`tests/fem/test_uniform_tower_analytical.py`), validating the FEM solver against the Blevins (1979) / Karnovsky & Lebed (2001) frequency equation across tip-mass ratios from 0 to 5.
- Comprehensive unit-test coverage of FEM building blocks: boundary conditions, generalised eigensolver, non-dimensionalisation, mode-shape extraction, polynomial-fit edge cases, and parser primitives.
- **OpenFAST deck adapters.** New classmethod constructors that consume OpenFAST input files directly:
  - `Tower.from_elastodyn(main_dat_path)` — parses the ElastoDyn main file plus the tower file referenced via `TwrFile` and the first blade file via `BldFile(1)` (the latter only to lump rotor mass into the tower-top assembly). Lands the *NREL 5MW Reference Turbine* (Jonkman et al. 2009) tower modal solve within ~ 1 % of the published target.
  - `Tower.from_elastodyn_with_subdyn(main_dat_path, subdyn_dat_path)` — splices a SubDyn pile geometry below the ElastoDyn tower into a single combined cantilever clamped at the SubDyn reaction joint. Designed for OC3-style fixed-base monopiles (no soil flexibility, no hydrodynamic added mass).
  - `RotatingBlade.from_elastodyn(main_dat_path)` — synthesises a BMI-equivalent from the ElastoDyn main + blade files, including centrifugal stiffening from the deck's `RotSpeed`.
  - `Tower.from_bmi(bmi_path)` — explicit classmethod alias of `Tower(...)` for symmetry with the other constructors.
- **`pybmodes.io.elastodyn_reader`** module — full ElastoDyn `.dat` parser + canonical writer + adapter helpers. Three dataclasses (`ElastoDynMain`, `ElastoDynTower`, `ElastoDynBlade`); label-based scanning robust across FAST v8 / OpenFAST v3+ format drift; semantic round-trip via `write_elastodyn_*` (parse → emit → re-parse equality with `np.allclose` rtol = 1e-12). Adapter helpers `to_pybmodes_tower` and `to_pybmodes_blade` synthesise BMI / SectionProperties in memory; `run_fem` accepts an optional pre-built `SectionProperties` so adapter paths skip the on-disk round-trip.
- **`pybmodes.io.subdyn_reader`** module — minimal SubDyn parser + pile/tower combiner (joints, members, circular cross-section properties, base reaction joint, interface joint). Sufficient for OC3-style monopiles; non-circular sections and SSI files are not parsed.
- **Cross-solver certification suite (`tests/test_certtest.py`).** Six certification cases now compared against the BModes Fortran reference solver (Bir 2010) at strict tolerances:
  - BModes v3.00 CertTest Test01-04 (rotating blades, cantilever tower with top mass, tension-wire-supported tower) at < 1 % / < 3 % per-mode.
  - `CS_Monopile.bmi` — *NREL 5MW Reference Turbine* on the *OC3 Monopile* (Jonkman & Musial 2010) at 0.01 %, < 0.005 % observed.
  - `OC3Hywind.bmi` — *NREL 5MW* on the *OC3 Hywind* floating spar (Jonkman 2010) at 0.01 %, ≤ 0.0003 % observed across the first 9 modes.
- **Degenerate-eigenpair resolver (`pybmodes.elastodyn.params._rotate_degenerate_pairs`).** Detects consecutive modes whose relative frequency gap is below 1e-4 and rotates the pair inside its 2D eigenspace so the first comes out FA-pure and the second SS-pure. Handles the symmetric-tower case where the eigensolver returns an arbitrary basis of the degenerate subspace.
- **Polynomial-fit conditioning instrumentation.** `PolyFitResult.cond_number` reports the 2-norm condition number of the reduced design matrix solved by `lstsq`. `compute_tower_params_report` emits a `RuntimeWarning` above 1e4 (WARN) and a stronger one above 1e6 (FAIL) so basis-conditioning artefacts on poorly-sampled meshes don't pass silently.
- **Case studies** (`cases/` directory). Three exploratory case directories — `nrel5mw_land/` (*NREL 5MW Reference Turbine*, Jonkman et al. 2009), `iea3mw_land/` (*IEA-3.4-130-RWT*, Bortolotti et al. 2019, IEA Wind Task 37), and `nrel5mw_monopile/` (*NREL 5MW* on rigid OC3-style monopile) — each with a `run.py` that prints a coefficient-comparison table (`coefficients.txt`) and writes mode-shape PNGs. `cases/ECOSYSTEM_FINDING.md` is the cross-deck summary documenting that the polynomial-coefficient blocks shipped in industry `_ElastoDyn.dat` files are typically not reproducible from the structural-property blocks in the same files.

### Changed

- **FEM core vectorisation.** Element-matrix construction is now vectorised over both Gauss points and elements via `numpy.einsum`, replacing the per-element Python loop. Inner double sums over Gauss points and local DOF pairs collapse to a single tensor contraction. Net speedup is ~2–3× on small cases and ~1.6× on larger meshes where the dense `eigh` solve dominates.
- **Validation contract.** Switched from bundled reference data files to published closed-form formulas as the source of truth for FEM accuracy. The reference list now contains only textbook material (Euler-Bernoulli cantilever frequency series; Blevins / Karnovsky cantilever-with-tip-mass equation), supplemented by cross-solver certification against BModes (see "Added" above).
- README rewritten to drop external-program framing; Windows + conda install instructions added.
- **Tower-top mass kinematic coupling for offshore / free-base towers.** `nondim_tip_mass` now uses the BMI's literal `cm_loc` / `cm_axial` pair directly when `hub_conn ∈ {2, 3}`. The previous code path applied the cantilever convention (which folds `cm_axial` into the internal `cm_loc` lever arm and drops the literal `cm_loc`) regardless of `hub_conn`, which on OC3 Hywind effectively dropped the `cm_axial` bending lever arm and made the 1st tower-bending pair too stiff — 0.4997 / 0.5087 Hz instead of BModes' 0.4816 / 0.4908 Hz (~ 3.8 % high). The cantilever path is preserved for `hub_conn = 1` because the four BModes v3.00 CertTest cases depend on the older convention to pass at 6-digit precision.
- **Eigensolver dispatch for asymmetric platform support.** OC3 Hywind has genuinely asymmetric platform-support contributions after the rigid-arm transformation. `solver.py` now detects asymmetry in the assembled `K` / `M` and routes those cases through `scipy.linalg.eig` (general dense eigensolver), matching BModes. Symmetric problems — all cantilever cases plus the soft-monopile CS_Monopile case — still use `scipy.linalg.eigh`.
- **PlatformSupport detection** in `models/_pipeline.py` keys off `isinstance(bmi.support, PlatformSupport)` rather than `bmi.tow_support == 2`. Both BMI dialects (legacy `tow_support = 2` and inline `tow_support = 1` with a numeric draft follow-up) get normalised to `PlatformSupport` by the parser; the new check picks up both consistently and also handles hand-built `BMIFile` instances that don't set `tow_support`.
- **CLAUDE.md naming convention clarified.** Citable published reference turbines (*NREL 5MW Reference Turbine*, *OC3 Monopile* / *OC3 Hywind*, *IEA-3.4-130-RWT* and the wider IEA Wind Task 37 family) are now explicitly named in validation tables, README content, and case-study reports — they're standard citations in the field. Restraint on ambient name-dropping in source comments and commit messages is unchanged.
- Test count expanded from 159 to 364 across this release window (159 → 197 with the analytical-validation pass; 197 → 252 with the cross-solver certification + offshore work; 252 → 338 with the Bir 2010 reproduction suite + the new `hub_conn=4` cable test; 338 → 364 with the coefficient-validator + reference-decks deliverable + professional-polish pass that landed test markers, public-API declaration, the unified plot style, and per-module mypy strict overrides).

### Removed

- All bundled reference-data files under `tests/data/` (`.bmi`, `.dat`, `.out`). The library is now a self-contained Python implementation validated only against analytical references and locally-supplied (uncommitted) BModes / OpenFAST decks.
- `examples/` directory — the demo scripts depended on the removed reference data; the walkthrough notebook supersedes them.

### Fixed

- **OC3 Hywind 1st tower-bending pair** — was running ≈ 3.7-3.8 % HIGH versus the BModes JJ reference (pyBmodes 0.4997 / 0.5087 Hz vs BModes 0.4816 / 0.4908 Hz). The fix combined the three changes listed under "Changed" above: the literal `cm_loc` / `cm_axial` interpretation for `hub_conn = 2`, the asymmetric-eigensolver routing, and the `PlatformSupport`-keyed pipeline. Post-fix, OC3 Hywind matches BModes JJ across the first 9 modes to **0.0000 – 0.0003 %** — > 30× headroom under the 0.01 % cert tolerance. CS_Monopile (which has zero `hydro_M` and a symmetric support matrix) was already exact; it remains so.
- **`patch_dat` no longer demotes CRLF line endings to LF** on Windows OpenFAST `.dat` files. The writer used to rstrip the matched line and rewrite it with a hardcoded `\n`, silently mixing endings; now the original line ending is captured per line and re-applied, with `newline=''` set on both read and write to defeat Python's universal-newline translation.
- Removed README claim of distributed-hydrodynamic-added-mass support for monopile towers — `distr_m` is parsed but not yet wired into the mass matrix; only distributed soil stiffness flows through to the FEM assembly.

## [0.1.0] — 2025-04-22

### Added

- Rotating blade modal analysis (flap, edge, torsion modes)
- Onshore tower analysis — cantilevered and tension-wire supported
- Offshore tower analysis — floating spar (`hub_conn=2`) and bottom-fixed monopile (`hub_conn=3`)
- Constrained 6th-order polynomial mode shape fitting (C₂ + C₃ + C₄ + C₅ + C₆ = 1)
- In-place patching of OpenFAST ElastoDyn `.dat` files
- Initial validation against bundled reference cases (later removed in the independence pass)
