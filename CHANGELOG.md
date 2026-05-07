<!-- markdownlint-disable MD024 -->

# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Self-contained walkthrough notebook (`notebooks/walkthrough.ipynb`) demonstrating the full public API on synthetic uniform blade and tower cases.
- Inline synthetic-fixture helpers (`tests/_synthetic_bmi.py`) that build `.bmi` and section-property files at test time, with numbers freely chosen by the project author.
- Closed-form analytical regression suite for the cantilever-with-tip-mass configuration (`tests/fem/test_uniform_tower_analytical.py`), validating the FEM solver against the Blevins (1979) / Karnovsky & Lebed (2001) frequency equation across tip-mass ratios from 0 to 5.
- Comprehensive unit-test coverage of FEM building blocks: boundary conditions, generalised eigensolver, non-dimensionalisation, mode-shape extraction, polynomial-fit edge cases, and parser primitives.

### Changed

- **FEM core vectorisation.** Element-matrix construction is now vectorised over both Gauss points and elements via `numpy.einsum`, replacing the per-element Python loop. Inner double sums over Gauss points and local DOF pairs collapse to a single tensor contraction. Net speedup is ~2–3× on small cases and ~1.6× on larger meshes where the dense `eigh` solve dominates.
- **Validation contract.** Switched from bundled reference data files to published closed-form formulas as the source of truth for FEM accuracy. The reference list now contains only textbook material (Euler-Bernoulli cantilever frequency series; Blevins / Karnovsky cantilever-with-tip-mass equation).
- Test count expanded from 159 to 197.
- README rewritten to drop external-program framing; Windows + conda install instructions added.

### Removed

- All bundled reference-data files under `tests/data/` (`.bmi`, `.dat`, `.out`). The library is now a self-contained Python implementation validated only against analytical references.
- `examples/` directory — the demo scripts depended on the removed reference data; the walkthrough notebook supersedes them.

### Fixed

- `patch_dat` no longer demotes CRLF line endings to LF on Windows OpenFAST `.dat` files. The writer used to rstrip the matched line and rewrite it with a hardcoded `\n`, silently mixing endings; now the original line ending is captured per line and re-applied, with `newline=''` set on both read and write to defeat Python's universal-newline translation.
- Removed README claim of distributed-hydrodynamic-added-mass support for monopile towers — `distr_m` is parsed but not yet wired into the mass matrix; only distributed soil stiffness flows through to the FEM assembly.

## [0.1.0] — 2025-04-22

### Added

- Rotating blade modal analysis (flap, edge, torsion modes)
- Onshore tower analysis — cantilevered and tension-wire supported
- Offshore tower analysis — floating spar (`hub_conn=2`) and bottom-fixed monopile (`hub_conn=3`)
- Constrained 6th-order polynomial mode shape fitting (C₂ + C₃ + C₄ + C₅ + C₆ = 1)
- In-place patching of OpenFAST ElastoDyn `.dat` files
- Initial validation against bundled reference cases (later removed in the independence pass)
