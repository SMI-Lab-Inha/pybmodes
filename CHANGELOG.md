<!-- markdownlint-disable MD024 -->

# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed (second post-1.0 static-review pass)

- **`MooringSystem.from_moordyn` — silent malformed-row drops.** The
  LINE TYPES / POINTS / LINES section parsers used to `continue` on
  rows that failed `len(parts) < N` or per-column `float` / `int`
  parsing, which turned a typo into an incomplete mooring model with
  no diagnostic. The parsers now raise `ValueError` with the section
  name, the offending row text, and the source file path on rows
  that look like data but fail strict parsing. Rows that look like
  column-name or units headers are still skipped — `_looks_like_header_row`
  identifies them by checking whether every first-four token parses
  as numeric (data) or whether the entire row is parenthesised
  (units). Pass-2 static review.
- **`MooringSystem._split_sections` — hardcoded 2-row header
  assumption.** The splitter previously skipped exactly two rows
  after every section divider. A MoorDyn variant shipped with only
  one header row (column names but no units line) had its first
  data row silently eaten. The splitter now inspects each post-
  divider row and only skips it if `_looks_like_header_row` flags
  it; the moment a data-looking row appears, the inspect-and-skip
  loop stops. Handles 0 / 1 / 2 header rows. Pass-2 static review.
- **SubDyn adapter — bare `StopIteration` on missing reaction /
  interface joint IDs.** A SubDyn deck whose `BASE REACTION` or
  `INTERFACE JOINTS` block referenced a joint ID absent from
  `STRUCTURE JOINTS` produced an uninformative `StopIteration`
  from the `next(...)` generator expression. Now routed through a
  `_lookup_joint(subdyn, joint_id, role)` helper that raises
  `ValueError` naming the missing ID, the role ("reaction" or
  "interface"), the source file, and the known joint IDs — matches
  the existing `_circ_prop_for` error-message style. Pass-2 static
  review.

### Changed

- **`cases/ECOSYSTEM_FINDING.md` — refreshed OC3 spar footnote.**
  The footnote on the polynomial-comparison table said pyBmodes had
  no `from_elastodyn` path for floating decks because parsing
  HydroDyn + MoorDyn into a 6 × 6 PlatformSupport was "out of scope".
  Stale since `Tower.from_elastodyn_with_mooring` landed before 1.0.
  The case-study table continues to use the BMI deck (the
  cantilever `hub_conn = 1` basis is the only one ElastoDyn's `SHP`
  ansatz can represent) — the footnote now spells out *why* rather
  than implying the path doesn't exist. Pass-2 static review.
- **`src/pybmodes/_examples/sample_inputs/README.md` — broken
  `../../docs/BModes` link.** The relative link to
  `../../docs/BModes/docs/examples/` didn't resolve from the
  packaged-wheel location, and the target is gitignored under the
  Independence stance anyway. Replaced with plain text that names
  the path and explains the local-only / upstream-clone story
  inline. Pass-2 static review.
- **`notebooks/walkthrough.ipynb` — closing-section "sparse eigsh
  on the roadmap" claim refreshed.** The sparse shift-invert
  `scipy.sparse.linalg.eigsh(sigma=0, mode='normal')` path landed in
  `pybmodes.fem.solver` before 1.0 and activates automatically on
  symmetric problems with a small mode-count request. The closing
  text now describes that and points at
  `scripts/benchmark_sparse_solver.py` for the timing study. Pass-2
  static review.

### Added

- **`wheel-smoke` CI job.** New matrix job (Python 3.11 + 3.12) that
  builds the wheel via `python -m build`, installs it into a fresh
  `python -m venv`, asserts `importlib.metadata.version("pybmodes")`
  matches the `[project] version` line in `pyproject.toml`,
  exercises `pybmodes examples --copy` end-to-end, and runs the
  sample verifier against the installed wheel. Catches packaging
  regressions the editable-install `test` job can't see: missing
  `package_data` entries, wheel-build failures, `sys.path` leaks
  from the source tree, and version drift between `pyproject.toml`
  and the dist-info metadata.
- **`tests/test_notebooks.py`.** Three tests covering the two
  bundled walkthrough notebooks:
  - `notebooks/walkthrough.ipynb` (synthetic, default suite) —
    executes every cell via `nbclient` and asserts no
    `CellExecutionError`. Was previously not in CI; a refactor
    that silently broke a cell would ship without warning (and
    one did — see *Fixed*).
  - `cases/iea15_umainesemi_walkthrough.ipynb` — split into two
    paths. The default-suite test asserts that with the upstream
    OpenFAST decks absent, the first code cell raises a friendly
    `FileNotFoundError` carrying the documented "Clone the
    upstream IEA-15-240-RWT" hint. Previously this was just a
    design contract; now it's a tested invariant. The
    `@integration`-marked counterpart executes every cell when
    the upstream data IS present.
- **New `[notebook]` optional extra** in `pyproject.toml` —
  `nbclient` / `nbformat` / `ipykernel`, test-time-only deps for
  the headless notebook execution path above. CI installs
  `.[dev,plots,notebook]` so the notebook tests run in the default
  suite.

### Fixed

- **`notebooks/walkthrough.ipynb` — missing `PALETTE` and `OI`
  references.** The `plot_mode_shapes_paper` and
  `plot_fit_quality_paper` helpers in the setup cell referenced
  `PALETTE[1:]` and `OI['verm']` / `OI['blue']` / `OI['green']` but
  never imported / defined them — silent `NameError` since the
  apply_style refactor in commit `00ff4c7`. The new notebook-
  execution CI step caught this on the first run. Setup cell now
  imports `from pybmodes.plots.style import PALETTE, apply_style`
  and defines `OI = {'verm': PALETTE[1], 'blue': PALETTE[2], 'green': PALETTE[3]}`.

- **Default-suite tests for three previously integration-only modules.**
  `tests/test_coords.py` covers the `pybmodes.coords` 6-DOF naming
  contract (`DOF_NAMES` ↔ `DOF_INDEX` agreement). `tests/test_elastodyn_writer.py`
  exercises the ElastoDyn writer's parse → emit → re-parse fixed point
  against the bundled NREL 5MW reference deck under
  `src/pybmodes/_examples/reference_decks/nrel5mw_land/` (no upstream-
  data dependency). `tests/test_subdyn_reader.py` exercises the SubDyn
  parser and the `SubDynCircProp` derived properties against a
  synthetic snippet emitted to `tmp_path`. Coverage on
  `src/pybmodes/coords.py` rose from 0 → 100 %, `io/_elastodyn/writer.py`
  from 6 → 82 %, and `io/subdyn_reader.py` from 0 → 71 % in the default
  pytest run.

### Changed

- **IEA-15 UMaineSemi walkthrough relocated `notebooks/ → cases/`.** The
  walkthrough at `notebooks/iea15_umainesemi_walkthrough.ipynb` depends
  on upstream OpenFAST decks under `docs/OpenFAST_files/` (gitignored
  per the Independence stance), so a fresh clone got a notebook that
  errored on the first cell. The `notebooks/` directory is contractually
  self-contained (`notebooks/walkthrough.ipynb` runs entirely on
  inline synthetic cases); data-dependent walkthroughs belong under
  `cases/` alongside the existing `run.py` case studies. Moved to
  `cases/iea15_umainesemi_walkthrough.ipynb` and the `sys.path`
  prologue rewritten to walk up from CWD looking for `src/pybmodes`
  so the notebook works regardless of where Jupyter launches.

## [1.0.0] — 2026-05-13

This is the stable 1.x baseline. Semver-protected public API
enumerated in [`src/pybmodes/__init__.py`](src/pybmodes/__init__.py)
and the *Public API* section of [`README.md`](README.md). The
following constructors / dataclasses / functions are now frozen
across 1.x minor releases: every name in the *Public API* list,
including the floating-platform additions originally documented as
"Provisional API" in 0.4.0 — `pybmodes.mooring` (`LineType`,
`Point`, `Line`, `MooringSystem`), `pybmodes.io` (`HydroDynReader`,
`WamitReader`, `WamitData`), and `Tower.from_elastodyn_with_mooring`.

### Planned for 1.1+

- **Additional floating reference-turbine samples** — sub-cases 08
  OC4 DeepCwind semi (Robertson 2014), 09 IEA-15 UMaine VolturnUS-S
  semi (Allen 2020), 10 IEA-22 floating. Needs a public ``.bmi``
  writer for ``PlatformSupport`` blocks; the writer landed
  in-development inside 1.0 (round-trips the existing OC3 sample
  through ``read_bmi → write → read``) but is reverted from the 1.0
  surface pending sample-authoring follow-up. Non-breaking under
  semver so this is safe to slot into a 1.1 minor release.

### Highlights for 1.0

- **Validated FEM core** — 1.0 ships frequency-accuracy validation
  on six BModes-JJ reference decks (Test01–04 land / tension-wire,
  CS_Monopile, OC3 Hywind) at ≤ 0.01 % cert tolerance plus the full
  closed-form analytical regression suite (Euler-Bernoulli
  cantilever, cantilever + tip mass, Wright 1982 / Bir 2009
  rotating uniform blade, Bir 2010 Table 5 rotating + tip mass,
  Bir 2009 Eq. 8 pinned-free cable). The matrix is enumerated in
  [`VALIDATION.md`](VALIDATION.md).
- **ElastoDyn-deck adapters** — `Tower.from_elastodyn` /
  `Tower.from_elastodyn_with_subdyn` / `Tower.from_elastodyn_with_mooring`
  / `RotatingBlade.from_elastodyn` cover the land + monopile +
  floating configurations.
- **OpenFAST polynomial-coefficient workflows** — six CLI
  subcommands (`validate` / `patch` / `campbell` / `batch` /
  `report` / `examples`) plus `pybmodes.elastodyn` Python API for
  programmatic use. Six patched reference decks ship under
  `src/pybmodes/_examples/reference_decks/` (3 fixed + 3 floating).
- **Quasi-static mooring linearisation** — `pybmodes.mooring`
  parses MoorDyn v1 / v2 and produces a 6 × 6 stiffness matrix
  reproducing the OC3 Hywind surge stiffness to better than 0.01 %
  vs Jonkman 2010 Table 5-1 (41,180 N/m).
- **WAMIT output reader** — `pybmodes.io.WamitReader` extracts
  `A_inf` / `A_0` / `C_hst` from a HydroDyn-pointed WAMIT `.1` /
  `.hst` pair, validated against the IEA-15-240-RWT-UMaineSemi
  upstream files at 1 % tolerance.
- **Bundled examples ship inside the wheel** — 4 analytical-
  reference BMIs, 7 RWT samples, 6 patched ElastoDyn decks. The
  `pybmodes examples --copy <dir>` CLI vendors them out from any
  install (source, editable, or wheel).
- **CI required on master** — branch-protection ruleset gates merges
  on `test (3.11)` + `test (3.12)` green; the merge model went
  through a one-time conversion to PR-required flow in 0.4.x.

### Fixed

- **`WamitReader` — upper-triangle-only WAMIT outputs are now mirrored.** Some WAMIT runs write only the upper triangle of a symmetric matrix (`A_inf`, `A_0`, `C_hst`); the parser previously assigned only `C[i, j]` per row and left the transpose at zero, silently losing half of the off-diagonal coupling for those files. The parsers for `.1` and `.hst` now mirror non-zero entries into the corresponding `[j, i]` slot after reading, preserving explicit zeros from fully-written matrices. Pre-1.0 review (pass 2).
- **`WamitReader` / `HydroDynReader` — Fortran D-exponent notation.** WAMIT and HydroDyn output writers occasionally emit Fortran-style `1.234D+02` instead of `1.234E+02`. The parsers previously used `float(value)` directly and silently dropped rows with `D` / `d` exponents (`ValueError` swallowed in the row loop). A shared `_parse_fortran_float` helper now normalises both forms across `pybmodes.io.wamit_reader` and `pybmodes.models.tower._scan_platform_fields`. Pre-1.0 review (pass 2).
- **`_scan_platform_fields` — raise on malformed critical scalars.** A typo or unsupported numeric token in `PtfmMass` / `PtfmRIner` / `PtfmPIner` / `PtfmYIner` previously fell through to the default `0.0`, producing a physically meaningless floating model with no hard failure. Those four scalars now raise `ValueError` on parse failure (after the Fortran-D normalisation pass); the non-critical fields (CM offsets, additional-stiffness scalars) still fall back to `0.0`. Pre-1.0 review (pass 2).
- **`campbell_sweep` — defensive bound on returned mode count.** The blade-sweep loop indexed `f_step[k]` for `k in range(n_modes)` after slicing whatever the eigensolver returned. On the rare general-eig fallback path (floating platforms with non-symmetric `K` at certain rotor speeds, dropping NaN eigenvalues) this could index past the slice. The loop now raises `RuntimeError` with a clear message naming the offending rotor speed if the solver returns fewer than `n_modes` rows. Pre-1.0 review (pass 2).
- **`Tower.from_elastodyn_with_mooring` — BMI radius / draft length mismatch.** The cantilever adapter sets `bmi.radius = TowerHt - TowerBsHt` (the flexible-tower length), but the floating BMI convention pairs `radius = TowerHt` with `draft = -TowerBsHt` so that `radius + draft` recovers the flexible length after the nondim step. The 0.4.0 `from_elastodyn_with_mooring` set the signed draft without overriding the radius, so for OC3 the flexible length came out as `TowerHt - 2·TowerBsHt = 67.6 m` instead of the intended 77.6 m. The constructor now overrides `bmi.radius = TowerHt` to match the bundled `OC3Hywind.bmi` convention. Pre-1.0 review.
- **`Tower.from_elastodyn_with_mooring` — `Ptfm*Stiff` scalars folded into `mooring_K`.** ElastoDyn carries six additional platform linear-stiffness scalars (`PtfmSurgeStiff` / `PtfmSwayStiff` / `PtfmHeaveStiff` / `PtfmRollStiff` / `PtfmPitchStiff` / `PtfmYawStiff`) that act on top of HydroDyn / MoorDyn contributions. The OC3 spec carries the delta-line crowfoot's yaw spring via `PtfmYawStiff` (~ 9.83e7 N·m/rad) and it isn't in the MoorDyn `.dat`; the 0.4.0 constructor ignored these so the coupled OC3 yaw frequency came out ~ 8× low. They are now scanned alongside the geometry/inertia scalars and added to the diagonal of `mooring_K`. Pre-1.0 review.
- **`_scan_platform_fields` — Fortran D-exponent notation.** ElastoDyn `.dat` scalars may be written as `7.466D+06` rather than `7.466E+06`; the scanner used `float(value)` directly and silently dropped any field that hit the Fortran form, producing a zero in `PtfmMass` / `PtfmRIner` / etc. The scanner now normalises `D` / `d` to `E` before parsing. Pre-1.0 review.
- **`pybmodes.campbell._solve_tower_sweep` — restore caller's `rot_rpm`.** The tower-only Campbell branch was setting `tbmi.rot_rpm = 0.0` without restoring it, mutating the caller's BMI. The blade-sweep path already used `try / finally`; the tower path now mirrors it. Cosmetic — tower modes are rotor-speed-independent so the mutation didn't change any computed value, but the API hygiene matters. Pre-1.0 review.
- **`Tower.from_elastodyn_with_mooring` — i_matrix double parallel-axis.** The previous build added `M·dz²` to `i_matrix[3,3]` / `i_matrix[4,4]` to "transfer the platform inertia from CM to body origin," but `pybmodes.fem.nondim.nondim_platform` already applies the rigid-arm transform itself using `cm_pform - draft` — so the parallel-axis term was being counted twice (~6e10 kg·m² for OC3, ~3.6e9 for IEA-15 UMaine), overstating roll/pitch inertia. The same bug also wrote spurious cross-coupling terms into `i_matrix[0,4]` / `i_matrix[1,3]`. The `i_matrix` is now stored AT THE CM exactly as the BMI parser expects (bottom-right 3×3 of the rotational block, no coupling). Reported by Pre-1.0 review.
- **`Tower.from_elastodyn_with_mooring` — BMI sign convention for `cm_pform` / `draft` / `ref_msl`.** The previous build stored these in ElastoDyn signed-z (so OC3 came out with `cm_pform = -89.9155`, `draft = 0`), but the downstream BMI consumer reads them in BModes file convention (positive distance below MSL for `cm_pform` and `ref_msl`; signed `draft` with negative = above MSL). For OC3 the corrected values match the canonical `OC3Hywind.bmi` deck: `draft = -10`, `cm_pform = 89.9155`, `ref_msl = 0`. Pulls `TowerBsHt` from the ElastoDyn main file to compute `draft = -TowerBsHt`. Reported by Pre-1.0 review.
- **`MooringSystem.from_moordyn` — MoorDyn v1 LINE PROPERTIES column order.** Older MoorDyn v1 line-properties rows use the column order `ID LineType UnstrLen NumSegs NodeAnch NodeFair`, not v2's `ID LineType AttachA AttachB UnstrLen ...`. The previous parser tried to read v2 columns unconditionally, so v1 rows like `1 main 902.2 20 1 4` failed with a ValueError on `int("902.2")` and got silently skipped — leaving the system with zero or incorrect lines. The parser now probes v2 column order first and validates AttachA/AttachB as known point IDs; on mismatch it falls back to v1 column order. `Point.__post_init__` also accepts MoorDyn v1 attachment aliases (`Fix` → `Fixed`, `Connect` → `Free`, `Body` / `Coupled` → `Vessel`, `Anchor` → `Fixed`). Reported by Pre-1.0 review.

### Added

- **`pybmodes.mooring`** — new module with `LineType`, `Point`, `Line`, and `MooringSystem`. Solves the extensible elastic catenary per line (Jonkman 2007 NREL/TP-500-41958 Appendix B equations B-1 / B-2 fully-suspended; B-7 / B-8 with `CB = 0` for the seabed-contact branch; damped Newton on `(H, V_F)` with analytical 2×2 Jacobian, `tol = 1e-6` m, MaxIter = 100). Multi-line platform restoring force assembled from world-frame fairlead positions through 3-2-1 intrinsic rotations; central-difference 6×6 linearisation about an arbitrary or zero offset, trans-rot off-diagonals symmetrised. `MooringSystem.from_moordyn(...)` parses MoorDyn v1 (`CONNECTION`) and v2 (`POINT`) `.dat` files; OC3 Hywind surge stiffness reproduced to better than 0.01 % vs Jonkman 2010 Table 5-1 (41,180 N/m).
- **`Tower.from_elastodyn_with_mooring(main_dat, moordyn_dat, hydrodyn_dat=None)`** — new classmethod that assembles a free-free (`hub_conn = 2`) floating-tower BMI with a populated `PlatformSupport` block: mooring K from MoorDyn, hydrodynamic A_inf + C_hst from HydroDyn / WAMIT (optional), platform inertia from ElastoDyn `PtfmMass` / `PtfmRIner` etc. with parallel-axis transfer from CM to body origin. End-to-end OC3 Hywind solve hits the 1st tower-bending FA pair within 1.2 % of Jonkman 2010's published 0.482 Hz. For ElastoDyn polynomial-coefficient generation use the standard cantilever `Tower.from_elastodyn` instead — that path is unchanged.
- **`pybmodes.io.wamit_reader`** — new module with `WamitReader`, `WamitData`, and `HydroDynReader`. Parses the WAMIT v7 `.1` (added mass / radiation damping) and `.hst` (hydrostatic restoring) output files an OpenFAST floating-platform deck points at via the HydroDyn `PotFile` value, redimensionalises them per the WAMIT v7 convention (`ρ · L^k` for added mass, `ρ · g · L^k` for hydrostatic stiffness — exponents pick up +1 per rotational DOF in the index pair), and returns SI 6 × 6 `A_inf` / `A_0` / `C_hst` matrices in a `WamitData` dataclass. `HydroDynReader` surfaces the four scalars needed to drive `WamitReader` from a HydroDyn `.dat` (`WAMITULEN`, `PotMod`, `PotFile`, `PtfmRefzt`) and chains them via `read_platform_matrices()`. `WtrDens` and `Gravity` defaults are ISO sea-water values since HydroDyn ≥ v2.03 delegates those to the paired SeaState input file. Path resolution handles surrounding quotes, Windows-style backslashes, and relative-vs-absolute `PotFile` values. Integration tests under `tests/test_wamit_reader.py` validate against the upstream IEA-15-240-RWT-UMaineSemi WAMIT files at the 1 % tolerance.

## [0.4.0] — 2026-05-11

### Added

- **`pybmodes examples --copy <dir> [--kind all|samples|decks] [--force]`** — new CLI subcommand. Vendors the bundled `sample_inputs/` and/or `reference_decks/` trees from the installed package into a user-supplied directory so a `pip install pybmodes` user can seed a working tree without keeping a git clone around. Resolves bundle paths relative to `pybmodes.__file__`. Destination conflicts return exit code 2 unless `--force` is set. Tests under `tests/test_examples_cli.py`.
- **Example bundles ship inside the wheel.** The previously top-level `cases/sample_inputs/` (analytical references + 7 RWT samples) and `reference_decks/` (6 patched ElastoDyn decks — 3 fixed + 3 floating) trees were moved into `src/pybmodes/_examples/sample_inputs/` and `src/pybmodes/_examples/reference_decks/` and declared as `package-data` in `pyproject.toml`. Every wheel and editable install now carries the trees alongside the package; the `pybmodes examples --copy` CLI uses this to work regardless of installation source. Delivers the *Repo assets accessible from a wheel install* item on the README *1.0 milestone* list.

### Changed

- **Breaking — bundle paths moved into the package tree.** Anything that hard-coded `repo_root / "cases" / "sample_inputs"` or `repo_root / "reference_decks"` now needs `repo_root / "src" / "pybmodes" / "_examples" / "sample_inputs"` (resp. `... / "_examples" / "reference_decks"`). The cleanest replacement is `pybmodes.cli._resolve_examples_root() / "sample_inputs"` (resp. `... / "reference_decks"`), which works for both source and wheel installs. Tests, scripts, and docs in the repo were updated mechanically. The bundles themselves are byte-identical to the 0.3.x payload — only the on-disk location changed.
- **`pybmodes report` no longer accepts `--rated-rpm`.** The flag was reserved / informational only in 0.3.0 and never surfaced in textual report output; it was removed for 0.4.0 to keep the CLI surface honest ahead of the 1.0 freeze. `pybmodes campbell --rated-rpm` (where the value is wired through to `plot_campbell`) is unchanged.

## [0.3.0] — 2026-05-11

### Added

- **Pre-solve sanity checks** (`pybmodes.checks`). New module shipping `check_model(model, n_modes=None) -> list[ModelWarning]` with eight gated checks: non-monotonic span stations (WARN), zero / negative mass density (ERROR), stiffness jumps > 5× between adjacent stations (WARN per FA + SS axis), EI_FA / EI_SS ratio outside `[0.1, 10]` (INFO), RNA mass > integrated tower mass (INFO), singular `PlatformSupport` 6×6 matrix (`cond > 1e10`, ERROR), `n_modes > 6 × n_nodes` (ERROR), polynomial-fit design-matrix condition number > 1e4 / 1e6 (WARN / ERROR, computed pre-solve from `bmi.el_loc`). `Tower.run()` / `RotatingBlade.run()` gain a keyword-only `check_model: bool = True` parameter; when True, WARN + ERROR findings emit `UserWarning`s and INFO findings stay silent. Internal validator / batch / patch service paths pass `check_model=False` to avoid duplicate warnings.
- **Mode-by-mode comparison** (`pybmodes.mac`). New module with `mac_matrix(shapes_A, shapes_B) -> ndarray (n, m)` (the standard `MAC_ij = |φ_i·φ_j|² / ((φ_i·φ_i)(φ_j·φ_j))` formula on the concatenated `[flap_disp, lag_disp, twist]` vector), `compare_modes(result_A, result_B, label_A, label_B) -> ModeComparison` (full MAC + Hungarian-optimal pairing via `scipy.optimize.linear_sum_assignment` + per-pair `(f_B − f_A)/f_A × 100` shift), and `plot_mac(comparison, ax=None) -> Figure` heatmap with paired cells outlined in red. The Campbell tracker's existing `_mac_matrix` is now a thin wrapper around the public function.
- **Bundled analysis report** (`pybmodes.report`). `generate_report(result, output_path, format='md'|'html'|'csv', model=…, validation=…, check_warnings=…, tower_params=…, blade_params=…, campbell=…, source_file=…)` builds a structured eight-section report (model summary, assumptions, frequencies, mode classification with FA / SS / twist participation, polynomial coefficients with fit residuals, validation verdict, `check_model` warnings, Campbell sweep). HTML output is emitted directly as self-contained HTML5 with inline CSS; no runtime dependency on the `markdown` package. CSV output is narrower (frequencies + coefficient rows) and suitable for spreadsheet ingestion.
- **Result serialisation.** `ModalResult` gains `save(path)` / `load(path)` (compressed NPZ) and `to_json(path)` / `from_json(path)` (UTF-8 JSON with `"schema_version": "1"`); new optional fields `participation` (N × 3) and `fit_residuals` (`dict[str, float]`); metadata block (pyBmodes version, UTC timestamp, source-file path, best-effort git hash) auto-populated at save time. `CampbellResult` gains `save(path)` / `load(path)` (NPZ) and `to_csv(path)`; per-step MAC tracking confidence rides in the new `mac_to_previous` array (NaN on row 0 and on tower columns). Shared `pybmodes.io._serialize` helper captures the metadata dict; `git rev-parse --short HEAD` runs with a 2-second timeout and silently records `None` on any failure.
- **`pybmodes report` CLI subcommand.** `pybmodes report ElastoDyn.dat --format md|html|csv --out PATH [--campbell --rated-rpm R --max-rpm R --n-steps N --n-blade-modes N --n-tower-modes N] [--n-modes N] [--no-validate]` runs the modal solve, optional coefficient validation, and optional Campbell sweep on one deck and writes a single bundled report via `pybmodes.report.generate_report`.
- **`pybmodes batch` CLI subcommand.** `pybmodes batch ROOT [--kind elastodyn] [--out OUT/] [--n-modes N] [--validate] [--patch]` walks `ROOT` recursively for ElastoDyn main `.dat` files (two-stage filter: name heuristic plus parse confirmation), runs `validate` and / or `patch` per deck, writes a per-deck validation report under `OUT/`, and emits a `summary.csv` with columns `filename, overall_verdict, TwFAM2Sh_ratio, TwSSM2Sh_ratio, n_fail, n_warn`. Exits 0 when every deck reaches PASS or WARN; 1 if any FAIL or ERROR remains; 2 on unsupported `--kind` or missing `ROOT`.
- **Sparse shift-invert eigensolver path** in `pybmodes.fem.solver`. When the FEM matrices are effectively symmetric AND `ngd > 500` AND the caller asked for a small subset of modes, `solve_modes` routes through `scipy.sparse.linalg.eigsh(K, k=n_modes, M=M, sigma=0, which='LM', mode='normal')`. The `mode='normal'` choice (not `'buckling'`) is documented inline — `mode='buckling'` with `sigma=0` reduces to `OP = K⁻¹ K = I` (degenerate). On ARPACK non-convergence or any other failure the solver logs a `WARNING` and falls back to dense `eigh`. The selected path is announced as a `logging.INFO` message on the `pybmodes.fem.solver` logger. `scripts/benchmark_sparse_solver.py` reports 5-18× speedups across `n_elements ∈ {20, 50, 100, 200, 500}` and asserts sparse beats dense for `n_elements > 100` within a 10 % margin.
- **Torsion-contamination filter** in `_select_tower_family` (`pybmodes.elastodyn.params`). Tower family candidates whose modal-kinetic-energy torsion fraction `T_tor ≥ 0.10` are dropped from the selection. New helper `_kinetic_participation(shape) -> (T_FA, T_SS, T_tor)` computes per-mode energy fractions under the unit-mass approximation. `TowerFamilyMemberReport` gains `fa_participation` / `ss_participation` / `torsion_participation` / `torsion_rejected` fields; `TowerSelectionReport` gains `rejected_fa_modes` / `rejected_ss_modes`; `CoeffBlockResult` gains the four corresponding fields for tower blocks (NaN / empty defaults on blade blocks).
- **`pybmodes patch` safe-review modes.** `--dry-run` computes the patched coefficients and prints a per-block change summary without writing; `--diff` prints a PR-ready coefficient-only unified-diff format (`old → new` lines per block plus a per-block `RMS improvement: file_rms → pyb_rms (Nx better)` annotation) and also writes nothing; `--output-dir DIR` (alias `--output DIR`) writes the patched tower + blade `.dat` to `DIR/` instead of in-place, leaving the originals untouched. Combining `--output*` with `--dry-run` or `--diff` exits 2 with a clear "incompatible flags" message. The default in-place path with no `--backup` emits a one-line first-time-run hint pointing at `--dry-run --diff`; suppressed when any of `--backup`, `--output-dir`, `--dry-run`, or `--diff` is set.
- **Hungarian MAC tracking on the Campbell sweep.** The greedy `argmax(mac)` mode-pairing inside `_solve_blade_sweep` is replaced with a global Hungarian assignment via `scipy.optimize.linear_sum_assignment(maximize=True)`. The old `_greedy_assignment` symbol is kept as a deprecated alias. `CampbellResult.mac_to_previous` (new field, `(N, n_total_modes)`) exposes per-step tracking confidence — NaN on row 0 (no previous step) and on tower columns (tower modes don't change with rotor speed). `_solve_blade_sweep` now restores `bbmi.rot_rpm` via `try`/`finally` so the caller's BMI is unmutated by the sweep.
- **Campbell input-validation hardening.** `campbell_sweep` rejects NaN, inf, negative, and unsorted `omega_rpm` arrays with explicit `ValueError`s naming the offending element.
- **`pybmodes.io._elastodyn` sub-package.** The 1315-line `elastodyn_reader.py` is split into `types.py` (dataclasses), `lex.py` (line + token scanning helpers), `parser.py` (line-driven flavour parsers), `writer.py` (canonical re-emitters), and `adapter.py` (`to_pybmodes_tower` / `to_pybmodes_blade` plus the `_stack_*` / `_rotary_inertia_floor` / `_build_bmi_skeleton` / `_tower_top_assembly_mass` helpers). `pybmodes.io.elastodyn_reader` becomes a re-export shim — every public name plus the private helpers `pybmodes.io.subdyn_reader` depends on (`_rotary_inertia_floor`, `_stack_*_section_props`, `_tower_top_assembly_mass`, `_build_bmi_skeleton`, `_resolve_relative`) stay importable from the historical dotted path.
- **`scripts/audit_validation_claims.py`.** Parses every `tests/...` link in `VALIDATION.md`, asserts each path exists and contains at least one `def test_…` method. Runs as a required CI step alongside ruff and mypy, plus step 4.5 of `docs/RELEASE_CHECKLIST.md`. Gates "claim ahead of test" drift mechanically.
- **`docs/RELEASE_CHECKLIST.md`** — 11-step pre-tag verification sequence (default + integration pytest, ruff + mypy, sample-input verifier, validation-matrix audit, reference-deck regeneration, notebook smoke, case-script regen, version + CHANGELOG promotion, tag + push, GitHub Release, post-release sanity).
- **Three floating reference decks** under `reference_decks/`: `nrel5mw_oc3spar/` (NREL 5MW on the OC3 Hywind spar), `nrel5mw_oc4semi/` (NREL 5MW on the OC4 DeepCwind semi), `iea15mw_umainesemi/` (IEA-15-240-RWT on the UMaine VolturnUS-S semi). All generated via the existing `Tower.from_elastodyn(...)` cantilever path with no platform / hydro / mooring matrices in the modal eigenproblem — matching what ElastoDyn assumes at runtime. The IEA-15 UMaine case ends at `Overall: WARN` on `TwSSM2Sh` (1.6 % RMS, ratio 1.00) — an unavoidable representation limit of the constrained 6th-order polynomial form for that tower's section-property gradient, documented inline in the deck's `validation_report.txt`.
- **`pybmodes.io._elastodyn` cantilever path used for floating polynomial generation.** OpenFAST ElastoDyn source-code audit (May 2026) established that the polynomial ansatz `SHP = Σ c_i · (h/H)^(i+1)` algebraically forces `SHP(0) = SHP'(0) = 0` and the modal eigenproblem in `Coeff` (lines 5141-5267) integrates only the tower beam plus `TwrTpMass` — no platform / hydro / mooring matrices. Platform 6-DOF motion is added at runtime via the rigid-body sum (lines 7485-7540). The correct polynomial basis for every ElastoDyn configuration (land, monopile, floating) is therefore the clamped-base cantilever. Findings published in `reference_decks/FLOATING_CASES.md` (rewritten) and `cases/ECOSYSTEM_FINDING.md` (new "Floating-deck polynomials" section).

### Changed

- **CI step hardening.** The integration-test step no longer uses `continue-on-error: true`; instead it tolerates pytest exit code 5 ("no tests collected" — the normal case on the default GA runner) but fails the build on any other non-zero exit, so a custom workflow run that does have the data surfaces real failures. Ruff scope expanded from `src/ tests/` to `src/ tests/ scripts/`; user-facing workflow scripts (`build_reference_decks`, `audit_validation_claims`, `benchmark_sparse_solver`, `campbell`, `visualise_polynomial_comparison_*`) are gated alongside the package and tests. The validation-matrix audit (`scripts/audit_validation_claims.py`) is now a required CI step between tests and lint.
- **Validator service paths skip pre-solve checks.** `validate_dat_coefficients` passes `check_model=False` to its internal `Tower.run()` / `RotatingBlade.run()` calls. Without this, batch over real RWT decks emitted noisy stiffness-jump warnings on every blade (the wind-turbine blades genuinely have stiffness gradients at the tip transition — not a bug worth re-reporting once per deck × per validate call). Direct `Tower.run()` calls from user code keep the default-on behaviour.
- **Standard engineering-paper plot palette.** `pybmodes.plots.style.apply_style` switches from the MATLAB R2014b lines colour order to a black / red / blue / green / magenta / orange / cyan ordering. Black first so single-line plots read as line art; red / blue / green next for grayscale-printability. Backwards-compatibility alias `MATLAB_LINES` points at the new palette. All committed plot-producing scripts under `cases/` and `scripts/` were regenerated; the visualise-polynomial-comparison scripts had hardcoded MATLAB RGB triples that were also updated.
- **README documentation.** New top-level *Sample inputs* section listing the four analytical-reference cases and the seven RWT samples; new *Validation* + *Compatibility policy* / *1.0 milestone* sections; refreshed *Quick Start* with mode-comparison, save/load, report, and batch examples.

### Fixed

- **Float reference-deck polynomial coefficient story corrected.** The earlier `FLOATING_CASES.md` claim that floating tower polynomials need `Tower.from_bmi()` with `hub_conn=2` and a populated `PlatformSupport` block was technically wrong: that path solves the coupled tower + platform eigenproblem (correct for matching BModes JJ frequency, validated to ~ 0.0003 % per `test_certtest_oc3hywind`) but produces eigenvectors that include platform rigid-body motion — incompatible with ElastoDyn's `SHP(0) = SHP'(0) = 0` ansatz. The correct path is the cantilever solve `Tower.from_elastodyn(...)`, the same as the land and monopile sides.
- **CHANGELOG and README test-count drift.** Earlier README / CHANGELOG entries quoted hardcoded test counts that aged out on every test addition. Both files now defer to `VALIDATION.md` (the structured single-source-of-truth matrix) and `pytest --collect-only` for the current count.

### Known limitations

- **`iea15mw_umainesemi/TwSSM2Sh`** stays at `Overall: WARN` (1.6 % RMS) after patching — an unavoidable representation limit of the constrained 6th-order polynomial form for that tower's section-property gradient. The patched polynomial IS pyBmodes' best constrained fit (ratio = 1.00 against pyBmodes' own reference); improving it further would require a higher polynomial order or a piecewise basis, neither of which ElastoDyn's `SHP` ansatz supports. Auto-emitted explanatory footer ships in the deck's `validation_report.txt`.

## [0.2.0] — 2026-05-09

### Added

- **Sample-input library** (`cases/sample_inputs/`) — pyBmodes-authored, MIT-licensed `.bmi` and section-property `.dat` files committed to the repo. Four analytical-reference cases at the top level (uniform isotropic cantilever blade, uniform tower with concentrated top mass, rotating uniform blade per Wright 1982 / Bir 2009 Table 3a, rotating pinned-free cable per Bir 2009 Eq. 8) exercising all four `hub_conn` BCs plus tower / blade and rotating / non-rotating splits. `cases/sample_inputs/verify.py` runs all four against closed-form references at < 1 % RMS. Plus `reference_turbines/` sub-directory with seven RWT samples (NREL 5MW land + OC3 monopile + OC3 Hywind, IEA-3.4-130-RWT land, IEA-10-198-RWT / IEA-15-240-RWT / IEA-22-280-RWT monopile), each shipping tower BMI + blade BMI + per-side section-properties + per-turbine README; regenerable from upstream ElastoDyn decks via `reference_turbines/build.py`.
- **Three floating reference decks** under `reference_decks/`: NREL 5MW on the OC3 Hywind floating spar (`nrel5mw_oc3spar/`), NREL 5MW on the OC4 DeepCwind semi-submersible (`nrel5mw_oc4semi/`), and IEA-15-240-RWT on the UMaine VolturnUS-S semi (`iea15mw_umainesemi/`). All three generated via the existing `Tower.from_elastodyn(...)` cantilever path; post-patch validation reports shipped per case. Two reach `Overall: PASS`; the IEA-15 UMaine case reaches `Overall: WARN` on `TwSSM2Sh` (1.6 % RMS) — a representation limit of the constrained 6th-order polynomial form for that specific tower's section-property gradient, with an auto-emitted explanatory footer in the report.
- **Canonical `tow_support = 1` block on monopile BMI samples** in `cases/sample_inputs/reference_turbines/` — full CS_Monopile.bmi-format section structure (3×3 platform-inertia, 6×6 hydro_M / hydro_K / mooring_K, distributed added-mass + distributed elastic-stiffness, tension wires) with zero-valued matrices for the rigid-clamp combined pile + tower model. Layout is BModes-JJ-readable unmodified; the all-zero matrices add nothing to the eigenvalue problem.
- **OpenFAST ElastoDyn and WISDEM source-code audit** documenting the load-bearing question of which boundary condition ElastoDyn assumes for the tower modal basis at runtime. Findings published in `cases/ECOSYSTEM_FINDING.md` (new "Floating-deck polynomials" section) and `reference_decks/FLOATING_CASES.md` (rewritten end-to-end). Conclusion: the ElastoDyn polynomial ansatz `SHP = Σ_{i=1..PolyOrd-1} c_i · (h/H)^(i+1)` (`ElastoDyn.f90:2486-2495`) algebraically forces `SHP(0) = SHP'(0) = 0`; the modal eigenproblem in `Coeff` (lines 5141-5267) integrates only the tower beam plus `TwrTpMass` with no platform / hydro / mooring matrices; platform 6-DOF motion enters at runtime via the rigid-body sum (lines 7485-7540). Therefore the **correct polynomial basis for ALL ElastoDyn configurations — land, monopile, floating — is the clamped-base cantilever in the platform-attached frame**.
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
