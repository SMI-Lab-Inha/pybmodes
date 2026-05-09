<!-- markdownlint-disable MD013 -->
# Floating cases — the correct ElastoDyn polynomial basis is *cantilever*

For floating platforms, ElastoDyn polynomial coefficients
(`TwFAM1Sh`, `TwFAM2Sh`, `TwSSM1Sh`, `TwSSM2Sh`) must be derived
from a **cantilever** (`hub_conn = 1`) tower model with the RNA
lumped at the tower top — **NOT** from a platform-coupled floating
model.

## Why

ElastoDyn represents floating tower dynamics as a clamped-base
cantilever **in the platform-attached frame**, with platform 6-DOF
motion handled separately as independent generalised coordinates
(`Sg / Sw / Hv / R / P / Y`). Three independent code-level evidences
in OpenFAST `modules/elastodyn/src/ElastoDyn.f90` (main branch):

1. The polynomial ansatz in `SHP` evaluates
   `Σ_{i=1..PolyOrd-1} c_i · (h/H)^(i+1)` (lines 2486–2495). The
   lowest power is `Fract²`, so `SHP(0) = SHP'(0) = 0`
   *identically*. A free-free or pinned-pinned mode shape with
   non-zero base slope cannot be represented in this format.
2. The base node is hard-coded zero: `p%TwrFASF(:,0,0:1) = 0`,
   `p%TwrSSSF(:,0,0:1) = 0` (lines 5147–5148).
3. The internal tower modal eigenproblem (`Coeff` subroutine,
   lines 5141–5267) integrates `MTFA = TwrTpMass + ∫ ρA φ² dh`
   and `KTFA = ∫ EI φ'' φ'' dh + KTFAGrav`. **No** `PlatformMass`,
   **no** `hydro_K`, **no** `mooring_K`, **no** `i_matrix` enter
   this assembly. The only tip-end inertia is the scalar
   `TwrTpMass` (lumped RNA mass).

Platform 6-DOF motion enters the absolute tower kinematics via the
**rigid-body sum** (lines 7485–7540):

```text
v_T(J) = v_Z + ω_X × rZT(J) + Σ_k φ_k(h_J) · q̇_k
```

`Sg/Sw/Hv/R/P/Y` and the tower modal coordinates `q_TFA1 / q_TFA2 /
q_TSS1 / q_TSS2` are **independent** generalised coordinates;
platform motion does NOT appear as forcing on `q_TFA1`. Feeding
ElastoDyn polynomials that already encode platform-coupling
**double-counts** the platform restoring forces because ElastoDyn
re-derives those effects independently through the platform DOFs.

Same BC for land and floating — only the runtime treatment of the
clamp point differs (locked in Earth for land; rigidly attached to
the moving platform for floating).

## How

Floating-platform polynomial coefficients are generated with the
**existing** pyBmodes path — `Tower.from_elastodyn(...)` is
*already* the cantilever path. It clamps at `TowerBsHt` with the RNA
lumped at the top, ignores any platform / hydro / mooring matrices,
and produces exactly the basis ElastoDyn assumes. No flag is needed.

```python
from pybmodes.models import Tower
from pybmodes.elastodyn import compute_tower_params, patch_dat
from pybmodes.io.elastodyn_reader import read_elastodyn_main

main_path = "Floating_ElastoDyn.dat"
tower = Tower.from_elastodyn(main_path)        # cantilever, RNA at top, no platform
result = tower.run(n_modes=10)
params = compute_tower_params(result)

# patch_dat rewrites the *tower* .dat file (where the polynomial
# blocks live), not the main ElastoDyn .dat.
main = read_elastodyn_main(main_path)
patch_dat(main_path.replace("ElastoDyn.dat", main.twr_file), params)
```

No WAMIT files, no HydroDyn parsing, and no MoorDyn parsing are
required. The cantilever path is correct and self-contained — the
ElastoDyn `.dat` (plus the tower file it references) carries every
input needed.

## What about `Tower.from_bmi()` with `hub_conn = 2`?

`Tower.from_bmi("OC3Hywind.bmi")` and similar BModes-format decks
with a populated `PlatformSupport` block solve the **coupled**
tower-and-platform eigenproblem (free-free root, full 6×6 hydro /
mooring / inertia matrices). That path:

- **Correctly predicts coupled-system frequencies** for validation
  against BModes JJ. pyBmodes matches BModes JJ to ~ 0.0003 % across
  the first nine OC3 Hywind modes (`test_certtest_oc3hywind`). If
  the goal is "what does the floating tower vibrate at when coupled
  to its platform?", this is the right path.
- **Produces eigenvectors that include platform rigid-body motion**
  — i.e. the modes have non-zero base displacement and non-zero
  base slope, which the ElastoDyn `SHP` ansatz cannot represent.
  Feeding these eigenvectors into a polynomial fit produces
  coefficients ElastoDyn cannot consume without double-counting the
  platform.

The two paths answer different questions; both are correct for
their intended use:

| Goal | Use | BC |
| --- | --- | --- |
| ElastoDyn polynomial coefficients (any floating deck) | `Tower.from_elastodyn(...)` | `hub_conn = 1`, RNA at top |
| Coupled-system frequency validation against BModes JJ | `Tower.from_bmi("OC3Hywind.bmi")` | `hub_conn = 2`, full PlatformSupport |

## Configurations included in `reference_decks/`

This directory now ships pre-patched ElastoDyn decks for three
floating configurations alongside the original three fixed-base
decks:

- [`nrel5mw_oc3spar/`](nrel5mw_oc3spar/) — *NREL 5MW* on the OC3
  Hywind spar (Jonkman 2010). Source: OpenFAST `r-test`
  `5MW_OC3Spar_DLL_WTurb_WavesIrr/`.
- [`nrel5mw_oc4semi/`](nrel5mw_oc4semi/) — *NREL 5MW* on the OC4
  DeepCwind semi-submersible (Robertson et al. 2014). Source:
  OpenFAST `r-test` `5MW_OC4Semi_WSt_WavesWN/`.
- [`iea15mw_umainesemi/`](iea15mw_umainesemi/) — *IEA-15-240-RWT*
  on the UMaine VolturnUS-S semi (Allen et al. 2020). Source:
  upstream `IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-UMaineSemi/`.

Each deck is built by `scripts/build_reference_decks.py` using the
cantilever path documented above; the validator passes on all four
tower coefficient blocks after patching.

## Citations

- Jonkman, J., Butterfield, S., Musial, W., & Scott, G. (2009).
  *Definition of a 5-MW Reference Wind Turbine for Offshore System
  Development*. NREL/TP-500-38060.
- Jonkman, J. (2010). *Definition of the Floating System for Phase
  IV of OC3*. NREL/TP-500-47535.
- Robertson, A., Jonkman, J., Masciola, M., Song, H., Goupee, A.,
  Coulling, A., & Luan, C. (2014). *Definition of the
  Semisubmersible Floating System for Phase II of OC4*.
  NREL/TP-5000-60601.
- Allen, C., Viselli, A., Dagher, H., Goupee, A., Gaertner, E.,
  Abbas, N., Hall, M., & Barter, G. (2020). *Definition of the
  UMaine VolturnUS-S Reference Platform Developed for the
  IEA Wind 15-Megawatt Offshore Reference Wind Turbine*.
  NREL/TP-5000-76773.
- Gaertner, E., Rinker, J., Sethuraman, L., Zahle, F., Anderson, B.,
  Barter, G., et al. (2020). *Definition of the IEA 15-Megawatt
  Offshore Reference Wind Turbine*. NREL/TP-5000-75698.
