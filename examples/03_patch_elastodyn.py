"""
Example 3 - Patch ElastoDyn input files
=======================================

This example shows the complete end-to-end workflow:

1. Run modal analysis for blade and tower
2. Fit ElastoDyn-compatible mode-shape polynomials
3. Patch existing ElastoDyn blade and tower files in place

The script uses small CertTest cases by default, but the file paths at the top
can be replaced with your own BModes-style inputs. This mirrors a common
practical use case for pybmodes: keeping a familiar legacy BModes input style
while using a modern Python workflow.

Run from the repository root:
    python examples/03_patch_elastodyn.py
"""

import pathlib

from pybmodes.elastodyn import compute_blade_params, compute_tower_params, patch_dat
from pybmodes.models import RotatingBlade, Tower

CERT_DIR = pathlib.Path(__file__).parent.parent / "tests" / "data" / "certtest"

# Replace these with your own paths.
BLADE_BMI = CERT_DIR / "Test01_nonunif_blade.bmi"
TOWER_BMI = CERT_DIR / "Test03_tower.bmi"
BLADE_DAT = pathlib.Path("ElastoDyn_blade.dat")
TOWER_DAT = pathlib.Path("ElastoDyn_tower.dat")


def main() -> None:
    # 1. Run modal analyses
    print("Running blade analysis...")
    blade_result = RotatingBlade(BLADE_BMI).run(n_modes=10)

    print("Running tower analysis...")
    tower_result = Tower(TOWER_BMI).run(n_modes=10)

    # 2. Fit mode-shape polynomials
    blade_params = compute_blade_params(blade_result)
    tower_params = compute_tower_params(tower_result)

    # 3. Print fitted coefficients
    print("\nBlade mode shape coefficients:")
    print(f"  {'Name':<12}  {'C2':>10}  {'C3':>10}  {'C4':>10}  {'C5':>10}  {'C6':>10}  {'RMS':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for name, fit in [
        ("BldFl1Sh", blade_params.BldFl1Sh),
        ("BldFl2Sh", blade_params.BldFl2Sh),
        ("BldEdgSh", blade_params.BldEdgSh),
    ]:
        print(
            f"  {name:<12}  {fit.c2:>10.5f}  {fit.c3:>10.5f}  {fit.c4:>10.5f}"
            f"  {fit.c5:>10.5f}  {fit.c6:>10.5f}  {fit.rms_residual:>8.5f}"
        )

    print("\nTower mode shape coefficients:")
    print(f"  {'Name':<12}  {'C2':>10}  {'C3':>10}  {'C4':>10}  {'C5':>10}  {'C6':>10}  {'RMS':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for name, fit in [
        ("TwFAM1Sh", tower_params.TwFAM1Sh),
        ("TwFAM2Sh", tower_params.TwFAM2Sh),
        ("TwSSM1Sh", tower_params.TwSSM1Sh),
        ("TwSSM2Sh", tower_params.TwSSM2Sh),
    ]:
        print(
            f"  {name:<12}  {fit.c2:>10.5f}  {fit.c3:>10.5f}  {fit.c4:>10.5f}"
            f"  {fit.c5:>10.5f}  {fit.c6:>10.5f}  {fit.rms_residual:>8.5f}"
        )

    # 4. Patch ElastoDyn files if they exist
    for dat, params, label in [
        (BLADE_DAT, blade_params, "blade"),
        (TOWER_DAT, tower_params, "tower"),
    ]:
        if dat.exists():
            print(f"\nPatching {dat}...")
            patch_dat(dat, params)
            print("Done.")
        else:
            print(f"\n{dat} not found - skipping patch for {label}.")
            print("  Update BLADE_DAT / TOWER_DAT at the top of this script to patch your own files.")


if __name__ == "__main__":
    main()
