"""
Example 3 — Patch ElastoDyn input files
========================================

Shows the complete end-to-end workflow:
  1. Run BModes analysis for blade and tower
  2. Fit mode shapes to 6th-order polynomials
  3. Patch the ElastoDyn blade and tower sub-files in place

This example patches copies of the uploaded IEA 10 MW ElastoDyn files using
the CertTest geometry as stand-ins (the IEA turbine needs its own .bmi files).
To use with real IEA 10 MW inputs, replace the .bmi paths below.

Run from the repository root:
    conda run -n pybmodes python examples/03_patch_elastodyn.py
"""

import pathlib
import shutil
import tempfile

from pybmodes.models import RotatingBlade, Tower
from pybmodes.elastodyn import (
    compute_blade_params,
    compute_tower_params,
    patch_dat,
)

CERT_DIR  = pathlib.Path(__file__).parent.parent / "tests" / "data" / "certtest"
REPO_ROOT = pathlib.Path(__file__).parent.parent

# Paths to the uploaded IEA 10 MW ElastoDyn files
BLADE_DAT = REPO_ROOT / "IEA-10.0-198-RWT_ElastoDyn_blade.dat"
TOWER_DAT = REPO_ROOT / "IEA-10.0-198-RWT_ElastoDyn_tower.dat"


def main() -> None:
    # ── 1. Run modal analyses ─────────────────────────────────────────────────
    print("Running blade analysis (CertTest01) …")
    blade_result = RotatingBlade(CERT_DIR / "Test01_nonunif_blade.bmi").run(n_modes=10)

    print("Running tower analysis (CertTest03) …")
    tower_result = Tower(CERT_DIR / "Test03_tower.bmi").run(n_modes=10)

    # ── 2. Fit polynomials ────────────────────────────────────────────────────
    blade_params = compute_blade_params(blade_result)
    tower_params = compute_tower_params(tower_result)

    # ── 3. Patch working copies of the ElastoDyn files ────────────────────────
    # Work on copies so the originals are not modified.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)

        blade_copy = tmp / BLADE_DAT.name
        tower_copy = tmp / TOWER_DAT.name
        shutil.copy(BLADE_DAT, blade_copy)
        shutil.copy(TOWER_DAT, tower_copy)

        print(f"\nPatching {blade_copy.name} …")
        patch_dat(blade_copy, blade_params)

        print(f"Patching {tower_copy.name} …")
        patch_dat(tower_copy, tower_params)

        # ── 4. Show the patched mode-shape sections ───────────────────────────
        for path, label in [(blade_copy, "BLADE MODE SHAPES"),
                            (tower_copy, "TOWER FORE-AFT MODE SHAPES")]:
            text = path.read_text(encoding="utf-8")
            in_section = False
            print(f"\n--- Patched {path.name} ({label}) ---")
            for line in text.splitlines():
                if label in line:
                    in_section = True
                if in_section:
                    print(f"  {line}")
                    # Stop after 12 lines (header + 10 coefficients + blank)
                    if in_section and line.strip() == "" and "---" not in line:
                        break

    print("\nDone.  To patch in place, pass the real file paths to patch_dat().")


if __name__ == "__main__":
    main()
