"""
Example 2 - Tower modal analysis
================================

This example solves the CertTest03 onshore tower reference case and prints
frequencies plus ElastoDyn-ready tower mode-shape coefficients.

It is a compact demonstration of how pybmodes modernizes the legacy NREL
BModes workflow for tower modal analysis while keeping the familiar engineering
inputs and outputs.

Run from the repository root:
    python examples/02_tower_analysis.py
"""

import pathlib

import numpy as np

from pybmodes.elastodyn import compute_tower_params
from pybmodes.models import Tower

CERT_DIR = pathlib.Path(__file__).parent.parent / "tests" / "data" / "certtest"


def main() -> None:
    # 1. Solve the tower modal problem
    tower = Tower(CERT_DIR / "Test03_tower.bmi")
    result = tower.run(n_modes=10)

    # 2. Print a simple frequency table
    print("Onshore tower natural frequencies")
    print(f"  {'Mode':>4}  {'Freq (Hz)':>10}  {'Type':>4}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*4}")
    for s in result.shapes:
        dominant = "FA" if abs(s.flap_disp[-1]) >= abs(s.lag_disp[-1]) else "SS"
        print(f"  {s.mode_number:>4}  {s.freq_hz:>10.4f}  {dominant:>4}")

    # 3. Fit ElastoDyn-compatible tower mode-shape polynomials
    params = compute_tower_params(result)

    print("\nElastoDyn tower mode shape coefficients")
    print(f"  {'Name':<12}  {'C2':>10}  {'C3':>10}  {'C4':>10}  {'C5':>10}  {'C6':>10}  {'RMS':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for name, fit in [
        ("TwFAM1Sh", params.TwFAM1Sh),
        ("TwFAM2Sh", params.TwFAM2Sh),
        ("TwSSM1Sh", params.TwSSM1Sh),
        ("TwSSM2Sh", params.TwSSM2Sh),
    ]:
        print(
            f"  {name:<12}  {fit.c2:>10.5f}  {fit.c3:>10.5f}  {fit.c4:>10.5f}"
            f"  {fit.c5:>10.5f}  {fit.c6:>10.5f}  {fit.rms_residual:>8.5f}"
        )

    # 4. Show fit quality for the first fore-aft mode
    fa1 = next(s for s in result.shapes if abs(s.flap_disp[-1]) >= abs(s.lag_disp[-1]))
    print("\nFirst fore-aft mode shape (normalized to tip = 1)")
    tip = fa1.flap_disp[-1]
    print(f"  {'span':>6}  {'FEM':>8}  {'poly fit':>8}  {'residual':>9}")
    for x, y in zip(fa1.span_loc, fa1.flap_disp / tip):
        y_poly = params.TwFAM1Sh.evaluate(np.array([x]))[0]
        print(f"  {x:>6.3f}  {y:>8.5f}  {y_poly:>8.5f}  {y - y_poly:>+9.5f}")


if __name__ == "__main__":
    main()
