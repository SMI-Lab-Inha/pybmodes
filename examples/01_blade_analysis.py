"""
Example 1 - Rotating blade modal analysis
=========================================

This example solves the CertTest01 rotating blade reference case and prints
frequencies plus ElastoDyn-ready polynomial coefficients.

The example is intentionally small and GitHub-friendly: it shows the basic
workflow of using pybmodes as a modern Python interpretation of the legacy
NREL BModes workflow for blade modal analysis.

Run from the repository root:
    python examples/01_blade_analysis.py
"""

import pathlib

import numpy as np

from pybmodes.elastodyn import compute_blade_params
from pybmodes.models import RotatingBlade

CERT_DIR = pathlib.Path(__file__).parent.parent / "tests" / "data" / "certtest"


def main() -> None:
    # 1. Solve the blade modal problem
    blade = RotatingBlade(CERT_DIR / "Test01_nonunif_blade.bmi")
    result = blade.run(n_modes=10)

    # 2. Print a simple frequency table
    print("Rotating blade natural frequencies")
    print(f"  {'Mode':>4}  {'Freq (Hz)':>10}  {'Type':>6}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*6}")
    for s in result.shapes:
        dominant = "flap" if abs(s.flap_disp[-1]) >= abs(s.lag_disp[-1]) else "edge"
        print(f"  {s.mode_number:>4}  {s.freq_hz:>10.4f}  {dominant:>6}")

    # 3. Fit ElastoDyn-compatible blade mode-shape polynomials
    params = compute_blade_params(result)

    print("\nElastoDyn mode shape coefficients")
    print(f"  {'Name':<12}  {'C2':>10}  {'C3':>10}  {'C4':>10}  {'C5':>10}  {'C6':>10}  {'RMS':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for name, fit in [
        ("BldFl1Sh", params.BldFl1Sh),
        ("BldFl2Sh", params.BldFl2Sh),
        ("BldEdgSh", params.BldEdgSh),
    ]:
        print(
            f"  {name:<12}  {fit.c2:>10.5f}  {fit.c3:>10.5f}  {fit.c4:>10.5f}"
            f"  {fit.c5:>10.5f}  {fit.c6:>10.5f}  {fit.rms_residual:>8.5f}"
        )
        assert abs(fit.c2 + fit.c3 + fit.c4 + fit.c5 + fit.c6 - 1.0) < 1e-10

    # 4. Show fit quality for the first flap mode
    flap1 = result.shapes[0]
    print("\nFirst flap mode shape (normalized to tip = 1)")
    tip = flap1.flap_disp[-1]
    print(f"  {'span':>6}  {'FEM':>8}  {'poly fit':>8}  {'residual':>9}")
    for x, y in zip(flap1.span_loc, flap1.flap_disp / tip):
        y_poly = params.BldFl1Sh.evaluate(np.array([x]))[0]
        print(f"  {x:>6.3f}  {y:>8.5f}  {y_poly:>8.5f}  {y - y_poly:>+9.5f}")


if __name__ == "__main__":
    main()
