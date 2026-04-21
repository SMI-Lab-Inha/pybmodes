"""
Example 4 — Mode shape and fit-quality plots
=============================================

Demonstrates the ``pybmodes.plots`` module.  Produces four figures:

1. Blade mode shapes (flap + edge panels)
2. Blade polynomial fit quality (BldFl1Sh, BldFl2Sh, BldEdgSh)
3. Tower mode shapes (fore-aft + side-side panels)
4. Tower polynomial fit quality (TwFAM1, TwFAM2, TwSSM1, TwSSM2)

Requires matplotlib (``pip install "pybmodes[plots]"``).

Run from the repository root::

    conda run -n pybmodes python examples/04_plot_results.py

Figures are saved to ``examples/`` as PNG files (300 dpi).
"""

import pathlib

from pybmodes.elastodyn import compute_blade_params, compute_tower_params
from pybmodes.models import RotatingBlade, Tower
from pybmodes.plots import blade_fit_pairs, plot_fit_quality, plot_mode_shapes, tower_fit_pairs

CERT_DIR = pathlib.Path(__file__).parent.parent / "tests" / "data" / "certtest"
OUT_DIR = pathlib.Path(__file__).parent


def _save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")


def main() -> None:
    # ── Blade ─────────────────────────────────────────────────────────────────
    print("Solving rotating blade …")
    blade_result = RotatingBlade(CERT_DIR / "Test01_nonunif_blade.bmi").run(n_modes=10)
    blade_params = compute_blade_params(blade_result)

    print("Plotting blade mode shapes …")
    fig1 = plot_mode_shapes(
        blade_result,
        n_modes=6,
        component="both",
        title="CertTest01 — Non-uniform rotating blade (60 RPM)",
    )
    _save(fig1, "blade_mode_shapes.png")

    print("Plotting blade fit quality …")
    fig2 = plot_fit_quality(
        blade_fit_pairs(blade_result, blade_params),
        title="CertTest01 — Blade polynomial fits",
    )
    _save(fig2, "blade_fit_quality.png")

    # ── Tower ─────────────────────────────────────────────────────────────────
    print("Solving onshore tower …")
    tower_result = Tower(CERT_DIR / "Test03_tower.bmi").run(n_modes=10)
    tower_params = compute_tower_params(tower_result)

    print("Plotting tower mode shapes …")
    fig3 = plot_mode_shapes(
        tower_result,
        n_modes=6,
        component="both",
        title="CertTest03 — Onshore cantilevered tower",
    )
    _save(fig3, "tower_mode_shapes.png")

    print("Plotting tower fit quality …")
    fig4 = plot_fit_quality(
        tower_fit_pairs(tower_result, tower_params),
        title="CertTest03 — Tower polynomial fits",
    )
    _save(fig4, "tower_fit_quality.png")

    print("\nDone.  Four PNG files written to examples/")


if __name__ == "__main__":
    main()
