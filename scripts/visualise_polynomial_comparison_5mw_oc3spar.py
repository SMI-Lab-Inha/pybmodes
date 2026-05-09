"""Polynomial-vs-FEM comparison for the NREL 5MW OC3 Hywind floating spar.

Same comparison the monopile scripts perform — file polynomial vs
pyBmodes-fitted polynomial vs raw FEM tower-segment shape — but for
the floating-spar substructure where the entire flexible-tower base
is *free* (six rigid-body DOFs reacted by hydrostatics and mooring
stiffness). pyBmodes has no ``from_elastodyn`` path for floating
ElastoDyn decks (parsing HydroDyn + MoorDyn into a 6×6 platform
support is out of scope), so we use the validated
``Tower(OC3Hywind.bmi)`` BMI path as the FEM reference — it solves
the same NREL 5MW Hywind tower with its 6×6 hydro / mooring / inertia
matrices to within 0.0003 % of BModes JJ across the first 9 modes
per ``tests/test_certtest.py::test_certtest_oc3hywind``.

Caveat on length: the BMI deck's flexible tower spans the full
``radius = 87.6 m`` from MSL up to the tower top, while the r-test
ElastoDyn deck's polynomial is defined over ``TowerBsHt..TowerHt =
10..87.6 m`` (77.6 m, the lowest 10 m absorbed into the platform).
That length difference shifts the RMS-residual metric by a constant
factor but doesn't affect the amplitude metric, which is purely a
property of the polynomial coefficients themselves.

Outputs ``scripts/outputs/polynomial_comparison_5mw_oc3spar_TwFA2_TwSS2.png``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (SRC_DIR, SCRIPTS_DIR):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from visualise_polynomial_comparison import (  # noqa: E402
    _eval_poly,
    _print_diagnostic_table,
)
from visualise_polynomial_comparison_5mw_monopile import (  # noqa: E402
    _amplitude_ratio,
    _coeffs_from_fit,
    _extract_tower_segment_bending,
    _participation,
    _rms_residuals,
)

from pybmodes.fitting.poly_fit import PolyFitResult, fit_mode_shape  # noqa: E402
from pybmodes.io.elastodyn_reader import read_elastodyn_tower  # noqa: E402
from pybmodes.models import Tower  # noqa: E402

_OC3_BMI = REPO_ROOT / "docs" / "BModes" / "docs" / "examples" / "OC3Hywind.bmi"
_RTEST_DECK = (
    REPO_ROOT
    / "docs" / "OpenFAST_files" / "r-test" / "glue-codes" / "openfast"
    / "5MW_OC3Spar_DLL_WTurb_WavesIrr"
)
_RTEST_TOWER = _RTEST_DECK / "NRELOffshrBsline5MW_OC3Hywind_ElastoDyn_Tower.dat"


def _select_first_two_above_freq(
    shapes,
    *,
    want_fa: bool,
    min_freq_hz: float,
) -> tuple[int, int]:
    """``_select_first_two`` for floating decks: skip platform rigid-body
    modes (below ``min_freq_hz``) so the picked indices land on the
    1st and 2nd tower-bending modes.

    OC3 Hywind has six low-frequency platform DOFs (surge / sway /
    heave at ~0.008 Hz, pitch / roll at ~0.034 Hz, yaw at ~0.121 Hz)
    that the participation-only ``_select_first_two`` would otherwise
    pick first because surge / pitch are FA-dominated and sway / roll
    are SS-dominated. Anything above 0.3 Hz on this turbine is already
    a flexible-tower mode.
    """
    out: list[int] = []
    for idx, shape in enumerate(shapes):
        if shape.freq_hz < min_freq_hz:
            continue
        p_fa, p_ss, _ = _participation(shape)
        if want_fa and p_fa >= 0.6:
            out.append(idx)
        elif (not want_fa) and p_ss >= 0.6:
            out.append(idx)
        if len(out) == 2:
            return (out[0], out[1])
    raise RuntimeError(
        f"could not find two {'FA' if want_fa else 'SS'}-dominated "
        f"flexible modes above {min_freq_hz} Hz "
        f"(got indices {out!r})"
    )


def _plot_one(
    ax, *, title: str, x_local: np.ndarray, phi_local: np.ndarray,
    pyb_coeffs: np.ndarray, ref_coeffs: np.ndarray,
) -> None:
    x = np.linspace(0.0, 1.0, 100)
    ax.plot(x, _eval_poly(ref_coeffs, x), "-",
            color=(0.85, 0.33, 0.10),
            linewidth=1.6, label="r-test polynomial")
    ax.plot(x, _eval_poly(pyb_coeffs, x), "-",
            color=(0.0, 0.45, 0.74),
            linewidth=1.6, label="pyBmodes polynomial")
    ax.scatter(x_local, phi_local, s=22, facecolors="none",
               edgecolors=(0.20, 0.20, 0.20), linewidths=0.9,
               label="pyBmodes FEM (raw, tower segment)")
    ax.axhline(0.0, color=(0.6, 0.6, 0.6), linewidth=0.5, zorder=0)
    ax.axhline(1.0, color=(0.6, 0.6, 0.6), linewidth=0.5,
               linestyle=":", zorder=0)
    ax.set_xlabel("Normalised tower height  z / H")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bmi", type=pathlib.Path, default=_OC3_BMI,
        help="OC3 Hywind BMI (default: docs/BModes/docs/examples/OC3Hywind.bmi)",
    )
    parser.add_argument(
        "--rtest-tower", type=pathlib.Path, default=_RTEST_TOWER,
        help="r-test floating-spar ElastoDyn tower .dat (for reference polynomial)",
    )
    parser.add_argument(
        "--out", type=pathlib.Path,
        default=REPO_ROOT / "scripts" / "outputs"
                          / "polynomial_comparison_5mw_oc3spar_TwFA2_TwSS2.png",
    )
    parser.add_argument(
        "--min-freq-hz", type=float, default=0.3,
        help="cutoff above which to look for flexible tower modes "
             "(skips platform rigid-body modes; default 0.3 Hz)",
    )
    args = parser.parse_args(argv)

    for label, path in (("OC3 Hywind BMI", args.bmi),
                         ("r-test tower .dat", args.rtest_tower)):
        if not path.is_file():
            print(f"error: {label} not found: {path}", file=sys.stderr)
            print(
                "  Both BMI and r-test data are gitignored under docs/;\n"
                "  see CLAUDE.md \"Independence stance\" for how to clone them.",
                file=sys.stderr,
            )
            return 2

    print(f"Reading OC3 Hywind BMI : {args.bmi}")
    print(f"Reading r-test tower   : {args.rtest_tower}")
    rtest_tower = read_elastodyn_tower(args.rtest_tower)

    print("  building Tower(OC3Hywind.bmi) + solving FEM ...")
    with warnings.catch_warnings():
        # Distributed-hydro warning is irrelevant here.
        warnings.simplefilter("ignore", UserWarning)
        model = Tower(args.bmi)
        modal = model.run(n_modes=12)

    fa_idxs = _select_first_two_above_freq(
        modal.shapes, want_fa=True, min_freq_hz=args.min_freq_hz,
    )
    ss_idxs = _select_first_two_above_freq(
        modal.shapes, want_fa=False, min_freq_hz=args.min_freq_hz,
    )
    fa1, fa2 = modal.shapes[fa_idxs[0]], modal.shapes[fa_idxs[1]]
    ss1, ss2 = modal.shapes[ss_idxs[0]], modal.shapes[ss_idxs[1]]

    print(f"  rigid-body / platform modes (skipped, < {args.min_freq_hz} Hz):")
    for s in modal.shapes:
        if s.freq_hz < args.min_freq_hz:
            print(f"    mode {s.mode_number}: {s.freq_hz:.4f} Hz")
    print(f"  FA1 = mode {fa1.mode_number} ({fa1.freq_hz:.4f} Hz)")
    print(f"  FA2 = mode {fa2.mode_number} ({fa2.freq_hz:.4f} Hz)")
    print(f"  SS1 = mode {ss1.mode_number} ({ss1.freq_hz:.4f} Hz)")
    print(f"  SS2 = mode {ss2.mode_number} ({ss2.freq_hz:.4f} Hz)")

    # tp_frac=0: the BMI's flexible beam IS the tower; no pile splice to
    # exclude. The helper still subtracts base rigid-body motion (which is
    # genuinely non-zero on a floating platform — the spar pitches/surges
    # under load), then tip-normalises.
    fits: dict[str, tuple[np.ndarray, np.ndarray, PolyFitResult, np.ndarray]] = {}
    for name, shape, is_fa, ref_coeffs in (
        ("TwFAM1Sh", fa1, True,
         np.asarray(rtest_tower.tw_fa_m1_sh, dtype=float)),
        ("TwSSM1Sh", ss1, False,
         np.asarray(rtest_tower.tw_ss_m1_sh, dtype=float)),
        ("TwFAM2Sh", fa2, True,
         np.asarray(rtest_tower.tw_fa_m2_sh, dtype=float)),
        ("TwSSM2Sh", ss2, False,
         np.asarray(rtest_tower.tw_ss_m2_sh, dtype=float)),
    ):
        x_local, phi_local = _extract_tower_segment_bending(
            shape, tp_frac=0.0, is_fa=is_fa,
        )
        fit = fit_mode_shape(x_local, phi_local)
        fits[name] = (x_local, phi_local, fit, ref_coeffs)

    for name, (_, _, fit, ref_coeffs) in fits.items():
        _print_diagnostic_table(name, _coeffs_from_fit(fit), ref_coeffs)

    print()
    print("=== Amplitude ratios (max|phi_rtest| / max|phi_pyB| on [0, 1]) ===")
    for name, (_, _, fit, ref_coeffs) in fits.items():
        ratio = _amplitude_ratio(_coeffs_from_fit(fit), ref_coeffs)
        print(f"  {name}: {ratio:>6.2f} x")

    print()
    print("=== RMS residuals at FEM stations (polynomial - tip-normalised FEM) ===")
    print(f"  {'block':<8}  {'file_rms':>10}  {'pyB_rms':>10}  {'ratio':>9}")
    for name, (x_local, phi_local, fit, ref_coeffs) in fits.items():
        file_rms, pyb_rms = _rms_residuals(
            ref_coeffs, x_local, phi_local, _coeffs_from_fit(fit),
        )
        ratio = file_rms / pyb_rms if pyb_rms > 0 else float("inf")
        print(f"  {name:<8}  {file_rms:>10.4f}  {pyb_rms:>10.5f}  {ratio:>7.2f} x")

    try:
        from pybmodes.plots.style import apply_style
        apply_style()
    except ImportError:
        print("note: matplotlib style helpers unavailable; "
              "install pybmodes[plots] for journal defaults")
    import matplotlib.pyplot as plt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=False)
    fa2_x, fa2_y, fa2_fit, fa2_ref = fits["TwFAM2Sh"]
    ss2_x, ss2_y, ss2_fit, ss2_ref = fits["TwSSM2Sh"]
    _plot_one(
        axes[0], title="TwFAM2Sh — fore-aft 2nd bending",
        x_local=fa2_x, phi_local=fa2_y,
        pyb_coeffs=_coeffs_from_fit(fa2_fit), ref_coeffs=fa2_ref,
    )
    _plot_one(
        axes[1], title="TwSSM2Sh — side-side 2nd bending",
        x_local=ss2_x, phi_local=ss2_y,
        pyb_coeffs=_coeffs_from_fit(ss2_fit), ref_coeffs=ss2_ref,
    )
    axes[0].set_ylabel("Modal displacement  phi(z)  (base-rigid-motion subtracted)")
    fig.suptitle(
        f"NREL 5MW OC3 Hywind tower: 2nd-mode polynomial vs FEM tower segment — "
        f"{args.rtest_tower.name}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out)
    print()
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
