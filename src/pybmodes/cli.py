"""Command-line interface for pyBmodes.

Currently exposes three subcommands:

* ``pybmodes validate <main.dat>`` — coefficient-consistency report for
  an OpenFAST ElastoDyn deck. Compares the polynomial blocks shipped in
  the deck against pyBmodes' own fits to the FEM mode shapes produced
  by the deck's structural inputs.
* ``pybmodes patch <main.dat> [--backup]`` — regenerate the polynomial
  blocks in the deck's tower and blade ``.dat`` files in place from the
  pyBmodes fits. Optional ``--backup`` saves a ``.bak`` copy of each
  modified file first.
* ``pybmodes campbell <input> --rated-rpm R --max-rpm M [--orders 1,2,3,6,9]
  [--out PATH]`` — sweep a blade across rotor speeds 0..max_rpm and emit a
  Campbell diagram (PNG by default) plus a per-step CSV summary. Accepts
  either a ``.bmi`` deck or an ElastoDyn main ``.dat``.

The script entry point is wired up in ``pyproject.toml`` as
``pybmodes = "pybmodes.cli:main"``.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from typing import Sequence

# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

def _fmt_ratio(ratio: float) -> str:
    if ratio != ratio:  # NaN
        return "  nan"
    if ratio == float("inf"):
        return "   inf"
    if ratio >= 1000.0:
        return f"{ratio:>5.0f}"
    if ratio >= 100.0:
        return f"{ratio:>5.0f}"
    if ratio >= 10.0:
        return f"{ratio:>5.1f}"
    return f"{ratio:>5.2f}"


def _format_block_row(block) -> str:
    flag = ""
    if block.verdict == "FAIL":
        flag = "  FAIL <-"
    elif block.verdict == "WARN":
        flag = "  WARN"
    elif block.verdict == "PASS":
        flag = "  PASS"
    return (
        f"  {block.name:<8}  file RMS={block.file_rms:7.4f}  "
        f"pyB RMS={block.pybmodes_rms:7.4f}  "
        f"ratio={_fmt_ratio(block.ratio)}  {flag}"
    )


def _print_validation_report(result, file=sys.stdout) -> None:
    print("pyBmodes coefficient validator", file=file)
    print("==============================", file=file)
    print(f"File: {result.dat_path.name}", file=file)
    print("", file=file)
    print("Tower modes:", file=file)
    for block in result.tower_results.values():
        print(_format_block_row(block), file=file)
    print("", file=file)
    print("Blade modes:", file=file)
    for block in result.blade_results.values():
        print(_format_block_row(block), file=file)
    print("", file=file)

    n_fail = len(result.failing_blocks())
    n_warn = len(result.warning_blocks())
    if result.overall == "FAIL":
        print(f"Overall: FAIL ({n_fail} block(s))", file=file)
        print(
            f"Recommendation: run `pybmodes patch {result.dat_path}` to "
            f"update coefficients from structural inputs.",
            file=file,
        )
    elif result.overall == "WARN":
        print(f"Overall: WARN ({n_warn} block(s))", file=file)
    else:
        print("Overall: PASS", file=file)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _cmd_validate(args: argparse.Namespace) -> int:
    from pybmodes.elastodyn import validate_dat_coefficients

    dat_path = pathlib.Path(args.dat_file).resolve()
    if not dat_path.is_file():
        print(f"error: file not found: {dat_path}", file=sys.stderr)
        return 2

    result = validate_dat_coefficients(dat_path)
    _print_validation_report(result)

    if result.overall == "FAIL":
        return 1
    if result.overall == "WARN":
        return 0  # warnings are informational, not a hard failure
    return 0


def _cmd_patch(args: argparse.Namespace) -> int:
    """Regenerate tower + blade polynomial blocks in place."""
    from pybmodes.elastodyn import (
        compute_blade_params,
        compute_tower_params,
        patch_dat,
    )
    from pybmodes.io.elastodyn_reader import read_elastodyn_main
    from pybmodes.models import RotatingBlade, Tower

    main_dat = pathlib.Path(args.dat_file).resolve()
    if not main_dat.is_file():
        print(f"error: file not found: {main_dat}", file=sys.stderr)
        return 2

    main = read_elastodyn_main(main_dat)
    tower_dat = main_dat.parent / main.twr_file
    blade_dat = main_dat.parent / main.bld_file[0]

    if not tower_dat.is_file():
        print(f"error: tower file not found: {tower_dat}", file=sys.stderr)
        return 2
    if not blade_dat.is_file():
        print(f"error: blade file not found: {blade_dat}", file=sys.stderr)
        return 2

    print("pyBmodes coefficient patch")
    print("==========================")
    print(f"Main:  {main_dat}")
    print(f"Tower: {tower_dat}")
    print(f"Blade: {blade_dat}")
    print("")

    if args.backup:
        for target in (tower_dat, blade_dat):
            backup = target.with_suffix(target.suffix + ".bak")
            shutil.copy2(target, backup)
            print(f"  backed up {target.name} -> {backup.name}")
        print("")

    print("  building tower model + fitting polynomials ...")
    tower = Tower.from_elastodyn(main_dat)
    tower_modal = tower.run(n_modes=args.n_modes)
    tower_params = compute_tower_params(tower_modal)
    print("  patching tower .dat ...")
    patch_dat(tower_dat, tower_params)

    print("  building blade model + fitting polynomials ...")
    blade = RotatingBlade.from_elastodyn(main_dat)
    blade_modal = blade.run(n_modes=args.n_modes)
    blade_params = compute_blade_params(blade_modal)
    print("  patching blade .dat ...")
    patch_dat(blade_dat, blade_params)

    print("")
    print("Done. Re-run `pybmodes validate` to confirm consistency.")
    return 0


def _cmd_campbell(args: argparse.Namespace) -> int:
    """Run a rotor-speed sweep and write a Campbell diagram + CSV."""
    import numpy as np

    from pybmodes.campbell import campbell_sweep, plot_campbell

    src = pathlib.Path(args.input).resolve()
    if not src.is_file():
        print(f"error: file not found: {src}", file=sys.stderr)
        return 2

    try:
        orders = [int(x) for x in args.orders.split(",") if x.strip()]
    except ValueError:
        print(f"error: --orders must be a comma-separated list of integers; "
              f"got {args.orders!r}", file=sys.stderr)
        return 2
    if not orders:
        print("error: --orders must list at least one integer", file=sys.stderr)
        return 2

    if args.max_rpm <= 0.0:
        print(f"error: --max-rpm must be > 0; got {args.max_rpm}", file=sys.stderr)
        return 2
    if args.n_steps < 2:
        print(f"error: --n-steps must be >= 2; got {args.n_steps}", file=sys.stderr)
        return 2

    rpm = np.linspace(0.0, args.max_rpm, args.n_steps)
    tower_input = pathlib.Path(args.tower).resolve() if args.tower else None
    print(f"Campbell sweep: {src.name}")
    print(f"  rpm grid       : 0..{args.max_rpm} ({args.n_steps} points)")
    print(f"  blade modes    : {args.n_blade_modes}")
    print(f"  tower modes    : {args.n_tower_modes}")
    if tower_input is not None:
        print(f"  tower override : {tower_input}")
    result = campbell_sweep(
        src,
        rpm,
        n_blade_modes=args.n_blade_modes,
        n_tower_modes=args.n_tower_modes,
        tower_input=tower_input,
    )

    out_path = pathlib.Path(args.out).resolve() if args.out else \
        src.with_name(src.stem + "_campbell.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_path.with_suffix(".csv")
    header = "rpm," + ",".join(result.labels)
    table = np.column_stack([result.omega_rpm, result.frequencies])
    np.savetxt(csv_path, table, delimiter=",", header=header, comments="")
    print(f"  wrote {csv_path}")

    try:
        from pybmodes.plots.style import apply_style
        apply_style()
    except ImportError:
        pass

    fig = plot_campbell(result, excitation_orders=orders, rated_rpm=args.rated_rpm)
    fig.savefig(out_path)
    print(f"  wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Argparse setup
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pybmodes",
        description=(
            "pyBmodes — pure-Python finite-element library for "
            "wind-turbine blade and tower modal analysis."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser(
        "validate",
        help="validate ElastoDyn polynomial coefficients vs structural "
             "inputs",
    )
    p_validate.add_argument(
        "dat_file",
        help="path to the ElastoDyn main .dat file",
    )
    p_validate.set_defaults(func=_cmd_validate)

    p_patch = sub.add_parser(
        "patch",
        help="regenerate ElastoDyn polynomial coefficients from "
             "structural inputs (modifies tower and blade .dat in place)",
    )
    p_patch.add_argument(
        "dat_file",
        help="path to the ElastoDyn main .dat file",
    )
    p_patch.add_argument(
        "--backup",
        action="store_true",
        help="save .bak copies of the tower and blade .dat files before "
             "patching",
    )
    p_patch.add_argument(
        "--n-modes",
        type=int,
        default=10,
        help="number of FEM modes to extract before fitting (default: 10)",
    )
    p_patch.set_defaults(func=_cmd_patch)

    p_camp = sub.add_parser(
        "campbell",
        help="sweep a blade or tower across rotor speeds and emit a Campbell "
             "diagram (PNG) plus a CSV summary",
    )
    p_camp.add_argument(
        "input",
        help="path to a .bmi deck or an ElastoDyn main .dat file",
    )
    p_camp.add_argument(
        "--rated-rpm",
        type=float,
        default=None,
        help="operating rotor speed (rpm); drawn as a vertical reference line",
    )
    p_camp.add_argument(
        "--max-rpm",
        type=float,
        required=True,
        help="upper end of the rotor-speed sweep (rpm)",
    )
    p_camp.add_argument(
        "--n-steps",
        type=int,
        default=16,
        help="number of rotor-speed points in the sweep, including 0 and "
             "max-rpm (default: 16)",
    )
    p_camp.add_argument(
        "--orders",
        type=str,
        default="1,2,3,6,9",
        help="comma-separated per-rev excitation orders to overlay "
             "(default: 1,2,3,6,9)",
    )
    p_camp.add_argument(
        "--n-blade-modes",
        type=int,
        default=4,
        help="number of blade modes to track across the sweep (default: 4 — "
             "1st/2nd flap and 1st/2nd edge)",
    )
    p_camp.add_argument(
        "--n-tower-modes",
        type=int,
        default=2,
        help="number of tower modes to overlay as horizontal lines (default: 2 — "
             "1st FA and 1st SS); set to 0 to suppress",
    )
    p_camp.add_argument(
        "--tower",
        type=str,
        default=None,
        help="optional tower .bmi file; overrides the deck-supplied tower when the "
             "primary input is an ElastoDyn .dat, or pairs with a blade-only .bmi",
    )
    p_camp.add_argument(
        "--out",
        type=str,
        default=None,
        help="output PNG path (default: <input>_campbell.png alongside the input)",
    )
    p_camp.set_defaults(func=_cmd_campbell)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
