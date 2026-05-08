"""Command-line interface for pyBmodes.

Currently exposes two subcommands:

* ``pybmodes validate <main.dat>`` — coefficient-consistency report for
  an OpenFAST ElastoDyn deck. Compares the polynomial blocks shipped in
  the deck against pyBmodes' own fits to the FEM mode shapes produced
  by the deck's structural inputs.
* ``pybmodes patch <main.dat> [--backup]`` — regenerate the polynomial
  blocks in the deck's tower and blade ``.dat`` files in place from the
  pyBmodes fits. Optional ``--backup`` saves a ``.bak`` copy of each
  modified file first.

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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
