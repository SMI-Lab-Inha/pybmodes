<!-- markdownlint-disable MD013 -->
# Reference-deck coefficient validation summary

Per-block RMS residual of the polynomial coefficients shipped in each upstream deck (Before) and after pyBmodes regenerated them from the structural inputs in the same deck (After). The ratio column is the upstream `file_rms / pybmodes_rms` — values >> 1 indicate the upstream polynomial does not represent the mode shape produced by the deck's structural inputs.

| Case | Block | Before RMS | After RMS | Ratio before | Status |
| --- | --- | ---: | ---: | ---: | :---: |
| nrel5mw_land | TwFAM1Sh | 0.0081 | 0.0000 | 313× | PASS |
| nrel5mw_land | TwFAM2Sh | 5.0783 | 0.0024 | 2101× | PASS |
| nrel5mw_land | TwSSM1Sh | 0.0075 | 0.0000 | 293× | PASS |
| nrel5mw_land | TwSSM2Sh | 5.9009 | 0.0023 | 2529× | PASS |
| nrel5mw_land | BldFl1Sh | 0.0020 | 0.0008 | 2.43× | PASS |
| nrel5mw_land | BldFl2Sh | 0.0090 | 0.0036 | 2.54× | PASS |
| nrel5mw_land | BldEdgSh | 0.0020 | 0.0002 | 11.9× | PASS |
| nrel5mw_oc3monopile | TwFAM1Sh | 0.0037 | 0.0000 | 140× | PASS |
| nrel5mw_oc3monopile | TwFAM2Sh | 5.7805 | 0.0032 | 1813× | PASS |
| nrel5mw_oc3monopile | TwSSM1Sh | 0.0045 | 0.0000 | 173× | PASS |
| nrel5mw_oc3monopile | TwSSM2Sh | 7.3266 | 0.0033 | 2220× | PASS |
| nrel5mw_oc3monopile | BldFl1Sh | 0.0020 | 0.0008 | 2.43× | PASS |
| nrel5mw_oc3monopile | BldFl2Sh | 0.0090 | 0.0036 | 2.54× | PASS |
| nrel5mw_oc3monopile | BldEdgSh | 0.0020 | 0.0002 | 11.9× | PASS |
| iea34_land | TwFAM1Sh | 0.0098 | 0.0002 | 56.9× | PASS |
| iea34_land | TwFAM2Sh | 0.7230 | 0.0042 | 172× | PASS |
| iea34_land | TwSSM1Sh | 0.0110 | 0.0002 | 64.0× | PASS |
| iea34_land | TwSSM2Sh | 1.5494 | 0.0041 | 380× | PASS |
| iea34_land | BldFl1Sh | 0.0112 | 0.0014 | 8.05× | PASS |
| iea34_land | BldFl2Sh | 0.0052 | 0.0047 | 1.11× | PASS |
| iea34_land | BldEdgSh | 0.0055 | 0.0014 | 3.88× | PASS |

## Pattern

- **2nd-mode tower coefficients** (`TwFAM2Sh`, `TwSSM2Sh`) show the largest inconsistency on every upstream deck: ratios from ~170× (IEA-3.4) to ~2,500× (NREL 5MW). The shipped polynomials do not represent the 2nd bending mode of the structural inputs by any reasonable metric.
- **1st-mode tower coefficients** (`TwFAM1Sh`, `TwSSM1Sh`) and blade coefficients (`BldFl1Sh`, `BldFl2Sh`, `BldEdgSh`) show a smaller but non-zero inconsistency (typical ratio ~ 2–300×). Their absolute file RMS values still classify as PASS under the 1 % per-block gate, but they are still drift artefacts from the same generation pipeline.
- **All blocks pass after patching.** The After-RMS column matches the pyBmodes-RMS column from the Before report; the polynomials in the patched files are exactly pyBmodes' fits, so the file polynomial reproduces the pyBmodes mode shape modulo the writer's text-precision (~7 sig figs).

## How to reproduce

```bash
python scripts/build_reference_decks.py
```

The script copies the upstream sources, runs `pybmodes patch`, and re-runs the validator. See `before_patch.txt` and `validation_report.txt` in each case directory for the raw CLI output.
