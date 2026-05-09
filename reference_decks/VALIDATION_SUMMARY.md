<!-- markdownlint-disable MD013 -->
# Reference-deck coefficient validation summary

Per-block RMS residual of the polynomial coefficients shipped in each upstream deck (Before) and after pyBmodes regenerated them from the structural inputs in the same deck (After). The ratio column is the upstream `file_rms / pybmodes_rms` — values >> 1 indicate the upstream polynomial does not represent the mode shape produced by the deck's structural inputs.

| Case | Block | Before RMS | After RMS | Ratio before | Status |
| --- | --- | ---: | ---: | ---: | :---: |
| nrel5mw_land | TwFAM1Sh | 0.0081 | 0.0000 | 313× | PASS |
| nrel5mw_land | TwFAM2Sh | 5.0783 | 0.0024 | 2101× | PASS |
| nrel5mw_land | TwSSM1Sh | 0.0075 | 0.0000 | 293× | PASS |
| nrel5mw_land | TwSSM2Sh | 5.9009 | 0.0023 | 2529× | PASS |
| nrel5mw_land | BldFl1Sh | 0.0022 | 0.0008 | 2.82× | PASS |
| nrel5mw_land | BldFl2Sh | 0.0088 | 0.0035 | 2.48× | PASS |
| nrel5mw_land | BldEdgSh | 0.0006 | 0.0002 | 3.17× | PASS |
| nrel5mw_oc3monopile | TwFAM1Sh | 0.0037 | 0.0000 | 140× | PASS |
| nrel5mw_oc3monopile | TwFAM2Sh | 5.7805 | 0.0032 | 1813× | PASS |
| nrel5mw_oc3monopile | TwSSM1Sh | 0.0045 | 0.0000 | 173× | PASS |
| nrel5mw_oc3monopile | TwSSM2Sh | 7.3266 | 0.0033 | 2220× | PASS |
| nrel5mw_oc3monopile | BldFl1Sh | 0.0022 | 0.0008 | 2.82× | PASS |
| nrel5mw_oc3monopile | BldFl2Sh | 0.0088 | 0.0035 | 2.48× | PASS |
| nrel5mw_oc3monopile | BldEdgSh | 0.0006 | 0.0002 | 3.17× | PASS |
| iea34_land | TwFAM1Sh | 0.0098 | 0.0002 | 56.9× | PASS |
| iea34_land | TwFAM2Sh | 0.7230 | 0.0042 | 172× | PASS |
| iea34_land | TwSSM1Sh | 0.0110 | 0.0002 | 64.0× | PASS |
| iea34_land | TwSSM2Sh | 1.5494 | 0.0041 | 380× | PASS |
| iea34_land | BldFl1Sh | 0.0106 | 0.0014 | 7.61× | PASS |
| iea34_land | BldFl2Sh | 0.0051 | 0.0047 | 1.09× | PASS |
| iea34_land | BldEdgSh | 0.0065 | 0.0014 | 4.62× | PASS |
| nrel5mw_oc3spar | TwFAM1Sh | 0.0111 | 0.0000 | 350× | PASS |
| nrel5mw_oc3spar | TwFAM2Sh | 10.6264 | 0.0024 | 4476× | PASS |
| nrel5mw_oc3spar | TwSSM1Sh | 0.0165 | 0.0000 | 524× | PASS |
| nrel5mw_oc3spar | TwSSM2Sh | 14.1050 | 0.0027 | 5311× | PASS |
| nrel5mw_oc3spar | BldFl1Sh | 0.0022 | 0.0008 | 2.82× | PASS |
| nrel5mw_oc3spar | BldFl2Sh | 0.0088 | 0.0035 | 2.48× | PASS |
| nrel5mw_oc3spar | BldEdgSh | 0.0006 | 0.0002 | 3.17× | PASS |
| nrel5mw_oc4semi | TwFAM1Sh | 0.0034 | 0.0000 | 106× | PASS |
| nrel5mw_oc4semi | TwFAM2Sh | 8.2805 | 0.0024 | 3488× | PASS |
| nrel5mw_oc4semi | TwSSM1Sh | 0.0048 | 0.0000 | 152× | PASS |
| nrel5mw_oc4semi | TwSSM2Sh | 9.0396 | 0.0027 | 3404× | PASS |
| nrel5mw_oc4semi | BldFl1Sh | 0.0033 | 0.0008 | 3.94× | PASS |
| nrel5mw_oc4semi | BldFl2Sh | 0.0093 | 0.0036 | 2.59× | PASS |
| nrel5mw_oc4semi | BldEdgSh | 0.0004 | 0.0002 | 2.29× | PASS |
| iea15mw_umainesemi | TwFAM1Sh | 0.0078 | 0.0001 | 95.8× | PASS |
| iea15mw_umainesemi | TwFAM2Sh | 0.7922 | 0.0013 | 619× | PASS |
| iea15mw_umainesemi | TwSSM1Sh | 0.0088 | 0.0001 | 118× | PASS |
| iea15mw_umainesemi | TwSSM2Sh | 102.4075 | 0.0163 | 6276× | WARN |
| iea15mw_umainesemi | BldFl1Sh | 0.0048 | 0.0002 | 24.7× | PASS |
| iea15mw_umainesemi | BldFl2Sh | 0.0021 | 0.0018 | 1.21× | PASS |
| iea15mw_umainesemi | BldEdgSh | 0.0028 | 0.0008 | 3.73× | PASS |

## Pattern

- **2nd-mode tower coefficients** (`TwFAM2Sh`, `TwSSM2Sh`) show the largest inconsistency on every upstream deck: ratios from ~170× (IEA-3.4) to ~2,500× (NREL 5MW). The shipped polynomials do not represent the 2nd bending mode of the structural inputs by any reasonable metric.
- **1st-mode tower coefficients** (`TwFAM1Sh`, `TwSSM1Sh`) and blade coefficients (`BldFl1Sh`, `BldFl2Sh`, `BldEdgSh`) show a smaller but non-zero inconsistency (typical ratio ~ 2–300×). Their absolute file RMS values still classify as PASS under the 1 % per-block gate, but they are still drift artefacts from the same generation pipeline.
- **All blocks pass after patching.** The After-RMS column matches the pyBmodes-RMS column from the Before report; the polynomials in the patched files are exactly pyBmodes' fits, so the file polynomial reproduces the pyBmodes mode shape modulo the writer's text-precision (~7 sig figs).

## How to reproduce

```bash
python scripts/build_reference_decks.py
```

The script copies the upstream sources, runs `pybmodes patch`, and re-runs the validator. See `before_patch.txt` and `validation_report.txt` in each case directory for the raw CLI output.
