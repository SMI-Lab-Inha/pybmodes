# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — 2025-04-22

### Added
- Rotating blade modal analysis (flap, edge, torsion modes)
- Onshore tower analysis — cantilevered and tension-wire supported
- Offshore tower analysis — floating spar (`hub_conn=2`) and bottom-fixed monopile (`hub_conn=3`)
- Constrained 6th-order polynomial mode shape fitting (C₂ + C₃ + C₄ + C₅ + C₆ = 1)
- In-place patching of OpenFAST ElastoDyn `.dat` files
- All six reference test cases pass within 0.5 % frequency tolerance
