"""I/O sub-package: BModes ``.bmi`` and OpenFAST ``.dat`` parsers.

Most callers reach the parsers through their submodules
(``pybmodes.io.bmi``, ``pybmodes.io.elastodyn_reader``,
``pybmodes.io.subdyn_reader``); the WAMIT / HydroDyn readers below are
also surfaced here for convenience because the typical entry point —
``HydroDynReader(...).read_platform_matrices()`` — wants a single import
line.
"""

from pybmodes.io.wamit_reader import HydroDynReader, WamitData, WamitReader

__all__ = ["HydroDynReader", "WamitData", "WamitReader"]
