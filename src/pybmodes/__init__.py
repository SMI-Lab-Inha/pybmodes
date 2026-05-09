"""pybmodes — Python finite-element library for wind turbine modal analysis.

Public API
==========

The following subpackage entry points are stable and supported across
0.x releases (see the *Compatibility policy* section in the README).
Anything else in the package tree is internal and may change between
patch releases.

    from pybmodes.models    import RotatingBlade, Tower, ModalResult
    from pybmodes.elastodyn import (
        compute_blade_params,
        compute_tower_params,
        compute_tower_params_report,
        patch_dat,
        validate_dat_coefficients,
        BladeElastoDynParams,
        TowerElastoDynParams,
        ValidationResult,
        CoeffBlockResult,
    )
    from pybmodes.fitting   import PolyFitResult, fit_mode_shape
    from pybmodes.campbell  import (
        CampbellResult,
        campbell_sweep,
        plot_campbell,
    )
    from pybmodes.plots     import (
        apply_style,
        plot_mode_shapes,
        plot_fit_quality,
        bir_mode_shape_plot,
        bir_mode_shape_subplot,
    )

Internal modules (``pybmodes.fem.*``, ``pybmodes.io.*``, the
underscore-prefixed module ``pybmodes.models._pipeline``) carry the
implementation and should not be imported directly by user code; their
signatures may change between 0.x releases.

The CLI is exposed via ``pybmodes`` (see ``pybmodes --help``) and is
declared in ``[project.scripts]``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pybmodes")
except PackageNotFoundError:
    __version__ = "0.2.0-dev"

__all__ = ["__version__"]
