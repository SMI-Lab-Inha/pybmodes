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
        CoeffBlockResult,           # tower blocks carry fa/ss/torsion
                                    # participation + rejected_modes
    )
    from pybmodes.fitting   import PolyFitResult, fit_mode_shape
    from pybmodes.campbell  import (
        CampbellResult,             # save / load / to_csv
        campbell_sweep,
        plot_campbell,
    )
    from pybmodes.checks    import check_model, ModelWarning
    from pybmodes.mac       import (
        mac_matrix,
        compare_modes,
        ModeComparison,
        plot_mac,
        shape_to_vector,
    )
    from pybmodes.report    import generate_report
    from pybmodes.plots     import (
        apply_style,
        plot_mode_shapes,
        plot_fit_quality,
        bir_mode_shape_plot,
        bir_mode_shape_subplot,
    )

``ModalResult`` ships ``save(.npz)`` / ``load(.npz)`` /
``to_json(.json)`` / ``from_json(.json)`` with metadata (pyBmodes
version, source file, timestamp, git hash) and optional
``participation`` + ``fit_residuals`` fields. ``CampbellResult``
ships ``save(.npz)`` / ``load(.npz)`` / ``to_csv(.csv)``.

Internal modules (``pybmodes.fem.*``, ``pybmodes.io.*``, the
underscore-prefixed module ``pybmodes.models._pipeline``, and the
private sub-package ``pybmodes.io._elastodyn``) carry the
implementation and should not be imported directly by user code;
their signatures may change between 0.x releases.

The CLI is exposed via ``pybmodes`` (see ``pybmodes --help``) and is
declared in ``[project.scripts]``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pybmodes")
except PackageNotFoundError:
    __version__ = "0.3.0-dev"

__all__ = ["__version__"]
