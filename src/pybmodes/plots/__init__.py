"""Professional plotting utilities for pybmodes results.

Requires matplotlib >= 3.7.  Install with::

    pip install "pybmodes[plots]"

All functions return a :class:`matplotlib.figure.Figure` object; call
``fig.show()`` or ``fig.savefig(path)`` as needed.
"""

from .mode_shapes import (
    bir_mode_shape_plot,
    bir_mode_shape_subplot,
    blade_fit_pairs,
    plot_fit_quality,
    plot_mode_shapes,
    tower_fit_pairs,
)
from .style import OKABE_ITO, PALETTE, apply_style

__all__ = [
    "plot_mode_shapes",
    "plot_fit_quality",
    "blade_fit_pairs",
    "tower_fit_pairs",
    "bir_mode_shape_plot",
    "bir_mode_shape_subplot",
    "apply_style",
    "OKABE_ITO",
    "PALETTE",
]
