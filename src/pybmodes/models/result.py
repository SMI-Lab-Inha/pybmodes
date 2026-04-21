"""ModalResult dataclass returned by RotatingBlade and Tower."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pybmodes.fem.normalize import NodeModeShape


@dataclass
class ModalResult:
    """Frequencies and mode shapes from a single FEM solve."""
    frequencies: np.ndarray      # Hz, shape (n_modes,)
    shapes: list[NodeModeShape]  # one entry per mode, root-to-tip order
