"""Eigenvector post-processing: extract nodal mode shapes from global DOF vector.

The global DOF layout (0-based Python indices) for a system with nselt elements:

  For each element i (0=tip … nselt-1=root), the outboard-end node occupies
  global DOFs 9*i+0 … 9*i+5:
    9*i+0 : axial u
    9*i+1 : v (lag/edge) displacement
    9*i+2 : v (lag/edge) slope
    9*i+3 : w (flap) displacement
    9*i+4 : w (flap) slope
    9*i+5 : phi (twist)

  Internal DOFs for element i: 9*i+6, 9*i+7, 9*i+8 (axial 2/3, phi mid, axial 1/3)

  Root node (constrained, zero): global DOFs 9*nselt+0 … 9*nselt+5.

Output convention for each mode (matching the .out file column order):
  span_loc | w(flap)_disp | w(flap)_slope | v(lag)_disp | v(lag)_slope | twist
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class NodeModeShape:
    """Nodal mode shape data for one mode."""
    mode_number: int
    freq_hz: float
    span_loc: np.ndarray    # normalised span, 0=root … 1=tip (root-to-tip order)
    flap_disp: np.ndarray   # w(flap) displacement
    flap_slope: np.ndarray  # w(flap) slope
    lag_disp: np.ndarray    # v(lag/edge) displacement
    lag_slope: np.ndarray   # v(lag/edge) slope
    twist: np.ndarray       # phi (torsion)


def extract_mode_shapes(
    eigvecs: np.ndarray,
    eigvals_hz: np.ndarray,
    nselt: int,
    el: np.ndarray,
    xb: np.ndarray,
    radius: float,
    hub_rad: float,
    bl_len: float,
    hub_conn: int = 1,
    active_dofs: Optional[np.ndarray] = None,
) -> list[NodeModeShape]:
    """Extract per-node mode shape data from eigenvectors.

    The output includes nselt+1 stations: root + one per element outboard end.

    Parameters
    ----------
    eigvecs    : (ngd, n_modes) compact eigenvectors (L2-normalised)
    eigvals_hz : (n_modes,) frequencies in Hz
    nselt      : number of elements
    el         : element lengths (non-dim, tip-to-root)
    xb         : inboard end positions (non-dim, tip-to-root)
    radius     : total radius/height (m)
    hub_rad    : hub radius or tower base height (m)
    bl_len     : flexible length = radius - hub_rad (m)
    hub_conn   : root BC (1=cantilever, 2=free-free, 3=axial+torsion only)
    active_dofs: (ngd,) sorted 0-based indices of active DOFs used to expand
                 compact eigvecs to full ndt size.  If None, assumes hub_conn=1.

    Returns
    -------
    List of NodeModeShape, one per mode.
    """
    ndt     = 9 * nselt + 6
    n_modes = eigvecs.shape[1]
    shapes  = []

    # Build output span locations: root + outboard ends of elements in root-to-tip order
    x_nodes_nd = np.empty(nselt + 1)
    x_nodes_nd[0] = xb[nselt - 1]    # inboard end of root element = root position
    for k, ie in enumerate(range(nselt - 1, -1, -1)):
        x_nodes_nd[k + 1] = xb[ie] + el[ie]

    span_loc = (x_nodes_nd * radius - hub_rad) / bl_len

    for mode_idx in range(n_modes):
        ev_compact = eigvecs[:, mode_idx]

        # Expand compact eigvec to full ndt length (zeros at constrained DOFs)
        ev = np.zeros(ndt)
        if active_dofs is not None:
            ev[active_dofs] = ev_compact
        else:
            # hub_conn=1: compact = full first 9*nselt DOFs
            ev[:len(ev_compact)] = ev_compact

        flap_d  = np.zeros(nselt + 1)
        flap_s  = np.zeros(nselt + 1)
        lag_d   = np.zeros(nselt + 1)
        lag_s   = np.zeros(nselt + 1)
        phi     = np.zeros(nselt + 1)

        # Root station (k=0): values come from root node DOFs (9*nselt + offset)
        root_base = 9 * nselt
        flap_d[0] = ev[root_base + 3]   # w_disp
        flap_s[0] = ev[root_base + 4]   # w_slope
        lag_d[0]  = ev[root_base + 1]   # v_disp
        lag_s[0]  = ev[root_base + 2]   # v_slope
        phi[0]    = ev[root_base + 5]   # phi

        # Outboard end of element ie corresponds to output station k+1
        for k, ie in enumerate(range(nselt - 1, -1, -1)):
            base = 9 * ie
            flap_d[k + 1]  = ev[base + 3]
            flap_s[k + 1]  = ev[base + 4]
            lag_d[k + 1]   = ev[base + 1]
            lag_s[k + 1]   = ev[base + 2]
            phi[k + 1]     = ev[base + 5]

        shapes.append(NodeModeShape(
            mode_number=mode_idx + 1,
            freq_hz=eigvals_hz[mode_idx],
            span_loc=span_loc.copy(),
            flap_disp=flap_d,
            flap_slope=flap_s,
            lag_disp=lag_d,
            lag_slope=lag_s,
            twist=phi,
        ))

    return shapes
