"""Connectivity vector construction and boundary-condition application.

Global DOF layout (0-based):

  Each element i (1=tip ... nselt=root) owns a 9-DOF "block" at global positions
  9*(i-1)+1 ... 9*(i-1)+9, plus a shared 6-DOF outboard-end node that overlaps
  with block i-1 (or the tip node for i=1).

  Within block for outboard-end node of element i (global DOFs 9*(i-1)+1..6):
    9*(i-1)+1 : axial  u
    9*(i-1)+2 : v (lag/edge) displacement
    9*(i-1)+3 : v (lag/edge) slope
    9*(i-1)+4 : w (flap) displacement
    9*(i-1)+5 : w (flap) slope
    9*(i-1)+6 : phi (twist)

  Internal DOFs for element i (global DOFs 9*(i-1)+7..9):
    9*(i-1)+7 : axial at 2/3 of element
    9*(i-1)+8 : twist at midpoint
    9*(i-1)+9 : axial at 1/3 of element

  Root node DOFs (shared inboard end of element nselt):
    9*nselt+1 .. 9*nselt+6  (= ndt-5 .. ndt)
    These are zeroed out for cantilevered BC.

  ndt  = 9*nselt + 6
  ngd  = ndt - ncon  where ncon=6 (cantilever) -> ngd = 9*nselt
"""

from __future__ import annotations

import numpy as np

# Hardwired 15-DOF element connectivity vector (1-based global offsets).
# Maps local DOF j (0-based) -> offset to add to (i-1)*9+1 to get global DOF.
_IVECBE = np.array([10, 9, 7, 1, 11, 12, 2, 3, 13, 14, 4, 5, 15, 8, 6], dtype=int)
NEDOF = 15
NNDOF = 6
NINTN = 3
NESH = 9


def build_connectivity(nselt: int, hub_conn: int = 1) -> np.ndarray:
    """Build element connectivity array indeg[j, i] (1-based global DOF, or 0 if constrained)."""
    indeg = np.zeros((NEDOF, nselt), dtype=int)
    for i in range(nselt):
        nsh = i * NESH
        for j in range(NEDOF):
            indeg[j, i] = _IVECBE[j] + nsh

    # Apply root BC: zero out the appropriate local DOFs of the root element.
    # Root DOF mapping (local index -> DOF type):
    #   0=axial@ib, 4=v@ib, 5=v'@ib, 8=w@ib, 9=w'@ib, 12=phi@ib
    if hub_conn == 1:
        root_local = [0, 4, 5, 8, 9, 12]
    elif hub_conn == 3:
        root_local = [0, 12]
    elif hub_conn == 4:
        # Pinned-free (Bir 2009 §III.B inextensible-cable convention):
        # lock translations + twist, leave bending slopes FREE.
        root_local = [0, 4, 8, 12]
    else:
        root_local = []

    last = nselt - 1
    for j in root_local:
        indeg[j, last] = 0

    return indeg


def n_total_dof(nselt: int) -> int:
    """Total (unreduced) DOFs: ndt = 9*nselt + 6."""
    return NESH * nselt + NNDOF


def n_free_dof(nselt: int, hub_conn: int = 1) -> int:
    """Free (reduced) DOFs after BC application."""
    ndt = NESH * nselt + NNDOF
    if hub_conn == 1:
        return ndt - NNDOF
    if hub_conn == 3:
        return ndt - 2
    if hub_conn == 4:
        return ndt - 4
    return ndt


def active_dof_indices(nselt: int, hub_conn: int = 1) -> np.ndarray:
    """Return sorted 0-based indices of unconstrained (active) global DOFs.

    Root-node DOF map (offset from ``root_base = NESH * nselt``):
      +0 axial · +1 v_disp · +2 v_slope · +3 w_disp · +4 w_slope · +5 phi

    Constrained sets:
      hub_conn=1 (cantilever):    all six       (+0 +1 +2 +3 +4 +5)
      hub_conn=3 (axial+torsion): two           (+0 +5)
      hub_conn=4 (pinned-free):   four          (+0 +1 +3 +5) — slopes FREE
    """
    ndt = NESH * nselt + NNDOF
    root_base = NESH * nselt
    if hub_conn == 1:
        constrained = set(range(root_base, ndt))
    elif hub_conn == 3:
        constrained = {root_base, root_base + 5}
    elif hub_conn == 4:
        constrained = {root_base, root_base + 1, root_base + 3, root_base + 5}
    else:
        constrained = set()
    return np.array([i for i in range(ndt) if i not in constrained], dtype=int)
