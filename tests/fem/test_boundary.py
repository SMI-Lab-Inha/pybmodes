"""Unit tests for :mod:`pybmodes.fem.boundary`.

Cover:
  * connectivity layout for cantilever (hub_conn=1), free-free (=2), and
    bottom-fixed monopile (=3) boundary conditions
  * the relationship between ``n_total_dof``, ``n_free_dof``, and
    ``active_dof_indices`` for all three BC types
  * shared-node bookkeeping across element pairs
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fem.boundary import (
    NEDOF,
    NESH,
    NNDOF,
    active_dof_indices,
    build_connectivity,
    n_free_dof,
    n_total_dof,
)

# ===========================================================================
# Constants
# ===========================================================================

class TestConstants:

    def test_layout_constants(self):
        assert NEDOF == 15
        assert NESH == 9
        assert NNDOF == 6


# ===========================================================================
# n_total_dof / n_free_dof
# ===========================================================================

class TestDofCounts:

    @pytest.mark.parametrize("nselt", [1, 4, 12, 50])
    def test_total_dof(self, nselt):
        assert n_total_dof(nselt) == 9 * nselt + 6

    @pytest.mark.parametrize("nselt", [1, 4, 12, 50])
    def test_cantilever_free_dof(self, nselt):
        assert n_free_dof(nselt, hub_conn=1) == 9 * nselt

    @pytest.mark.parametrize("nselt", [1, 4, 12, 50])
    def test_free_free_full_dof(self, nselt):
        assert n_free_dof(nselt, hub_conn=2) == 9 * nselt + 6

    @pytest.mark.parametrize("nselt", [1, 4, 12, 50])
    def test_monopile_constrains_two_dofs(self, nselt):
        # hub_conn=3 only constrains axial + twist at the root.
        assert n_free_dof(nselt, hub_conn=3) == 9 * nselt + 4

    def test_default_is_cantilever(self):
        assert n_free_dof(5) == n_free_dof(5, hub_conn=1)


# ===========================================================================
# active_dof_indices
# ===========================================================================

class TestActiveDofIndices:

    def test_cantilever_excludes_root_block(self):
        nselt = 3
        active = active_dof_indices(nselt, hub_conn=1)
        root_base = 9 * nselt
        assert active.tolist() == list(range(root_base))
        assert len(active) == n_free_dof(nselt, 1)

    def test_free_free_returns_all_dofs(self):
        nselt = 3
        active = active_dof_indices(nselt, hub_conn=2)
        ndt = n_total_dof(nselt)
        assert active.tolist() == list(range(ndt))

    def test_monopile_excludes_axial_and_twist_only(self):
        nselt = 3
        active = active_dof_indices(nselt, hub_conn=3)
        root_base = 9 * nselt
        # constrained: root_base (axial) and root_base+5 (phi/twist)
        assert root_base not in active
        assert (root_base + 5) not in active
        # but v_disp/slope and w_disp/slope at the root remain free
        for j in (1, 2, 3, 4):
            assert (root_base + j) in active
        assert len(active) == n_free_dof(nselt, 3)

    def test_indices_sorted(self):
        for hc in (1, 2, 3):
            arr = active_dof_indices(5, hub_conn=hc)
            assert np.all(np.diff(arr) > 0)


# ===========================================================================
# build_connectivity
# ===========================================================================

class TestBuildConnectivity:

    def test_shape(self):
        for nselt in (1, 5, 20):
            indeg = build_connectivity(nselt)
            assert indeg.shape == (NEDOF, nselt)

    def test_cantilever_zeros_root_local_dofs(self):
        nselt = 4
        indeg = build_connectivity(nselt, hub_conn=1)
        for j in (0, 4, 5, 8, 9, 12):
            assert indeg[j, nselt - 1] == 0, f"local DOF {j} not zeroed"

    def test_free_free_keeps_all_dofs(self):
        nselt = 3
        indeg = build_connectivity(nselt, hub_conn=2)
        # No DOF should be 0 for hub_conn=2
        assert (indeg > 0).all()

    def test_monopile_zeros_axial_and_twist_only(self):
        nselt = 3
        indeg = build_connectivity(nselt, hub_conn=3)
        last = nselt - 1
        # local DOFs 0 and 12 are constrained
        assert indeg[0, last] == 0
        assert indeg[12, last] == 0
        # but local DOFs 4, 5, 8, 9 remain free for monopile
        for j in (4, 5, 8, 9):
            assert indeg[j, last] > 0

    def test_global_dof_range(self):
        nselt = 6
        indeg = build_connectivity(nselt, hub_conn=2)
        # All non-zero connectivity entries must be valid 1-based global DOFs
        ndt = n_total_dof(nselt)
        nz = indeg[indeg > 0]
        assert nz.min() >= 1
        assert nz.max() <= ndt

    def test_unique_global_dofs_per_element(self):
        # Within a single element, each local DOF maps to a distinct global DOF
        indeg = build_connectivity(4, hub_conn=2)
        for i in range(indeg.shape[1]):
            col = indeg[:, i]
            assert len(np.unique(col)) == NEDOF

    def test_inboard_outboard_node_sharing(self):
        # The inboard end of element i and the outboard end of element i+1
        # share the same physical node, so their bending DOFs must coincide.
        # Inboard-end locals of element i:    v_disp=4, v_slope=5,
        #                                     w_disp=8, w_slope=9, phi=12
        # Outboard-end locals of element i+1: v_disp=6, v_slope=7,
        #                                     w_disp=10, w_slope=11, phi=14
        nselt = 4
        indeg = build_connectivity(nselt, hub_conn=2)
        for i in range(nselt - 1):
            # v_disp
            assert indeg[4, i] == indeg[6, i + 1]
            # v_slope
            assert indeg[5, i] == indeg[7, i + 1]
            # w_disp
            assert indeg[8, i] == indeg[10, i + 1]
            # w_slope
            assert indeg[9, i] == indeg[11, i + 1]
            # phi (twist)
            assert indeg[12, i] == indeg[14, i + 1]


# ===========================================================================
# Shared hub_conn validation across the three entry points
# ===========================================================================

class TestHubConnValidation:
    """All three boundary entry points must reject the same set of
    unsupported ``hub_conn`` values. Pre-1.0 review caught an
    inconsistency where ``build_connectivity`` raised but
    ``n_free_dof`` and ``active_dof_indices`` silently fell back to
    free-free — that meant a typo in a BMI ``hub_conn`` field could
    slip through whenever the model bypassed ``build_connectivity``.
    """

    @pytest.mark.parametrize("bad", [0, 5, 99, -1])
    def test_build_connectivity_rejects(self, bad):
        with pytest.raises(ValueError, match="Unsupported hub_conn"):
            build_connectivity(4, hub_conn=bad)

    @pytest.mark.parametrize("bad", [0, 5, 99, -1])
    def test_n_free_dof_rejects(self, bad):
        with pytest.raises(ValueError, match="Unsupported hub_conn"):
            n_free_dof(4, hub_conn=bad)

    @pytest.mark.parametrize("bad", [0, 5, 99, -1])
    def test_active_dof_indices_rejects(self, bad):
        with pytest.raises(ValueError, match="Unsupported hub_conn"):
            active_dof_indices(4, hub_conn=bad)
