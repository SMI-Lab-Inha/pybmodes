"""Unit tests for :mod:`pybmodes.fem.normalize`.

Verifies that ``extract_mode_shapes`` correctly maps compact eigenvectors back
to per-node ``NodeModeShape`` objects, with the right span ordering, root-node
extraction, and shape data layout for cantilever and free-free boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fem.boundary import active_dof_indices, n_total_dof
from pybmodes.fem.normalize import NodeModeShape, extract_mode_shapes

# ===========================================================================
# Helpers
# ===========================================================================

def _uniform_geometry(nselt: int, radius: float = 50.0):
    """Build per-element lengths/positions for a uniform mesh, tip-to-root.

    Non-dim element coordinates run from 0 (root) to 1 (tip), so each element
    has length 1/nselt and ``xb[i]`` is the inboard end of element ``i`` in
    tip-to-root ordering.
    """
    eli = 1.0 / nselt
    el = np.full(nselt, eli)
    xb = np.array([1.0 - (i + 1) * eli for i in range(nselt)])
    return el, xb


def _make_eigvec_full(nselt: int) -> np.ndarray:
    """Build a synthetic full-length eigenvector with distinct DOF values.

    Each global DOF is assigned a unique label so the test can verify the
    per-node DOF extraction logic exactly.
    """
    return np.arange(1.0, n_total_dof(nselt) + 1.0)


# ===========================================================================
# extract_mode_shapes — cantilever (hub_conn=1)
# ===========================================================================

class TestExtractCantilever:
    """For cantilever, root-node DOFs are constrained; root values must be zero."""

    NSELT = 3
    RADIUS = 30.0
    HUB_RAD = 0.0

    def setup_method(self):
        self.el, self.xb = _uniform_geometry(self.NSELT)
        # Compact eigvec covers only the active (free) 9*nselt DOFs
        n_free = 9 * self.NSELT
        self.compact = np.arange(1.0, n_free + 1.0)
        eigvecs = self.compact.reshape(-1, 1)
        eigvals_hz = np.array([0.5])
        self.shapes = extract_mode_shapes(
            eigvecs=eigvecs,
            eigvals_hz=eigvals_hz,
            nselt=self.NSELT,
            el=self.el,
            xb=self.xb,
            radius=self.RADIUS,
            hub_rad=self.HUB_RAD,
            bl_len=self.RADIUS - self.HUB_RAD,
            hub_conn=1,
            active_dofs=active_dof_indices(self.NSELT, hub_conn=1),
        )

    def test_returns_list_of_one_mode(self):
        assert len(self.shapes) == 1
        assert isinstance(self.shapes[0], NodeModeShape)

    def test_n_stations_is_nselt_plus_one(self):
        assert len(self.shapes[0].span_loc) == self.NSELT + 1

    def test_root_values_zero(self):
        s = self.shapes[0]
        assert s.flap_disp[0] == pytest.approx(0.0)
        assert s.flap_slope[0] == pytest.approx(0.0)
        assert s.lag_disp[0] == pytest.approx(0.0)
        assert s.lag_slope[0] == pytest.approx(0.0)
        assert s.twist[0] == pytest.approx(0.0)

    def test_span_loc_root_to_tip_monotone(self):
        s = self.shapes[0]
        assert np.all(np.diff(s.span_loc) > 0)
        # Ends are 0 at root and 1 at tip in normalised span
        assert s.span_loc[0] == pytest.approx(0.0)
        assert s.span_loc[-1] == pytest.approx(1.0)

    def test_frequency_attached(self):
        assert self.shapes[0].freq_hz == pytest.approx(0.5)
        assert self.shapes[0].mode_number == 1


# ===========================================================================
# extract_mode_shapes — free-free (hub_conn=2)
# ===========================================================================

class TestExtractFreeFree:
    """For free-free, all DOFs (incl. root) are active and root values may be nonzero."""

    NSELT = 2
    RADIUS = 20.0
    HUB_RAD = 0.0

    def setup_method(self):
        self.el, self.xb = _uniform_geometry(self.NSELT, self.RADIUS)
        ndt = n_total_dof(self.NSELT)
        # Compact = full ndt-length here because hub_conn=2 has zero constraints
        self.compact = np.arange(1.0, ndt + 1.0)
        eigvecs = self.compact.reshape(-1, 1)
        eigvals_hz = np.array([1.5])
        self.shapes = extract_mode_shapes(
            eigvecs=eigvecs,
            eigvals_hz=eigvals_hz,
            nselt=self.NSELT,
            el=self.el,
            xb=self.xb,
            radius=self.RADIUS,
            hub_rad=self.HUB_RAD,
            bl_len=self.RADIUS - self.HUB_RAD,
            hub_conn=2,
            active_dofs=active_dof_indices(self.NSELT, hub_conn=2),
        )

    def test_root_values_nonzero(self):
        # For free-free the root node carries the synthetic eigvec values
        s = self.shapes[0]
        root_base = 9 * self.NSELT
        # flap_disp[0] is taken from ev[root_base+3] (1-indexed entry root_base+4)
        assert s.flap_disp[0] == pytest.approx(self.compact[root_base + 3])
        assert s.flap_slope[0] == pytest.approx(self.compact[root_base + 4])
        assert s.lag_disp[0] == pytest.approx(self.compact[root_base + 1])
        assert s.lag_slope[0] == pytest.approx(self.compact[root_base + 2])
        assert s.twist[0] == pytest.approx(self.compact[root_base + 5])

    def test_outboard_node_extraction(self):
        # Element 0 (tip) outboard end: global DOFs 0..5 -> station k=NSELT (the tip).
        s = self.shapes[0]
        # Tip station corresponds to outboard end of element 0 (tip element)
        # Mapping: flap_disp[k+1] = ev[9*ie+3] for ie = nselt-1, nselt-2, ..., 0.
        # The last station (k+1 = nselt) corresponds to ie = 0.
        assert s.flap_disp[-1] == pytest.approx(self.compact[3])
        assert s.lag_disp[-1] == pytest.approx(self.compact[1])
        assert s.twist[-1] == pytest.approx(self.compact[5])


# ===========================================================================
# Multiple modes
# ===========================================================================

class TestExtractMultipleModes:

    def test_multiple_modes_extracted(self):
        nselt = 3
        el, xb = _uniform_geometry(nselt)
        n_free = 9 * nselt
        eigvecs = np.column_stack([
            np.linspace(0.0, 1.0, n_free),
            np.linspace(0.0, 2.0, n_free),
            np.linspace(0.0, 3.0, n_free),
        ])
        eigvals_hz = np.array([1.0, 2.0, 3.0])
        shapes = extract_mode_shapes(
            eigvecs=eigvecs,
            eigvals_hz=eigvals_hz,
            nselt=nselt,
            el=el,
            xb=xb,
            radius=10.0,
            hub_rad=0.0,
            bl_len=10.0,
            hub_conn=1,
            active_dofs=active_dof_indices(nselt, hub_conn=1),
        )
        assert len(shapes) == 3
        assert [s.mode_number for s in shapes] == [1, 2, 3]
        assert [s.freq_hz for s in shapes] == [1.0, 2.0, 3.0]

    def test_shapes_independent_arrays(self):
        # Each mode should have its own copies of the span_loc array — modifying
        # one must not affect the others.
        nselt = 2
        el, xb = _uniform_geometry(nselt)
        n_free = 9 * nselt
        eigvecs = np.eye(n_free)[:, :2]
        shapes = extract_mode_shapes(
            eigvecs=eigvecs,
            eigvals_hz=np.array([1.0, 2.0]),
            nselt=nselt,
            el=el,
            xb=xb,
            radius=10.0,
            hub_rad=0.0,
            bl_len=10.0,
            hub_conn=1,
            active_dofs=active_dof_indices(nselt, hub_conn=1),
        )
        shapes[0].span_loc[0] = -999.0
        assert shapes[1].span_loc[0] != -999.0


# ===========================================================================
# Default active_dofs path (hub_conn=1, no active_dofs argument)
# ===========================================================================

class TestExtractDefaultActiveDofs:

    def test_default_assumes_cantilever(self):
        # When active_dofs is None the function assumes hub_conn=1 layout
        nselt = 2
        el, xb = _uniform_geometry(nselt)
        n_free = 9 * nselt
        eigvecs = np.arange(1.0, n_free + 1.0).reshape(-1, 1)
        shapes = extract_mode_shapes(
            eigvecs=eigvecs,
            eigvals_hz=np.array([1.0]),
            nselt=nselt,
            el=el,
            xb=xb,
            radius=10.0,
            hub_rad=0.0,
            bl_len=10.0,
            hub_conn=1,
            active_dofs=None,
        )
        # Cantilever -> root values should be zero
        s = shapes[0]
        assert s.flap_disp[0] == 0.0
        assert s.lag_disp[0] == 0.0
        assert s.twist[0] == 0.0
