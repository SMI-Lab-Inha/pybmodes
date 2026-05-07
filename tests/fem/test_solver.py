"""Unit tests for the generalised eigensolver in :mod:`pybmodes.fem.solver`.

These exercise ``solve_modes`` and ``eigvals_to_hz`` independently of the FEM
assembly pipeline, on small hand-built K, M matrices where the eigenpairs are
known analytically.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fem.solver import eigvals_to_hz, solve_modes

# ===========================================================================
# Diagonal toy systems with known eigenvalues
# ===========================================================================

class TestDiagonalSystem:
    """K and M diagonal -> eigenvalues are k_i / m_i, sorted ascending."""

    def setup_method(self):
        self.k = np.diag([4.0, 1.0, 9.0])
        self.m = np.diag([1.0, 1.0, 1.0])

    def test_returns_two_arrays(self):
        eigvals, eigvecs = solve_modes(self.k, self.m)
        assert isinstance(eigvals, np.ndarray)
        assert isinstance(eigvecs, np.ndarray)

    def test_eigvals_sorted_ascending(self):
        eigvals, _ = solve_modes(self.k, self.m)
        assert np.all(np.diff(eigvals) >= 0)
        assert eigvals == pytest.approx([1.0, 4.0, 9.0])

    def test_eigvec_shape(self):
        eigvals, eigvecs = solve_modes(self.k, self.m)
        assert eigvecs.shape == (3, 3)
        # Each column has unit L2 norm
        for j in range(eigvecs.shape[1]):
            assert np.linalg.norm(eigvecs[:, j]) == pytest.approx(1.0, abs=1e-12)

    def test_n_modes_subset(self):
        # Request only the lowest 2 modes
        eigvals, eigvecs = solve_modes(self.k, self.m, n_modes=2)
        assert eigvals.shape == (2,)
        assert eigvecs.shape == (3, 2)
        assert eigvals == pytest.approx([1.0, 4.0])

    def test_n_modes_exceeds_size_clipped(self):
        # If n_modes > ngd, it should be clamped to ngd.
        eigvals, eigvecs = solve_modes(self.k, self.m, n_modes=10)
        assert eigvals.shape == (3,)


class TestNonDiagonal:
    """A 2x2 SPD pair with analytically known eigenvalues."""

    def setup_method(self):
        # K v = lambda M v with K=[[2,-1],[-1,2]], M=[[1,0],[0,1]]
        # eigenvalues: 1 and 3
        self.k = np.array([[2.0, -1.0], [-1.0, 2.0]])
        self.m = np.eye(2)

    def test_eigvals(self):
        eigvals, _ = solve_modes(self.k, self.m)
        assert eigvals == pytest.approx([1.0, 3.0], abs=1e-12)

    def test_modal_orthogonality_wrt_M(self):
        eigvals, eigvecs = solve_modes(self.k, self.m)
        # v_i^T M v_j == 0 for i != j (eigh orthogonality through M)
        cross = eigvecs[:, 0] @ self.m @ eigvecs[:, 1]
        assert abs(cross) < 1e-12

    def test_eigenvectors_satisfy_problem(self):
        eigvals, eigvecs = solve_modes(self.k, self.m)
        for j in range(2):
            lhs = self.k @ eigvecs[:, j]
            rhs = eigvals[j] * (self.m @ eigvecs[:, j])
            np.testing.assert_allclose(lhs, rhs, atol=1e-12)


class TestSymmetrisation:
    """The solver should symmetrise inputs that have small floating-point asymmetry."""

    def test_asymmetric_input_still_solved(self):
        # Build symmetric K, M and add a tiny perturbation to one off-diagonal
        k = np.array([[2.0, -1.0], [-1.0, 2.0]])
        m = np.eye(2)
        k_asym = k.copy()
        k_asym[0, 1] += 1e-13
        eigvals, _ = solve_modes(k_asym, m)
        assert eigvals == pytest.approx([1.0, 3.0], abs=1e-10)


# ===========================================================================
# eigvals_to_hz
# ===========================================================================

class TestEigvalsToHz:

    def test_zero_eigenvalue(self):
        # A rigid-body mode -> 0 Hz
        out = eigvals_to_hz(np.array([0.0]), romg=10.0)
        assert out[0] == pytest.approx(0.0)

    def test_negative_eigenvalue_clipped(self):
        # Tiny negative numerical noise should be clipped to 0, not produce NaN
        out = eigvals_to_hz(np.array([-1e-16, 1.0]), romg=10.0)
        assert out[0] == pytest.approx(0.0)
        assert not np.any(np.isnan(out))

    def test_unit_eigenvalue(self):
        # lambda = 1, romg = 10 -> freq = 10 / (2 pi) Hz
        out = eigvals_to_hz(np.array([1.0]), romg=10.0)
        assert out[0] == pytest.approx(10.0 / (2.0 * np.pi))

    def test_general_formula(self):
        eigvals = np.array([0.0, 1.0, 4.0, 16.0])
        romg = 10.0
        out = eigvals_to_hz(eigvals, romg)
        expected = np.sqrt(eigvals) * romg / (2.0 * np.pi)
        np.testing.assert_allclose(out, expected, rtol=1e-12)

    def test_array_shape_preserved(self):
        eigvals = np.linspace(0.1, 10.0, 5)
        out = eigvals_to_hz(eigvals, romg=12.0)
        assert out.shape == eigvals.shape

    def test_monotone_for_sorted_input(self):
        eigvals = np.array([1.0, 2.0, 4.0, 9.0])
        out = eigvals_to_hz(eigvals, romg=10.0)
        assert np.all(np.diff(out) > 0.0)
