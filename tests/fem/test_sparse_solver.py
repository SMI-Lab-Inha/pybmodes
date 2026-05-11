"""Tests for the sparse-solver dispatch in ``pybmodes.fem.solver``.

The solver gates the sparse shift-invert path on three conditions:

  ngd > _SPARSE_NDOF_THRESHOLD AND symmetric matrices AND a small
  subset of modes was requested (``n_modes is not None`` and
  ``n_modes < ngd // 2``).

These tests build small synthetic symmetric / asymmetric problems
(no FEM pipeline needed) so they're fast and self-contained.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pybmodes.fem import solver
from pybmodes.fem.solver import (
    _SPARSE_NDOF_THRESHOLD,
    solve_modes,
)


def _spd_problem(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build a symmetric positive-definite (K, M) pair of size n×n with
    a known smooth eigenvalue spectrum. Constructed as random SPD
    matrices via ``A.T @ A + I`` so eigenvalues are well-spaced and
    the problem is far from degenerate."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(n, n))
    K = A.T @ A + np.eye(n) * (n + 1.0)
    B = rng.standard_normal(size=(n, n))
    M = B.T @ B + np.eye(n) * (n + 1.0)
    # Symmetric to machine precision
    K = 0.5 * (K + K.T)
    M = 0.5 * (M + M.T)
    return K, M


def _asymmetric_problem(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build a deliberately asymmetric (K, M) pair so the solver
    routes through the dense general ``eig`` path."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(n, n))
    K = A.T @ A + np.eye(n) * (n + 1.0)
    # Inject a non-symmetric coupling block big enough to clear the
    # _is_effectively_symmetric threshold (1e-12 * max|K|).
    K[0, n - 1] += 1.5 * float(np.max(np.abs(K)))
    B = rng.standard_normal(size=(n, n))
    M = B.T @ B + np.eye(n) * (n + 1.0)
    return K, M


# ---------------------------------------------------------------------------
# 1. Sparse vs dense numerical agreement
# ---------------------------------------------------------------------------

def test_sparse_matches_dense_frequencies() -> None:
    """For a symmetric SPD problem of size n > threshold, the sparse
    shift-invert path returns the same lowest-k eigenvalues as the
    dense ``eigh`` path to rtol = 1e-8."""
    n = _SPARSE_NDOF_THRESHOLD + 100
    K, M = _spd_problem(n, seed=1)
    n_modes = 8

    # Force the sparse path by calling the helper directly.
    sparse_eigvals, sparse_eigvecs = solver._solve_sparse_shift_invert(
        K, M, n_modes,
    )
    # Force the dense path the same way.
    dense_eigvals, _dense_eigvecs = solver._solve_dense_symmetric(
        K, M, n_modes,
    )

    np.testing.assert_allclose(
        np.sort(sparse_eigvals)[:n_modes],
        np.sort(dense_eigvals)[:n_modes],
        rtol=1.0e-8,
    )


# ---------------------------------------------------------------------------
# 2. Sparse path is taken above the size threshold
# ---------------------------------------------------------------------------

def test_sparse_triggered_above_threshold(caplog: pytest.LogCaptureFixture) -> None:
    """When n_dof > threshold AND symmetric AND n_modes is a small
    subset, ``solve_modes`` logs that it took the sparse path."""
    n = _SPARSE_NDOF_THRESHOLD + 50
    K, M = _spd_problem(n, seed=2)
    n_modes = 6
    with caplog.at_level(logging.INFO, logger="pybmodes.fem.solver"):
        solve_modes(K, M, n_modes=n_modes)
    sparse_msgs = [
        rec for rec in caplog.records
        if "sparse shift-invert" in rec.getMessage()
    ]
    assert sparse_msgs, (
        "expected an INFO log line announcing the sparse path; "
        f"saw records: {[r.getMessage() for r in caplog.records]}"
    )


def test_sparse_skipped_below_threshold(caplog: pytest.LogCaptureFixture) -> None:
    """At n_dof below the threshold the dense path runs; the sparse
    path is not invoked."""
    n = _SPARSE_NDOF_THRESHOLD - 100
    K, M = _spd_problem(n, seed=3)
    with caplog.at_level(logging.INFO, logger="pybmodes.fem.solver"):
        solve_modes(K, M, n_modes=6)
    # Either no log was emitted or it was the dense one.
    messages = [r.getMessage() for r in caplog.records]
    assert all("sparse" not in m for m in messages), (
        "small problem should not trigger the sparse path; "
        f"saw: {messages}"
    )


# ---------------------------------------------------------------------------
# 3. Asymmetric problems route through dense `eig`
# ---------------------------------------------------------------------------

def test_dense_fallback_on_asymmetric(caplog: pytest.LogCaptureFixture) -> None:
    """An asymmetric problem above the size threshold takes the dense
    general-matrix path; the sparse symmetric solver is skipped."""
    n = _SPARSE_NDOF_THRESHOLD + 50
    K, M = _asymmetric_problem(n, seed=4)
    with caplog.at_level(logging.INFO, logger="pybmodes.fem.solver"):
        eigvals, eigvecs = solve_modes(K, M, n_modes=6)
    messages = [r.getMessage() for r in caplog.records]
    # Sparse path should NOT have run.
    assert all("sparse" not in m for m in messages), (
        f"asymmetric problem should bypass the sparse path; "
        f"saw: {messages}"
    )
    # Dense general path should have run.
    assert any("dense general" in m for m in messages), (
        f"expected dense general-eig path; saw: {messages}"
    )
    # Sanity: results are well-formed.
    assert eigvals.size == 6
    assert eigvecs.shape == (n, 6)
    assert np.all(np.isfinite(eigvals))
