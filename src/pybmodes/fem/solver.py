"""Generalized eigenvalue solver for the reduced FEM system.

Solves: K ψ = λ M ψ  using scipy.linalg.eigh (symmetric positive-definite).
Returns eigenvalues sorted ascending and the corresponding eigenvectors.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def solve_modes(
    gk: np.ndarray,
    gm: np.ndarray,
    n_modes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the generalised eigenproblem K ψ = λ M ψ.

    Parameters
    ----------
    gk      : (ngd, ngd) global stiffness matrix
    gm      : (ngd, ngd) global mass matrix
    n_modes : number of lowest modes to return (None = all)

    Returns
    -------
    eigvals : (n_modes,) eigenvalues λ, sorted ascending (λ = (ω_nd)²)
    eigvecs : (ngd, n_modes) eigenvectors, columns correspond to eigvals,
              each normalised to unit L2 norm
    """
    ngd = gk.shape[0]
    if n_modes is None:
        subset = None
    else:
        subset = (0, min(n_modes, ngd) - 1)

    # Symmetrise to suppress any floating-point asymmetry
    gk_sym = 0.5 * (gk + gk.T)
    gm_sym = 0.5 * (gm + gm.T)

    if subset is not None:
        eigvals, eigvecs = eigh(gk_sym, gm_sym, subset_by_index=subset)
    else:
        eigvals, eigvecs = eigh(gk_sym, gm_sym)

    # eigh already returns eigenvalues in ascending order.
    # Normalise eigenvectors to unit L2 norm.
    for j in range(eigvecs.shape[1]):
        norm = np.sqrt(np.sum(eigvecs[:, j] ** 2))
        if norm > 0.0:
            eigvecs[:, j] /= norm

    return eigvals, eigvecs


def eigvals_to_hz(eigvals: np.ndarray, romg: float) -> np.ndarray:
    """Convert non-dimensional eigenvalues to Hz.

    freq_Hz = sqrt(λ_nd) * romg / (2π)

    Parameters
    ----------
    eigvals : non-dimensional eigenvalues (λ = (ω / romg)²)
    romg    : reference angular velocity (rad/s) used in non-dimensionalisation
              (typically romg = 10.0 rad/s)
    """
    return np.sqrt(np.maximum(eigvals, 0.0)) * romg / (2.0 * np.pi)
