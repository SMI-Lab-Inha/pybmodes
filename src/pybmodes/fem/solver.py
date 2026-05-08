"""Generalized eigenvalue solver for the reduced FEM system.

Solves: K ψ = λ M ψ.  Symmetric systems use ``scipy.linalg.eigh``; genuinely
asymmetric offshore-support systems use ``scipy.linalg.eig`` to match BModes'
general matrix solver path.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig, eigh


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

    if _is_effectively_symmetric(gk) and _is_effectively_symmetric(gm):
        # Symmetrise to suppress floating-point scatter before the SPD solve.
        gk_sym = 0.5 * (gk + gk.T)
        gm_sym = 0.5 * (gm + gm.T)
        if subset is not None:
            eigvals, eigvecs = eigh(gk_sym, gm_sym, subset_by_index=subset)
        else:
            eigvals, eigvecs = eigh(gk_sym, gm_sym)
    else:
        eigvals_all, eigvecs_all = eig(gk, gm)
        eigvals_real = np.real_if_close(eigvals_all, tol=1000)
        valid = (
            np.isreal(eigvals_real)
            & np.isfinite(eigvals_real.real)
            & (eigvals_real.real > 0.0)
        )
        eigvals = eigvals_real.real[valid]
        eigvecs = np.real_if_close(eigvecs_all[:, valid], tol=1000).real
        order = np.argsort(eigvals)
        if n_modes is not None:
            order = order[: min(n_modes, order.size)]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

    # eigh already returns eigenvalues in ascending order.
    # Normalise eigenvectors to unit L2 norm.
    for j in range(eigvecs.shape[1]):
        norm = np.sqrt(np.sum(eigvecs[:, j] ** 2))
        if norm > 0.0:
            eigvecs[:, j] /= norm

    return eigvals, eigvecs


def _is_effectively_symmetric(a: np.ndarray) -> bool:
    """Return True for exact/small roundoff asymmetry, False for input asymmetry."""
    scale = max(1.0, float(np.max(np.abs(a))))
    return bool(np.max(np.abs(a - a.T)) <= 1.0e-12 * scale)


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
