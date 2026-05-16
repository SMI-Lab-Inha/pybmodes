"""Name the floating-platform rigid-body modes.

For a free-free tower (``hub_conn = 2``) carrying a
:class:`~pybmodes.io.bmi.PlatformSupport`, the low modes are the
platform rigid-body modes — surge / sway / heave / roll / pitch / yaw.
They are identifiable from the tower-base node's 6-DOF motion in the
eigenvector, weighted by the platform 6×6 inertia (which supplies the
mass / moment-of-inertia metric that makes a translation amplitude
comparable to a rotation amplitude).

The classifier is deliberately *conservative*: it names a mode only
when one platform DOF clearly dominates the base-node modal kinetic
energy. A flexible tower mode, or a strongly coupled / eigensolver-
rotated near-degenerate pair (surge≈sway, roll≈pitch), is left
``None`` rather than mislabelled — consistent with the project's
"only name what's unambiguous" stance. Empirically every genuine
rigid-body mode on the validated floating decks (OC3 Hywind, the
IEA-15 / IEA-22 / OC4 / UPSCALE samples) is overwhelmingly single-DOF
and is named; the first flexible bending pair and above stay ``None``.
"""

from __future__ import annotations

import numpy as np

from pybmodes.fem.boundary import NESH

# Platform DOF names in OpenFAST order.
_PLATFORM_DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")

# FEM base-node DOF order is [axial, v_disp, v_slope, w_disp, w_slope,
# phi]; the platform (file) order is [surge, sway, heave, roll, pitch,
# yaw]. This index list reorders an FEM-ordered base 6-vector into
# platform order — the inverse of the ``P`` reorder in
# ``pybmodes.fem.nondim._rigid_arm_T``:
#   surge ← v_disp(1) · sway ← w_disp(3) · heave ← axial(0)
#   roll  ← w_slope(4) · pitch ← v_slope(2) · yaw ← phi(5)
_FEM_TO_PLATFORM = np.array([1, 3, 0, 4, 2, 5])

# A mode is named only if its dominant platform DOF carries at least
# this fraction of the base-node modal kinetic energy. 0.6 cleanly
# separates the genuine (single-DOF) rigid-body modes from
# coupled / rotated ones on every validated floating deck.
_DOMINANCE_THRESHOLD = 0.6

# A 6-DOF rigid platform has exactly 6 rigid-body modes, and for any
# real floating wind system they are the 6 lowest-frequency modes:
# the mooring / hydrostatic restoring is orders of magnitude softer
# than the tower bending stiffness, so the rigid-body periods
# (10–100 s) sit far below the first flexible tower mode (1–2 s) —
# a large spectral gap on every validated deck (OC3 0.12→0.48 Hz,
# IEA-15 0.049→0.525 Hz). Only the lowest ``_N_RIGID`` modes are
# rigid-body candidates; a free-free flexible bending mode also moves
# the base and would otherwise be mis-named.
_N_RIGID = 6


def classify_platform_modes(
    eigvecs: np.ndarray,
    active_dofs: np.ndarray,
    nselt: int,
    platform_mass: np.ndarray,
) -> list[str | None]:
    """Return a per-mode label list naming the platform rigid-body
    modes (``surge`` / … / ``yaw``) or ``None`` where no single
    platform DOF dominates.

    Parameters
    ----------
    eigvecs : (ngd, n_modes) compact (active-DOF) eigenvectors, the
        array :func:`pybmodes.fem.solver.solve_modes` returns.
    active_dofs : (ngd,) sorted global indices of the active DOFs
        (from :func:`pybmodes.fem.boundary.active_dof_indices`), used
        to scatter the compact eigenvector back to full DOF size — the
        same expansion :func:`pybmodes.fem.normalize.extract_mode_shapes`
        performs.
    nselt : number of beam elements.
    platform_mass : the platform 6×6 inertia at the tower base in FEM
        DOF order (``PlatformND.mass`` from
        :func:`pybmodes.fem.nondim.nondim_platform`). Supplies the
        mass / inertia metric for the energy weighting.

    Caller must invoke this only for a floating model
    (``hub_conn == 2`` with a ``PlatformSupport``); for any other
    model there are no rigid-body modes to name.
    """
    ndt = NESH * nselt + 6
    n_modes = eigvecs.shape[1]

    ev_full = np.zeros((ndt, n_modes))
    ev_full[active_dofs, :] = eigvecs

    root_base = NESH * nselt          # base-node block start
    base = ev_full[root_base:root_base + 6, :]   # (6, n_modes), FEM order

    Mp = np.asarray(platform_mass, dtype=float)

    # Modes are returned ascending in frequency, so the rigid-body
    # candidates are the first _N_RIGID columns.
    n_rigid = min(_N_RIGID, n_modes)

    labels: list[str | None] = []
    used: set[str] = set()
    for m in range(n_modes):
        if m >= n_rigid:
            labels.append(None)                 # flexible tower mode
            continue
        b = base[:, m]
        # Per-DOF modal kinetic energy contribution b_i·(M_p b)_i. The
        # platform mass matrix is the metric that puts translation and
        # rotation amplitudes on a comparable footing.
        e = np.abs(b * (Mp @ b))
        total = float(e.sum())
        if total <= 0.0 or not np.isfinite(total):
            labels.append(None)
            continue
        frac = e[_FEM_TO_PLATFORM] / total      # platform-DOF order
        k = int(np.argmax(frac))
        if frac[k] < _DOMINANCE_THRESHOLD:
            labels.append(None)                 # coupled / rotated pair
            continue
        name = _PLATFORM_DOF_NAMES[k]
        if name in used:
            # A 6-DOF platform has exactly one rigid-body mode per DOF.
            # Seeing the same dominant DOF twice within the lowest six
            # means the one-mode-per-DOF assumption has already failed
            # (a degenerate / rotated pair, or a flexible mode leaking
            # in) — emitting a suffixed physical label like "surge (2)"
            # would mislead a downstream plot/report into treating the
            # duplicate as meaningful. Stay conservative: leave it
            # ``None``, consistent with this classifier's "name only
            # the unambiguous" contract.
            labels.append(None)
            continue
        used.add(name)
        labels.append(name)

    return labels
