"""Campbell-diagram support: rotor-speed sweep with MAC-tracked blade
modes and constant-frequency tower modes overlaid on the same plot.

A Campbell diagram plots a turbine's natural frequencies against rotor
speed and overlays the per-revolution excitation lines (1P, 2P, 3P,
…); crossings between excitation lines and structural-mode lines flag
resonance risks. For a wind-turbine blade the centrifugal-stiffening
contribution to the FEM stiffness matrix raises flap-dominated
frequencies markedly with rotor speed while edgewise (lag-dominated)
modes barely move. The tower lives in an Earth-fixed frame, so its
fore-aft / side-to-side bending frequencies don't depend on rotor
speed at all and show up as horizontal lines on the diagram. The
NREL 5MW turbine's canonical resonance call-out — 3P crossing the
1st tower fore-aft mode near ~6.4 rpm — sits right where the cut-in
operating envelope begins, which is exactly the kind of constraint
this diagram is designed to surface.

Public API
----------

- :func:`campbell_sweep` — given an OpenFAST ElastoDyn main ``.dat``,
  loads the blade and tower from the same deck, sweeps the blade
  across ``omega_rpm`` (with MAC-based mode tracking), solves the
  tower once, and packs both into a single :class:`CampbellResult`.
  ``.bmi`` inputs are also accepted and route to blade-only or
  tower-only sweeps based on ``beam_type``; an explicit
  ``tower_input=...`` keyword adds a tower file alongside a blade
  ``.bmi``.
- :func:`plot_campbell` — renders the result with blade modes as
  solid coloured lines, tower modes as horizontal dashed dark-grey
  lines, and the per-rev excitation family as light grey rays from
  the origin. Optional vertical marker at the rated rotor speed.

Defaults are deliberately spare (``n_blade_modes=4``, ``n_tower_modes=4``)
so the diagram shows the modes that actually drive resonance design —
1st/2nd flap, 1st/2nd edge, 1st/2nd tower FA, 1st/2nd tower SS —
without crowding the plot with high-order modes that the per-rev
family doesn't reach inside any realistic operating envelope.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np

from pybmodes.fem.normalize import NodeModeShape
from pybmodes.io.bmi import BMIFile, read_bmi
from pybmodes.io.sec_props import SectionProperties
from pybmodes.models._pipeline import run_fem


@dataclass
class CampbellResult:
    """Frequencies and labels from a Campbell sweep — blade + tower combined.

    Attributes
    ----------
    omega_rpm : (N,) array of rotor speeds in rpm.
    frequencies : (N, n_total_modes) array of natural frequencies in Hz.
        Columns are ordered *blade modes first, then tower modes*. With
        MAC tracking enabled, blade columns hold the same physical mode
        across all rotor speeds. Tower columns are constant across rows
        (tower frequencies don't depend on rotor speed).
    labels : list of length ``n_total_modes`` with human-readable mode
        names — blade modes look like ``"1st flap"`` / ``"2nd edge"``,
        tower modes are prefixed with ``"tower"`` (e.g.
        ``"1st tower FA"``, ``"1st tower SS"``) so callers can split
        the two by string match if needed.
    participation : (N, n_total_modes, 3) array of energy fractions in
        the FEM's per-mode (flap or FA, edge or SS, torsion) axes.
        Each row sums to 1. Note the axis interpretation is
        beam-type-specific: blade columns use flap/edge/torsion, tower
        columns use FA/SS/torsion.
    mac_to_previous : (N, n_total_modes) array of per-step MAC values
        between each output slot's mode shape at step ``k`` and the
        same slot at step ``k - 1`` (i.e. the tracking confidence).
        Row 0 is filled with NaN (no previous step). Tower columns are
        also NaN (tower modes don't change with rotor speed, so a MAC
        confidence is not meaningful for them).
    n_blade_modes : how many of the leading columns are blade modes.
    n_tower_modes : how many of the trailing columns are tower modes.
    """

    omega_rpm: np.ndarray
    frequencies: np.ndarray
    labels: list[str]
    participation: np.ndarray
    n_blade_modes: int
    n_tower_modes: int
    mac_to_previous: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # ------------------------------------------------------------------
    # NPZ round-trip
    # ------------------------------------------------------------------

    def save(
        self, path: str | pathlib.Path, *,
        source_file: str | pathlib.Path | None = None,
    ) -> None:
        """Write the sweep result to a ``.npz`` archive.

        Arrays go in as named keys; labels and the two integer scalars
        ride in via the embedded JSON ``__meta__`` blob alongside the
        standard pyBmodes-version / timestamp / source-file / git-hash
        metadata captured by :func:`pybmodes.io._serialize._capture_metadata`.
        """
        from pybmodes.io._serialize import _capture_metadata, _metadata_to_npz_value

        meta = _capture_metadata(source_file=source_file)
        meta["labels"] = list(self.labels)
        meta["n_blade_modes"] = int(self.n_blade_modes)
        meta["n_tower_modes"] = int(self.n_tower_modes)

        np.savez_compressed(
            pathlib.Path(path),
            omega_rpm=np.asarray(self.omega_rpm, dtype=float),
            frequencies=np.asarray(self.frequencies, dtype=float),
            participation=np.asarray(self.participation, dtype=float),
            mac_to_previous=np.asarray(self.mac_to_previous, dtype=float),
            __meta__=_metadata_to_npz_value(meta),
        )

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "CampbellResult":
        """Read a sweep result back from a ``.npz`` archive saved by
        :meth:`save`."""
        from pybmodes.io._serialize import _metadata_from_npz_value

        with np.load(pathlib.Path(path), allow_pickle=True) as npz:
            meta = _metadata_from_npz_value(npz["__meta__"])
            return cls(
                omega_rpm=np.asarray(npz["omega_rpm"], dtype=float),
                frequencies=np.asarray(npz["frequencies"], dtype=float),
                labels=list(meta["labels"]),
                participation=np.asarray(npz["participation"], dtype=float),
                n_blade_modes=int(meta["n_blade_modes"]),
                n_tower_modes=int(meta["n_tower_modes"]),
                mac_to_previous=np.asarray(npz["mac_to_previous"], dtype=float),
            )

    # ------------------------------------------------------------------
    # CSV emission
    # ------------------------------------------------------------------

    def to_csv(self, path: str | pathlib.Path) -> None:
        """Write a spreadsheet-friendly CSV with one row per rotor-speed
        step.

        Columns: ``rpm``, then one frequency column per mode (named by
        the mode's label), then one MAC-confidence column per mode
        suffixed with ``_mac``. Tower-mode MAC columns are NaN
        throughout because tower modes don't change with rotor speed —
        kept as columns for shape-stability across blade-only / tower-
        only / mixed sweeps.
        """
        import csv

        n_steps, n_modes = self.frequencies.shape
        freq_cols = list(self.labels)
        mac_cols = [f"{lbl}_mac" for lbl in self.labels]
        header = ["rpm", *freq_cols, *mac_cols]

        with pathlib.Path(path).open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for step in range(n_steps):
                row: list[object] = [float(self.omega_rpm[step])]
                row.extend(float(self.frequencies[step, k]) for k in range(n_modes))
                # Per-mode MAC confidence (NaN where unset / not meaningful).
                if self.mac_to_previous.shape == self.frequencies.shape:
                    row.extend(
                        float(self.mac_to_previous[step, k])
                        for k in range(n_modes)
                    )
                else:
                    row.extend([float("nan")] * n_modes)
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# A "model" is a (BMIFile, SectionProperties|None) tuple. ``None`` for the
# section-properties slot signals that ``run_fem`` should re-read them
# from disk via ``BMIFile.resolve_sec_props_path``; ElastoDyn-derived
# models supply them directly.
_Model = tuple[BMIFile, SectionProperties | None]


def _load_models(
    input_path: pathlib.Path,
    tower_input: pathlib.Path | None,
) -> tuple[_Model | None, _Model | None]:
    """Resolve the input path(s) to (blade, tower) model pairs.

    For an ElastoDyn ``.dat`` file the deck carries both, so we load
    both unless the corresponding files can't be resolved. ``.bmi``
    inputs are routed to blade or tower by their ``beam_type``. The
    ``tower_input`` keyword lets a caller pair a blade ``.bmi`` with
    an explicit tower ``.bmi``; if the primary input was an ElastoDyn
    deck and ``tower_input`` is also given, ``tower_input`` overrides
    the deck-supplied tower (useful when the deck's tower file points
    somewhere unhelpful).
    """
    suffix = input_path.suffix.lower()

    blade: _Model | None = None
    tower: _Model | None = None

    if suffix == ".dat":
        from pybmodes.io.elastodyn_reader import (
            read_elastodyn_blade,
            read_elastodyn_main,
            read_elastodyn_tower,
            to_pybmodes_blade,
            to_pybmodes_tower,
        )
        main = read_elastodyn_main(input_path)
        bld_path = input_path.parent / main.bld_file[0]
        blade_data = read_elastodyn_blade(bld_path)
        blade = to_pybmodes_blade(main, blade_data)

        twr_path = input_path.parent / main.twr_file
        if twr_path.is_file():
            tower_data = read_elastodyn_tower(twr_path)
            tower = to_pybmodes_tower(main, tower_data, blade=blade_data)
    elif suffix == ".bmi":
        bmi = read_bmi(input_path)
        if bmi.beam_type == 1:
            blade = (bmi, None)
        elif bmi.beam_type == 2:
            tower = (bmi, None)
        else:
            raise ValueError(
                f"unsupported beam_type {bmi.beam_type} in {input_path}"
            )
    else:
        raise ValueError(
            f"campbell_sweep input must be .bmi or ElastoDyn .dat; "
            f"got {input_path.suffix!r}"
        )

    if tower_input is not None:
        if tower_input.suffix.lower() != ".bmi":
            raise ValueError(
                f"tower_input must be a .bmi file; got {tower_input.suffix!r}"
            )
        tower_bmi = read_bmi(tower_input)
        if tower_bmi.beam_type != 2:
            raise ValueError(
                f"tower_input {tower_input} has beam_type {tower_bmi.beam_type}, "
                f"expected 2 (tower)"
            )
        tower = (tower_bmi, None)

    return blade, tower


def _shape_vector(shape: NodeModeShape) -> np.ndarray:
    return np.concatenate([shape.flap_disp, shape.lag_disp, shape.twist])


def _participation(shape: NodeModeShape) -> np.ndarray:
    """Energy fractions in axes 0 / 1 / 2 (sum to 1; zeros if shape is null).

    For a blade these read flap / edge / torsion; for a tower they read
    FA / SS / torsion (same FEM DOF layout, different physical naming).
    """
    flap = float(np.dot(shape.flap_disp, shape.flap_disp))
    edge = float(np.dot(shape.lag_disp, shape.lag_disp))
    twist = float(np.dot(shape.twist, shape.twist))
    total = flap + edge + twist
    if total <= 0.0:
        return np.zeros(3)
    return np.array([flap, edge, twist]) / total


def _mac_matrix(
    curr: list[NodeModeShape],
    prev: list[NodeModeShape],
) -> np.ndarray:
    """``mac[i, j] = (curr_i · prev_j)² / (||curr_i||² · ||prev_j||²)``."""
    curr_v = np.array([_shape_vector(s) for s in curr])
    prev_v = np.array([_shape_vector(s) for s in prev])
    inner = curr_v @ prev_v.T
    curr_n = np.einsum("ij,ij->i", curr_v, curr_v)
    prev_n = np.einsum("ij,ij->i", prev_v, prev_v)
    denom = np.outer(curr_n, prev_n)
    safe = denom > 0.0
    mac = np.zeros_like(inner)
    mac[safe] = (inner[safe] ** 2) / denom[safe]
    return mac


def _hungarian_assignment(mac: np.ndarray) -> np.ndarray:
    """Global MAC-maximising assignment via the Hungarian (Munkres)
    algorithm.

    Returns ``order[i] = j`` mapping current-step mode ``i`` to the
    previous-step slot ``j`` that maximises the sum of MAC values
    across all matched pairs. This is the standard industry approach
    for mode tracking — it avoids the failure mode of the older
    greedy ``argmax(mac)`` scheme, which can lock in a slightly-
    better first match and force later modes into worse pairings.

    Non-square inputs are handled natively by
    ``scipy.optimize.linear_sum_assignment``: it returns
    ``min(n_curr, n_prev)`` matched pairs, and any current-step row
    that did not receive a previous-step pairing stays at the
    sentinel ``-1`` in the output. The caller (``_solve_blade_sweep``)
    fills those slots from any free previous-step indices, so a
    non-square call still produces a well-defined ordering for every
    current-step mode. In practice the Campbell sweep always supplies
    square ``(n_modes, n_modes)`` inputs; the non-square fallback is
    defensive.
    """
    from scipy.optimize import linear_sum_assignment

    n_curr, _ = mac.shape
    row_ind, col_ind = linear_sum_assignment(mac, maximize=True)
    order = -np.ones(n_curr, dtype=int)
    order[row_ind] = col_ind
    return order


# Kept as a thin wrapper for backwards compatibility — older callers
# (and tests) may import ``_greedy_assignment`` by name. Delegates to
# the Hungarian-based implementation.
def _greedy_assignment(mac: np.ndarray) -> np.ndarray:
    """Deprecated alias for :func:`_hungarian_assignment` — kept for
    backwards compatibility; new code should call the Hungarian
    version directly."""
    return _hungarian_assignment(mac)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _label_blade_modes(participation_row: np.ndarray) -> list[str]:
    """``"1st flap"`` / ``"2nd edge"`` / … from participation at one rotor speed."""
    n = participation_row.shape[0]
    counts = [0, 0, 0]
    names = ("flap", "edge", "torsion")
    out: list[str] = []
    for i in range(n):
        axis = int(np.argmax(participation_row[i]))
        counts[axis] += 1
        out.append(f"{_ordinal(counts[axis])} {names[axis]}")
    return out


def _label_tower_modes(participation_row: np.ndarray) -> list[str]:
    """``"1st tower FA"`` / ``"1st tower SS"`` / …."""
    n = participation_row.shape[0]
    counts = [0, 0, 0]
    names = ("FA", "SS", "torsion")
    out: list[str] = []
    for i in range(n):
        axis = int(np.argmax(participation_row[i]))
        counts[axis] += 1
        out.append(f"{_ordinal(counts[axis])} tower {names[axis]}")
    return out


def _solve_blade_sweep(
    blade: _Model,
    omega_rpm: np.ndarray,
    n_modes: int,
    track_by_mac: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Run the rotor-speed sweep on the blade model.

    Returns ``(frequencies, participation, labels, mac_to_previous)``
    with shapes ``(n_steps, n_modes)``, ``(n_steps, n_modes, 3)``, a
    list of ``n_modes`` labels, and ``(n_steps, n_modes)`` per-step
    MAC values vs the immediately-preceding step (row 0 is NaN).
    Restores the original ``bbmi.rot_rpm`` after the sweep so the
    caller's BMI object is unmutated.
    """
    bbmi, bsp = blade
    original_rpm = float(getattr(bbmi, "rot_rpm", 0.0))
    n_steps = omega_rpm.size
    freqs = np.zeros((n_steps, n_modes))
    parts = np.zeros((n_steps, n_modes, 3))
    mac_to_prev = np.full((n_steps, n_modes), np.nan, dtype=float)
    slot_shapes: list[NodeModeShape] | None = None

    try:
        for step, rpm in enumerate(omega_rpm):
            bbmi.rot_rpm = float(rpm)
            modal = run_fem(bbmi, n_modes=n_modes, sp=bsp)
            shapes = list(modal.shapes[:n_modes])
            f_step = np.asarray(modal.frequencies[:n_modes], dtype=float)
            p_step = np.array([_participation(s) for s in shapes])

            if step == 0 or not track_by_mac or slot_shapes is None:
                order = np.arange(n_modes, dtype=int)
                mac_row = np.full(n_modes, np.nan, dtype=float)
            else:
                mac = _mac_matrix(shapes, slot_shapes)
                order = _hungarian_assignment(mac)
                free = [s for s in range(n_modes) if s not in order]
                for k in range(n_modes):
                    if order[k] < 0 and free:
                        order[k] = free.pop(0)
                # MAC confidence of the chosen pairing per output slot.
                mac_row = np.empty(n_modes, dtype=float)
                for k in range(n_modes):
                    slot = int(order[k])
                    mac_row[slot] = float(mac[k, slot]) if slot >= 0 else np.nan

            for k in range(n_modes):
                slot = int(order[k])
                freqs[step, slot] = f_step[k]
                parts[step, slot, :] = p_step[k]
            mac_to_prev[step, :] = mac_row

            new_slot_shapes: list[NodeModeShape | None] = [None] * n_modes
            for k in range(n_modes):
                new_slot_shapes[int(order[k])] = shapes[k]
            slot_shapes = [s for s in new_slot_shapes if s is not None]
    finally:
        bbmi.rot_rpm = original_rpm

    labels = _label_blade_modes(parts[0])
    return freqs, parts, labels, mac_to_prev


def _solve_tower_once(
    tower: _Model,
    n_modes: int,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Solve the tower once at ``rot_rpm = 0`` and broadcast across the sweep.

    Tower modes are rotor-speed-independent (the tower lives in an
    Earth-fixed frame), so a single eigensolve is enough; we tile the
    result across ``n_steps`` rows for shape compatibility with the
    blade-sweep output.
    """
    tbmi, tsp = tower
    tbmi.rot_rpm = 0.0
    modal = run_fem(tbmi, n_modes=n_modes, sp=tsp)
    tshapes = list(modal.shapes[:n_modes])
    tfreqs = np.asarray(modal.frequencies[:n_modes], dtype=float)
    tparts = np.array([_participation(s) for s in tshapes])

    freqs = np.broadcast_to(tfreqs, (n_steps, n_modes)).copy()
    parts = np.broadcast_to(tparts, (n_steps, n_modes, 3)).copy()
    labels = _label_tower_modes(tparts)
    return freqs, parts, labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def campbell_sweep(
    input_path: str | pathlib.Path,
    omega_rpm: np.ndarray,
    n_blade_modes: int = 4,
    n_tower_modes: int = 4,
    *,
    tower_input: str | pathlib.Path | None = None,
    track_by_mac: bool = True,
) -> CampbellResult:
    """Build a Campbell-diagram dataset for the given turbine.

    Parameters
    ----------
    input_path :
        Path to either:

        - an OpenFAST ElastoDyn main ``.dat`` file — the function
          loads the blade *and* the tower from the deck and runs both;
        - a blade ``.bmi`` (``beam_type = 1``) — blade-only sweep
          unless ``tower_input`` is also supplied;
        - a tower ``.bmi`` (``beam_type = 2``) — tower-only result
          (frequencies are constant across ``omega_rpm``; the result
          is mostly useful for overlay against the per-rev family).
    omega_rpm :
        1-D array of rotor speeds in rpm. ``Ω = 0`` is fine and
        produces the parked-rotor frequencies.
    n_blade_modes :
        Number of blade modes to extract per speed and report in
        ``frequencies[:, :n_blade_modes]``. Default 4 covers
        1st/2nd flap and 1st/2nd edge — the modes that actually drive
        resonance design. Pushing this much higher just adds
        high-order flap modes that no realistic per-rev family
        crosses inside the operating envelope; raise it deliberately
        when you need them.
    n_tower_modes :
        Number of tower modes (default 4 — 1st/2nd FA + 1st/2nd SS).
        Drop to 2 to overlay only the 1st FA + 1st SS pair, or push
        higher for offshore decks where 3rd-mode crossings matter.
        Ignored when no tower model is available.
    tower_input :
        Optional explicit tower ``.bmi`` (keyword-only). Useful when
        ``input_path`` is a blade-only deck. Overrides the deck-
        supplied tower if ``input_path`` was an ElastoDyn ``.dat``.
    track_by_mac :
        Whether to use MAC across consecutive rotor speeds to keep
        each blade output column corresponding to the same physical
        mode. ``False`` returns the eigensolver's native order (useful
        for debugging mode re-ordering issues). Tower modes don't
        change with rotor speed and are unaffected by this flag.

    Returns
    -------
    :class:`CampbellResult`.
    """
    path = pathlib.Path(input_path)
    tower_path = pathlib.Path(tower_input) if tower_input is not None else None
    blade, tower = _load_models(path, tower_path)

    omega_rpm = np.asarray(omega_rpm, dtype=float).ravel()
    if omega_rpm.size == 0:
        raise ValueError("omega_rpm must contain at least one rotor speed")
    if not np.all(np.isfinite(omega_rpm)):
        raise ValueError(
            "omega_rpm must be finite; found NaN or inf in "
            f"{omega_rpm.tolist()!r}"
        )
    if np.any(omega_rpm < 0.0):
        raise ValueError(
            "omega_rpm must be non-negative (rotor speeds in rpm); found "
            f"min = {float(omega_rpm.min())!r}"
        )
    if omega_rpm.size >= 2 and np.any(np.diff(omega_rpm) < 0.0):
        raise ValueError(
            "omega_rpm must be sorted ascending so MAC tracking can pair "
            "consecutive steps; got "
            f"{omega_rpm.tolist()!r}"
        )
    if not isinstance(n_blade_modes, int) or n_blade_modes < 0:
        raise ValueError(
            f"n_blade_modes must be a non-negative integer; got {n_blade_modes!r}"
        )
    if not isinstance(n_tower_modes, int) or n_tower_modes < 0:
        raise ValueError(
            f"n_tower_modes must be a non-negative integer; got {n_tower_modes!r}"
        )

    # Silently zero-out mode counts for components that aren't present —
    # easier on the caller than raising for the common "no tower" case.
    if blade is None:
        n_blade_modes = 0
    if tower is None:
        n_tower_modes = 0
    if n_blade_modes + n_tower_modes < 1:
        raise ValueError(
            "no modes to compute: input had neither a blade nor a tower "
            "component, or both n_blade_modes and n_tower_modes were 0"
        )

    n_steps = omega_rpm.size
    blade_freqs = blade_parts = None
    blade_labels: list[str] = []
    blade_mac: np.ndarray | None = None
    if blade is not None and n_blade_modes > 0:
        blade_freqs, blade_parts, blade_labels, blade_mac = _solve_blade_sweep(
            blade, omega_rpm, n_blade_modes, track_by_mac,
        )

    tower_freqs = tower_parts = None
    tower_labels: list[str] = []
    if tower is not None and n_tower_modes > 0:
        tower_freqs, tower_parts, tower_labels = _solve_tower_once(
            tower, n_tower_modes, n_steps,
        )

    parts_pieces = [a for a in (blade_parts, tower_parts) if a is not None]
    freqs_pieces = [a for a in (blade_freqs, tower_freqs) if a is not None]
    frequencies = np.concatenate(freqs_pieces, axis=1)
    participation = np.concatenate(parts_pieces, axis=1)
    labels = blade_labels + tower_labels

    # Build the per-step MAC table: blade columns get the tracked
    # MACs from the sweep; tower columns are NaN (no rotor-speed
    # dependence, so a MAC confidence is not meaningful).
    mac_pieces: list[np.ndarray] = []
    if blade_mac is not None:
        mac_pieces.append(blade_mac)
    if tower_freqs is not None:
        mac_pieces.append(np.full((n_steps, n_tower_modes), np.nan))
    if mac_pieces:
        mac_to_previous = np.concatenate(mac_pieces, axis=1)
    else:
        mac_to_previous = np.empty((n_steps, 0))

    return CampbellResult(
        omega_rpm=omega_rpm,
        frequencies=frequencies,
        labels=labels,
        participation=participation,
        n_blade_modes=n_blade_modes,
        n_tower_modes=n_tower_modes,
        mac_to_previous=mac_to_previous,
    )


def plot_campbell(
    result: CampbellResult,
    excitation_orders: list[int] | None = None,
    rated_rpm: float | None = None,
    ax=None,
):
    """Render a Campbell diagram from a :class:`CampbellResult`.

    Blade modes are drawn as solid coloured lines (using whatever
    matplotlib ``axes.prop_cycle`` is active — call
    :func:`pybmodes.plots.apply_style` first for the MATLAB-styled
    defaults), tower modes as horizontal dashed dark-grey lines with a
    right-margin frequency label so the dashes are unambiguous, and
    the per-rev excitation family as red dotted rays shaded
    medium-to-dark by ascending order.

    Note on blade-line jitter
    -------------------------
    For ElastoDyn-derived blade FEMs the 1st-flap line typically shows
    ~5 % step-to-step scatter — *not* real Southwell dynamics. The
    BMI adapter floors rotary inertia and forces near-rigid axial
    behaviour (``EA / EI ≈ 1e6``), leaving the dense FEM matrices
    ill-conditioned (κ(M) ≈ 1e11), which makes LAPACK's subset
    eigenvalue routines wobble on the lowest mode even when the
    underlying eigenvector is identical step to step. The MAC tracker
    catches this — the participation array stays > 98 % flap-dominant
    in the 1st-flap slot — so the mode *identity* is correct, only
    the eigenvalue precision suffers. Centrifugal stiffening is
    monotonic in physics (Wright 1982); endpoint-to-endpoint
    comparisons (parked vs rated) are reliable, individual-step
    monotonicity is not.

    Parameters
    ----------
    result :
        Output of :func:`campbell_sweep`.
    excitation_orders :
        Per-rev orders to overlay; default ``[1, 2, 3, 6, 9]`` covers
        1P (rotor) + 3P (3-bladed blade-passing) + the harmonics most
        often called out in design reviews.
    rated_rpm :
        If supplied, draws a vertical reference line at the operating
        rotor speed.
    ax :
        Existing matplotlib Axes to draw into; if ``None`` a fresh
        figure is created.

    Returns
    -------
    :class:`matplotlib.figure.Figure` for the rendered axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_campbell; install with "
            'pip install "pybmodes[plots]"'
        ) from exc

    if excitation_orders is None:
        excitation_orders = [1, 2, 3, 6, 9]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
    else:
        fig = ax.figure

    rpm = result.omega_rpm
    rpm_max = float(rpm.max()) if rpm.size > 0 else 0.0
    rpm_grid = np.array([0.0, rpm_max])

    # Per-rev excitation rays — drawn behind the mode lines but in red
    # so they read as the resonance-warning lines they are. Sample the
    # ``Reds`` colormap from medium to dark so consecutive orders are
    # visually distinguishable without a legend lookup; thicker stroke
    # than the structural-mode lines so the rays stay legible when they
    # cross dense mode clusters.
    n_orders = max(len(excitation_orders), 1)
    cmap = plt.get_cmap("Reds")
    for i, order in enumerate(excitation_orders):
        shade = cmap(0.45 + 0.50 * (i / max(n_orders - 1, 1)))
        ax.plot(
            rpm_grid,
            order * rpm_grid / 60.0,
            ":",
            color=shade,
            linewidth=1.4,
            label=f"{order}P",
            zorder=1,
        )

    n_blade = result.n_blade_modes
    n_tower = result.n_tower_modes

    # Blade modes: solid coloured lines from the active prop_cycle.
    for k in range(n_blade):
        ax.plot(
            rpm,
            result.frequencies[:, k],
            "-o",
            markersize=3.5,
            label=result.labels[k],
            zorder=3,
        )

    # Tower modes: horizontal dashed dark-grey lines.
    #
    # Right-margin labels replace per-line legend entries — the dashed-
    # grey style already encodes "this is tower" and the legend would
    # otherwise carry redundant rows. Modes within 2 % of each other in
    # frequency get a single merged label (e.g. "1st FA / SS") so a
    # near-symmetric tower (FA ≈ SS, common case) doesn't stack two
    # text labels on top of each other.
    label_x = rpm_max if rpm_max > 0 else 1.0
    tower_groups: list[dict] = []
    for k in range(n_blade, n_blade + n_tower):
        f = float(result.frequencies[0, k])
        ax.axhline(
            f,
            linestyle="--",
            color=(0.25, 0.25, 0.25),
            linewidth=1.1,
            zorder=2,
        )
        short = result.labels[k].replace("tower ", "")
        merged = False
        for g in tower_groups:
            if abs(g["f"] - f) / max(g["f"], 1e-9) < 0.02:
                g["names"].append(short)
                # Take the mean for the printed frequency so a slight
                # FA/SS asymmetry shows up rounded sensibly.
                g["f"] = 0.5 * (g["f"] + f)
                merged = True
                break
        if not merged:
            tower_groups.append({"f": f, "names": [short]})

    for g in tower_groups:
        text = " / ".join(g["names"]) + f" ({g['f']:.2f} Hz)"
        ax.text(
            label_x,
            g["f"],
            f" {text}",
            color=(0.20, 0.20, 0.20),
            fontsize=8,
            va="bottom",
            ha="left",
            zorder=4,
            clip_on=False,
        )

    if rated_rpm is not None:
        ax.axvline(
            rated_rpm,
            color=(0.35, 0.35, 0.35),
            linestyle="-.",
            linewidth=0.8,
            label=f"rated {rated_rpm:g} rpm",
            zorder=1.5,
        )

    ax.set_xlabel("Rotor speed (rpm)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Campbell diagram")
    ax.set_xlim(0.0, rpm_max if rpm_max > 0.0 else 1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    return fig


__all__ = ["CampbellResult", "campbell_sweep", "plot_campbell"]
