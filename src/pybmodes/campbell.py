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

Defaults are deliberately spare (``n_blade_modes=4``, ``n_tower_modes=2``)
so the diagram shows the modes that actually drive resonance design —
1st/2nd flap, 1st/2nd edge, 1st tower FA, 1st tower SS — without
crowding the plot with high-order modes that the per-rev family doesn't
reach inside any realistic operating envelope.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

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
    n_blade_modes : how many of the leading columns are blade modes.
    n_tower_modes : how many of the trailing columns are tower modes.
    """

    omega_rpm: np.ndarray
    frequencies: np.ndarray
    labels: list[str]
    participation: np.ndarray
    n_blade_modes: int
    n_tower_modes: int


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
        tp = pathlib.Path(tower_input)
        if tp.suffix.lower() != ".bmi":
            raise ValueError(
                f"tower_input must be a .bmi file; got {tp.suffix!r}"
            )
        tower_bmi = read_bmi(tp)
        if tower_bmi.beam_type != 2:
            raise ValueError(
                f"tower_input {tp} has beam_type {tower_bmi.beam_type}, "
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


def _greedy_assignment(mac: np.ndarray) -> np.ndarray:
    """Return ``order[i] = j`` mapping current mode ``i`` to prev slot ``j``.

    Greedy descending-MAC assignment; unmatched rows return ``-1`` and
    are filled in by the caller from any free slots.
    """
    n_curr, n_prev = mac.shape
    order = -np.ones(n_curr, dtype=int)
    used_prev = np.zeros(n_prev, dtype=bool)
    used_curr = np.zeros(n_curr, dtype=bool)
    flat = np.argsort(mac.ravel())[::-1]
    for idx in flat:
        i, j = divmod(int(idx), n_prev)
        if used_curr[i] or used_prev[j]:
            continue
        order[i] = j
        used_curr[i] = True
        used_prev[j] = True
        if used_curr.all() or used_prev.all():
            break
    return order


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
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the rotor-speed sweep on the blade model.

    Returns ``(frequencies, participation, labels)`` with shapes
    ``(n_steps, n_modes)``, ``(n_steps, n_modes, 3)``, and a list of
    ``n_modes`` labels.
    """
    bbmi, bsp = blade
    n_steps = omega_rpm.size
    freqs = np.zeros((n_steps, n_modes))
    parts = np.zeros((n_steps, n_modes, 3))
    slot_shapes: list[NodeModeShape] | None = None

    for step, rpm in enumerate(omega_rpm):
        bbmi.rot_rpm = float(rpm)
        modal = run_fem(bbmi, n_modes=n_modes, sp=bsp)
        shapes = list(modal.shapes[:n_modes])
        f_step = np.asarray(modal.frequencies[:n_modes], dtype=float)
        p_step = np.array([_participation(s) for s in shapes])

        if step == 0 or not track_by_mac or slot_shapes is None:
            order = np.arange(n_modes, dtype=int)
        else:
            mac = _mac_matrix(shapes, slot_shapes)
            order = _greedy_assignment(mac)
            free = [s for s in range(n_modes) if s not in order]
            for k in range(n_modes):
                if order[k] < 0 and free:
                    order[k] = free.pop(0)

        for k in range(n_modes):
            slot = int(order[k])
            freqs[step, slot] = f_step[k]
            parts[step, slot, :] = p_step[k]

        new_slot_shapes: list[NodeModeShape | None] = [None] * n_modes
        for k in range(n_modes):
            new_slot_shapes[int(order[k])] = shapes[k]
        slot_shapes = [s for s in new_slot_shapes if s is not None]

    labels = _label_blade_modes(parts[0])
    return freqs, parts, labels


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
    n_tower_modes: int = 2,
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
        Number of tower modes (default 2 — 1st FA + 1st SS). 4 also
        works well if you want 2nd FA / 2nd SS for offshore decks.
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
    blade, tower = _load_models(path, tower_input)

    omega_rpm = np.asarray(omega_rpm, dtype=float).ravel()
    if omega_rpm.size == 0:
        raise ValueError("omega_rpm must contain at least one rotor speed")
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
    if blade is not None and n_blade_modes > 0:
        blade_freqs, blade_parts, blade_labels = _solve_blade_sweep(
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

    return CampbellResult(
        omega_rpm=omega_rpm,
        frequencies=frequencies,
        labels=labels,
        participation=participation,
        n_blade_modes=n_blade_modes,
        n_tower_modes=n_tower_modes,
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
    defaults), tower modes as horizontal dashed dark-grey lines, and
    the per-rev excitation family as light-grey dotted rays from the
    origin.

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

    # Per-rev excitation rays — drawn first so they sit behind the modes.
    for order in excitation_orders:
        ax.plot(
            rpm_grid,
            order * rpm_grid / 60.0,
            ":",
            color=(0.55, 0.55, 0.55),
            linewidth=0.9,
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
    for k in range(n_blade, n_blade + n_tower):
        f = float(result.frequencies[0, k])  # constant across rpm
        ax.axhline(
            f,
            linestyle="--",
            color=(0.25, 0.25, 0.25),
            linewidth=1.1,
            label=result.labels[k],
            zorder=2,
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
