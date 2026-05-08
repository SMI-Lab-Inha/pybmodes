"""Unified pyBmodes plot style.

A single :func:`apply_style` call sets the journal-paper matplotlib
defaults used across the walkthrough notebook, the Bir 2010 case
scripts (``cases/bir_2010_*``), and the ecosystem-finding case scripts
(``cases/nrel5mw_land``, ``cases/iea3mw_land``, ``cases/nrel5mw_monopile``).

The style is built on top of:

- The Okabe-Ito 8-colour palette, which is colour-blind-safe across the
  most common forms of colour vision deficiency
  (`Okabe & Ito 2008 <https://jfly.uni-koeln.de/color/>`_).
- Serif fonts (STIX Two Text → STIX → DejaVu Serif fallback).
- Inward ticks on all four axes, thin frame, dashed grid.
- 110 DPI inline (notebook), 300 DPI savefig.

Typical usage at the top of a case script::

    import matplotlib
    matplotlib.use("Agg")
    from pybmodes.plots.style import apply_style
    apply_style()

Once :func:`apply_style` has run, every subsequent ``plt.subplots`` call
inherits the unified defaults. The function is idempotent — calling it
multiple times has no further effect.
"""

from __future__ import annotations


# Okabe-Ito colour-blind-safe palette (Okabe & Ito, 2008).
OKABE_ITO: dict[str, str] = {
    "black":  "#000000",
    "orange": "#E69F00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "verm":   "#D55E00",
    "purple": "#CC79A7",
}

#: Ordered list of Okabe-Ito colours used as the matplotlib prop_cycle.
PALETTE: list[str] = list(OKABE_ITO.values())


def apply_style() -> None:
    """Apply the unified pyBmodes journal-paper plot style.

    Mutates ``matplotlib.rcParams`` in place. Safe to call multiple
    times. Raises :class:`ImportError` if matplotlib isn't installed.
    """
    try:
        import matplotlib as mpl
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for pybmodes.plots.style; "
            'install it with: pip install "pybmodes[plots]"'
        ) from exc

    mpl.rcParams.update({
        # --- Fonts ---------------------------------------------------------
        "font.family":        "serif",
        "font.serif":         ["STIX Two Text", "STIX",
                               "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset":   "stix",
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.titleweight":   "bold",
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.titlesize":   12,
        "figure.titleweight": "bold",
        # --- Lines + colours ----------------------------------------------
        "axes.prop_cycle":    mpl.cycler(color=PALETTE),
        "lines.linewidth":    1.6,
        "lines.markersize":   4.0,
        # --- Frame + grid -------------------------------------------------
        "axes.linewidth":     0.8,
        "axes.spines.top":    True,
        "axes.spines.right":  True,
        "axes.grid":          True,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.4,
        "grid.alpha":         0.6,
        # --- Ticks --------------------------------------------------------
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.top":          True,
        "ytick.right":        True,
        "xtick.major.width":  0.7,
        "ytick.major.width":  0.7,
        # --- Legend -------------------------------------------------------
        "legend.frameon":     True,
        "legend.framealpha":  0.95,
        "legend.edgecolor":   "0.85",
        # --- Output -------------------------------------------------------
        "figure.dpi":         110,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })
