"""Tests for ``pybmodes.mac``.

Three spec-named tests:

* ``test_mac_identity_same_shapes`` — MAC matrix between a set of
  shapes and itself has 1.0 on the diagonal and < 1 off it (for
  shapes that aren't pairwise parallel).
* ``test_mac_orthogonal_shapes`` — MAC matrix between two mutually
  orthogonal sets is the zero matrix.
* ``test_compare_modes_frequency_shift_sign`` — when the modified
  result has higher frequencies than baseline, the per-pair
  frequency_shift entries are positive; lower → negative.

Plus a few support tests covering the empty / mismatched-DOF inputs
and the basic plot_mac smoke (matplotlib only — skipped when the
``[plots]`` extra isn't installed).
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fem.normalize import NodeModeShape
from pybmodes.mac import compare_modes, mac_matrix, shape_to_vector
from pybmodes.models.result import ModalResult


def _shape(idx: int, flap: np.ndarray, lag: np.ndarray, twist: np.ndarray,
           freq: float = 1.0) -> NodeModeShape:
    """Build a NodeModeShape from explicit flap / lag / twist arrays.

    Slopes are filled with zeros (MAC doesn't use them); span_loc is
    a regular [0, 1] grid matching the array length."""
    n = flap.size
    return NodeModeShape(
        mode_number=idx,
        freq_hz=freq,
        span_loc=np.linspace(0.0, 1.0, n),
        flap_disp=flap,
        flap_slope=np.zeros(n),
        lag_disp=lag,
        lag_slope=np.zeros(n),
        twist=twist,
    )


# ---------------------------------------------------------------------------
# Spec-named tests
# ---------------------------------------------------------------------------

def test_mac_identity_same_shapes() -> None:
    """MAC of a set of distinct shapes against itself has 1.0 on the
    diagonal. Off-diagonal entries are < 1 for shapes that aren't
    pairwise parallel."""
    n_nodes = 8
    # Three FA-dominated shapes with different spanwise profiles.
    s1 = _shape(1, np.linspace(0.0, 1.0, n_nodes),
                np.zeros(n_nodes), np.zeros(n_nodes))
    s2 = _shape(2, np.linspace(0.0, 1.0, n_nodes) ** 2,
                np.zeros(n_nodes), np.zeros(n_nodes))
    s3 = _shape(3, np.sin(np.linspace(0.0, np.pi, n_nodes)),
                np.zeros(n_nodes), np.zeros(n_nodes))
    shapes = [s1, s2, s3]

    mac = mac_matrix(shapes, shapes)
    assert mac.shape == (3, 3)
    np.testing.assert_allclose(np.diag(mac), np.ones(3), atol=1e-12)
    # Off-diagonal entries are bounded above by 1 and not all unity.
    off = mac - np.diag(np.diag(mac))
    assert np.all(off <= 1.0 + 1e-12)
    assert np.any(off < 0.999), (
        "off-diagonal MAC should be < 1 for the three distinct shapes; "
        f"saw matrix:\n{mac}"
    )


def test_mac_orthogonal_shapes() -> None:
    """Two shape sets that live on orthogonal FEM axes (one FA-only,
    one SS-only) produce a zero MAC matrix."""
    n_nodes = 6
    zeros = np.zeros(n_nodes)
    # Pure-FA shapes: only flap component.
    fa_only = [
        _shape(1, np.linspace(0.0, 1.0, n_nodes), zeros, zeros),
        _shape(2, np.linspace(0.0, 1.0, n_nodes) ** 2, zeros, zeros),
    ]
    # Pure-SS shapes: only lag component.
    ss_only = [
        _shape(1, zeros, np.linspace(0.0, 1.0, n_nodes), zeros),
        _shape(2, zeros, np.linspace(0.0, 1.0, n_nodes) ** 2, zeros),
    ]
    mac = mac_matrix(fa_only, ss_only)
    np.testing.assert_allclose(mac, np.zeros((2, 2)), atol=1e-12)


def test_compare_modes_frequency_shift_sign() -> None:
    """When modified-result frequencies are uniformly higher than
    baseline-result frequencies, every Hungarian-paired pair gets a
    positive frequency_shift. Negative direction works symmetrically."""
    n_nodes = 6
    zeros = np.zeros(n_nodes)
    # Same shapes for both results — so the MAC diagonal is 1.0 and
    # pairing is the identity. Frequencies differ by a controlled
    # +20 % across all three modes.
    shapes_A = [
        _shape(k + 1, np.sin((k + 1) * np.linspace(0.0, np.pi, n_nodes)),
               zeros, zeros, freq=0.5 + k * 0.4)
        for k in range(3)
    ]
    shapes_B = [
        _shape(k + 1, np.sin((k + 1) * np.linspace(0.0, np.pi, n_nodes)),
               zeros, zeros, freq=(0.5 + k * 0.4) * 1.2)
        for k in range(3)
    ]
    result_A = ModalResult(
        frequencies=np.array([s.freq_hz for s in shapes_A]),
        shapes=shapes_A,
    )
    result_B = ModalResult(
        frequencies=np.array([s.freq_hz for s in shapes_B]),
        shapes=shapes_B,
    )
    cmp = compare_modes(result_A, result_B, label_A="land", label_B="modified")
    # Identity pairing because shapes match.
    assert cmp.paired_modes == [(0, 0), (1, 1), (2, 2)]
    # +20 % shift on every pair.
    np.testing.assert_allclose(cmp.frequency_shift, [20.0, 20.0, 20.0], atol=1e-10)
    # Labels travel through.
    assert cmp.label_A == "land"
    assert cmp.label_B == "modified"

    # Sanity in the opposite direction. (f_A - f_B) / f_B = -0.2 / 1.2.
    expected_back_pct = -100.0 * 0.2 / 1.2
    cmp_back = compare_modes(result_B, result_A, label_A="B", label_B="A")
    np.testing.assert_allclose(
        cmp_back.frequency_shift,
        [expected_back_pct] * 3,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Support tests
# ---------------------------------------------------------------------------

def test_mac_empty_inputs_return_empty_matrix() -> None:
    """Either input being empty yields an empty (correctly-shaped)
    ndarray rather than raising."""
    assert mac_matrix([], []).shape == (0, 0)
    n = 4
    one = [_shape(1, np.ones(n), np.zeros(n), np.zeros(n))]
    assert mac_matrix(one, []).shape == (1, 0)
    assert mac_matrix([], one).shape == (0, 1)


def test_mac_mismatched_dof_raises() -> None:
    """MAC requires shape vectors of equal length; mismatched node
    counts raise ValueError."""
    a = [_shape(1, np.ones(4), np.zeros(4), np.zeros(4))]
    b = [_shape(1, np.ones(6), np.zeros(6), np.zeros(6))]
    with pytest.raises(ValueError, match="same length"):
        mac_matrix(a, b)


def test_shape_to_vector_layout() -> None:
    """shape_to_vector concatenates flap, lag, twist in that order."""
    n = 3
    flap = np.array([1.0, 2.0, 3.0])
    lag = np.array([4.0, 5.0, 6.0])
    twist = np.array([7.0, 8.0, 9.0])
    shape = _shape(1, flap, lag, twist)
    vec = shape_to_vector(shape)
    np.testing.assert_array_equal(vec, np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ))
    assert vec.size == 3 * n


def test_plot_mac_returns_figure() -> None:
    """Smoke test: plot_mac yields a matplotlib Figure with one axes
    populated and the MAC colorbar attached. Skipped when matplotlib
    isn't available."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    n = 4
    zeros = np.zeros(n)
    shapes_A = [
        _shape(1, np.ones(n), zeros, zeros, freq=0.5),
        _shape(2, np.linspace(0.0, 1.0, n), zeros, zeros, freq=1.0),
    ]
    result_A = ModalResult(
        frequencies=np.array([0.5, 1.0]), shapes=shapes_A,
    )
    cmp = compare_modes(result_A, result_A)
    from pybmodes.mac import plot_mac
    fig = plot_mac(cmp)
    # Sanity on the rendered figure.
    assert fig is not None
    axes = fig.axes
    assert len(axes) >= 1, "expected at least the MAC axes"
    # Title should mention the labels.
    assert "A" in axes[0].get_title()
    import matplotlib.pyplot as plt
    plt.close(fig)
