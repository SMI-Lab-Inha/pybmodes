"""Synthetic-fixture tests for ``pybmodes.io.wamit_reader``.

These run in the default-pytest suite (no external data required) and
cover corner cases surfaced during pre-1.0 review:

- Upper-triangle-only `.hst` / `.1` files get mirrored after parsing
  so the returned matrices are full symmetric 6×6.
- Fortran-style ``D`` / ``d`` exponent notation (``1.234D+02``) is
  accepted alongside the C-convention ``E`` / ``e``.
"""

from __future__ import annotations

import pathlib

import pytest

from pybmodes.io import WamitReader


def test_wamit_upper_triangle_only_is_mirrored(tmp_path: pathlib.Path) -> None:
    """Mirror the missing half when WAMIT emits only one triangle of a
    symmetric matrix."""
    (tmp_path / "upper_only.hst").write_text(
        "     3     3   4.430486E+02\n"
        "     3     5  -4.012296E-01\n",
        encoding="utf-8",
    )
    (tmp_path / "upper_only.1").write_text(
        " -1.000000E+00     1     1  1.233416E+04\n"
        " -1.000000E+00     1     3 -2.975529E-02\n",
        encoding="utf-8",
    )
    data = WamitReader(
        "upper_only", tmp_path, rho=1025.0, g=9.80665, ulen=1.0,
    ).read()
    # Heave-pitch coupling: only (3,5) was in the file; both (2,4) and
    # (4,2) must end up non-zero and equal.
    assert data.C_hst[2, 4] != 0.0
    assert data.C_hst[4, 2] == data.C_hst[2, 4]
    # Surge-heave for A_inf: only (1,3) was written; both (0,2) and
    # (2,0) must end up non-zero and equal.
    assert data.A_inf[0, 2] != 0.0
    assert data.A_inf[2, 0] == data.A_inf[0, 2]


def test_wamit_fortran_d_exponent(tmp_path: pathlib.Path) -> None:
    """Accept Fortran-style ``D`` / ``d`` exponent notation alongside
    the C convention ``E`` / ``e``."""
    (tmp_path / "fortran.hst").write_text(
        "     3     3   4.430486D+02\n",
        encoding="utf-8",
    )
    (tmp_path / "fortran.1").write_text(
        " -1.000000D+00     1     1  1.233416d+04\n",
        encoding="utf-8",
    )
    data = WamitReader(
        "fortran", tmp_path, rho=1025.0, g=9.80665, ulen=1.0,
    ).read()
    assert data.C_hst[2, 2] == pytest.approx(4.430486e2 * 1025.0 * 9.80665)
    assert data.A_inf[0, 0] == pytest.approx(1.233416e4 * 1025.0)
