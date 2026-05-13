"""Sanity tests for :mod:`pybmodes.coords`.

The module is documentation + two constants — no runtime logic, so
the tests just lock in the public contract: ``DOF_NAMES`` is the
OpenFAST 6-DOF order, ``DOF_INDEX`` is the inverse mapping, and the
two stay in lockstep if anyone edits the file.

The OpenFAST 6-DOF order is the load-bearing invariant for every
``PlatformSupport`` 6×6 matrix in the public API; flipping it would
silently mis-couple surge↔sway / roll↔pitch on asymmetric mooring or
hydro decks. The cert tolerance test on OC3 cannot catch the flip
because OC3 is symmetric, so this module's contract has to be checked
directly.
"""

from __future__ import annotations

from pybmodes.coords import DOF_INDEX, DOF_NAMES


def test_dof_names_is_openfast_order() -> None:
    """The canonical 6-DOF order per Jonkman (2010) NREL/TP-500-47535
    Table 5-1 and every subsequent OpenFAST publication.
    """
    assert DOF_NAMES == ("surge", "sway", "heave", "roll", "pitch", "yaw")


def test_dof_index_is_inverse_of_dof_names() -> None:
    """``DOF_INDEX[name] == DOF_NAMES.index(name)`` for every name —
    the two constants must stay in lockstep or downstream code that
    looks up by name will mis-index 6×6 matrices.
    """
    assert len(DOF_INDEX) == len(DOF_NAMES) == 6
    for i, name in enumerate(DOF_NAMES):
        assert DOF_INDEX[name] == i, (
            f"DOF_INDEX[{name!r}] = {DOF_INDEX[name]} but "
            f"DOF_NAMES[{i}] = {name!r}"
        )


def test_dof_index_keys_match_dof_names() -> None:
    """No spurious extra keys in ``DOF_INDEX``."""
    assert set(DOF_INDEX) == set(DOF_NAMES)


def test_translational_dofs_come_before_rotational() -> None:
    """Indices 0–2 are translations (surge/sway/heave), 3–5 are
    rotations (roll/pitch/yaw). The translational-first convention is
    baked into the rigid-arm transform in
    ``pybmodes.fem.nondim.nondim_platform`` and the WAMIT v7 re-
    dimensionalisation in ``pybmodes.io.wamit_reader``.
    """
    translational = {"surge", "sway", "heave"}
    rotational = {"roll", "pitch", "yaw"}
    assert {DOF_NAMES[i] for i in range(3)} == translational
    assert {DOF_NAMES[i] for i in range(3, 6)} == rotational
