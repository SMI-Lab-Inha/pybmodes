"""Tests for ``RotatingBlade.from_elastodyn(elastodyn_compatible=...)``.

Per Jonkman (NREL forum, March 2015) the polynomial mode shapes
ElastoDyn consumes must come from a model that shares ElastoDyn's
structural assumptions: pure flapwise + edgewise bending of a straight
isotropic beam, no axial / torsional DOFs, no mass / shear-centre /
tension-centre offsets, no inertial-vs-structural twist split. The
``elastodyn_compatible=True`` path (the default) overrides the parsed
section properties to match those assumptions; ``False`` keeps the
parsed values and warns.

Two of Jonkman's BModes recommendations — "very small rotary
inertia" and "very large axial / torsional stiffness" — are already
covered by
:func:`pybmodes.io.elastodyn_reader._stack_blade_section_props`
(rotary-inertia floor at 1e-6 · char² · ρA, axial_stff = 1e6 · EI,
tor_stff = 100 · EI). The remaining gaps the compatibility flag
closes are ``str_tw``, ``tw_iner``, and the three section offsets.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest

from pybmodes.models import RotatingBlade

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
NREL5MW_DECK = (
    REPO_ROOT
    / "reference_decks"
    / "nrel5mw_land"
    / "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
)

if not NREL5MW_DECK.is_file():
    pytest.skip(
        f"NREL 5MW reference deck not present at {NREL5MW_DECK}; "
        "run `python scripts/build_reference_decks.py` to generate.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Section-property overrides (sp object after build)
# ---------------------------------------------------------------------------

class TestElastoDynCompatibleSectionProperties:

    def test_zeroes_structural_twist(self) -> None:
        blade = RotatingBlade.from_elastodyn(NREL5MW_DECK)
        sp = blade._sp
        assert sp is not None
        assert np.all(sp.str_tw == 0.0)

    def test_zeroes_inertial_twist(self) -> None:
        blade = RotatingBlade.from_elastodyn(NREL5MW_DECK)
        sp = blade._sp
        assert sp is not None
        assert np.all(sp.tw_iner == 0.0)

    def test_zeroes_offsets(self) -> None:
        blade = RotatingBlade.from_elastodyn(NREL5MW_DECK)
        sp = blade._sp
        assert sp is not None
        assert np.all(sp.cg_offst == 0.0)
        assert np.all(sp.sc_offst == 0.0)
        assert np.all(sp.tc_offst == 0.0)

    def test_default_is_compatible(self) -> None:
        """Calling without the kwarg must use the compatible path —
        that's the documented default for users generating ElastoDyn
        polynomial inputs."""
        # No UserWarning expected on the default path; an info-level
        # log message fires but doesn't surface as a warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            blade = RotatingBlade.from_elastodyn(NREL5MW_DECK)
        sp = blade._sp
        assert sp is not None
        assert np.all(sp.str_tw == 0.0)


class TestElastoDynIncompatible:

    def test_preserves_structural_twist(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            blade = RotatingBlade.from_elastodyn(
                NREL5MW_DECK, elastodyn_compatible=False
            )
        sp = blade._sp
        assert sp is not None
        # NREL 5MW blade has ~13° pretwist at the root — definitely
        # non-zero in the parsed deck, definitely zeroed in the
        # compatibility path. Each path is distinguishable by the
        # str_tw column.
        assert np.any(np.abs(sp.str_tw) > 0.05)

    def test_preserves_inertial_twist(self) -> None:
        """The ElastoDyn adapter sets tw_iner = str_tw (the deck
        doesn't carry an independent inertial-twist column), so the
        compatibility-False path keeps that copy."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            blade = RotatingBlade.from_elastodyn(
                NREL5MW_DECK, elastodyn_compatible=False
            )
        sp = blade._sp
        assert sp is not None
        assert np.any(np.abs(sp.tw_iner) > 0.05)

    def test_emits_warning(self) -> None:
        with pytest.warns(UserWarning, match="elastodyn_compatible=False"):
            RotatingBlade.from_elastodyn(
                NREL5MW_DECK, elastodyn_compatible=False
            )


# ---------------------------------------------------------------------------
# Frequency comparison: the compatibility shifts non-physical DOFs but
# leaves the dominant flap/edge spectrum essentially unchanged.
# ---------------------------------------------------------------------------

class TestFrequencyComparison:
    """The compatibility flag's main effect is suppressing flap/edge
    cross-coupling via str_tw. For the NREL 5MW blade — moderately
    twisted, otherwise close to isotropic — the lowest-mode
    frequencies stay within ~10 % between the two paths.
    """

    def _build(self, compatible: bool) -> RotatingBlade:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return RotatingBlade.from_elastodyn(
                NREL5MW_DECK, elastodyn_compatible=compatible,
            )

    def test_first_flap_within_tolerance(self) -> None:
        comp = self._build(True).run(n_modes=8)
        phys = self._build(False).run(n_modes=8)
        f_comp = float(np.asarray(comp.frequencies)[0])
        f_phys = float(np.asarray(phys.frequencies)[0])
        rel = abs(f_comp - f_phys) / f_phys
        assert rel < 0.15, (
            f"1st-mode frequency differs by {rel*100:.1f} % between the "
            f"two paths (comp={f_comp:.4f} Hz, phys={f_phys:.4f} Hz); "
            f"compatibility-flag effect on 1st flap should be < 15 % on "
            f"the NREL 5MW deck"
        )

    def test_first_edge_within_tolerance(self) -> None:
        comp = self._build(True).run(n_modes=8)
        phys = self._build(False).run(n_modes=8)
        f_comp = float(np.asarray(comp.frequencies)[1])
        f_phys = float(np.asarray(phys.frequencies)[1])
        rel = abs(f_comp - f_phys) / f_phys
        assert rel < 0.15, (
            f"2nd-mode (1st edge) frequency differs by {rel*100:.1f} % "
            f"between the two paths (comp={f_comp:.4f} Hz, "
            f"phys={f_phys:.4f} Hz); expected < 15 %"
        )


# ---------------------------------------------------------------------------
# 6. Direct override-helper unit test (synthetic input — no NREL5MW
#    needed, runs in default suite without integration data).
# ---------------------------------------------------------------------------

class TestApplyHelperDirect:
    """Unit-test ``_apply_elastodyn_compatibility`` on a hand-built
    SectionProperties so the invariants hold regardless of what
    ``to_pybmodes_blade`` happens to produce on a given deck.
    """

    def test_overrides_zero_target_columns(self) -> None:
        from pybmodes.io.sec_props import SectionProperties
        from pybmodes.models.blade import _apply_elastodyn_compatibility

        n = 5
        ones = np.ones(n)
        sp = SectionProperties(
            title="synthetic",
            n_secs=n,
            span_loc=np.linspace(0.0, 1.0, n),
            str_tw=0.2 * ones,
            tw_iner=0.3 * ones,
            mass_den=100.0 * ones,
            flp_iner=5.0 * ones,
            edge_iner=5.0 * ones,
            flp_stff=1.0e9 * ones,
            edge_stff=2.0e9 * ones,
            tor_stff=1.0e11 * ones,
            axial_stff=1.0e15 * ones,
            cg_offst=0.1 * ones,
            sc_offst=0.05 * ones,
            tc_offst=-0.05 * ones,
        )
        _apply_elastodyn_compatibility(sp)
        assert np.all(sp.str_tw == 0.0)
        assert np.all(sp.tw_iner == 0.0)
        assert np.all(sp.cg_offst == 0.0)
        assert np.all(sp.sc_offst == 0.0)
        assert np.all(sp.tc_offst == 0.0)

    def test_preserves_other_columns(self) -> None:
        """Bending stiffness and mass density must NOT be touched —
        they're shared with ElastoDyn's model."""
        from pybmodes.io.sec_props import SectionProperties
        from pybmodes.models.blade import _apply_elastodyn_compatibility

        n = 4
        sp = SectionProperties(
            title="synthetic",
            n_secs=n,
            span_loc=np.linspace(0.0, 1.0, n),
            str_tw=np.full(n, 0.1),
            tw_iner=np.full(n, 0.1),
            mass_den=np.full(n, 200.0),
            flp_iner=np.full(n, 0.5),
            edge_iner=np.full(n, 0.5),
            flp_stff=np.full(n, 1.0e10),
            edge_stff=np.full(n, 2.0e10),
            tor_stff=np.full(n, 1.0e12),
            axial_stff=np.full(n, 1.0e16),
            cg_offst=np.zeros(n),
            sc_offst=np.zeros(n),
            tc_offst=np.zeros(n),
        )
        flp_stff_before = sp.flp_stff.copy()
        edge_stff_before = sp.edge_stff.copy()
        mass_den_before = sp.mass_den.copy()

        _apply_elastodyn_compatibility(sp)

        np.testing.assert_array_equal(sp.flp_stff, flp_stff_before)
        np.testing.assert_array_equal(sp.edge_stff, edge_stff_before)
        np.testing.assert_array_equal(sp.mass_den, mass_den_before)
