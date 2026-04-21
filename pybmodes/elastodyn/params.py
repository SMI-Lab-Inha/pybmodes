"""Map ModalResult + poly fit to named ElastoDyn input parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pybmodes.fem.normalize import NodeModeShape
from pybmodes.fitting.poly_fit import PolyFitResult, fit_mode_shape
from pybmodes.models.result import ModalResult


@dataclass
class BladeElastoDynParams:
    """Polynomial fits for the three blade mode shapes required by ElastoDyn."""

    BldFl1Sh: PolyFitResult
    BldFl2Sh: PolyFitResult
    BldEdgSh: PolyFitResult

    def as_dict(self) -> dict[str, float]:
        """Flat {ElastoDyn_param_name: coefficient_value} for writer."""
        out: dict[str, float] = {}
        for ed_name, fit in [
            ("BldFl1Sh", self.BldFl1Sh),
            ("BldFl2Sh", self.BldFl2Sh),
            ("BldEdgSh", self.BldEdgSh),
        ]:
            for k, c in zip(range(2, 7), fit.coefficients()):
                out[f"{ed_name}({k})"] = float(c)
        return out


@dataclass
class TowerElastoDynParams:
    """Polynomial fits for the four tower mode shapes required by ElastoDyn.

    Field names match the OpenFAST tower sub-file (*_ElastoDyn_tower.dat):
      TwFAM1Sh / TwFAM2Sh - fore-aft modes 1 and 2
      TwSSM1Sh / TwSSM2Sh - side-side modes 1 and 2
    """

    TwFAM1Sh: PolyFitResult
    TwFAM2Sh: PolyFitResult
    TwSSM1Sh: PolyFitResult
    TwSSM2Sh: PolyFitResult

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for ed_name, fit in [
            ("TwFAM1Sh", self.TwFAM1Sh),
            ("TwFAM2Sh", self.TwFAM2Sh),
            ("TwSSM1Sh", self.TwSSM1Sh),
            ("TwSSM2Sh", self.TwSSM2Sh),
        ]:
            for k, c in zip(range(2, 7), fit.coefficients()):
                out[f"{ed_name}({k})"] = float(c)
        return out


def _is_fa_dominated(shape: NodeModeShape) -> bool:
    """True if the flap (fore-aft) tip displacement dominates over lag (side-side)."""
    return abs(shape.flap_disp[-1]) >= abs(shape.lag_disp[-1])


def _sorted_modes(
    shapes: list[NodeModeShape],
    fa_dominated: bool,
) -> list[NodeModeShape]:
    """Return modes of the given type, in ascending frequency order."""
    return [s for s in shapes if _is_fa_dominated(s) == fa_dominated]


@dataclass
class _TowerModeCandidate:
    """One tower mode interpreted as either FA or SS bending for ElastoDyn fitting."""

    shape: NodeModeShape
    fit_disp: np.ndarray
    fit: PolyFitResult
    is_fa: bool


def _remove_root_rigid_motion(
    span_loc: np.ndarray,
    displacement: np.ndarray,
    slope: np.ndarray,
) -> np.ndarray:
    """Remove rigid-body root translation/rotation before tower polynomial fitting.

    ElastoDyn tower polynomials enforce zero displacement and zero slope at the
    base. Offshore/support-compliant tower modes can include rigid-body motion at
    the base, so we subtract the affine root contribution first and fit the
    bending part only.
    """

    x = np.asarray(span_loc, dtype=float)
    y = np.asarray(displacement, dtype=float)
    yp = np.asarray(slope, dtype=float)
    return y - y[0] - yp[0] * x


def _tower_candidate(shape: NodeModeShape) -> _TowerModeCandidate:
    """Build the best FA/SS interpretation of a tower mode for ElastoDyn fitting."""

    fa_disp = _remove_root_rigid_motion(shape.span_loc, shape.flap_disp, shape.flap_slope)
    ss_disp = _remove_root_rigid_motion(shape.span_loc, shape.lag_disp, shape.lag_slope)

    fa_strength = float(np.sqrt(np.mean(fa_disp**2)))
    ss_strength = float(np.sqrt(np.mean(ss_disp**2)))

    is_fa = fa_strength >= ss_strength
    fit_disp = fa_disp if is_fa else ss_disp
    fit = fit_mode_shape(shape.span_loc, fit_disp)

    return _TowerModeCandidate(
        shape=shape,
        fit_disp=fit_disp,
        fit=fit,
        is_fa=is_fa,
    )


def _select_tower_family(
    candidates: list[_TowerModeCandidate],
    is_fa: bool,
) -> tuple[_TowerModeCandidate, _TowerModeCandidate]:
    """Select the 1st/2nd FA or SS tower bending modes for ElastoDyn.

    We keep the lowest-frequency candidate as the first family member, then pick
    the next higher-frequency candidate whose clamped-base polynomial fit is
    still good. This skips support-dominated modes that happen to align with the
    same direction but are poor ElastoDyn bending-shape representatives.
    """

    family = [c for c in candidates if c.is_fa == is_fa]
    family.sort(key=lambda c: c.shape.freq_hz)

    if len(family) < 2:
        kind = "FA" if is_fa else "SS"
        raise ValueError(
            f"Need >= 2 {kind} modes; found {len(family)}. "
            "Increase n_modes in Tower.run()."
        )

    first = family[0]
    good_rms = 0.09

    for cand in family[1:]:
        if cand.fit.rms_residual <= good_rms:
            return first, cand

    second = min(family[1:], key=lambda c: c.fit.rms_residual)
    return first, second


def compute_blade_params(modal: ModalResult) -> BladeElastoDynParams:
    """Fit polynomials to the 1st/2nd flap and 1st edge modes."""

    flap_modes = _sorted_modes(modal.shapes, fa_dominated=True)
    edge_modes = _sorted_modes(modal.shapes, fa_dominated=False)

    if len(flap_modes) < 2:
        raise ValueError(
            f"Need >= 2 flap modes; found {len(flap_modes)}. "
            "Increase n_modes in RotatingBlade.run()."
        )
    if len(edge_modes) < 1:
        raise ValueError(
            f"Need >= 1 edge mode; found {len(edge_modes)}. "
            "Increase n_modes in RotatingBlade.run()."
        )

    return BladeElastoDynParams(
        BldFl1Sh=fit_mode_shape(flap_modes[0].span_loc, flap_modes[0].flap_disp),
        BldFl2Sh=fit_mode_shape(flap_modes[1].span_loc, flap_modes[1].flap_disp),
        BldEdgSh=fit_mode_shape(edge_modes[0].span_loc, edge_modes[0].lag_disp),
    )


def compute_tower_params(modal: ModalResult) -> TowerElastoDynParams:
    """Fit polynomials to the 1st/2nd FA and 1st/2nd SS tower modes."""

    candidates = [_tower_candidate(shape) for shape in modal.shapes]
    fa1, fa2 = _select_tower_family(candidates, is_fa=True)
    ss1, ss2 = _select_tower_family(candidates, is_fa=False)

    return TowerElastoDynParams(
        TwFAM1Sh=fa1.fit,
        TwFAM2Sh=fa2.fit,
        TwSSM1Sh=ss1.fit,
        TwSSM2Sh=ss2.fit,
    )
