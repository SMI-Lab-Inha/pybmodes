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


@dataclass(frozen=True)
class TowerFamilyMemberReport:
    """Diagnostic view of one scored FA/SS tower family candidate."""

    mode_number: int
    frequency_hz: float
    family_rank: int
    is_fa: bool
    fa_rms: float
    ss_rms: float
    direction_ratio: float
    fit_rms: float
    fit_is_good: bool
    selected: bool


@dataclass(frozen=True)
class TowerSelectionReport:
    """Structured report of tower-mode family scoring and final selection."""

    fa_family: tuple[TowerFamilyMemberReport, ...]
    ss_family: tuple[TowerFamilyMemberReport, ...]
    selected_fa_modes: tuple[int, int]
    selected_ss_modes: tuple[int, int]


def _component_strength(span_loc: np.ndarray, displacement: np.ndarray) -> float:
    """Return a spanwise RMS-like displacement strength for classification."""
    x = np.asarray(span_loc, dtype=float)
    y = np.asarray(displacement, dtype=float)

    if x.shape != y.shape:
        raise ValueError(
            "span_loc and displacement must have the same shape for mode classification"
        )
    if y.size == 0:
        return 0.0
    if y.size == 1:
        return float(abs(y[0]))

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    span = float(x_sorted[-1] - x_sorted[0])
    if span <= 0.0:
        return float(np.sqrt(np.mean(y_sorted**2)))

    dx = np.diff(x_sorted)
    y2 = y_sorted**2
    area = np.sum(0.5 * dx * (y2[:-1] + y2[1:]))
    return float(np.sqrt(area / span))


def _classify_fa_dominant(
    span_loc: np.ndarray,
    fa_disp: np.ndarray,
    ss_disp: np.ndarray,
) -> bool:
    """Return True iff the FA component dominates the SS component over the span.

    Compares the spanwise RMS-like strengths of the two displacement curves.
    On a near-tie (relative gap below the default ``np.isclose`` tolerance),
    the tip displacement breaks the tie — that is the only place the tip
    appears in the rule.

    This is the single source of truth for FA-vs-SS classification across
    the blade and tower paths; both ``_is_fa_dominated`` and
    ``_tower_candidate`` route through it so a borderline mode lands on the
    same side regardless of which classifier is invoked.
    """
    fa = _component_strength(span_loc, fa_disp)
    ss = _component_strength(span_loc, ss_disp)
    if np.isclose(fa, ss):
        return abs(fa_disp[-1]) >= abs(ss_disp[-1])
    return fa > ss


def _is_fa_dominated(shape: NodeModeShape) -> bool:
    """True if spanwise flap/fore-aft motion dominates lag/side-side motion."""
    return _classify_fa_dominant(
        shape.span_loc, shape.flap_disp, shape.lag_disp
    )


def _sorted_modes(
    shapes: list[NodeModeShape],
    fa_dominated: bool,
) -> list[NodeModeShape]:
    """Return modes of the given type, sorted by ascending frequency.

    Sorting is what callers of ``compute_blade_params`` rely on to pick
    1st-flap / 2nd-flap / 1st-edge: the lowest-frequency match comes first.
    A user constructing ``ModalResult`` by hand may not feed shapes in
    frequency order, so we sort defensively rather than trust the caller.
    """
    selected = [s for s in shapes if _is_fa_dominated(s) == fa_dominated]
    selected.sort(key=lambda s: s.freq_hz)
    return selected


@dataclass
class _TowerModeCandidate:
    """One tower mode interpreted as either FA or SS bending for ElastoDyn fitting."""

    shape: NodeModeShape
    fa_disp: np.ndarray
    ss_disp: np.ndarray
    fa_rms: float
    ss_rms: float
    fit_disp: np.ndarray
    fit: PolyFitResult
    is_fa: bool


@dataclass(frozen=True)
class _TowerFamilySelectionConfig:
    """Selection knobs for choosing ElastoDyn FA/SS tower mode families."""

    good_fit_rms: float = 0.09


@dataclass(frozen=True)
class _TowerFamilyMemberScore:
    """Scored view of one candidate within an FA or SS tower family."""

    candidate: _TowerModeCandidate
    family_rank: int
    fit_is_good: bool
    direction_ratio: float


@dataclass(frozen=True)
class _TowerFamilySelectionResult:
    """Selected family members plus their scored candidate list."""

    first: _TowerModeCandidate
    second: _TowerModeCandidate
    scores: tuple[_TowerFamilyMemberScore, ...]

    def __iter__(self):
        """Support tuple-style unpacking as `(first, second)`."""
        yield self.first
        yield self.second


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

    fa_strength = _component_strength(shape.span_loc, fa_disp)
    ss_strength = _component_strength(shape.span_loc, ss_disp)

    # Route through the shared classifier so blade and tower paths agree on
    # tie-handling: spanwise strength wins, the tip breaks ties.
    is_fa = _classify_fa_dominant(shape.span_loc, fa_disp, ss_disp)
    fit_disp = fa_disp if is_fa else ss_disp
    fit = fit_mode_shape(shape.span_loc, fit_disp)

    return _TowerModeCandidate(
        shape=shape,
        fa_disp=fa_disp,
        ss_disp=ss_disp,
        fa_rms=fa_strength,
        ss_rms=ss_strength,
        fit_disp=fit_disp,
        fit=fit,
        is_fa=is_fa,
    )


def _tower_family_candidates(
    candidates: list[_TowerModeCandidate],
    is_fa: bool,
) -> list[_TowerModeCandidate]:
    """Return one directional tower family sorted by ascending frequency."""
    family = [c for c in candidates if c.is_fa == is_fa]
    family.sort(key=lambda c: c.shape.freq_hz)
    return family


def _score_tower_family(
    family: list[_TowerModeCandidate],
    is_fa: bool,
    config: _TowerFamilySelectionConfig,
) -> list[_TowerFamilyMemberScore]:
    """Annotate a directional tower family with explicit selection metrics."""
    scores: list[_TowerFamilyMemberScore] = []
    for idx, candidate in enumerate(family):
        major = candidate.fa_rms if is_fa else candidate.ss_rms
        minor = candidate.ss_rms if is_fa else candidate.fa_rms
        direction_ratio = float("inf") if minor == 0.0 else major / minor
        scores.append(
            _TowerFamilyMemberScore(
                candidate=candidate,
                family_rank=idx + 1,
                fit_is_good=candidate.fit.rms_residual <= config.good_fit_rms,
                direction_ratio=direction_ratio,
            )
        )
    return scores


def _select_tower_family(
    candidates: list[_TowerModeCandidate],
    is_fa: bool,
    config: _TowerFamilySelectionConfig | None = None,
) -> _TowerFamilySelectionResult:
    """Select the 1st/2nd FA or SS tower bending modes for ElastoDyn.

    We keep the lowest-frequency candidate as the first family member, then pick
    the next higher-frequency candidate whose clamped-base polynomial fit is
    still good. This skips support-dominated modes that happen to align with the
    same direction but are poor ElastoDyn bending-shape representatives.
    """

    config = config or _TowerFamilySelectionConfig()
    family = _tower_family_candidates(candidates, is_fa=is_fa)

    if len(family) < 2:
        kind = "FA" if is_fa else "SS"
        raise ValueError(
            f"Need >= 2 {kind} modes; found {len(family)}. "
            "Increase n_modes in Tower.run()."
        )

    scores = _score_tower_family(family, is_fa=is_fa, config=config)
    first = scores[0].candidate

    for score in scores[1:]:
        if score.fit_is_good:
            return _TowerFamilySelectionResult(
                first=first,
                second=score.candidate,
                scores=tuple(scores),
            )

    second = min(scores[1:], key=lambda s: s.candidate.fit.rms_residual).candidate
    return _TowerFamilySelectionResult(
        first=first,
        second=second,
        scores=tuple(scores),
    )


def _family_report(
    selection: _TowerFamilySelectionResult,
) -> tuple[TowerFamilyMemberReport, ...]:
    """Convert one internal family-selection result to a public report."""
    selected_modes = {
        selection.first.shape.mode_number,
        selection.second.shape.mode_number,
    }
    return tuple(
        TowerFamilyMemberReport(
            mode_number=score.candidate.shape.mode_number,
            frequency_hz=score.candidate.shape.freq_hz,
            family_rank=score.family_rank,
            is_fa=score.candidate.is_fa,
            fa_rms=score.candidate.fa_rms,
            ss_rms=score.candidate.ss_rms,
            direction_ratio=score.direction_ratio,
            fit_rms=score.candidate.fit.rms_residual,
            fit_is_good=score.fit_is_good,
            selected=score.candidate.shape.mode_number in selected_modes,
        )
        for score in selection.scores
    )


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

    params, _ = compute_tower_params_report(modal)
    return params


def compute_tower_params_report(
    modal: ModalResult,
) -> tuple[TowerElastoDynParams, TowerSelectionReport]:
    """Compute tower ElastoDyn parameters and return a selection report."""
    candidates = [_tower_candidate(shape) for shape in modal.shapes]
    fa_sel = _select_tower_family(candidates, is_fa=True)
    ss_sel = _select_tower_family(candidates, is_fa=False)

    params = TowerElastoDynParams(
        TwFAM1Sh=fa_sel.first.fit,
        TwFAM2Sh=fa_sel.second.fit,
        TwSSM1Sh=ss_sel.first.fit,
        TwSSM2Sh=ss_sel.second.fit,
    )
    report = TowerSelectionReport(
        fa_family=_family_report(fa_sel),
        ss_family=_family_report(ss_sel),
        selected_fa_modes=(fa_sel.first.shape.mode_number, fa_sel.second.shape.mode_number),
        selected_ss_modes=(ss_sel.first.shape.mode_number, ss_sel.second.shape.mode_number),
    )
    return params, report
