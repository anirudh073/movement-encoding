"""Nested cross-validation utilities for movement encoding models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import warnings

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from scipy.optimize import minimize
from scipy.special import gammaln, i0e, i1e
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


@dataclass(frozen=True)
class SplineSpec:
    """Frozen outer-training spline boundaries and internal knots."""

    predictor: str
    kind: str
    df: int
    lower_bound: float
    upper_bound: float
    knots: tuple[float, ...]
    degree: int = 3
    constraints: str | None = None
    by: str | None = None


@dataclass
class FittedSplineBasis:
    """A Patsy basis fitted on one training-only predictor distribution."""

    spec: SplineSpec
    design_info: object
    column_names: tuple[str, ...]


@dataclass
class DesignMatrixTransformer:
    """Frozen training-only transformations for one CV split and model."""

    included_groups: tuple[str, ...]
    direct_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    categorical_levels: dict[str, tuple]
    categorical_design_info: object
    categorical_column_names: tuple[str, ...]
    spline_bases: dict[str, FittedSplineBasis]
    branch_spline_bases: dict[object, FittedSplineBasis]
    retained_columns: tuple[str, ...]
    means: pd.Series
    scales: pd.Series
    zero_variance_report: pd.DataFrame


@dataclass
class RidgeAlphaSearchResult:
    """Inner-CV ridge selection outputs for one neuron and outer fold."""

    selected_alpha: float | None
    selected_null_model: bool
    selected_candidate_score: float
    best_finite_alpha: float
    best_finite_alpha_score: float
    null_model_score: float
    alpha_grid: np.ndarray
    n_grid_expansions: int
    score_table: pd.DataFrame
    inner_folds: list[dict]
    clipping_summary: pd.DataFrame
    clipping_detail: pd.DataFrame
    zero_variance_report: pd.DataFrame
    permutation_audit: pd.DataFrame

@dataclass
class BandedRidgeSearchResult:
    """Inner-CV banded-ridge selection outputs for one outer fold."""

    selected_penalties: dict[str, float]
    selected_null_model: bool
    selected_candidate_score: float
    best_finite_alpha: float
    best_finite_alpha_score: float
    null_model_score: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_n_iterations: int
    optimizer_n_evaluations: int
    evaluation_table: pd.DataFrame
    initialization_score_table: pd.DataFrame
    inner_folds: list[dict]
    clipping_summary: pd.DataFrame
    clipping_detail: pd.DataFrame
    zero_variance_report: pd.DataFrame
    permutation_audit: pd.DataFrame


@dataclass
class FittedCircularGAM:
    """Penalized von Mises regression of circular phase on trial progress."""

    coefficients: np.ndarray
    concentration: float
    smoothing_penalty: float
    basis_transformer: SplineTransformer
    n_knots: int
    degree: int
    progress_bin_width: float
    initialization_cycles: float
    objective_value: float
    optimization_success: bool
    optimization_message: str

    def predict_latent(self, progress):
        """Return the fitted unwrapped latent phase trajectory in radians."""
        progress = np.asarray(progress, dtype=float).reshape(-1, 1)
        return self.basis_transformer.transform(progress) @ self.coefficients

    def predict(self, progress):
        """Return predicted phase in radians on ``[-pi, pi)``."""
        return np.angle(np.exp(1j * self.predict_latent(progress)))



@dataclass
class FittedPoissonRidgeModel:
    """One fitted Poisson ridge estimator with an unpenalized intercept."""

    estimator: PoissonRegressor | None
    feature_columns: tuple[str, ...]
    intercept_only_mean: float | None
    feature_scales: pd.Series | None = None

    def predict(self, design_matrix):
        """Predict counts from a compatible standardized design matrix."""
        missing_columns = set(self.feature_columns).difference(design_matrix.columns)
        if missing_columns:
            raise ValueError(
                "Design matrix is missing fitted feature columns: "
                f"{sorted(missing_columns)}"
            )
        if self.estimator is None:
            return np.full(len(design_matrix), self.intercept_only_mean, dtype=float)
        features = design_matrix.loc[:, self.feature_columns]
        if self.feature_scales is not None:
            features = features / self.feature_scales
        return np.asarray(self.estimator.predict(features), dtype=float)


@dataclass
class OuterFoldPoissonRidgeFit:
    """Final outer-training fit for one neuron and one predictor model."""

    outer_fold: int | str
    selected_alpha: float | None
    selected_penalties: dict[str, float] | None
    selected_null_model: bool
    train_trial_ids: tuple
    n_training_bins: int
    null_mean_count: float
    transformer: DesignMatrixTransformer
    model: FittedPoissonRidgeModel
    held_out_design_matrix: pd.DataFrame | None = None
    clipping_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    clipping_detail: pd.DataFrame = field(default_factory=pd.DataFrame)
    permutation_audit: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class OuterFoldPoissonRidgeEvaluation:
    """Untouched outer-test predictions and Poisson likelihood for one fit."""

    outer_fold: int | str
    test_trial_ids: tuple
    n_test_bins: int
    log_likelihood: float
    mean_log_likelihood_per_bin: float
    model_deviance: float
    null_deviance: float
    deviance_explained: float
    prediction_table: pd.DataFrame
    clipping_summary: pd.DataFrame
    clipping_detail: pd.DataFrame


@dataclass
class CrossValidatedPoissonRidgeResult:
    """Nested-CV results for one neuron and one specified predictor model."""

    mean_fold_log_likelihood_per_bin: float
    pooled_log_likelihood_per_bin: float
    total_log_likelihood: float
    n_held_out_bins: int
    mean_fold_deviance_explained: float
    pooled_deviance_explained: float
    fold_scores: pd.DataFrame
    out_of_fold_predictions: pd.DataFrame
    tuning_results: dict
    outer_fits: dict
    clipping_summary: pd.DataFrame
    clipping_detail: pd.DataFrame
    permutation_audit: pd.DataFrame
    penalty_search: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_movement_predictor_groups(paw_prefixes: Mapping):
    """Return the approved compact movement-encoding predictor groups."""
    paw_prefixes = dict(paw_prefixes)
    required_paws = {"forepawL", "forepawR", "hindpawL", "hindpawR"}
    missing_paws = required_paws.difference(paw_prefixes)
    if missing_paws:
        raise ValueError(
            "paw_prefixes is missing required paws: " f"{sorted(missing_paws)}"
        )
    limb_columns = (
        "gait_phase_sin",
        "gait_phase_cos",
        "lateral_phase_sin",
        "lateral_phase_cos",
        "relative_paw_vigor",
        "fore_minus_hind_vigor",
        "mean_stance_width",
        "fore_minus_hind_width",
        "mean_fore_hind_spacing",
        "fore_hind_spacing_asymmetry",
    )
    return {
        "task": (
            "trial_type",
            "branch_id",
            "branch_pos_scaled",
            "trial_progress_distance",
        ),
        "translation": ("speed", "tangential_acceleration_cm_s2"),
        "turning": ("body_angular_velocity_rad_s",),
        "limb": limb_columns,
    }


def build_movement_penalty_bands(predictor_groups: Mapping):
    """Return the six approved raw-predictor bands for banded ridge."""
    required_groups = {"task", "translation", "turning", "limb"}
    missing_groups = required_groups.difference(predictor_groups)
    if missing_groups:
        raise ValueError(
            "predictor_groups is missing required groups: "
            f"{sorted(missing_groups)}"
        )
    bands = {
        "trial_type": ("trial_type",),
        "position": ("branch_id", "branch_pos_scaled"),
        "progress": ("trial_progress_distance",),
        "translation": tuple(predictor_groups["translation"]),
        "turning": tuple(predictor_groups["turning"]),
        "limb": tuple(predictor_groups["limb"]),
    }
    flattened = [predictor for values in bands.values() for predictor in values]
    duplicates = sorted(
        predictor for predictor in set(flattened) if flattened.count(predictor) > 1
    )
    expected = {
        predictor
        for group_name in required_groups
        for predictor in predictor_groups[group_name]
    }
    missing_predictors = sorted(expected.difference(flattened))
    unexpected_predictors = sorted(set(flattened).difference(expected))
    if duplicates or missing_predictors or unexpected_predictors:
        raise ValueError(
            "Penalty bands must partition the approved predictors exactly; "
            f"duplicates={duplicates}, missing={missing_predictors}, "
            f"unexpected={unexpected_predictors}"
        )
    return bands


def build_movement_term_model_group(predictor_groups: Mapping, terms):
    """Build a complete, self-contained model family for selected terms.

    A term is one of the keys returned by :func:`build_movement_penalty_bands`.
    The returned family contains a selected-term ``full`` model, an
    ``intercept_only`` model, and matched ``<term>_alone`` and
    ``full_minus_<term>`` models.  The comparison models reuse the full
    model's fold-wise penalties.  When the runner uses its default shuffle
    ablation mode, ``<term>_alone`` keeps the selected term intact and
    shuffles every other selected term, while ``full_minus_<term>`` shuffles
    only that term.

    The result is intentionally independent of the legacy task/movement
    registry so model families can live in separate result directories while
    using the same short, predictable model names.
    """
    predictor_groups = {
        str(name): tuple(predictors)
        for name, predictors in dict(predictor_groups).items()
    }
    all_penalty_bands = build_movement_penalty_bands(predictor_groups)
    terms = tuple(dict.fromkeys(str(term) for term in terms))
    if not terms:
        raise ValueError("A movement term model group must contain at least one term")
    unknown_terms = set(terms).difference(all_penalty_bands)
    if unknown_terms:
        raise ValueError(
            "Unknown movement-model terms: "
            f"{sorted(unknown_terms)}; available={list(all_penalty_bands)}"
        )

    selected_penalty_bands = {
        term: tuple(all_penalty_bands[term]) for term in terms
    }
    selected_predictors = tuple(
        predictor
        for term in terms
        for predictor in selected_penalty_bands[term]
    )
    selected_predictor_set = set(selected_predictors)
    included_groups = tuple(
        group_name
        for group_name, group_predictors in predictor_groups.items()
        if selected_predictor_set.intersection(group_predictors)
    )
    included_predictors = {
        predictor
        for group_name in included_groups
        for predictor in predictor_groups[group_name]
    }
    excluded_predictors = tuple(
        predictor
        for group_name in included_groups
        for predictor in predictor_groups[group_name]
        if predictor not in selected_predictor_set
    )
    if selected_predictor_set.difference(included_predictors):
        raise RuntimeError("Selected terms contain predictors outside predictor groups")

    full_model_name = "full"
    intercept_model_name = "intercept_only"
    registry = {
        full_model_name: {
            "included_groups": included_groups,
            "excluded_predictors": excluded_predictors,
            "label": "Full (" + " + ".join(terms) + ")",
            "penalty_source_model": None,
        },
        intercept_model_name: {
            "included_groups": (),
            "label": "Intercept only",
            "penalty_source_model": full_model_name,
        },
    }
    individual_term_model_pairs = {}
    for term in terms:
        term_predictors = tuple(selected_penalty_bands[term])
        other_predictors = tuple(
            predictor
            for predictor in selected_predictors
            if predictor not in set(term_predictors)
        )
        alone_model_name = f"{term}_alone"
        reduced_model_name = f"full_minus_{term}"
        registry[alone_model_name] = {
            "included_groups": included_groups,
            "excluded_predictors": excluded_predictors,
            "ablated_predictors": other_predictors,
            "label": f"{term} alone",
            "penalty_source_model": full_model_name,
        }
        registry[reduced_model_name] = {
            "included_groups": included_groups,
            "excluded_predictors": excluded_predictors,
            "ablated_predictors": term_predictors,
            "label": f"Full - {term}",
            "penalty_source_model": full_model_name,
        }
        individual_term_model_pairs[term] = (
            alone_model_name,
            reduced_model_name,
        )

    model_names = (
        full_model_name,
        intercept_model_name,
        *(
            model_name
            for model_pair in individual_term_model_pairs.values()
            for model_name in model_pair
        ),
    )
    return {
        "terms": terms,
        "penalty_bands": selected_penalty_bands,
        "registry": registry,
        "full_model": full_model_name,
        "intercept_model": intercept_model_name,
        "individual_term_model_pairs": individual_term_model_pairs,
        "model_names": model_names,
    }



def build_movement_model_registry(predictor_groups: Mapping, additional_models=None):
    """Return initial and optional user-defined declarative model specs.

    ``additional_models`` maps a new model name either to an iterable of group
    names or to a mapping with ``included_groups`` and optional ``label``,
    ``excluded_predictors``, ``ablated_predictors``, and ``penalty_source_model`` entries. Existing
    model names cannot be replaced implicitly.
    """
    available_groups = set(predictor_groups)
    registry = {
        "intercept_only": {
            "included_groups": (),
            "label": "Intercept only",
        },
        "task_only": {
            "included_groups": ("task",),
            "label": "Task",
        },
        "movement_only": {
            "included_groups": ("translation", "turning", "limb"),
            "label": "Movement",
        },
        "task_movement": {
            "included_groups": ("task", "translation", "turning", "limb"),
            "label": "Task + movement (full model)",
        },
        "trial_type_only": {
            "included_groups": ("task",),
            "excluded_predictors": (
                "branch_id",
                "branch_pos_scaled",
                "trial_progress_distance",
            ),
            "label": "Trial type only",
        },
        "position_only": {
            "included_groups": ("task",),
            "excluded_predictors": (
                "trial_type",
                "trial_progress_distance",
            ),
            "label": "Position only",
        },
        "progress_only": {
            "included_groups": ("task",),
            "excluded_predictors": (
                "trial_type",
                "branch_id",
                "branch_pos_scaled",
            ),
            "label": "Progress only",
        },
        "translation_only": {
            "included_groups": ("translation",),
            "label": "Translation only",
        },
        "turning_only": {
            "included_groups": ("turning",),
            "label": "Turning only",
        },
        "limb_only": {
            "included_groups": ("limb",),
            "label": "Limb only",
        },
        "task_movement_minus_trial_type": {
            "included_groups": ("task", "translation", "turning", "limb"),
            "ablated_predictors": ("trial_type",),
            "label": "Task + movement − trial type",
        },
        "task_movement_minus_position": {
            "included_groups": ("task", "translation", "turning", "limb"),
            "ablated_predictors": ("branch_id", "branch_pos_scaled"),
            "label": "Task + movement − position",
        },
        "task_movement_minus_progress": {
            "included_groups": ("task", "translation", "turning", "limb"),
            "ablated_predictors": ("trial_progress_distance",),
            "label": "Task + movement − progress",
        },
        "task_movement_minus_translation": {
            "included_groups": ("task", "translation", "turning", "limb"),
            "ablated_predictors": tuple(predictor_groups["translation"]),
            "label": "Task + movement − translation",
        },
        "task_movement_minus_turning": {
            "included_groups": ("task", "translation", "turning", "limb"),
            "ablated_predictors": tuple(predictor_groups["turning"]),
            "label": "Task + movement − turning",
        },
        "task_movement_minus_limb": {
            "included_groups": ("task", "translation", "turning", "limb"),
            "ablated_predictors": tuple(predictor_groups["limb"]),
            "label": "Task + movement − limb",
        },
    }
    registry["task_movement"]["penalty_source_model"] = None
    for inherited_model_name in set(registry).difference({"intercept_only", "task_movement"}):
        registry[inherited_model_name]["penalty_source_model"] = "task_movement"

    for model_name, model_spec in (additional_models or {}).items():
        if model_name in registry:
            raise ValueError(
                f"{model_name!r} is an existing registry name and cannot be replaced"
            )
        if isinstance(model_spec, Mapping):
            if "included_groups" not in model_spec:
                raise ValueError(
                    f"{model_name!r} must define 'included_groups'"
                )
            registry[model_name] = {
                "included_groups": tuple(model_spec["included_groups"]),
                "label": model_spec.get("label", model_name),
            }
            if "excluded_predictors" in model_spec:
                registry[model_name]["excluded_predictors"] = tuple(
                    model_spec["excluded_predictors"]
                )
            if "penalty_source_model" in model_spec:
                registry[model_name]["penalty_source_model"] = model_spec[
                    "penalty_source_model"
                ]
            if "ablated_predictors" in model_spec:
                registry[model_name]["ablated_predictors"] = tuple(
                    model_spec["ablated_predictors"]
                )
        else:
            registry[model_name] = {
                "included_groups": tuple(model_spec),
                "label": model_name,
            }

    for model_name, model_spec in registry.items():
        included_groups = set(model_spec["included_groups"])
        unknown_groups = included_groups.difference(available_groups)
        if unknown_groups:
            raise ValueError(
                f"{model_name!r} uses unavailable groups: {sorted(unknown_groups)}"
            )
        selected_predictors = {
            predictor
            for group_name in model_spec["included_groups"]
            for predictor in predictor_groups[group_name]
        }
        excluded_predictors = set(model_spec.get("excluded_predictors", ()))
        unavailable_exclusions = excluded_predictors.difference(selected_predictors)
        if unavailable_exclusions:
            raise ValueError(
                f"{model_name!r} excludes predictors not present in its included "
                f"groups: {sorted(unavailable_exclusions)}"
            )
        ablated_predictors = set(model_spec.get("ablated_predictors", ()))
        unavailable_ablations = ablated_predictors.difference(selected_predictors)
        if unavailable_ablations:
            raise ValueError(
                f"{model_name!r} ablates predictors not present in its included "
                f"groups: {sorted(unavailable_ablations)}"
            )
        penalty_source_model = model_spec.get("penalty_source_model")
        if penalty_source_model is not None:
            if penalty_source_model not in registry:
                raise ValueError(
                    f"{model_name!r} uses unavailable penalty source "
                    f"{penalty_source_model!r}"
                )
            if penalty_source_model == model_name:
                raise ValueError(f"{model_name!r} cannot inherit penalties from itself")
        if "Intercept" in ablated_predictors:
            raise ValueError(f"{model_name!r} cannot ablate 'Intercept'")
    return registry


def movement_spline_config(paw_prefixes):
    """Return the approved spline specification for each continuous predictor.

    ``by='branch_id'`` identifies the existing branch-specific position term;
    its separate per-branch fits are created in a later design-matrix step.
    """
    config = {
        "branch_pos_scaled": {"kind": "bs", "df": 4, "by": "branch_id"},
        "trial_progress_distance": {"kind": "cr", "df": 4},
        "speed": {"kind": "bs", "df": 4},
        "tangential_acceleration_cm_s2": {"kind": "cr", "df": 4},
        "body_angular_velocity_rad_s": {"kind": "cr", "df": 4},
        "relative_paw_vigor": {"kind": "cr", "df": 4},
        "fore_minus_hind_vigor": {"kind": "cr", "df": 4},
        "mean_stance_width": {"kind": "cr", "df": 4},
        "fore_minus_hind_width": {"kind": "cr", "df": 4},
        "mean_fore_hind_spacing": {"kind": "cr", "df": 4},
        "fore_hind_spacing_asymmetry": {"kind": "cr", "df": 4},
    }
    return config


def _number_of_internal_knots(kind, df, degree=3, constraints="center"):
    """Match Patsy's df-to-knot convention for the approved spline terms."""
    if kind == "bs":
        # Patsy bs(..., include_intercept=False), which is the intended basis.
        n_knots = df - (degree + 1) + 1
    elif kind == "cr":
        n_constraints = 1 if constraints == "center" else 0
        n_knots = df - 2 + n_constraints
    else:
        raise ValueError(f"Unsupported spline kind: {kind!r}")
    if n_knots < 0:
        raise ValueError(
            f"df={df} is too small for {kind!r} with degree={degree}"
        )
    return n_knots


def fit_quantile_spline_spec(
    values,
    predictor,
    kind,
    df,
    degree=3,
    constraints="center",
    by=None,
):
    """Fit one frozen spline specification from outer-training values only.

    Boundaries are the finite outer-training minimum and maximum. Internal
    knots use equally spaced quantiles of the outer-training distribution.
    Duplicate quantile knots are rejected rather than silently changing the
    requested spline representation.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError(f"{predictor!r} has no finite outer-training values")

    lower_bound = float(values.min())
    upper_bound = float(values.max())
    if not lower_bound < upper_bound:
        raise ValueError(
            f"{predictor!r} has no outer-training range: "
            f"min=max={lower_bound}"
        )

    n_knots = _number_of_internal_knots(
        kind,
        df,
        degree=degree,
        constraints=constraints,
    )
    quantile_levels = np.linspace(0, 1, n_knots + 2)[1:-1]
    knots = np.quantile(values, quantile_levels).astype(float)
    all_locations = np.concatenate(([lower_bound], knots, [upper_bound]))
    if len(np.unique(all_locations)) != len(all_locations):
        raise ValueError(
            f"{predictor!r} has duplicate outer-training quantile knots or "
            "a knot equal to a boundary. Reduce spline df or explicitly "
            "approve an alternative representation."
        )

    return SplineSpec(
        predictor=predictor,
        kind=kind,
        df=int(df),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        knots=tuple(knots.tolist()),
        degree=int(degree),
        constraints=constraints if kind == "cr" else None,
        by=by,
    )


def clip_to_spline_bounds(
    values,
    spline_spec: SplineSpec,
    fold,
    trial_numbers,
    time_bin_centers,
):
    """Clip held-out values and return detailed clipping audit tables.

    The summary always has one row per boundary side. The detail table has one
    row per clipped finite sample, including its trial and time-bin location.
    A warning is emitted whenever clipping occurs.
    """
    values = np.asarray(values, dtype=float)
    trial_numbers = np.asarray(trial_numbers)
    time_bin_centers = np.asarray(time_bin_centers)
    if not (len(values) == len(trial_numbers) == len(time_bin_centers)):
        raise ValueError("values, trial_numbers, and time_bin_centers must align")

    finite = np.isfinite(values)
    lower_mask = finite & (values < spline_spec.lower_bound)
    upper_mask = finite & (values > spline_spec.upper_bound)
    clipped_values = values.copy()
    clipped_values[lower_mask] = spline_spec.lower_bound
    clipped_values[upper_mask] = spline_spec.upper_bound

    n_finite = int(finite.sum())
    test_min = float(values[finite].min()) if n_finite else np.nan
    test_max = float(values[finite].max()) if n_finite else np.nan
    summary_rows = []
    detail_rows = []
    for side, mask, boundary in (
        ("lower", lower_mask, spline_spec.lower_bound),
        ("upper", upper_mask, spline_spec.upper_bound),
    ):
        affected_trials = pd.unique(trial_numbers[mask]).tolist()
        n_clipped = int(mask.sum())
        summary_rows.append(
            {
                "fold": fold,
                "predictor": spline_spec.predictor,
                "spline_kind": spline_spec.kind,
                "side": side,
                "training_boundary": boundary,
                "held_out_min": test_min,
                "held_out_max": test_max,
                "n_held_out_finite": n_finite,
                "n_clipped": n_clipped,
                "fraction_clipped": n_clipped / n_finite if n_finite else np.nan,
                "affected_trial_ids": affected_trials,
            }
        )
        for index in np.flatnonzero(mask):
            detail_rows.append(
                {
                    "fold": fold,
                    "predictor": spline_spec.predictor,
                    "spline_kind": spline_spec.kind,
                    "side": side,
                    "training_boundary": boundary,
                    "trial_number": trial_numbers[index],
                    "time_bin_center": time_bin_centers[index],
                    "original_value": values[index],
                    "clipped_value": clipped_values[index],
                }
            )

    total_clipped = int(lower_mask.sum() + upper_mask.sum())
    if total_clipped:
        warnings.warn(
            f"Outer fold {fold}: clipped {total_clipped}/{n_finite} finite "
            f"held-out samples for {spline_spec.predictor!r}; inspect the "
            "returned clipping summary and detail tables.",
            stacklevel=2,
        )
    summary_columns = [
        "fold", "predictor", "spline_kind", "side", "training_boundary",
        "held_out_min", "held_out_max", "n_held_out_finite", "n_clipped",
        "fraction_clipped", "affected_trial_ids",
    ]
    detail_columns = [
        "fold", "predictor", "spline_kind", "side", "training_boundary",
        "trial_number", "time_bin_center", "original_value", "clipped_value",
    ]
    return (
        clipped_values,
        pd.DataFrame(summary_rows, columns=summary_columns),
        pd.DataFrame(detail_rows, columns=detail_columns),
    )


def _spline_formula(spline_spec: SplineSpec):
    """Return a Patsy term with fixed knots and boundaries."""
    knots = repr(tuple(float(knot) for knot in spline_spec.knots))
    lower_bound = repr(float(spline_spec.lower_bound))
    upper_bound = repr(float(spline_spec.upper_bound))
    if spline_spec.kind == "bs":
        return (
            "0 + bs(value, "
            f"knots={knots}, degree={spline_spec.degree}, "
            "include_intercept=False, "
            f"lower_bound={lower_bound}, upper_bound={upper_bound})"
        )
    if spline_spec.kind == "cr":
        return (
            "0 + cr(value, "
            f"knots={knots}, lower_bound={lower_bound}, "
            f"upper_bound={upper_bound}, constraints={spline_spec.constraints!r})"
        )
    raise ValueError(f"Unsupported spline kind: {spline_spec.kind!r}")


def _fit_spline_basis(values, spline_spec: SplineSpec):
    """Fit a Patsy spline basis on finite training values."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{spline_spec.predictor!r} has non-finite training values"
        )
    basis = dmatrix(
        _spline_formula(spline_spec),
        {"value": values},
        return_type="dataframe",
    )
    column_names = tuple(
        f"{spline_spec.predictor}__basis_{index}"
        for index in range(basis.shape[1])
    )
    basis.columns = column_names
    return basis, FittedSplineBasis(
        spec=spline_spec,
        design_info=basis.design_info,
        column_names=column_names,
    )


def _transform_spline_basis(
    fitted_basis: FittedSplineBasis,
    values,
    fold,
    trial_numbers,
    time_bin_centers,
):
    """Clip and transform held-out values using a frozen training basis."""
    clipped_values, clipping_summary, clipping_detail = clip_to_spline_bounds(
        values,
        fitted_basis.spec,
        fold=fold,
        trial_numbers=trial_numbers,
        time_bin_centers=time_bin_centers,
    )
    if not np.isfinite(clipped_values).all():
        raise ValueError(
            f"{fitted_basis.spec.predictor!r} has non-finite held-out values"
        )
    basis = build_design_matrices(
        [fitted_basis.design_info],
        {"value": clipped_values},
        return_type="dataframe",
    )[0]
    basis.columns = fitted_basis.column_names
    return basis, clipping_summary, clipping_detail


def _selected_columns(predictor_groups: Mapping, included_groups):
    unknown_groups = set(included_groups).difference(predictor_groups)
    if unknown_groups:
        raise KeyError(f"Unknown predictor groups: {sorted(unknown_groups)}")
    return [
        column
        for group in included_groups
        for column in predictor_groups[group]
    ]


def _validate_required_columns(data, columns, dataset_name):
    missing = set(columns).difference(data.columns)
    if missing:
        raise KeyError(f"{dataset_name} is missing columns: {sorted(missing)}")
    null_columns = [column for column in columns if data[column].isna().any()]
    if null_columns:
        raise ValueError(
            f"{dataset_name} has missing values in selected columns: "
            f"{sorted(null_columns)}"
        )


def _fit_categorical_basis(train_data, categorical_columns):
    """Fit a training-only Patsy categorical basis with an intercept."""
    levels = {}
    categorical_data = pd.DataFrame(index=train_data.index)
    for column in categorical_columns:
        observed = pd.unique(train_data[column].astype(object))
        observed = [value for value in observed if pd.notna(value)]
        if not observed:
            raise ValueError(f"{column!r} has no observed training levels")
        levels[column] = tuple(sorted(observed))
        categorical_data[column] = pd.Categorical(
            train_data[column], categories=levels[column]
        )

    formula = "1"
    if categorical_columns:
        formula += " + " + " + ".join(f"C({column})" for column in categorical_columns)
    basis = dmatrix(formula, categorical_data, return_type="dataframe")
    return basis, levels


def _transform_categorical_basis(
    transformer: DesignMatrixTransformer,
    test_data,
    fold,
):
    """Transform categories, failing explicitly for unseen held-out levels."""
    categorical_data = pd.DataFrame(index=test_data.index)
    for column in transformer.categorical_columns:
        observed = set(test_data[column].dropna().astype(object))
        unseen = observed.difference(transformer.categorical_levels[column])
        if unseen:
            affected_trials = (
                test_data.loc[
                    test_data[column].isin(unseen), "trial_number"
                ].unique().tolist()
            )
            raise ValueError(
                f"Outer fold {fold}: held-out {column!r} levels absent from "
                f"training: {sorted(unseen)}; affected trial IDs: "
                f"{affected_trials}"
            )
        categorical_data[column] = pd.Categorical(
            test_data[column], categories=transformer.categorical_levels[column]
        )
    basis = build_design_matrices(
        [transformer.categorical_design_info],
        categorical_data,
        return_type="dataframe",
    )[0]
    basis.columns = transformer.categorical_column_names
    return basis


def _concat_audit_frames(frames):
    """Concatenate audit-table frames without the empty/all-NA concat FutureWarning."""
    non_empty = [frame for frame in frames if not frame.empty]
    if non_empty:
        return pd.concat(non_empty, ignore_index=True)
    return pd.concat(frames, ignore_index=True)


def _zero_variance_report(matrix, fold):
    """Return an audit table for non-intercept training columns with zero SD."""
    non_intercept = [column for column in matrix.columns if column != "Intercept"]
    means = matrix[non_intercept].mean()
    scales = matrix[non_intercept].std(ddof=0)
    zero_variance = scales.index[~np.isfinite(scales) | scales.eq(0)].tolist()
    report = pd.DataFrame(
        [
            {
                "fold": fold,
                "column": column,
                "training_mean": means[column],
                "training_sd": scales[column],
                "action": "dropped_from_train_and_test",
            }
            for column in zero_variance
        ],
        columns=["fold", "column", "training_mean", "training_sd", "action"],
    )
    if zero_variance:
        warnings.warn(
            f"Outer fold {fold}: dropped {len(zero_variance)} zero-variance "
            "training design columns; inspect zero_variance_report.",
            stacklevel=2,
        )
    return zero_variance, report


def fit_design_matrix_transformer(
    train_data,
    predictor_groups: Mapping,
    included_groups,
    spline_config: Mapping,
    fold,
):
    """Fit a training-only design-matrix transformer for one CV split.

    The returned training matrix is standardized after basis construction.
    The transformer freezes category levels, spline design information,
    standardization parameters, and zero-variance column removal for the
    corresponding held-out transform.
    """
    included_groups = tuple(included_groups)
    selected = _selected_columns(predictor_groups, included_groups)
    selected_without_intercept = [column for column in selected if column != "Intercept"]
    _validate_required_columns(train_data, selected_without_intercept, "training data")

    categorical_columns = tuple(
        column for column in ("trial_type", "branch_id") if column in selected
    )
    direct_columns = tuple(
        column
        for column in selected_without_intercept
        if column not in categorical_columns and column not in spline_config
    )
    spline_predictors = [
        column for column in selected_without_intercept if column in spline_config
    ]

    categorical_basis, categorical_levels = _fit_categorical_basis(
        train_data,
        categorical_columns,
    )
    categorical_basis.index = train_data.index
    pieces = [categorical_basis]
    if direct_columns:
        pieces.append(train_data.loc[:, direct_columns].astype(float).copy())

    spline_bases = {}
    branch_spline_bases = {}
    for predictor in spline_predictors:
        config = spline_config[predictor]
        if config.get("by") is None:
            spline_spec = fit_quantile_spline_spec(
                train_data[predictor],
                predictor=predictor,
                **config,
            )
            basis, fitted_basis = _fit_spline_basis(
                train_data[predictor], spline_spec
            )
            basis.index = train_data.index
            pieces.append(basis)
            spline_bases[predictor] = fitted_basis
            continue

        by_column = config["by"]
        if by_column not in categorical_columns:
            raise ValueError(
                f"{predictor!r} requires {by_column!r} as a selected category"
            )
        for level in categorical_levels[by_column]:
            level_mask = train_data[by_column].eq(level).to_numpy()
            branch_predictor = f"{predictor}[{by_column}={level}]"
            spline_spec = fit_quantile_spline_spec(
                train_data.loc[level_mask, predictor],
                predictor=branch_predictor,
                kind=config["kind"],
                df=config["df"],
                degree=config.get("degree", 3),
                constraints=config.get("constraints", "center"),
                by=by_column,
            )
            level_basis, fitted_basis = _fit_spline_basis(
                train_data.loc[level_mask, predictor], spline_spec
            )
            basis = pd.DataFrame(
                0.0,
                index=train_data.index,
                columns=fitted_basis.column_names,
            )
            basis.loc[level_mask, :] = level_basis.to_numpy()
            pieces.append(basis)
            branch_spline_bases[level] = fitted_basis

    raw_train_matrix = pd.concat(pieces, axis=1)
    if raw_train_matrix.columns.duplicated().any():
        duplicates = raw_train_matrix.columns[raw_train_matrix.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate design columns: {duplicates}")

    zero_variance_columns, zero_variance_report = _zero_variance_report(
        raw_train_matrix,
        fold,
    )
    retained_columns = tuple(
        column for column in raw_train_matrix.columns if column not in zero_variance_columns
    )
    train_matrix = raw_train_matrix.loc[:, retained_columns].copy()
    standardized_columns = [
        column for column in retained_columns if column != "Intercept"
    ]
    means = train_matrix[standardized_columns].mean()
    scales = train_matrix[standardized_columns].std(ddof=0)
    train_matrix.loc[:, standardized_columns] = (
        train_matrix[standardized_columns] - means
    ) / scales

    transformer = DesignMatrixTransformer(
        included_groups=included_groups,
        direct_columns=direct_columns,
        categorical_columns=categorical_columns,
        categorical_levels=categorical_levels,
        categorical_design_info=categorical_basis.design_info,
        categorical_column_names=tuple(categorical_basis.columns),
        spline_bases=spline_bases,
        branch_spline_bases=branch_spline_bases,
        retained_columns=retained_columns,
        means=means,
        scales=scales,
        zero_variance_report=zero_variance_report,
    )
    return train_matrix, transformer


def transform_design_matrix(transformer: DesignMatrixTransformer, test_data, fold):
    """Apply one frozen training-only design-matrix transformer to test data."""
    required_columns = [
        *transformer.direct_columns,
        *transformer.categorical_columns,
        "trial_number",
        "time_bin_center",
    ]
    required_columns.extend(
        fitted_basis.spec.predictor.split("[")[0]
        for fitted_basis in transformer.spline_bases.values()
    )
    required_columns.extend(
        "branch_pos_scaled" for _ in transformer.branch_spline_bases
    )
    _validate_required_columns(test_data, required_columns, "held-out data")

    categorical_basis = _transform_categorical_basis(transformer, test_data, fold)
    categorical_basis.index = test_data.index
    pieces = [categorical_basis]
    if transformer.direct_columns:
        pieces.append(test_data.loc[:, transformer.direct_columns].astype(float).copy())

    clipping_summaries = []
    clipping_details = []
    for predictor, fitted_basis in transformer.spline_bases.items():
        basis, summary, detail = _transform_spline_basis(
            fitted_basis,
            test_data[predictor],
            fold=fold,
            trial_numbers=test_data["trial_number"],
            time_bin_centers=test_data["time_bin_center"],
        )
        basis.index = test_data.index
        summary["branch_id"] = pd.NA
        detail["branch_id"] = pd.NA
        pieces.append(basis)
        clipping_summaries.append(summary)
        clipping_details.append(detail)

    if transformer.branch_spline_bases:
        branch_basis_columns = [
            column
            for fitted_basis in transformer.branch_spline_bases.values()
            for column in fitted_basis.column_names
        ]
        branch_basis = pd.DataFrame(
            0.0,
            index=test_data.index,
            columns=branch_basis_columns,
        )
        for level, fitted_basis in transformer.branch_spline_bases.items():
            level_mask = test_data["branch_id"].eq(level).to_numpy()
            if not level_mask.any():
                continue
            basis, summary, detail = _transform_spline_basis(
                fitted_basis,
                test_data.loc[level_mask, "branch_pos_scaled"],
                fold=fold,
                trial_numbers=test_data.loc[level_mask, "trial_number"],
                time_bin_centers=test_data.loc[level_mask, "time_bin_center"],
            )
            branch_basis.loc[level_mask, fitted_basis.column_names] = basis.to_numpy()
            summary["branch_id"] = level
            detail["branch_id"] = level
            clipping_summaries.append(summary)
            clipping_details.append(detail)
        pieces.append(branch_basis)

    raw_test_matrix = pd.concat(pieces, axis=1)
    dropped_columns = set(transformer.zero_variance_report["column"])
    expected_columns = set(transformer.retained_columns) | dropped_columns
    unexpected_columns = set(raw_test_matrix.columns).difference(expected_columns)
    if unexpected_columns:
        raise ValueError(
            f"Outer fold {fold}: held-out design matrix has unexpected columns: "
            f"{sorted(unexpected_columns)}"
        )
    missing_columns = set(transformer.retained_columns).difference(raw_test_matrix.columns)
    if missing_columns:
        raise ValueError(
            f"Outer fold {fold}: held-out design matrix is missing columns: "
            f"{sorted(missing_columns)}"
        )
    test_matrix = raw_test_matrix.loc[:, transformer.retained_columns].copy()
    standardized_columns = [
        column for column in transformer.retained_columns if column != "Intercept"
    ]
    test_matrix.loc[:, standardized_columns] = (
        test_matrix[standardized_columns] - transformer.means
    ) / transformer.scales

    summary_columns = [
        "fold", "predictor", "spline_kind", "branch_id", "side",
        "training_boundary", "held_out_min", "held_out_max",
        "n_held_out_finite", "n_clipped", "fraction_clipped",
        "affected_trial_ids",
    ]
    detail_columns = [
        "fold", "predictor", "spline_kind", "branch_id", "side",
        "training_boundary", "trial_number", "time_bin_center",
        "original_value", "clipped_value",
    ]
    nonempty_summaries = [frame for frame in clipping_summaries if not frame.empty]
    nonempty_details = [frame for frame in clipping_details if not frame.empty]
    clipping_summary = (
        pd.concat(nonempty_summaries, ignore_index=True)
        if nonempty_summaries else pd.DataFrame(columns=summary_columns)
    )
    clipping_detail = (
        pd.concat(nonempty_details, ignore_index=True)
        if nonempty_details else pd.DataFrame(columns=detail_columns)
    )
    clipping_summary = clipping_summary.reindex(columns=summary_columns)
    clipping_detail = clipping_detail.reindex(columns=detail_columns)
    return test_matrix, clipping_summary, clipping_detail


def _design_columns_for_predictor(transformer, predictor):
    """Return retained transformed columns derived from one raw predictor."""
    retained = set(transformer.retained_columns)
    columns = []
    if predictor in transformer.direct_columns and predictor in retained:
        columns.append(predictor)
    if predictor in transformer.categorical_columns:
        prefix = f"C({predictor})["
        columns.extend(
            column
            for column in transformer.categorical_column_names
            if column in retained and column.startswith(prefix)
        )
    fitted_basis = transformer.spline_bases.get(predictor)
    if fitted_basis is not None:
        columns.extend(
            column for column in fitted_basis.column_names if column in retained
        )
    if predictor == "branch_pos_scaled":
        columns.extend(
            column
            for fitted_basis in transformer.branch_spline_bases.values()
            for column in fitted_basis.column_names
            if column in retained
        )
    return tuple(dict.fromkeys(columns))


def _feature_penalties_from_bands(transformer, penalty_bands, penalties):
    """Map raw-predictor band penalties onto retained design columns."""
    penalty_bands = {
        str(name): tuple(predictors)
        for name, predictors in dict(penalty_bands).items()
    }
    penalties = {str(name): float(value) for name, value in dict(penalties).items()}
    if set(penalties) != set(penalty_bands):
        raise ValueError(
            "Penalty values must match penalty bands exactly; "
            f"missing={sorted(set(penalty_bands).difference(penalties))}, "
            f"unexpected={sorted(set(penalties).difference(penalty_bands))}"
        )
    invalid = {
        name: value
        for name, value in penalties.items()
        if not np.isfinite(value) or value <= 0
    }
    if invalid:
        raise ValueError(f"Band penalties must be finite and positive: {invalid}")

    feature_columns = tuple(
        column for column in transformer.retained_columns if column != "Intercept"
    )
    column_to_band = {}
    for band_name, predictors in penalty_bands.items():
        for predictor in predictors:
            for column in _design_columns_for_predictor(transformer, predictor):
                previous_band = column_to_band.get(column)
                if previous_band is not None and previous_band != band_name:
                    raise ValueError(
                        f"Design column {column!r} belongs to both "
                        f"{previous_band!r} and {band_name!r}"
                    )
                column_to_band[column] = band_name
    unmapped = sorted(set(feature_columns).difference(column_to_band))
    if unmapped:
        raise ValueError(
            "Every retained non-intercept design column must belong to one "
            f"penalty band; unmapped={unmapped}"
        )
    feature_penalties = pd.Series(
        {column: penalties[column_to_band[column]] for column in feature_columns},
        index=feature_columns,
        dtype=float,
    )
    return feature_penalties, column_to_band



def shuffle_design_predictors(
    train_matrix,
    held_out_matrix,
    transformer,
    ablated_predictors,
    *,
    stage,
    outer_fold,
    inner_fold=None,
    seed=42,
):
    """Shuffle each selected transformed column across all split rows.

    Shuffling occurs on the unstandardized design values. Training means and
    scales are then recomputed and applied to both matrices. Each design
    column receives its own deterministic permutation.
    """
    if seed != 42:
        raise ValueError("The paper-style shuffle seed is fixed internally at 42")
    train_raw = train_matrix.copy()
    held_out_raw = held_out_matrix.copy()
    standardized_columns = [
        column for column in transformer.retained_columns if column != "Intercept"
    ]
    train_raw.loc[:, standardized_columns] = (
        train_raw[standardized_columns] * transformer.scales + transformer.means
    )
    held_out_raw.loc[:, standardized_columns] = (
        held_out_raw[standardized_columns] * transformer.scales + transformer.means
    )
    combined = pd.concat((train_raw, held_out_raw), axis=0, ignore_index=True)
    n_train = len(train_raw)
    audit_rows = []
    shuffled_columns = set()
    for predictor in tuple(ablated_predictors):
        design_columns = _design_columns_for_predictor(transformer, predictor)
        if not design_columns:
            audit_rows.append(
                {
                    "seed": 42,
                    "stage": stage,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "target_predictor": predictor,
                    "design_column": pd.NA,
                    "n_rows": len(combined),
                    "permutation_sha256": pd.NA,
                    "status": "no_retained_design_columns",
                }
            )
            continue
        for column in design_columns:
            if column in shuffled_columns:
                continue
            key = f"42\x1f{stage}\x1f{column}".encode("utf-8")
            derived_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
            permutation = np.random.default_rng(derived_seed).permutation(len(combined))
            combined.loc[:, column] = combined[column].to_numpy()[permutation]
            checksum = hashlib.sha256(
                np.asarray(permutation, dtype="<i8").tobytes()
            ).hexdigest()
            audit_rows.append(
                {
                    "seed": 42,
                    "stage": stage,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "target_predictor": predictor,
                    "design_column": column,
                    "n_rows": len(combined),
                    "permutation_sha256": checksum,
                    "status": "shuffled",
                }
            )
            shuffled_columns.add(column)

    shuffled_train = combined.iloc[:n_train].copy()
    shuffled_train.index = train_matrix.index
    shuffled_held_out = combined.iloc[n_train:].copy()
    shuffled_held_out.index = held_out_matrix.index
    post_shuffle_zero, post_shuffle_report = _zero_variance_report(
        shuffled_train,
        stage,
    )
    if not post_shuffle_report.empty:
        post_shuffle_report.loc[:, "action"] = (
            "dropped_after_shuffle_from_train_and_test"
        )
    retained_columns = tuple(
        column
        for column in transformer.retained_columns
        if column not in post_shuffle_zero
    )
    shuffled_train = shuffled_train.loc[:, retained_columns]
    shuffled_held_out = shuffled_held_out.loc[:, retained_columns]
    standardized_columns = [
        column for column in retained_columns if column != "Intercept"
    ]
    means = shuffled_train[standardized_columns].mean()
    scales = shuffled_train[standardized_columns].std(ddof=0)
    shuffled_train.loc[:, standardized_columns] = (
        shuffled_train[standardized_columns] - means
    ) / scales
    shuffled_held_out.loc[:, standardized_columns] = (
        shuffled_held_out[standardized_columns] - means
    ) / scales
    if transformer.zero_variance_report.empty:
        zero_variance_report = post_shuffle_report.reset_index(drop=True)
    elif post_shuffle_report.empty:
        zero_variance_report = transformer.zero_variance_report.copy()
    else:
        zero_variance_report = pd.concat(
            (transformer.zero_variance_report, post_shuffle_report),
            ignore_index=True,
            sort=False,
        )
    shuffled_transformer = replace(
        transformer,
        retained_columns=retained_columns,
        means=means,
        scales=scales,
        zero_variance_report=zero_variance_report,
    )
    return (
        shuffled_train,
        shuffled_held_out,
        shuffled_transformer,
        pd.DataFrame(audit_rows),
    )


def make_contiguous_inner_trial_folds(
    outer_training_trial_ids: Sequence,
    n_splits: int = 4,
    seed: int = 42,
):
    """Create deterministic inner CV folds from outer-training trial IDs."""
    folds = make_contiguous_outer_trial_folds(
        outer_training_trial_ids,
        n_splits=n_splits,
        seed=seed,
    )
    for fold in folds:
        fold["inner_fold"] = fold.pop("fold")
    return folds


def poisson_log_likelihood(observed_counts, predicted_counts):
    """Return the full Poisson log-likelihood for aligned count vectors."""
    observed_counts = np.asarray(observed_counts, dtype=float)
    predicted_counts = np.asarray(predicted_counts, dtype=float)
    if observed_counts.shape != predicted_counts.shape:
        raise ValueError("Observed and predicted count vectors must have the same shape")
    if not np.isfinite(observed_counts).all() or (observed_counts < 0).any():
        raise ValueError("Observed counts must be finite and non-negative")
    if not np.isfinite(predicted_counts).all() or (predicted_counts <= 0).any():
        raise ValueError("Predicted counts must be finite and positive")
    return float(np.sum(
        observed_counts * np.log(predicted_counts)
        - predicted_counts
        - gammaln(observed_counts + 1)
    ))


def poisson_deviance(observed_counts, predicted_counts):
    """Return Poisson deviance from the saturated model for aligned counts."""
    observed_counts = np.asarray(observed_counts, dtype=float)
    predicted_counts = np.asarray(predicted_counts, dtype=float)
    if observed_counts.shape != predicted_counts.shape:
        raise ValueError("Observed and predicted count vectors must have the same shape")
    if not np.isfinite(observed_counts).all() or (observed_counts < 0).any():
        raise ValueError("Observed counts must be finite and non-negative")
    if not np.isfinite(predicted_counts).all() or (predicted_counts <= 0).any():
        raise ValueError("Predicted counts must be finite and positive")
    deviance_terms = predicted_counts - observed_counts
    nonzero_observed = observed_counts > 0
    deviance_terms[nonzero_observed] += observed_counts[nonzero_observed] * np.log(
        observed_counts[nonzero_observed] / predicted_counts[nonzero_observed]
    )
    return float(2.0 * deviance_terms.sum())


def _fit_poisson_ridge(
    train_matrix,
    train_counts,
    alpha,
    estimator_kwargs=None,
):
    """Fit one Poisson ridge model with its intercept excluded from penalty."""
    estimator_kwargs = dict(estimator_kwargs or {})
    estimator_kwargs.setdefault("max_iter", 5000)
    estimator_kwargs.setdefault("tol", 1e-8)
    if "fit_intercept" in estimator_kwargs:
        raise ValueError("fit_intercept is fixed to True for the Poisson ridge model")
    if "alpha" in estimator_kwargs:
        raise ValueError("Pass ridge strength via alpha, not estimator_kwargs")

    train_counts = np.asarray(train_counts, dtype=float)
    if not np.isfinite(train_counts).all() or (train_counts < 0).any():
        raise ValueError("Training spike counts must be finite and non-negative")
    if train_counts.sum() <= 0:
        raise ValueError(
            "Cannot fit a Poisson model when the training split contains zero "
            "spikes for the selected neuron"
        )
    feature_columns = [
        column for column in train_matrix.columns if column != "Intercept"
    ]
    if feature_columns:
        estimator = PoissonRegressor(
            alpha=float(alpha),
            fit_intercept=True,
            **estimator_kwargs,
        )
        estimator.fit(train_matrix[feature_columns], train_counts)
        return FittedPoissonRidgeModel(
            estimator=estimator,
            feature_columns=tuple(feature_columns),
            intercept_only_mean=None,
        )

    # An intercept-only model has no coefficient subject to ridge shrinkage.
    return _fit_intercept_only(train_counts)


def _fit_poisson_banded_ridge(
    train_matrix,
    train_counts,
    transformer,
    penalty_bands,
    penalties,
    estimator_kwargs=None,
):
    """Fit banded Poisson ridge by exact per-feature rescaling."""
    estimator_kwargs = dict(estimator_kwargs or {})
    estimator_kwargs.setdefault("max_iter", 5000)
    estimator_kwargs.setdefault("tol", 1e-8)
    if "fit_intercept" in estimator_kwargs:
        raise ValueError("fit_intercept is fixed to True for banded ridge")
    if "alpha" in estimator_kwargs:
        raise ValueError("Band penalties, not estimator alpha, set regularization")

    train_counts = np.asarray(train_counts, dtype=float)
    if not np.isfinite(train_counts).all() or (train_counts < 0).any():
        raise ValueError("Training spike counts must be finite and non-negative")
    if train_counts.sum() <= 0:
        raise ValueError(
            "Cannot fit a Poisson model when the training split contains zero "
            "spikes for the selected neuron"
        )
    feature_penalties, _ = _feature_penalties_from_bands(
        transformer, penalty_bands, penalties
    )
    if feature_penalties.empty:
        return _fit_intercept_only(train_counts)
    feature_scales = np.sqrt(feature_penalties)
    estimator = PoissonRegressor(
        alpha=1.0,
        fit_intercept=True,
        **estimator_kwargs,
    )
    estimator.fit(
        train_matrix.loc[:, feature_penalties.index] / feature_scales,
        train_counts,
    )
    return FittedPoissonRidgeModel(
        estimator=estimator,
        feature_columns=tuple(feature_penalties.index),
        intercept_only_mean=None,
        feature_scales=feature_scales,
    )



def _fit_intercept_only(train_counts):
    """Fit the explicit unpenalized Poisson null model for equal-width bins."""
    train_counts = np.asarray(train_counts, dtype=float)
    if not np.isfinite(train_counts).all() or (train_counts < 0).any():
        raise ValueError("Training spike counts must be finite and non-negative")
    if train_counts.sum() <= 0:
        raise ValueError(
            "Cannot fit a Poisson model when the training split contains zero "
            "spikes for the selected neuron"
        )
    return FittedPoissonRidgeModel(
        estimator=None,
        feature_columns=(),
        intercept_only_mean=float(train_counts.mean()),
    )


def _fit_and_predict_poisson_ridge(
    train_matrix,
    train_counts,
    validation_matrix,
    alpha,
    estimator_kwargs=None,
):
    """Fit an unpenalized-intercept Poisson ridge model and predict validation."""
    model = _fit_poisson_ridge(
        train_matrix,
        train_counts,
        alpha=alpha,
        estimator_kwargs=estimator_kwargs,
    )
    return model.predict(validation_matrix)


def _fit_and_predict_intercept_only(train_counts, validation_matrix):
    """Predict held-out bins from the explicit intercept-only Poisson model."""
    return _fit_intercept_only(train_counts).predict(validation_matrix)


def fit_outer_fold_poisson_ridge(
    outer_training_data,
    spike_counts_unit,
    predictor_groups: Mapping,
    included_groups,
    spline_config: Mapping,
    outer_fold,
    selected_alpha=None,
    selected_penalties=None,
    penalty_bands=None,
    selected_null_model: bool = False,
    estimator_kwargs=None,
    outer_test_data=None,
    ablated_predictors=(),
    shuffle_seed: int = 42,
):
    """Fit one final scalar- or banded-ridge model on an outer train set."""
    outer_training_data = outer_training_data.copy()
    if "trial_number" not in outer_training_data.columns:
        raise KeyError("outer_training_data is missing trial_number")
    if len(outer_training_data) != len(spike_counts_unit):
        raise ValueError(
            "outer_training_data and spike_counts_unit must have equal length"
        )
    if selected_null_model:
        if selected_alpha is not None:
            raise ValueError(
                "selected_alpha must be None when selected_null_model is true"
            )
    elif selected_penalties is not None:
        if penalty_bands is None:
            raise ValueError("penalty_bands is required with selected_penalties")
        if selected_alpha is not None:
            raise ValueError("selected_alpha and selected_penalties are exclusive")
        selected_penalties = {
            str(name): float(value)
            for name, value in dict(selected_penalties).items()
        }
    else:
        selected_alpha = float(selected_alpha)
        if not np.isfinite(selected_alpha) or selected_alpha < 0:
            raise ValueError("selected_alpha must be finite and non-negative")

    train_matrix, transformer = fit_design_matrix_transformer(
        outer_training_data,
        predictor_groups=predictor_groups,
        included_groups=included_groups,
        spline_config=spline_config,
        fold=f"outer_{outer_fold}",
    )
    held_out_design_matrix = None
    clipping_summary = pd.DataFrame()
    clipping_detail = pd.DataFrame()
    permutation_audit = pd.DataFrame()
    if ablated_predictors:
        if outer_test_data is None:
            raise ValueError(
                "outer_test_data is required when predictors are shuffled"
            )
        held_out_design_matrix, clipping_summary, clipping_detail = (
            transform_design_matrix(
                transformer,
                outer_test_data,
                fold=f"outer_{outer_fold}",
            )
        )
        (
            train_matrix,
            held_out_design_matrix,
            transformer,
            permutation_audit,
        ) = shuffle_design_predictors(
            train_matrix,
            held_out_design_matrix,
            transformer,
            ablated_predictors,
            stage=f"outer_{outer_fold}",
            outer_fold=outer_fold,
            seed=shuffle_seed,
        )
    if selected_null_model:
        model = _fit_intercept_only(spike_counts_unit)
    elif selected_penalties is not None:
        model = _fit_poisson_banded_ridge(
            train_matrix,
            spike_counts_unit,
            transformer=transformer,
            penalty_bands=penalty_bands,
            penalties=selected_penalties,
            estimator_kwargs=estimator_kwargs,
        )
    else:
        model = _fit_poisson_ridge(
            train_matrix,
            spike_counts_unit,
            alpha=selected_alpha,
            estimator_kwargs=estimator_kwargs,
        )
    return OuterFoldPoissonRidgeFit(
        outer_fold=outer_fold,
        selected_alpha=selected_alpha,
        selected_penalties=selected_penalties,
        selected_null_model=bool(selected_null_model),
        train_trial_ids=tuple(
            np.sort(
                outer_training_data["trial_number"].dropna().unique()
            ).tolist()
        ),
        n_training_bins=len(outer_training_data),
        null_mean_count=float(np.mean(spike_counts_unit)),
        transformer=transformer,
        model=model,
        held_out_design_matrix=held_out_design_matrix,
        clipping_summary=clipping_summary,
        clipping_detail=clipping_detail,
        permutation_audit=permutation_audit,
    )


def evaluate_outer_fold_poisson_ridge(

    fitted_outer_model: OuterFoldPoissonRidgeFit,
    outer_test_data,
    spike_counts_unit,
):
    """Evaluate one final model on its untouched outer-test bins.

    The fitted transformer's spline basis and train-derived standardization are
    applied without refitting. Poisson log-likelihood is reported both as a
    total and per held-out bin. Deviance explained uses the intercept-only
    Poisson rate estimated from this outer fold's training bins as its null.
    """
    outer_test_data = outer_test_data.copy()
    if "trial_number" not in outer_test_data.columns:
        raise KeyError("outer_test_data is missing 'trial_number'")
    if len(outer_test_data) == 0:
        raise ValueError("outer_test_data has no held-out bins")
    if len(outer_test_data) != len(spike_counts_unit):
        raise ValueError(
            "outer_test_data and spike_counts_unit must have equal length"
        )

    if fitted_outer_model.held_out_design_matrix is None:
        test_matrix, clipping_summary, clipping_detail = transform_design_matrix(
            fitted_outer_model.transformer,
            outer_test_data,
            fold=f"outer_{fitted_outer_model.outer_fold}",
        )
    else:
        if not fitted_outer_model.held_out_design_matrix.index.equals(
            outer_test_data.index
        ):
            raise ValueError("outer_test_data does not match the prepared held-out data")
        test_matrix = fitted_outer_model.held_out_design_matrix
        clipping_summary = fitted_outer_model.clipping_summary
        clipping_detail = fitted_outer_model.clipping_detail
    observed_counts = np.asarray(spike_counts_unit, dtype=float)
    predicted_counts = fitted_outer_model.model.predict(test_matrix)
    log_likelihood = poisson_log_likelihood(observed_counts, predicted_counts)
    model_deviance = poisson_deviance(observed_counts, predicted_counts)
    null_predictions = np.full(
        len(observed_counts), fitted_outer_model.null_mean_count, dtype=float
    )
    null_deviance = poisson_deviance(observed_counts, null_predictions)
    deviance_explained = (
        np.nan
        if null_deviance == 0
        else 1.0 - (model_deviance / null_deviance)
    )
    prediction_columns = {
        "trial_number": outer_test_data["trial_number"].to_numpy(),
        "observed_spike_count": observed_counts,
        "predicted_spike_count": predicted_counts,
    }
    if "time_bin_center" in outer_test_data:
        prediction_columns["time_bin_center"] = outer_test_data[
            "time_bin_center"
        ].to_numpy()
    prediction_table = pd.DataFrame(prediction_columns, index=outer_test_data.index)
    return OuterFoldPoissonRidgeEvaluation(
        outer_fold=fitted_outer_model.outer_fold,
        test_trial_ids=tuple(
            np.sort(outer_test_data["trial_number"].dropna().unique()).tolist()
        ),
        n_test_bins=len(outer_test_data),
        log_likelihood=log_likelihood,
        mean_log_likelihood_per_bin=log_likelihood / len(outer_test_data),
        model_deviance=model_deviance,
        null_deviance=null_deviance,
        deviance_explained=deviance_explained,
        prediction_table=prediction_table,
        clipping_summary=clipping_summary,
        clipping_detail=clipping_detail,
    )


def _alpha_grid(lower_log10, upper_log10, log10_step=0.5):
    """Return an inclusive, evenly log-spaced ridge grid."""
    n_steps = int(round((upper_log10 - lower_log10) / log10_step))
    exponents = lower_log10 + log10_step * np.arange(n_steps + 1)
    return np.power(10.0, exponents)


def tune_poisson_ridge_alpha(
    outer_training_data,
    spike_counts_unit,
    predictor_groups: Mapping,
    included_groups,
    spline_config: Mapping,
    outer_fold,
    n_inner_splits: int = 4,
    seed: int = 42,
    lower_log10: float = -8.0,
    upper_log10: float = 2.0,
    log10_step: float = 0.5,
    edge_expansion_log10: float = 2.0,
    max_grid_expansions: int = 10,
    estimator_kwargs=None,
    ablated_predictors=(),
    shuffle_seed: int = 42,
):
    """Select finite ridge or an explicit null model with nested inner CV.

    Every finite alpha and the intercept-only Poisson model are evaluated on
    exactly the same inner-validation bins. A numerical tie with the null
    model selects the null; otherwise ties between finite ridge candidates
    select the larger alpha. Finite-grid expansion is retained only when a
    non-null ridge candidate is selected at a grid edge.
    """
    outer_training_data = outer_training_data.copy()
    if "trial_number" not in outer_training_data.columns:
        raise KeyError("outer_training_data is missing 'trial_number'")
    if len(outer_training_data) != len(spike_counts_unit):
        raise ValueError(
            "outer_training_data and spike_counts_unit must have equal length"
        )
    counts = pd.Series(
        np.asarray(spike_counts_unit, dtype=float),
        index=outer_training_data.index,
    )
    inner_folds = make_contiguous_inner_trial_folds(
        outer_training_data["trial_number"],
        n_splits=n_inner_splits,
        seed=seed,
    )

    prepared_folds = []
    clipping_summaries = []
    clipping_details = []
    zero_variance_reports = []
    permutation_audits = []
    for inner_fold in inner_folds:
        inner_fold_index = inner_fold["inner_fold"]
        fold_label = f"outer_{outer_fold}_inner_{inner_fold_index}"
        train_mask = outer_training_data["trial_number"].isin(
            inner_fold["train_trial_ids"]
        )
        validation_mask = outer_training_data["trial_number"].isin(
            inner_fold["test_trial_ids"]
        )
        inner_train = outer_training_data.loc[train_mask]
        inner_validation = outer_training_data.loc[validation_mask]
        train_matrix, transformer = fit_design_matrix_transformer(
            inner_train,
            predictor_groups=predictor_groups,
            included_groups=included_groups,
            spline_config=spline_config,
            fold=fold_label,
        )
        validation_matrix, clipping_summary, clipping_detail = transform_design_matrix(
            transformer,
            inner_validation,
            fold=fold_label,
        )
        if ablated_predictors:
            (
                train_matrix,
                validation_matrix,
                transformer,
                permutation_audit,
            ) = shuffle_design_predictors(
                train_matrix,
                validation_matrix,
                transformer,
                ablated_predictors,
                stage=fold_label,
                outer_fold=outer_fold,
                inner_fold=inner_fold_index,
                seed=shuffle_seed,
            )
            permutation_audits.append(permutation_audit)
        for audit in (clipping_summary, clipping_detail, transformer.zero_variance_report):
            audit["outer_fold"] = outer_fold
            audit["inner_fold"] = inner_fold_index
        clipping_summaries.append(clipping_summary)
        clipping_details.append(clipping_detail)
        zero_variance_reports.append(transformer.zero_variance_report)
        prepared_folds.append(
            {
                "inner_fold": inner_fold_index,
                "train_matrix": train_matrix,
                "validation_matrix": validation_matrix,
                "train_counts": counts.loc[inner_train.index].to_numpy(),
                "validation_counts": counts.loc[inner_validation.index].to_numpy(),
            }
        )

    expansions = 0
    while True:
        alpha_grid = _alpha_grid(lower_log10, upper_log10, log10_step)
        score_rows = []
        for alpha in alpha_grid:
            for prepared in prepared_folds:
                predicted_counts = _fit_and_predict_poisson_ridge(
                    prepared["train_matrix"],
                    prepared["train_counts"],
                    prepared["validation_matrix"],
                    alpha=alpha,
                    estimator_kwargs=estimator_kwargs,
                )
                log_likelihood = poisson_log_likelihood(
                    prepared["validation_counts"], predicted_counts
                )
                score_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": prepared["inner_fold"],
                        "candidate": "ridge",
                        "alpha": float(alpha),
                        "log_likelihood": log_likelihood,
                        "n_validation_bins": len(prepared["validation_counts"]),
                    }
                )
        for prepared in prepared_folds:
            predicted_counts = _fit_and_predict_intercept_only(
                prepared["train_counts"], prepared["validation_matrix"]
            )
            log_likelihood = poisson_log_likelihood(
                prepared["validation_counts"], predicted_counts
            )
            score_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": prepared["inner_fold"],
                    "candidate": "null",
                    "alpha": np.nan,
                    "log_likelihood": log_likelihood,
                    "n_validation_bins": len(prepared["validation_counts"]),
                }
            )

        score_table = pd.DataFrame(score_rows)
        candidate_scores = (
            score_table.groupby(["candidate", "alpha"], as_index=False, dropna=False)
            .agg(
                total_log_likelihood=("log_likelihood", "sum"),
                total_validation_bins=("n_validation_bins", "sum"),
            )
            .sort_values(["candidate", "alpha"])
            .reset_index(drop=True)
        )
        candidate_scores["mean_log_likelihood_per_bin"] = (
            candidate_scores["total_log_likelihood"]
            / candidate_scores["total_validation_bins"]
        )
        finite_scores = candidate_scores.loc[
            candidate_scores["candidate"].eq("ridge")
        ].copy()
        best_finite_score = float(finite_scores["mean_log_likelihood_per_bin"].max())
        finite_ties = finite_scores.loc[
            np.isclose(
                finite_scores["mean_log_likelihood_per_bin"],
                best_finite_score,
                rtol=1e-12,
                atol=1e-12,
            ),
            "alpha",
        ]
        best_finite_alpha = float(finite_ties.max())
        null_model_score = float(
            candidate_scores.loc[
                candidate_scores["candidate"].eq("null"),
                "mean_log_likelihood_per_bin",
            ].iloc[0]
        )
        best_score = max(best_finite_score, null_model_score)
        selected_null_model = bool(
            np.isclose(
                null_model_score, best_score, rtol=1e-12, atol=1e-12
            )
        )
        selected_alpha = None if selected_null_model else best_finite_alpha
        selected_candidate_score = (
            null_model_score if selected_null_model else best_finite_score
        )
        if selected_null_model:
            break
        at_lower_edge = selected_alpha == float(alpha_grid[0])
        at_upper_edge = selected_alpha == float(alpha_grid[-1])
        if not (at_lower_edge or at_upper_edge):
            break
        if expansions >= max_grid_expansions:
            raise RuntimeError(
                f"Outer fold {outer_fold}: ridge grid was not bracketed after "
                f"{max_grid_expansions} expansions; selected alpha={selected_alpha}"
            )
        if at_lower_edge:
            lower_log10 -= edge_expansion_log10
        if at_upper_edge:
            upper_log10 += edge_expansion_log10
        expansions += 1

    score_table = score_table.merge(
        candidate_scores,
        on=["candidate", "alpha"],
        how="left",
    )
    return RidgeAlphaSearchResult(
        selected_alpha=selected_alpha,
        selected_null_model=selected_null_model,
        selected_candidate_score=selected_candidate_score,
        best_finite_alpha=best_finite_alpha,
        best_finite_alpha_score=best_finite_score,
        null_model_score=null_model_score,
        alpha_grid=alpha_grid,
        n_grid_expansions=expansions,
        score_table=score_table,
        inner_folds=inner_folds,
        clipping_summary=_concat_audit_frames(clipping_summaries),
        clipping_detail=_concat_audit_frames(clipping_details),
        zero_variance_report=_concat_audit_frames(zero_variance_reports),
        permutation_audit=(
            pd.concat(permutation_audits, ignore_index=True)
            if permutation_audits else pd.DataFrame()
        ),
    )


def _banded_inner_cv_objective(
    log_penalties,
    band_names,
    prepared_folds,
    penalty_bands,
    estimator_kwargs=None,
):
    """Return pooled validation NLL/bin and its analytic log-penalty gradient."""
    log_penalties = np.asarray(log_penalties, dtype=float)
    penalties = {
        band_name: float(np.exp(value))
        for band_name, value in zip(band_names, log_penalties)
    }
    band_indices = {name: index for index, name in enumerate(band_names)}
    total_log_likelihood = 0.0
    total_validation_bins = 0
    total_gradient = np.zeros(len(band_names), dtype=float)

    for prepared in prepared_folds:
        train_matrix = prepared["train_matrix"]
        validation_matrix = prepared["validation_matrix"]
        train_counts = prepared["train_counts"]
        validation_counts = prepared["validation_counts"]
        transformer = prepared["transformer"]
        model = _fit_poisson_banded_ridge(
            train_matrix,
            train_counts,
            transformer=transformer,
            penalty_bands=penalty_bands,
            penalties=penalties,
            estimator_kwargs=estimator_kwargs,
        )
        if model.estimator is None:
            raise ValueError("Banded-ridge tuning requires at least one feature")

        feature_columns = model.feature_columns
        feature_penalties, column_to_band = _feature_penalties_from_bands(
            transformer, penalty_bands, penalties
        )
        train_features = train_matrix.loc[:, feature_columns].to_numpy(dtype=float)
        validation_features = validation_matrix.loc[
            :, feature_columns
        ].to_numpy(dtype=float)
        feature_scales = model.feature_scales.loc[list(feature_columns)].to_numpy()
        coefficients = model.estimator.coef_ / feature_scales
        train_predictions = model.predict(train_matrix)
        validation_predictions = model.predict(validation_matrix)

        n_train = len(train_counts)
        augmented_train = np.column_stack(
            (np.ones(n_train, dtype=float), train_features)
        )
        hessian = (
            augmented_train.T
            @ (train_predictions[:, None] * augmented_train)
            / n_train
        )
        hessian[1:, 1:] += np.diag(feature_penalties.to_numpy())

        penalty_derivatives = np.zeros(
            (len(feature_columns) + 1, len(band_names)), dtype=float
        )
        for feature_index, column in enumerate(feature_columns):
            band_index = band_indices[column_to_band[column]]
            penalty_derivatives[feature_index + 1, band_index] = (
                feature_penalties.loc[column] * coefficients[feature_index]
            )
        parameter_derivatives = -np.linalg.solve(
            hessian, penalty_derivatives
        )

        validation_residual = validation_predictions - validation_counts
        augmented_validation = np.column_stack(
            (np.ones(len(validation_counts), dtype=float), validation_features)
        )
        validation_parameter_gradient = (
            augmented_validation.T @ validation_residual
        )
        total_gradient += (
            validation_parameter_gradient @ parameter_derivatives
        )
        total_log_likelihood += poisson_log_likelihood(
            validation_counts, validation_predictions
        )
        total_validation_bins += len(validation_counts)

    objective = -total_log_likelihood / total_validation_bins
    gradient = total_gradient / total_validation_bins
    return float(objective), gradient


def tune_poisson_banded_ridge(
    outer_training_data,
    spike_counts_unit,
    predictor_groups: Mapping,
    included_groups,
    spline_config: Mapping,
    penalty_bands: Mapping,
    outer_fold,
    n_inner_splits: int = 4,
    seed: int = 42,
    lower_log10: float = -8.0,
    upper_log10: float = 2.0,
    log10_step: float = 0.5,
    edge_expansion_log10: float = 2.0,
    max_grid_expansions: int = 10,
    max_iter: int = 20,
    max_evaluations: int = 30,
    gtol: float = 1e-4,
    ftol: float = 1e-9,
    estimator_kwargs=None,
):
    """Tune six positive ridge penalties by analytic-hypergradient L-BFGS."""
    outer_training_data = outer_training_data.copy()
    band_names = tuple(penalty_bands)
    if not band_names:
        raise ValueError("penalty_bands must contain at least one band")
    if len(outer_training_data) != len(spike_counts_unit):
        raise ValueError(
            "outer_training_data and spike_counts_unit must have equal length"
        )
    if max_iter < 1 or max_evaluations < 1:
        raise ValueError("Banded optimizer limits must be positive")

    initialization = tune_poisson_ridge_alpha(
        outer_training_data,
        spike_counts_unit,
        predictor_groups=predictor_groups,
        included_groups=included_groups,
        spline_config=spline_config,
        outer_fold=outer_fold,
        n_inner_splits=n_inner_splits,
        seed=seed,
        lower_log10=lower_log10,
        upper_log10=upper_log10,
        log10_step=log10_step,
        edge_expansion_log10=edge_expansion_log10,
        max_grid_expansions=max_grid_expansions,
        estimator_kwargs=estimator_kwargs,
    )
    counts = pd.Series(
        np.asarray(spike_counts_unit, dtype=float),
        index=outer_training_data.index,
    )
    prepared_folds = []
    clipping_summaries = []
    clipping_details = []
    zero_variance_reports = []
    for inner_fold in initialization.inner_folds:
        inner_fold_index = inner_fold["inner_fold"]
        fold_label = f"outer_{outer_fold}_inner_{inner_fold_index}"
        train_mask = outer_training_data["trial_number"].isin(
            inner_fold["train_trial_ids"]
        )
        validation_mask = outer_training_data["trial_number"].isin(
            inner_fold["test_trial_ids"]
        )
        inner_train = outer_training_data.loc[train_mask]
        inner_validation = outer_training_data.loc[validation_mask]
        train_matrix, transformer = fit_design_matrix_transformer(
            inner_train,
            predictor_groups=predictor_groups,
            included_groups=included_groups,
            spline_config=spline_config,
            fold=fold_label,
        )
        validation_matrix, clipping_summary, clipping_detail = (
            transform_design_matrix(
                transformer, inner_validation, fold=fold_label
            )
        )
        for audit in (
            clipping_summary,
            clipping_detail,
            transformer.zero_variance_report,
        ):
            audit["outer_fold"] = outer_fold
            audit["inner_fold"] = inner_fold_index
        clipping_summaries.append(clipping_summary)
        clipping_details.append(clipping_detail)
        zero_variance_reports.append(transformer.zero_variance_report)
        prepared_folds.append(
            {
                "inner_fold": inner_fold_index,
                "train_matrix": train_matrix,
                "validation_matrix": validation_matrix,
                "train_counts": counts.loc[inner_train.index].to_numpy(),
                "validation_counts": counts.loc[inner_validation.index].to_numpy(),
                "transformer": transformer,
            }
        )

    lower_bound = float(np.log(10.0) * lower_log10)
    upper_bound = float(np.log(10.0) * upper_log10)
    initial_log_penalty = float(
        np.clip(
            np.log(initialization.best_finite_alpha),
            lower_bound,
            upper_bound,
        )
    )
    initial_point = np.full(len(band_names), initial_log_penalty, dtype=float)
    evaluation_rows = []

    def objective(log_penalties):
        value, gradient = _banded_inner_cv_objective(
            log_penalties,
            band_names,
            prepared_folds,
            penalty_bands,
            estimator_kwargs=estimator_kwargs,
        )
        row = {
            "outer_fold": outer_fold,
            "evaluation": len(evaluation_rows),
            "objective": value,
            "mean_log_likelihood_per_bin": -value,
            "gradient_norm": float(np.linalg.norm(gradient)),
        }
        row.update(
            {
                f"penalty_{name}": float(np.exp(log_penalty))
                for name, log_penalty in zip(band_names, log_penalties)
            }
        )
        evaluation_rows.append(row)
        return value, gradient

    optimizer_result = minimize(
        objective,
        initial_point,
        method="L-BFGS-B",
        jac=True,
        bounds=[(lower_bound, upper_bound)] * len(band_names),
        options={
            "maxiter": int(max_iter),
            "maxfun": int(max_evaluations),
            "gtol": float(gtol),
            "ftol": float(ftol),
        },
    )
    selected_penalties = {
        name: float(np.exp(value))
        for name, value in zip(band_names, optimizer_result.x)
    }
    best_finite_score = float(-optimizer_result.fun)
    null_model_score = float(initialization.null_model_score)
    selected_null_model = False
    selected_candidate_score = best_finite_score
    evaluation_table = pd.DataFrame(evaluation_rows)
    if not evaluation_table.empty:
        final_values = np.asarray(list(selected_penalties.values()))
        penalty_columns = [f"penalty_{name}" for name in band_names]
        evaluation_table["selected_optimizer_point"] = np.all(
            np.isclose(
                evaluation_table[penalty_columns].to_numpy(),
                final_values[None, :],
                rtol=1e-10,
                atol=1e-12,
            ),
            axis=1,
        )
        evaluation_table["optimizer_success"] = bool(optimizer_result.success)
        evaluation_table["optimizer_status"] = int(optimizer_result.status)
        evaluation_table["optimizer_message"] = str(optimizer_result.message)
        evaluation_table["optimizer_n_iterations"] = int(optimizer_result.nit)
        evaluation_table["optimizer_n_evaluations"] = int(optimizer_result.nfev)
        for band_name, value in selected_penalties.items():
            evaluation_table[f"selected_at_lower_{band_name}"] = np.isclose(
                np.log(value), lower_bound
            )
            evaluation_table[f"selected_at_upper_{band_name}"] = np.isclose(
                np.log(value), upper_bound
            )

    return BandedRidgeSearchResult(
        selected_penalties=selected_penalties,
        selected_null_model=selected_null_model,
        selected_candidate_score=selected_candidate_score,
        best_finite_alpha=initialization.best_finite_alpha,
        best_finite_alpha_score=best_finite_score,
        null_model_score=null_model_score,
        optimizer_success=bool(optimizer_result.success),
        optimizer_status=int(optimizer_result.status),
        optimizer_message=str(optimizer_result.message),
        optimizer_n_iterations=int(optimizer_result.nit),
        optimizer_n_evaluations=int(optimizer_result.nfev),
        evaluation_table=evaluation_table,
        initialization_score_table=initialization.score_table,
        inner_folds=initialization.inner_folds,
        clipping_summary=_concat_audit_frames(clipping_summaries),
        clipping_detail=_concat_audit_frames(clipping_details),
        zero_variance_report=_concat_audit_frames(zero_variance_reports),
        permutation_audit=pd.DataFrame(),
    )



def make_contiguous_outer_trial_folds(
    trial_ids: Sequence,
    n_splits: int = 5,
    seed: int = 42,
):
    """Create deterministic outer CV folds from contiguous trial blocks.

    Trial identifiers are sorted, then divided into ``n_splits`` contiguous
    test blocks. ``seed`` is retained as experiment metadata; it has no effect
    because this split deliberately contains no random assignment.
    """
    trial_ids = pd.Series(trial_ids).dropna().unique()
    if len(trial_ids) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique trials; found {len(trial_ids)}"
        )

    sorted_trial_ids = np.sort(trial_ids)
    test_blocks = np.array_split(sorted_trial_ids, n_splits)
    folds = []
    for fold_index, test_block in enumerate(test_blocks):
        if len(test_block) == 0:
            raise ValueError(f"Fold {fold_index} has no held-out trials")
        test_set = set(test_block.tolist())
        train_block = np.asarray(
            [trial_id for trial_id in sorted_trial_ids if trial_id not in test_set]
        )
        folds.append(
            {
                "fold": fold_index,
                "seed": int(seed),
                "train_trial_ids": train_block.tolist(),
                "test_trial_ids": test_block.tolist(),
            }
        )

    validate_outer_trial_folds(folds, sorted_trial_ids)
    return folds


def validate_outer_trial_folds(folds, trial_ids: Sequence):
    """Validate exhaustive, non-overlapping outer trial-fold assignments."""
    expected_trials = set(pd.Series(trial_ids).dropna().unique().tolist())
    if not expected_trials:
        raise ValueError("No trial identifiers were supplied")

    held_out_trials = []
    for fold in folds:
        train_trials = set(fold["train_trial_ids"])
        test_trials = set(fold["test_trial_ids"])
        if not train_trials:
            raise ValueError(f"Fold {fold['fold']} has no training trials")
        if not test_trials:
            raise ValueError(f"Fold {fold['fold']} has no held-out trials")
        if train_trials & test_trials:
            raise ValueError(f"Fold {fold['fold']} has overlapping train/test trials")
        if train_trials | test_trials != expected_trials:
            raise ValueError(f"Fold {fold['fold']} does not cover the expected trials")
        held_out_trials.extend(fold["test_trial_ids"])

    held_out_trials = pd.Series(held_out_trials)
    repeated = held_out_trials[held_out_trials.duplicated()].unique().tolist()
    missing = sorted(expected_trials.difference(held_out_trials.tolist()))
    if repeated or missing:
        raise ValueError(
            "Outer test-fold assignment must include each trial exactly once; "
            f"repeated={repeated}, missing={missing}"
        )


def summarize_outer_trial_folds(folds):
    """Return a compact table describing outer trial-fold membership."""
    rows = []
    for fold in folds:
        test_trial_ids = fold["test_trial_ids"]
        rows.append(
            {
                "fold": fold["fold"],
                "seed": fold["seed"],
                "n_train_trials": len(fold["train_trial_ids"]),
                "n_test_trials": len(test_trial_ids),
                "test_trial_start": min(test_trial_ids),
                "test_trial_end": max(test_trial_ids),
            }
        )
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def _progress_r2_sklearn(values, progress, trial_ids, n_knots, degree, ridge_alpha, n_splits):
    """Per-fold-averaged CV R^2 via SplineTransformer + Ridge (GroupKFold).

    Matches the method used in ``TCM_1_figures.ipynb`` (Fig1D speed-vs-progress
    R^2): a regularized cubic B-spline fit with scikit-learn's
    ``cross_val_score(..., scoring="r2")``, which averages R^2 across folds
    rather than pooling out-of-fold residuals.
    """
    values = np.asarray(values, dtype=float)
    progress = np.asarray(progress, dtype=float)
    trial_ids = pd.Series(trial_ids).to_numpy()

    valid = np.isfinite(values) & np.isfinite(progress) & pd.notna(trial_ids)
    values = values[valid]
    progress = progress[valid]
    trial_ids = trial_ids[valid]

    n_unique_trials = int(pd.Series(trial_ids).nunique())
    effective_splits = min(n_splits, n_unique_trials)
    model = make_pipeline(
        SplineTransformer(n_knots=n_knots, degree=degree),
        Ridge(alpha=ridge_alpha),
    )
    scores = cross_val_score(
        model,
        progress.reshape(-1, 1),
        values,
        groups=trial_ids,
        cv=GroupKFold(n_splits=effective_splits),
        scoring="r2",
    )
    return (
        float(scores.mean()),
        float(scores.std()),
        int(len(values)),
        n_unique_trials,
    )


def summarize_progress_predictability(
    data: pd.DataFrame,
    value_cols,
    labels=None,
    progress_col: str = "trial_progress_distance",
    trial_col: str = "trial_number",
    trial_type_col: str = "trial_type",
    split_by_trial_type: bool = True,
    conditions=("outbound", "inbound"),
    n_knots: int = 8,
    degree: int = 3,
    ridge_alpha: float = 1.0,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Summarize how well trial progress alone predicts each behavioral variable.

    For each variable, a regularized cubic B-spline (``SplineTransformer`` +
    ``Ridge``) is regressed on trial progress and scored with
    ``cross_val_score(..., scoring="r2")`` under trial-grouped ``GroupKFold``
    CV — the same method used for the speed-vs-progress R^2 in
    ``TCM_1_figures.ipynb``. A high R^2 flags a variable that is largely
    redundant with the task's trial-progress predictor.

    If ``split_by_trial_type`` is True (default), the fit is run separately
    per entry in ``conditions`` (e.g. outbound/inbound), producing one R^2
    column per condition. If False, all conditions are pooled into a single
    fit and the result has one ``combined_r2`` column.
    """
    value_cols = list(value_cols)
    labels = list(labels) if labels is not None else value_cols
    if len(labels) != len(value_cols):
        raise ValueError("labels must have one entry per value column")
    required = {progress_col, trial_col, trial_type_col, *value_cols}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {sorted(missing)}")

    if split_by_trial_type:
        subsets = {
            condition: data.loc[data[trial_type_col].eq(condition)]
            for condition in conditions
        }
    else:
        subsets = {"combined": data}

    rows = []
    for column, label in zip(value_cols, labels):
        row = {"variable": column, "label": label}
        for condition_name, subset in subsets.items():
            r2_mean, r2_std, n_samples, n_trials = _progress_r2_sklearn(
                subset[column],
                subset[progress_col],
                subset[trial_col],
                n_knots=n_knots,
                degree=degree,
                ridge_alpha=ridge_alpha,
                n_splits=n_splits,
            )
            row[f"{condition_name}_r2"] = r2_mean
            row[f"{condition_name}_r2_std"] = r2_std
            row[f"{condition_name}_n_samples"] = n_samples
            row[f"{condition_name}_n_trials"] = n_trials
        rows.append(row)
    return pd.DataFrame(rows)


def _circular_resultant_length(observed_phase, predicted_phase):
    """Return residual mean resultant length and mean circular loss."""
    residual = np.asarray(observed_phase) - np.asarray(predicted_phase)
    residual_vector = np.exp(1j * residual)
    return (
        float(np.abs(np.mean(residual_vector))),
        float(np.mean(1.0 - np.cos(residual))),
    )


def _aggregate_circular_progress(progress, phase, bin_width):
    """Compress circular observations into fine progress-bin sufficient sums."""
    progress = np.asarray(progress, dtype=float)
    phase = np.asarray(phase, dtype=float)
    if not 0 < bin_width <= 1:
        raise ValueError("bin_width must be in (0, 1]")
    if len(progress) != len(phase):
        raise ValueError("progress and phase must have equal length")
    if not len(progress):
        raise ValueError("At least one circular observation is required")
    if np.any((progress < -1e-9) | (progress > 1.0 + 1e-9)):
        raise ValueError("progress must lie within [0, 1]")

    progress = np.clip(progress, 0.0, 1.0)
    n_bins = int(np.ceil(1.0 / bin_width))
    bin_index = np.minimum((progress / bin_width).astype(int), n_bins - 1)
    counts = np.bincount(bin_index, minlength=n_bins).astype(float)
    retained = counts > 0
    summed_progress = np.bincount(
        bin_index, weights=progress, minlength=n_bins
    )
    summed_cos = np.bincount(
        bin_index, weights=np.cos(phase), minlength=n_bins
    )
    summed_sin = np.bincount(
        bin_index, weights=np.sin(phase), minlength=n_bins
    )
    return (
        summed_progress[retained] / counts[retained],
        summed_cos[retained],
        summed_sin[retained],
        counts[retained],
    )


def _make_circular_gam_basis(progress, n_knots, degree):
    """Return a fixed [0, 1] cubic B-spline basis and its transformer."""
    if n_knots < 2:
        raise ValueError("n_knots must be at least 2")
    if degree < 1:
        raise ValueError("degree must be positive")
    knot_values = np.linspace(0.0, 1.0, int(n_knots)).reshape(-1, 1)
    transformer = SplineTransformer(
        degree=int(degree),
        knots=knot_values,
        include_bias=True,
        extrapolation="linear",
    )
    transformer.fit(np.array([[0.0], [1.0]]))
    basis = transformer.transform(np.asarray(progress, dtype=float).reshape(-1, 1))
    return basis, transformer


def _von_mises_concentration_from_resultant(resultant_length):
    """Approximate the von Mises concentration from a sample resultant."""
    r = float(np.clip(resultant_length, 0.0, 1.0 - 1e-9))
    if r < 0.53:
        concentration = 2.0 * r + r**3 + 5.0 * r**5 / 6.0
    elif r < 0.85:
        concentration = -0.4 + 1.39 * r + 0.43 / (1.0 - r)
    else:
        concentration = 1.0 / (r**3 - 4.0 * r**2 + 3.0 * r)
    return float(np.clip(concentration, 1e-4, 1e3))


def _circular_gam_initializations(
    progress,
    summed_cos,
    summed_sin,
    counts,
    basis,
    max_initial_cycles,
    initial_cycle_step,
    n_initializations,
):
    """Generate multi-start latent trajectories from circular-linear fits."""
    if max_initial_cycles <= 0:
        raise ValueError("max_initial_cycles must be positive")
    if initial_cycle_step <= 0:
        raise ValueError("initial_cycle_step must be positive")
    if n_initializations < 1:
        raise ValueError("n_initializations must be positive")

    candidate_cycles = np.arange(
        0.0,
        max_initial_cycles + 0.5 * initial_cycle_step,
        initial_cycle_step,
    )
    complex_sums = summed_cos + 1j * summed_sin
    resultant = np.empty(len(candidate_cycles), dtype=float)
    total_count = float(np.sum(counts))
    chunk_size = 128
    for start in range(0, len(candidate_cycles), chunk_size):
        candidate_chunk = candidate_cycles[start : start + chunk_size]
        rotations = np.exp(
            -2j * np.pi * candidate_chunk[:, None] * progress[None, :]
        )
        resultant[start : start + len(candidate_chunk)] = np.abs(
            rotations @ complex_sums
        ) / total_count

    ranked = np.argsort(resultant)[::-1]
    selected = []
    minimum_separation = max(0.5, 2.0 * initial_cycle_step)
    for index in ranked:
        cycles = float(candidate_cycles[index])
        if all(abs(cycles - previous) >= minimum_separation for previous in selected):
            selected.append(cycles)
        if len(selected) == n_initializations:
            break

    starts = []
    sqrt_counts = np.sqrt(counts)
    weighted_basis = basis * sqrt_counts[:, None]
    for cycles in selected:
        residual_sum = np.sum(
            complex_sums * np.exp(-2j * np.pi * cycles * progress)
        )
        offset = float(np.angle(residual_sum))
        latent_target = 2.0 * np.pi * cycles * progress + offset
        coefficients = np.linalg.lstsq(
            weighted_basis,
            latent_target * sqrt_counts,
            rcond=None,
        )[0]
        cycle_index = int(np.argmin(np.abs(candidate_cycles - cycles)))
        concentration = _von_mises_concentration_from_resultant(
            resultant[cycle_index]
        )
        starts.append((cycles, coefficients, concentration))
    return starts


def fit_progress_phase_circular_gam(
    progress,
    phase,
    smoothing_penalty,
    n_knots=12,
    degree=3,
    progress_bin_width=0.0025,
    max_initial_cycles=30.0,
    initial_cycle_step=0.05,
    n_initializations=3,
    max_iter=500,
):
    """Fit a cubic P-spline circular GAM with a von Mises likelihood.

    The latent mean phase is a spline of normalized trial progress and is
    wrapped only when evaluated by the likelihood or returned as a prediction.
    A second-difference penalty on adjacent spline coefficients controls local
    changes in gait winding rate. Fine progress-bin circular sufficient sums
    make fitting fast while retaining cycle-scale structure.
    """
    progress = np.asarray(progress, dtype=float)
    phase = np.asarray(phase, dtype=float)
    valid = np.isfinite(progress) & np.isfinite(phase)
    progress = progress[valid]
    phase = phase[valid]
    if smoothing_penalty < 0:
        raise ValueError("smoothing_penalty must be nonnegative")

    (
        binned_progress,
        summed_cos,
        summed_sin,
        counts,
    ) = _aggregate_circular_progress(
        progress,
        phase,
        progress_bin_width,
    )
    basis, transformer = _make_circular_gam_basis(
        binned_progress,
        n_knots,
        degree,
    )
    second_difference = np.diff(np.eye(basis.shape[1]), n=2, axis=0)
    penalty_size = max(1, len(second_difference))
    total_count = float(np.sum(counts))
    starts = _circular_gam_initializations(
        binned_progress,
        summed_cos,
        summed_sin,
        counts,
        basis,
        max_initial_cycles=max_initial_cycles,
        initial_cycle_step=initial_cycle_step,
        n_initializations=n_initializations,
    )

    def objective_and_gradient(parameters):
        coefficients = parameters[:-1]
        log_concentration = parameters[-1]
        concentration = float(np.exp(log_concentration))
        latent_phase = basis @ coefficients
        cos_latent = np.cos(latent_phase)
        sin_latent = np.sin(latent_phase)
        aligned_cosine_sum = np.sum(
            summed_cos * cos_latent + summed_sin * sin_latent
        )
        log_i0 = float(np.log(i0e(concentration)) + concentration)
        coefficient_curvature = second_difference @ coefficients
        objective = (
            np.log(2.0 * np.pi)
            + log_i0
            - concentration * aligned_cosine_sum / total_count
            + 0.5
            * smoothing_penalty
            * np.mean(coefficient_curvature**2)
        )

        phase_gradient = (
            concentration
            * basis.T
            @ (summed_cos * sin_latent - summed_sin * cos_latent)
            / total_count
        )
        penalty_gradient = (
            smoothing_penalty
            * second_difference.T
            @ coefficient_curvature
            / penalty_size
        )
        mean_aligned_cosine = aligned_cosine_sum / total_count
        concentration_ratio = float(i1e(concentration) / i0e(concentration))
        log_concentration_gradient = concentration * (
            concentration_ratio - mean_aligned_cosine
        )
        gradient = np.concatenate(
            [
                phase_gradient + penalty_gradient,
                [log_concentration_gradient],
            ]
        )
        return float(objective), gradient

    candidates = []
    log_concentration_bounds = (np.log(1e-4), np.log(1e3))
    for cycles, initial_coefficients, initial_concentration in starts:
        initial_parameters = np.concatenate(
            [initial_coefficients, [np.log(initial_concentration)]]
        )
        optimization = minimize(
            objective_and_gradient,
            initial_parameters,
            method="L-BFGS-B",
            jac=True,
            bounds=[(None, None)] * len(initial_coefficients)
            + [log_concentration_bounds],
            options={"maxiter": int(max_iter), "ftol": 1e-10},
        )
        if np.isfinite(optimization.fun):
            candidates.append((float(optimization.fun), cycles, optimization))
    if not candidates:
        raise RuntimeError("All circular-GAM optimizations returned non-finite values")

    objective_value, initialization_cycles, optimization = min(
        candidates,
        key=lambda candidate: candidate[0],
    )
    return FittedCircularGAM(
        coefficients=np.asarray(optimization.x[:-1], dtype=float),
        concentration=float(np.exp(optimization.x[-1])),
        smoothing_penalty=float(smoothing_penalty),
        basis_transformer=transformer,
        n_knots=int(n_knots),
        degree=int(degree),
        progress_bin_width=float(progress_bin_width),
        initialization_cycles=float(initialization_cycles),
        objective_value=float(objective_value),
        optimization_success=bool(optimization.success),
        optimization_message=str(optimization.message),
    )


def _select_circular_gam_smoothing_penalty(
    progress,
    phase,
    trial_ids,
    smoothing_penalties,
    n_splits,
    **fit_kwargs,
):
    """Select circular-GAM roughness penalty by trial-grouped inner CV."""
    smoothing_penalties = np.asarray(smoothing_penalties, dtype=float)
    if (
        not len(smoothing_penalties)
        or np.any(~np.isfinite(smoothing_penalties))
        or np.any(smoothing_penalties < 0)
    ):
        raise ValueError("smoothing_penalties must be finite nonnegative values")
    unique_trials = pd.unique(trial_ids)
    effective_splits = min(int(n_splits), len(unique_trials))
    if effective_splits < 2:
        raise ValueError("At least two trials are required for smoothing selection")

    rows = []
    splitter = GroupKFold(n_splits=effective_splits)
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(progress, groups=trial_ids)
    ):
        for penalty in smoothing_penalties:
            fitted = fit_progress_phase_circular_gam(
                progress[train_index],
                phase[train_index],
                smoothing_penalty=float(penalty),
                **fit_kwargs,
            )
            predicted = fitted.predict(progress[validation_index])
            resultant, circular_loss = _circular_resultant_length(
                phase[validation_index],
                predicted,
            )
            rows.append(
                {
                    "inner_fold": fold,
                    "smoothing_penalty": float(penalty),
                    "validation_resultant_length": resultant,
                    "validation_circular_loss": circular_loss,
                    "optimization_success": fitted.optimization_success,
                }
            )
    score_table = pd.DataFrame(rows)
    penalty_summary = (
        score_table.groupby("smoothing_penalty", as_index=False)
        .agg(
            mean_validation_circular_loss=(
                "validation_circular_loss",
                "mean",
            ),
            mean_validation_resultant_length=(
                "validation_resultant_length",
                "mean",
            ),
        )
        .sort_values(
            ["mean_validation_circular_loss", "smoothing_penalty"],
            ascending=[True, False],
        )
    )
    selected_penalty = float(penalty_summary.iloc[0]["smoothing_penalty"])
    return selected_penalty, score_table, penalty_summary


def cross_validate_progress_phase_circular_gam(
    data,
    phase_sin_col="gait_phase_sin",
    phase_cos_col="gait_phase_cos",
    progress_col="trial_progress_distance",
    trial_col="trial_number",
    n_knots=12,
    degree=3,
    smoothing_penalties=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    outer_splits=5,
    inner_splits=3,
    progress_bin_width=0.0025,
    max_initial_cycles=30.0,
    initial_cycle_step=0.05,
    n_initializations=3,
    max_iter=500,
):
    """Evaluate progress-to-phase circular GAM on completely held-out trials.

    Smoothness is selected independently inside every outer training fold.
    The returned out-of-fold predictions can therefore be used to quantify
    phase predictability without evaluating any trial used to fit its model.
    """
    required = {phase_sin_col, phase_cos_col, progress_col, trial_col}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {sorted(missing)}")

    analysis_data = data.loc[:, list(required)].copy()
    analysis_data["_source_index"] = data.index
    valid = (
        analysis_data[[phase_sin_col, phase_cos_col, progress_col]].notna().all(axis=1)
        & analysis_data[trial_col].notna()
    )
    analysis_data = analysis_data.loc[valid].reset_index(drop=True)
    if not len(analysis_data):
        raise ValueError("No finite progress/phase samples are available")

    progress = analysis_data[progress_col].to_numpy(dtype=float)
    phase = np.arctan2(
        analysis_data[phase_sin_col].to_numpy(dtype=float),
        analysis_data[phase_cos_col].to_numpy(dtype=float),
    )
    trial_ids = analysis_data[trial_col].to_numpy()
    n_trials = int(pd.Series(trial_ids).nunique())
    effective_outer_splits = min(int(outer_splits), n_trials)
    if effective_outer_splits < 2:
        raise ValueError("At least two trials are required for outer CV")

    fit_kwargs = {
        "n_knots": n_knots,
        "degree": degree,
        "progress_bin_width": progress_bin_width,
        "max_initial_cycles": max_initial_cycles,
        "initial_cycle_step": initial_cycle_step,
        "n_initializations": n_initializations,
        "max_iter": max_iter,
    }
    fold_rows = []
    tuning_tables = []
    prediction_rows = []
    fitted_models = []
    splitter = GroupKFold(n_splits=effective_outer_splits)
    for outer_fold, (train_index, test_index) in enumerate(
        splitter.split(progress, groups=trial_ids)
    ):
        selected_penalty, score_table, penalty_summary = (
            _select_circular_gam_smoothing_penalty(
                progress[train_index],
                phase[train_index],
                trial_ids[train_index],
                smoothing_penalties=smoothing_penalties,
                n_splits=inner_splits,
                **fit_kwargs,
            )
        )
        score_table.insert(0, "outer_fold", outer_fold)
        score_table["selected"] = score_table["smoothing_penalty"].eq(
            selected_penalty
        )
        tuning_tables.append(score_table)

        fitted = fit_progress_phase_circular_gam(
            progress[train_index],
            phase[train_index],
            smoothing_penalty=selected_penalty,
            **fit_kwargs,
        )
        predicted_phase = fitted.predict(progress[test_index])
        resultant, circular_loss = _circular_resultant_length(
            phase[test_index],
            predicted_phase,
        )
        circular_residual = np.angle(
            np.exp(1j * (phase[test_index] - predicted_phase))
        )
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_smoothing_penalty": selected_penalty,
                "test_resultant_length": resultant,
                "test_circular_loss": circular_loss,
                "test_mean_absolute_circular_error_rad": float(
                    np.mean(np.abs(circular_residual))
                ),
                "concentration": fitted.concentration,
                "initialization_cycles": fitted.initialization_cycles,
                "n_train_samples": len(train_index),
                "n_test_samples": len(test_index),
                "n_train_trials": int(pd.Series(trial_ids[train_index]).nunique()),
                "n_test_trials": int(pd.Series(trial_ids[test_index]).nunique()),
                "optimization_success": fitted.optimization_success,
                "optimization_message": fitted.optimization_message,
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "source_index": analysis_data.loc[
                        test_index, "_source_index"
                    ].to_numpy(),
                    trial_col: trial_ids[test_index],
                    progress_col: progress[test_index],
                    "observed_phase_rad": phase[test_index],
                    "predicted_phase_rad": predicted_phase,
                    "circular_residual_rad": circular_residual,
                    "outer_fold": outer_fold,
                }
            )
        )
        fitted_models.append(fitted)

    oof_predictions = pd.concat(prediction_rows, ignore_index=True)
    pooled_resultant, pooled_loss = _circular_resultant_length(
        oof_predictions["observed_phase_rad"],
        oof_predictions["predicted_phase_rad"],
    )
    summary = pd.DataFrame(
        [
            {
                "oof_resultant_length": pooled_resultant,
                "oof_circular_loss": pooled_loss,
                "oof_mean_absolute_circular_error_rad": float(
                    np.mean(np.abs(oof_predictions["circular_residual_rad"]))
                ),
                "n_samples": len(oof_predictions),
                "n_trials": n_trials,
                "n_outer_folds": effective_outer_splits,
                "n_knots": int(n_knots),
                "n_basis_functions": len(fitted_models[0].coefficients),
                "degree": int(degree),
                "progress_bin_width": float(progress_bin_width),
            }
        ]
    )
    return {
        "summary": summary,
        "folds": pd.DataFrame(fold_rows),
        "tuning": pd.concat(tuning_tables, ignore_index=True),
        "oof_predictions": oof_predictions,
        "models": fitted_models,
    }


def summarize_progress_phase_circular_gam(
    data,
    phase_sin_col="gait_phase_sin",
    phase_cos_col="gait_phase_cos",
    progress_col="trial_progress_distance",
    trial_col="trial_number",
    trial_type_col="trial_type",
    split_by_trial_type=False,
    conditions=("outbound", "inbound"),
    **cv_kwargs,
):
    """Run the progress-to-phase circular GAM pooled or by trial type."""
    required = {
        phase_sin_col,
        phase_cos_col,
        progress_col,
        trial_col,
        trial_type_col,
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {sorted(missing)}")
    subsets = (
        {
            condition: data.loc[data[trial_type_col].eq(condition)]
            for condition in conditions
        }
        if split_by_trial_type
        else {"combined": data}
    )

    results = {}
    for condition, subset in subsets.items():
        result = cross_validate_progress_phase_circular_gam(
            subset,
            phase_sin_col=phase_sin_col,
            phase_cos_col=phase_cos_col,
            progress_col=progress_col,
            trial_col=trial_col,
            **cv_kwargs,
        )
        for table_name in ("summary", "folds", "tuning", "oof_predictions"):
            result[table_name].insert(0, "condition", condition)
        results[condition] = result

    return {
        "summary": pd.concat(
            [result["summary"] for result in results.values()],
            ignore_index=True,
        ),
        "folds": pd.concat(
            [result["folds"] for result in results.values()],
            ignore_index=True,
        ),
        "tuning": pd.concat(
            [result["tuning"] for result in results.values()],
            ignore_index=True,
        ),
        "oof_predictions": pd.concat(
            [result["oof_predictions"] for result in results.values()],
            ignore_index=True,
        ),
        "models": {
            condition: result["models"] for condition, result in results.items()
        },
    }


def circular_gam_trial_rotation_test(
    oof_predictions,
    trial_col="trial_number",
    condition_col="condition",
    observed_phase_col="observed_phase_rad",
    predicted_phase_col="predicted_phase_rad",
    n_permutations=1000,
    seed=42,
):
    """Trial-rotation null test for held-out circular-GAM predictions.

    Each trial's observed phase is rotated by one random offset while its
    within-trial gait dynamics remain intact. Predictions are fixed because
    every row is already out-of-fold for its trial. The resulting test asks
    whether held-out absolute phase alignment exceeds trial-rotated alignment.
    """
    required = {trial_col, observed_phase_col, predicted_phase_col}
    if condition_col in oof_predictions.columns:
        required.add(condition_col)
    missing = required.difference(oof_predictions.columns)
    if missing:
        raise ValueError(
            f"oof_predictions is missing required columns: {sorted(missing)}"
        )
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")

    condition_values = (
        pd.unique(oof_predictions[condition_col])
        if condition_col in oof_predictions.columns
        else ["combined"]
    )
    rng = np.random.default_rng(seed)
    summaries = []
    null_rows = []
    for condition in condition_values:
        subset = (
            oof_predictions.loc[oof_predictions[condition_col].eq(condition)]
            if condition_col in oof_predictions.columns
            else oof_predictions
        )
        valid = subset[
            [trial_col, observed_phase_col, predicted_phase_col]
        ].notna().all(axis=1)
        subset = subset.loc[valid]
        residual_vector = np.exp(
            1j
            * (
                subset[observed_phase_col].to_numpy(dtype=float)
                - subset[predicted_phase_col].to_numpy(dtype=float)
            )
        )
        trial_codes, trial_levels = pd.factorize(subset[trial_col], sort=True)
        trial_vector_sums = np.bincount(
            trial_codes,
            weights=residual_vector.real,
        ) + 1j * np.bincount(
            trial_codes,
            weights=residual_vector.imag,
        )
        observed_vector = np.sum(trial_vector_sums) / len(subset)
        observed_resultant = float(np.abs(observed_vector))
        observed_loss = float(1.0 - observed_vector.real)

        rotations = np.exp(
            1j
            * rng.uniform(
                0.0,
                2.0 * np.pi,
                size=(int(n_permutations), len(trial_levels)),
            )
        )
        null_vectors = rotations @ trial_vector_sums / len(subset)
        null_resultant = np.abs(null_vectors)
        null_loss = 1.0 - null_vectors.real
        resultant_p = float(
            (1 + np.sum(null_resultant >= observed_resultant))
            / (n_permutations + 1)
        )
        loss_p = float(
            (1 + np.sum(null_loss <= observed_loss))
            / (n_permutations + 1)
        )
        summaries.append(
            {
                "condition": condition,
                "oof_resultant_length": observed_resultant,
                "resultant_p_value": resultant_p,
                "oof_circular_loss": observed_loss,
                "circular_loss_p_value": loss_p,
                "n_permutations": int(n_permutations),
                "n_trials": len(trial_levels),
                "n_samples": len(subset),
            }
        )
        null_rows.append(
            pd.DataFrame(
                {
                    "condition": condition,
                    "permutation": np.arange(int(n_permutations)),
                    "null_resultant_length": null_resultant,
                    "null_circular_loss": null_loss,
                }
            )
        )
    return pd.DataFrame(summaries), pd.concat(null_rows, ignore_index=True)


def _task_formula_r2(formula, values, data, trial_ids, n_splits, ridge_alpha):
    """Per-fold-averaged CV R^2 for a Ridge fit of an arbitrary patsy formula.

    Uses the same trial-grouped ``GroupKFold`` + per-fold ``r2_score``
    averaging as :func:`_progress_r2_sklearn`, but builds the design matrix
    from a patsy formula each fold (via ``dmatrix``/``build_design_matrices``)
    so it can express categorical and branch-interaction task terms that
    ``SplineTransformer`` can't.
    """
    values = np.asarray(values, dtype=float)
    trial_ids = pd.Series(trial_ids).to_numpy()

    valid = np.isfinite(values) & pd.notna(trial_ids)
    values = values[valid]
    fit_data = data.loc[valid].reset_index(drop=True)
    trial_ids = trial_ids[valid]

    n_unique_trials = int(pd.Series(trial_ids).nunique())
    effective_splits = min(n_splits, n_unique_trials)
    group_kfold = GroupKFold(n_splits=effective_splits)
    scores = []
    for train_idx, test_idx in group_kfold.split(
        np.zeros((len(values), 1)), values, groups=trial_ids
    ):
        train_design = dmatrix(
            formula, fit_data.iloc[train_idx], return_type="dataframe"
        )
        test_design = build_design_matrices(
            [train_design.design_info],
            fit_data.iloc[test_idx],
            return_type="dataframe",
        )[0]
        model = Ridge(alpha=ridge_alpha)
        model.fit(train_design.to_numpy(), values[train_idx])
        predicted = model.predict(test_design.to_numpy())
        scores.append(r2_score(values[test_idx], predicted))
    return (
        float(np.mean(scores)),
        float(np.std(scores)),
        int(len(values)),
        n_unique_trials,
    )


def summarize_task_predictor_dependence(
    data: pd.DataFrame,
    value_cols,
    labels=None,
    progress_col: str = "trial_progress_distance",
    trial_col: str = "trial_number",
    trial_type_col: str = "trial_type",
    branch_id_col: str = "branch_id",
    branch_pos_col: str = "branch_pos_scaled",
    tp_df: int = 4,
    branch_pos_df: int = 4,
    n_splits: int = 5,
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """Break down how well each behavioral variable is predicted by task predictors.

    For each variable, fits a ridge-regularized patsy design (trial-grouped
    ``GroupKFold`` CV, per-fold-averaged R^2, matching
    :func:`summarize_progress_predictability`'s scoring convention)
    separately for four task-predictor sets:

    - ``progress``: ``cr(trial_progress_distance, df=tp_df)`` — the same
      trial-progress term used in the encoding models
      (``movement_spline_config``).
    - ``position``: ``C(branch_id) + bs(branch_pos_scaled, df=branch_pos_df,
      lower_bound=0, upper_bound=1):C(branch_id)`` — the exact branch-specific
      position term the encoding models use
      (``encoding_utils.load_and_prepare_data``'s position formula).
    - ``trial_type``: ``C(trial_type)``.
    - ``all_three``: all of the above combined.

    Outbound and inbound trials are pooled throughout, since ``trial_type``
    is itself one of the predictors being evaluated here. A high
    ``all_three`` R^2 that isn't matched by any single column flags a
    variable that is jointly, but not individually, redundant with the task
    predictors. Requires ``branch_id``/``branch_pos_scaled`` already added
    (e.g. via ``encoding_utils._add_branch_position_columns``).
    """
    value_cols = list(value_cols)
    labels = list(labels) if labels is not None else value_cols
    if len(labels) != len(value_cols):
        raise ValueError("labels must have one entry per value column")
    required = {
        progress_col,
        trial_col,
        trial_type_col,
        branch_id_col,
        branch_pos_col,
        *value_cols,
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {sorted(missing)}")

    task_predictor_columns = [
        progress_col, trial_type_col, branch_id_col, branch_pos_col
    ]
    data = data.loc[data[task_predictor_columns].notna().all(axis=1)]

    progress_formula = f"cr({progress_col}, df={tp_df})"
    position_formula = (
        f"C({branch_id_col}) + bs({branch_pos_col}, df={branch_pos_df}, "
        f"lower_bound=0.0, upper_bound=1.0):C({branch_id_col})"
    )
    trial_type_formula = f"C({trial_type_col})"
    formulas = {
        "progress": progress_formula,
        "position": position_formula,
        "trial_type": trial_type_formula,
        "all_three": (
            f"{trial_type_formula} + {progress_formula} + {position_formula}"
        ),
    }

    trial_ids = data[trial_col]
    rows = []
    for column, label in zip(value_cols, labels):
        row = {"variable": column, "label": label}
        for predictor_set, formula in formulas.items():
            r2_mean, r2_std, n_samples, n_trials = _task_formula_r2(
                formula,
                data[column],
                data,
                trial_ids,
                n_splits=n_splits,
                ridge_alpha=ridge_alpha,
            )
            row[f"{predictor_set}_r2"] = r2_mean
            row[f"{predictor_set}_r2_std"] = r2_std
        row["n_samples"] = n_samples
        row["n_trials"] = n_trials
        rows.append(row)
    return pd.DataFrame(rows)


def run_nested_outer_cv_poisson_ridge(
    data,
    spike_counts_unit,
    predictor_groups: Mapping,
    included_groups,
    spline_config: Mapping,
    outer_folds,
    inner_cv_config: Mapping,
    estimator_kwargs=None,
    progress_callback=None,
    ablated_predictors=(),
    shuffle_seed: int = 42,
    penalty_bands: Mapping | None = None,
    banded_optimizer_config: Mapping | None = None,
    fixed_penalty_plan: Mapping | None = None,
    penalty_source_model: str | None = None,
):
    """Run nested trial-wise CV with optimized or inherited ridge penalties."""
    data = data.copy()
    if "trial_number" not in data.columns:
        raise KeyError("data is missing trial_number")
    if data["trial_number"].isna().any():
        raise ValueError("data contains bins with missing trial_number")
    if len(data) != len(spike_counts_unit):
        raise ValueError("data and spike_counts_unit must have equal length")
    validate_outer_trial_folds(outer_folds, data["trial_number"])

    required_inner_keys = {
        "n_splits",
        "seed",
        "lower_log10",
        "upper_log10",
        "log10_step",
        "edge_expansion_log10",
        "max_grid_expansions",
    }
    missing_inner_keys = required_inner_keys.difference(inner_cv_config)
    unexpected_inner_keys = set(inner_cv_config).difference(required_inner_keys)
    if missing_inner_keys or unexpected_inner_keys:
        raise ValueError(
            "inner_cv_config must contain exactly the approved settings; "
            f"missing={sorted(missing_inner_keys)}, "
            f"unexpected={sorted(unexpected_inner_keys)}"
        )
    if penalty_bands is None and fixed_penalty_plan is not None:
        raise ValueError("fixed_penalty_plan requires penalty_bands")
    if penalty_bands is not None:
        required_optimizer_keys = {
            "max_iter",
            "max_evaluations",
            "gtol",
            "ftol",
        }
        optimizer_config = dict(banded_optimizer_config or {})
        missing_optimizer_keys = required_optimizer_keys.difference(
            optimizer_config
        )
        unexpected_optimizer_keys = set(optimizer_config).difference(
            required_optimizer_keys
        )
        if missing_optimizer_keys or unexpected_optimizer_keys:
            raise ValueError(
                "banded_optimizer_config must contain exactly the approved "
                f"settings; missing={sorted(missing_optimizer_keys)}, "
                f"unexpected={sorted(unexpected_optimizer_keys)}"
            )
    else:
        optimizer_config = {}

    tuning_kwargs = {
        "n_inner_splits": inner_cv_config["n_splits"],
        "seed": inner_cv_config["seed"],
        "lower_log10": inner_cv_config["lower_log10"],
        "upper_log10": inner_cv_config["upper_log10"],
        "log10_step": inner_cv_config["log10_step"],
        "edge_expansion_log10": inner_cv_config["edge_expansion_log10"],
        "max_grid_expansions": inner_cv_config["max_grid_expansions"],
        "estimator_kwargs": estimator_kwargs,
    }
    counts = np.asarray(spike_counts_unit, dtype=float)
    fold_score_rows = []
    prediction_tables = []
    clipping_summaries = []
    clipping_details = []
    permutation_audits = []
    penalty_search_tables = []
    tuning_results = {}
    outer_fits = {}

    for fold in outer_folds:
        outer_fold = fold["fold"]
        if progress_callback is not None:
            progress_callback(
                {"event": "outer_fold_started", "outer_fold": outer_fold}
            )
        train_mask = data["trial_number"].isin(fold["train_trial_ids"])
        test_mask = data["trial_number"].isin(fold["test_trial_ids"])
        outer_train = data.loc[train_mask]
        outer_test = data.loc[test_mask]
        tuning_result = None
        selected_alpha = None
        selected_penalties = None
        selected_null_model = False
        tuning_mode = "intercept_only"

        if included_groups:
            if fixed_penalty_plan is not None:
                if outer_fold not in fixed_penalty_plan:
                    raise KeyError(
                        f"Missing inherited penalty plan for outer fold {outer_fold}"
                    )
                fold_plan = fixed_penalty_plan[outer_fold]
                selected_penalties = dict(fold_plan["penalties"])
                if bool(fold_plan.get("selected_null_model", False)):
                    raise ValueError(
                        "Inherited banded plans must contain finite fits"
                    )
                selected_null_model = False
                tuning_mode = "inherited_banded"
            elif penalty_bands is not None:
                tuning_result = tune_poisson_banded_ridge(
                    outer_train,
                    counts[train_mask.to_numpy()],
                    predictor_groups=predictor_groups,
                    included_groups=included_groups,
                    spline_config=spline_config,
                    penalty_bands=penalty_bands,
                    outer_fold=outer_fold,
                    **tuning_kwargs,
                    **optimizer_config,
                )
                selected_penalties = tuning_result.selected_penalties
                selected_null_model = False
                tuning_mode = "optimized_banded"
            else:
                tuning_result = tune_poisson_ridge_alpha(
                    outer_train,
                    counts[train_mask.to_numpy()],
                    predictor_groups=predictor_groups,
                    included_groups=included_groups,
                    spline_config=spline_config,
                    outer_fold=outer_fold,
                    ablated_predictors=tuple(ablated_predictors),
                    shuffle_seed=shuffle_seed,
                    **tuning_kwargs,
                )
                selected_alpha = tuning_result.selected_alpha
                selected_null_model = tuning_result.selected_null_model
                tuning_mode = "optimized_scalar"
        else:
            selected_null_model = True

        outer_fit = fit_outer_fold_poisson_ridge(
            outer_train,
            counts[train_mask.to_numpy()],
            predictor_groups=predictor_groups,
            included_groups=included_groups,
            spline_config=spline_config,
            outer_fold=outer_fold,
            selected_alpha=selected_alpha,
            selected_penalties=selected_penalties,
            penalty_bands=penalty_bands,
            selected_null_model=selected_null_model,
            estimator_kwargs=estimator_kwargs,
            outer_test_data=outer_test,
            ablated_predictors=tuple(ablated_predictors),
            shuffle_seed=shuffle_seed,
        )
        evaluation = evaluate_outer_fold_poisson_ridge(
            outer_fit,
            outer_test,
            counts[test_mask.to_numpy()],
        )
        score_row = {
            "outer_fold": outer_fold,
            "penalty_tuning_mode": tuning_mode,
            "penalty_source_model": penalty_source_model,
            "selected_alpha": selected_alpha,
            "selected_null_model": selected_null_model,
            "selected_inner_cv_score": np.nan,
            "best_finite_alpha": np.nan,
            "best_finite_alpha_score": np.nan,
            "null_model_score": np.nan,
            "n_alpha_grid_expansions": np.nan,
            "optimizer_success": np.nan,
            "optimizer_status": np.nan,
            "optimizer_message": pd.NA,
            "optimizer_n_iterations": np.nan,
            "optimizer_n_evaluations": np.nan,
            "n_train_trials": len(fold["train_trial_ids"]),
            "n_test_trials": len(fold["test_trial_ids"]),
            "n_test_bins": evaluation.n_test_bins,
            "log_likelihood": evaluation.log_likelihood,
            "mean_log_likelihood_per_bin": (
                evaluation.mean_log_likelihood_per_bin
            ),
            "null_mean_count": outer_fit.null_mean_count,
            "model_deviance": evaluation.model_deviance,
            "null_deviance": evaluation.null_deviance,
            "deviance_explained": evaluation.deviance_explained,
        }
        if selected_penalties is not None:
            score_row.update(
                {
                    f"penalty_{name}": float(value)
                    for name, value in selected_penalties.items()
                }
            )
        if tuning_result is not None:
            score_row.update(
                {
                    "selected_inner_cv_score": (
                        tuning_result.selected_candidate_score
                    ),
                    "best_finite_alpha": tuning_result.best_finite_alpha,
                    "best_finite_alpha_score": (
                        tuning_result.best_finite_alpha_score
                    ),
                    "null_model_score": tuning_result.null_model_score,
                }
            )
            if isinstance(tuning_result, RidgeAlphaSearchResult):
                score_row["n_alpha_grid_expansions"] = (
                    tuning_result.n_grid_expansions
                )
            else:
                score_row.update(
                    {
                        "optimizer_success": tuning_result.optimizer_success,
                        "optimizer_status": tuning_result.optimizer_status,
                        "optimizer_message": tuning_result.optimizer_message,
                        "optimizer_n_iterations": (
                            tuning_result.optimizer_n_iterations
                        ),
                        "optimizer_n_evaluations": (
                            tuning_result.optimizer_n_evaluations
                        ),
                    }
                )
                penalty_search_tables.append(tuning_result.evaluation_table)
            if not tuning_result.permutation_audit.empty:
                permutation_audits.append(tuning_result.permutation_audit)
            tuning_results[outer_fold] = tuning_result

        fold_score_rows.append(score_row)
        prediction_table = evaluation.prediction_table.copy()
        prediction_table.insert(0, "outer_fold", outer_fold)
        prediction_tables.append(prediction_table)
        clipping_summaries.append(evaluation.clipping_summary)
        clipping_details.append(evaluation.clipping_detail)
        if not outer_fit.permutation_audit.empty:
            permutation_audits.append(outer_fit.permutation_audit)
        outer_fits[outer_fold] = outer_fit
        if progress_callback is not None:
            progress_callback(
                {"event": "outer_fold_completed", "outer_fold": outer_fold}
            )

    fold_scores = pd.DataFrame(fold_score_rows).sort_values("outer_fold")
    total_log_likelihood = float(fold_scores["log_likelihood"].sum())
    n_held_out_bins = int(fold_scores["n_test_bins"].sum())
    total_model_deviance = float(fold_scores["model_deviance"].sum())
    total_null_deviance = float(fold_scores["null_deviance"].sum())
    return CrossValidatedPoissonRidgeResult(
        mean_fold_log_likelihood_per_bin=float(
            fold_scores["mean_log_likelihood_per_bin"].mean()
        ),
        pooled_log_likelihood_per_bin=total_log_likelihood / n_held_out_bins,
        total_log_likelihood=total_log_likelihood,
        n_held_out_bins=n_held_out_bins,
        mean_fold_deviance_explained=float(
            fold_scores["deviance_explained"].mean()
        ),
        pooled_deviance_explained=(
            np.nan
            if total_null_deviance == 0
            else 1.0 - (total_model_deviance / total_null_deviance)
        ),
        fold_scores=fold_scores.reset_index(drop=True),
        out_of_fold_predictions=pd.concat(
            prediction_tables, axis=0
        ).sort_index(),
        tuning_results=tuning_results,
        outer_fits=outer_fits,
        clipping_summary=_concat_audit_frames(clipping_summaries),
        clipping_detail=_concat_audit_frames(clipping_details),
        permutation_audit=(
            pd.concat(permutation_audits, ignore_index=True)
            if permutation_audits else pd.DataFrame()
        ),
        penalty_search=(
            pd.concat(penalty_search_tables, ignore_index=True)
            if penalty_search_tables else pd.DataFrame()
        ),
    )
