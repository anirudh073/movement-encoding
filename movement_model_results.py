"""Load and visualize persisted nested movement-encoding model results."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata, wilcoxon
from tqdm.auto import tqdm


@dataclass
class MovementModelResults:
    """Saved outputs for a compatible set of movement-encoding models."""

    manifests: dict
    unit_scores: pd.DataFrame
    fold_scores: pd.DataFrame
    out_of_fold_predictions: pd.DataFrame
    clipping_summary: pd.DataFrame
    clipping_detail: pd.DataFrame
    zero_variance_audit: pd.DataFrame
    penalty_search: pd.DataFrame
    permutation_audit: pd.DataFrame


_RESULT_FILES = {
    "unit_scores": "unit_scores.csv",
    "fold_scores": "fold_scores.csv",
    "out_of_fold_predictions": "oof_predictions.csv.gz",
    "clipping_summary": "clipping_summary.csv",
    "clipping_detail": "clipping_detail.csv",
    "zero_variance_audit": "zero_variance_audit.csv",
    "penalty_search": "penalty_search.csv",
    "permutation_audit": "permutation_audit.csv",
}
_OOF_SHARD_DIRECTORY = "oof_prediction_shards"
_OOF_COLUMNS_FILENAME = "oof_prediction_columns.json"


def _read_csv_or_empty(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _compatibility_spec(manifest):
    """Fields that must agree before model scores can be compared."""
    run_spec = manifest["run_spec"]
    if run_spec.get("spike_history_enabled") is not False:
        raise ValueError("Refusing to load deprecated spike-history model results")
    return {
        key: run_spec[key]
        for key in (
            "input_bundle_manifest_sha256",
            "predictor_groups",
            "spline_config",
            "spike_history_enabled",
            "inner_cv_config",
            "penalty_bands",
            "banded_optimizer_config",
            "outer_trial_folds",
            "unit_ids",
        )
    }


def resolve_movement_model_result_dir(results_root, model_name):
    """Resolve an explicit run key or prefer shuffle for a base model name."""
    results_root = Path(results_root)
    explicit_mode = None
    base_name = model_name
    for mode in ("shuffle", "drop"):
        suffix = f"__{mode}"
        if model_name.endswith(suffix):
            explicit_mode = mode
            base_name = model_name[: -len(suffix)]
            break
    if explicit_mode is not None:
        candidates = ((results_root / model_name, explicit_mode),)
    else:
        candidates = (
            (results_root / f"{model_name}__shuffle", "shuffle"),
            (results_root / f"{model_name}__drop", "drop"),
            (results_root / model_name, None),
        )
    for model_dir, mode in candidates:
        manifest_path = model_dir / "run_manifest.json"
        if manifest_path.exists():
            if mode is None and base_name.startswith("task_movement_minus_"):
                mode = "drop"
            return model_dir, base_name, mode
    searched = ", ".join(str(path) for path, _ in candidates)
    raise FileNotFoundError(
        f"No result manifest found for {model_name!r}; searched: {searched}"
    )


def load_movement_model_results(
    results_root,
    model_names,
    load_oof_predictions: bool = True,
):
    """Load saved results and require identical data/CV provenance per model.

    Set ``load_oof_predictions=False`` for lightweight score/audit inspection;
    individual OOF traces can then be loaded from their per-unit shards.
    """
    results_root = Path(results_root)
    manifests = {}
    tables_by_name = {name: [] for name in _RESULT_FILES}
    reference_spec = None
    for requested_model_name in model_names:
        model_dir, model_name, resolved_ablation_mode = (
            resolve_movement_model_result_dir(results_root, requested_model_name)
        )
        manifest_path = model_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if "run_spec" not in manifest:
            raise ValueError(f"Malformed result manifest for {model_name!r}")
        compatibility_spec = _compatibility_spec(manifest)
        compatibility_json = json.dumps(
            compatibility_spec, sort_keys=True, separators=(",", ":")
        )
        if reference_spec is None:
            reference_spec = compatibility_json
        elif compatibility_json != reference_spec:
            raise ValueError(
                f"{model_name!r} was not fit using the same input data, folds, "
                "or preprocessing configuration as the other selected models"
            )
        manifests[requested_model_name] = manifest
        for table_name, filename in _RESULT_FILES.items():
            if table_name == "out_of_fold_predictions" and not load_oof_predictions:
                continue
            table = _read_csv_or_empty(model_dir / filename)
            if not table.empty and "model_name" not in table:
                table.insert(0, "model_name", model_name)
            if (
                not table.empty
                and resolved_ablation_mode is not None
                and "ablation_mode" not in table
            ):
                table.insert(1, "ablation_mode", resolved_ablation_mode)
            tables_by_name[table_name].append(table)

    combined = {
        name: pd.concat(frames, ignore_index=True, sort=False)
        if any(not frame.empty for frame in frames)
        else pd.DataFrame()
        for name, frames in tables_by_name.items()
    }
    return MovementModelResults(manifests=manifests, **combined)


def _oof_shard_columns(model_dir: Path):
    columns_path = model_dir / _OOF_COLUMNS_FILENAME
    if not columns_path.exists():
        raise FileNotFoundError(
            f"Missing OOF shard-column manifest: {columns_path}"
        )
    return json.loads(columns_path.read_text())


def _read_oof_prediction_shard(model_dir: Path, unit_index, usecols=None):
    columns = _oof_shard_columns(model_dir)
    shard_path = model_dir / _OOF_SHARD_DIRECTORY / f"unit_{int(unit_index):04d}.csv.gz"
    if not shard_path.exists():
        raise FileNotFoundError(
            f"No OOF prediction shard for unit {unit_index}: {shard_path}"
        )
    return pd.read_csv(shard_path, header=None, names=columns, usecols=usecols)


def available_oof_trials(results_root, model_name, unit_index):
    """Return held-out trial IDs for one completed unit without loading all OOF data."""
    model_dir, _, _ = resolve_movement_model_result_dir(results_root, model_name)
    trial_numbers = _read_oof_prediction_shard(
        model_dir, unit_index, usecols=["trial_number"]
    )["trial_number"]
    return sorted(trial_numbers.dropna().unique().tolist())


def load_oof_prediction_trace(results_root, model_name, unit_index, trial_number):
    """Load one unit's OOF shard and retain one requested held-out trial."""
    model_dir, _, _ = resolve_movement_model_result_dir(results_root, model_name)
    trace = _read_oof_prediction_shard(model_dir, unit_index)
    trace = trace.loc[trace["trial_number"].eq(trial_number)].copy()
    if trace.empty:
        raise ValueError(
            "No held-out predictions for the selected model, unit, and trial"
        )
    return trace


def load_oof_unit_predictions(results_root, model_name, unit_index):
    """Load all held-out predictions for one completed unit from its shard."""
    model_dir, _, _ = resolve_movement_model_result_dir(results_root, model_name)
    return _read_oof_prediction_shard(
        model_dir,
        unit_index,
    )


def _poisson_deviance_from_counts(observed_counts, predicted_counts):
    """Return Poisson deviance from the saturated model for positive means."""
    observed_counts = np.asarray(observed_counts, dtype=float)
    predicted_counts = np.asarray(predicted_counts, dtype=float)
    if observed_counts.shape != predicted_counts.shape:
        raise ValueError("Observed and predicted count vectors must have the same shape")
    if not np.isfinite(observed_counts).all() or (observed_counts < 0).any():
        raise ValueError("Observed counts must be finite and non-negative")
    if not np.isfinite(predicted_counts).all() or (predicted_counts <= 0).any():
        raise ValueError("Predicted counts must be finite and positive")
    terms = predicted_counts - observed_counts
    nonzero_observed = observed_counts > 0
    terms[nonzero_observed] += observed_counts[nonzero_observed] * np.log(
        observed_counts[nonzero_observed] / predicted_counts[nonzero_observed]
    )
    return float(2.0 * terms.sum())


def _calculate_saved_oof_d2_for_unit(
    results_root,
    model_name,
    unit_id,
    unit_index,
    ablation_mode=None,
):
    """Calculate one saved model/unit's fold and pooled D² values."""
    shard_columns = [
        "outer_fold",
        "observed_spike_count",
        "predicted_spike_count",
    ]
    selector = (
        f"{model_name}__{ablation_mode}"
        if ablation_mode in {"drop", "shuffle"} else model_name
    )
    model_dir, _, _ = resolve_movement_model_result_dir(results_root, selector)
    predictions = _read_oof_prediction_shard(
        model_dir,
        unit_index,
        usecols=shard_columns,
    )
    outer_folds = sorted(predictions["outer_fold"].dropna().unique().tolist())
    if len(outer_folds) < 2:
        raise ValueError(
            f"{model_name} unit {unit_index} has fewer than two outer folds "
            "in its OOF shard"
        )

    unit_fold_rows = []
    for outer_fold in outer_folds:
        test_mask = predictions["outer_fold"].eq(outer_fold)
        test_counts = predictions.loc[test_mask, "observed_spike_count"].to_numpy(
            dtype=float
        )
        predicted_counts = predictions.loc[
            test_mask, "predicted_spike_count"
        ].to_numpy(dtype=float)
        training_counts = predictions.loc[
            ~test_mask, "observed_spike_count"
        ].to_numpy(dtype=float)
        null_mean_count = float(training_counts.mean())
        if not np.isfinite(null_mean_count) or null_mean_count <= 0:
            raise ValueError(
                f"{model_name} unit {unit_index}, outer fold {outer_fold}: "
                "null mean count must be positive"
            )
        model_deviance = _poisson_deviance_from_counts(test_counts, predicted_counts)
        null_deviance = _poisson_deviance_from_counts(
            test_counts,
            np.full(len(test_counts), null_mean_count, dtype=float),
        )
        unit_fold_rows.append(
            {
                "model_name": model_name,
                "unit_id": unit_id,
                "unit_index": unit_index,
                "outer_fold": outer_fold,
                "n_test_bins": len(test_counts),
                "null_mean_count": null_mean_count,
                "model_deviance": model_deviance,
                "null_deviance": null_deviance,
                "deviance_explained": (
                    np.nan
                    if null_deviance == 0
                    else 1.0 - model_deviance / null_deviance
                ),
            }
        )
    unit_fold_scores = pd.DataFrame(unit_fold_rows)
    total_model_deviance = unit_fold_scores["model_deviance"].sum()
    total_null_deviance = unit_fold_scores["null_deviance"].sum()
    unit_row = {
        "model_name": model_name,
        "unit_id": unit_id,
        "unit_index": unit_index,
        "n_held_out_bins": int(unit_fold_scores["n_test_bins"].sum()),
        "mean_fold_deviance_explained": unit_fold_scores[
            "deviance_explained"
        ].mean(),
        "pooled_deviance_explained": (
            np.nan
            if total_null_deviance == 0
            else 1.0 - total_model_deviance / total_null_deviance
        ),
    }
    if ablation_mode in {"drop", "shuffle"}:
        unit_row["ablation_mode"] = ablation_mode
        for row in unit_fold_rows:
            row["ablation_mode"] = ablation_mode
    return unit_row, unit_fold_rows


def calculate_saved_oof_deviance_explained(
    results_root,
    model_names=None,
    unit_indices=None,
    n_jobs=1,
    show_progress=True,
):
    """Calculate outer-CV Poisson D² from saved OOF prediction shards.

    Each outer fold's null prediction is the mean observed count across the
    other outer folds. Because every retained bin is held out exactly once,
    those rows are exactly the fold's original outer-training bins. This is a
    read-only calculation: it neither refits nor modifies saved artifacts.
    Set ``n_jobs`` above one to process independent unit shards in parallel.
    """
    results_root = Path(results_root)
    if model_names is None:
        discovered = []
        for path in sorted(results_root.iterdir()):
            if not (path / "run_manifest.json").exists():
                continue
            base_name = path.name
            for mode in ("shuffle", "drop"):
                suffix = f"__{mode}"
                if base_name.endswith(suffix):
                    base_name = base_name[: -len(suffix)]
                    break
            if base_name not in discovered:
                discovered.append(base_name)
        model_names = tuple(discovered)
    model_names = tuple(model_names)
    results = load_movement_model_results(
        results_root,
        model_names,
        load_oof_predictions=False,
    )
    completed = completed_unit_scores(results)
    if unit_indices is not None:
        completed = completed.loc[
            completed["unit_index"].isin(set(unit_indices))
        ].copy()

    tasks = []
    has_ablation_mode = "ablation_mode" in completed
    for score_row in completed.itertuples(index=False):
        mode = getattr(score_row, "ablation_mode", None) if has_ablation_mode else None
        if pd.isna(mode):
            mode = None
        tasks.append(
            (
                score_row.model_name,
                score_row.unit_id,
                score_row.unit_index,
                mode,
            )
        )
    if n_jobs == 1:
        task_results = (
            _calculate_saved_oof_d2_for_unit(results_root, *task)
            for task in tasks
        )
    else:
        task_results = Parallel(
            n_jobs=n_jobs,
            return_as="generator_unordered",
        )(
            delayed(_calculate_saved_oof_d2_for_unit)(results_root, *task)
            for task in tasks
        )
    if show_progress:
        task_results = tqdm(task_results, total=len(tasks), desc="Saved OOF D²")

    unit_rows = []
    fold_rows = []
    for unit_row, unit_fold_rows in task_results:
        unit_rows.append(unit_row)
        fold_rows.extend(unit_fold_rows)
    return (
        pd.DataFrame(unit_rows).sort_values(["model_name", "unit_index"]).reset_index(drop=True),
        pd.DataFrame(fold_rows).sort_values(
            ["model_name", "unit_index", "outer_fold"]
        ).reset_index(drop=True),
    )


def calculate_task_movement_predictive_gains(

    unit_scores,
    score_column="pooled_deviance_explained",
):
    """Calculate four matched-unit predictive gains for task/movement models.

    The required models are intercept_only (I), task_only (I+T),
    movement_only (I+M), and task_movement (I+T+M).
    """
    required_columns = {"model_name", "unit_id", "unit_index", score_column}
    missing_columns = required_columns.difference(unit_scores.columns)
    if missing_columns:
        raise ValueError(
            "Unit score table is missing columns: "
            f"{sorted(missing_columns)}"
        )
    model_columns = {
        "intercept_only": "score_intercept_only",
        "task_only": "score_task_only",
        "movement_only": "score_movement_only",
        "task_movement": "score_task_movement",
    }
    scores = unit_scores.loc[
        unit_scores["model_name"].isin(model_columns),
        ["model_name", "unit_id", "unit_index", score_column],
    ].copy()
    if "status" in unit_scores:
        completed_pairs = unit_scores.loc[
            unit_scores["status"].eq("completed"),
            ["model_name", "unit_index"],
        ]
        scores = scores.merge(
            completed_pairs,
            on=["model_name", "unit_index"],
            how="inner",
        )
    duplicate_rows = scores.duplicated(["model_name", "unit_index"])
    if duplicate_rows.any():
        duplicate_units = scores.loc[duplicate_rows, "unit_index"].tolist()
        raise ValueError(
            "Each model/unit pair must have one score; duplicate unit indices "
            f"include {duplicate_units[:10]}"
        )

    wide_scores = scores.pivot(
        index=["unit_id", "unit_index"],
        columns="model_name",
        values=score_column,
    )
    missing_models = set(model_columns).difference(wide_scores.columns)
    if missing_models:
        raise ValueError(
            "No scores were found for required models: "
            f"{sorted(missing_models)}"
        )
    gains = (
        wide_scores.loc[:, list(model_columns)]
        .dropna()
        .rename(columns=model_columns)
        .copy()
    )
    gains["task_alone"] = (
        gains["score_task_only"] - gains["score_intercept_only"]
    )
    gains["movement_alone"] = (
        gains["score_movement_only"] - gains["score_intercept_only"]
    )
    gains["task_given_movement"] = (
        gains["score_task_movement"] - gains["score_movement_only"]
    )
    gains["movement_given_task"] = (
        gains["score_task_movement"] - gains["score_task_only"]
    )
    return gains.reset_index()


def calculate_saved_task_movement_predictive_gains(
    results_root,
    unit_indices=None,
    n_jobs=1,
    show_progress=True,
):
    """Calculate saved OOF D² and four task/movement gains for matched units."""
    required_models = (
        "intercept_only",
        "task_only",
        "movement_only",
        "task_movement",
    )
    unit_d2, fold_d2 = calculate_saved_oof_deviance_explained(
        results_root,
        model_names=required_models,
        unit_indices=unit_indices,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )
    gains = calculate_task_movement_predictive_gains(unit_d2)
    return gains, unit_d2, fold_d2


def calculate_task_movement_redundancy(
    task_movement_gains,
    additive_tolerance=0.0,
):
    """Calculate the task/movement redundancy contrast for each unit.

    With ``S`` denoting pooled out-of-fold D², the returned table contains
    ``task_alone = S_T - S_I``, ``movement_alone = S_M - S_I``,
    the corresponding unique gains, both alone-minus-unique contrasts,
    ``combined_alone_gain = task_alone + movement_alone``,
    ``joint_gain = S_TM - S_I``, and
    ``redundancy_index = combined_alone_gain - joint_gain``.

    The two alone-minus-unique contrasts are necessarily identical:
    ``A_T - U_T = A_M - U_M = S_T + S_M - S_I - S_TM``.

    Positive contrasts are labeled redundancy and negative contrasts are
    labeled complementarity/suppression. Values within ``additive_tolerance``
    of zero are labeled additive.
    """
    if additive_tolerance < 0:
        raise ValueError("additive_tolerance must be non-negative")
    score_columns = {
        "score_intercept_only",
        "score_task_only",
        "score_movement_only",
        "score_task_movement",
    }
    required_columns = {"unit_id", "unit_index", *score_columns}
    missing_columns = required_columns.difference(task_movement_gains.columns)
    if missing_columns:
        raise ValueError(
            "Task/movement gain table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    redundancy = task_movement_gains.loc[
        :, ["unit_id", "unit_index", *sorted(score_columns)]
    ].copy()
    finite_scores = np.isfinite(
        redundancy.loc[:, sorted(score_columns)].to_numpy(dtype=float)
    ).all(axis=1)
    redundancy = redundancy.loc[finite_scores].copy()
    redundancy["task_alone"] = (
        redundancy["score_task_only"]
        - redundancy["score_intercept_only"]
    )
    redundancy["movement_alone"] = (
        redundancy["score_movement_only"]
        - redundancy["score_intercept_only"]
    )
    redundancy["task_unique"] = (
        redundancy["score_task_movement"]
        - redundancy["score_movement_only"]
    )
    redundancy["movement_unique"] = (
        redundancy["score_task_movement"]
        - redundancy["score_task_only"]
    )
    redundancy["task_alone_minus_unique"] = (
        redundancy["task_alone"] - redundancy["task_unique"]
    )
    redundancy["movement_alone_minus_unique"] = (
        redundancy["movement_alone"] - redundancy["movement_unique"]
    )
    redundancy["combined_alone_gain"] = (
        redundancy["task_alone"] + redundancy["movement_alone"]
    )
    redundancy["joint_gain"] = (
        redundancy["score_task_movement"]
        - redundancy["score_intercept_only"]
    )
    redundancy["redundancy_index"] = (
        redundancy["combined_alone_gain"] - redundancy["joint_gain"]
    )
    if not np.allclose(
        redundancy["task_alone_minus_unique"],
        redundancy["movement_alone_minus_unique"],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    ):
        raise RuntimeError(
            "Task and movement alone-minus-unique contrasts should be "
            "algebraically identical"
        )
    contrast = redundancy["redundancy_index"]
    redundancy["interaction_type"] = np.select(
        [
            contrast.gt(additive_tolerance),
            contrast.lt(-additive_tolerance),
        ],
        ["Redundancy", "Complementarity / suppression"],
        default="Additive",
    )
    redundancy.columns.name = None
    return redundancy.reset_index(drop=True)


_INDIVIDUAL_TERM_MODELS = {
    "trial_type": (
        "trial_type_only",
        "task_movement_minus_trial_type",
    ),
    "position": (
        "position_only",
        "task_movement_minus_position",
    ),
    "progress": (
        "progress_only",
        "task_movement_minus_progress",
    ),
    "translation": (
        "translation_only",
        "task_movement_minus_translation",
    ),
    "turning": (
        "turning_only",
        "task_movement_minus_turning",
    ),
    "limb": (
        "limb_only",
        "task_movement_minus_limb",
    ),
}


def calculate_individual_term_predictive_gains(
    unit_scores,
    score_column="pooled_deviance_explained",
):
    """Calculate alone and conditional D² gains for task/movement terms.

    For each term, the alone gain is ``(I + term) - I`` and the conditional
    gain is ``full - (full - term)``, where ``full`` is the task/movement
    task-plus-movement model. Only units shared by every required model are
    retained so all displayed term distributions use the same population.
    """
    required_columns = {"model_name", "unit_id", "unit_index", score_column}
    missing_columns = required_columns.difference(unit_scores.columns)
    if missing_columns:
        raise ValueError(
            "Unit score table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    required_models = {
        "intercept_only",
        "task_movement",
        *(
            model_name
            for model_pair in _INDIVIDUAL_TERM_MODELS.values()
            for model_name in model_pair
        ),
    }
    scores = unit_scores.loc[
        unit_scores["model_name"].isin(required_models),
        ["model_name", "unit_id", "unit_index", score_column],
    ].copy()
    if "status" in unit_scores:
        completed_pairs = unit_scores.loc[
            unit_scores["status"].eq("completed"),
            ["model_name", "unit_index"],
        ]
        scores = scores.merge(
            completed_pairs,
            on=["model_name", "unit_index"],
            how="inner",
        )
    duplicate_rows = scores.duplicated(["model_name", "unit_index"])
    if duplicate_rows.any():
        duplicate_units = scores.loc[duplicate_rows, "unit_index"].tolist()
        raise ValueError(
            "Each model/unit pair must have one score; duplicate unit indices "
            f"include {duplicate_units[:10]}"
        )

    wide_scores = scores.pivot(
        index=["unit_id", "unit_index"],
        columns="model_name",
        values=score_column,
    )
    missing_models = required_models.difference(wide_scores.columns)
    if missing_models:
        raise ValueError(
            "No scores were found for required models: "
            f"{sorted(missing_models)}"
        )
    wide_scores = wide_scores.loc[:, sorted(required_models)].dropna().copy()
    gains = wide_scores.loc[:, []].copy()
    for term_name, (alone_model, reduced_model) in (
        _INDIVIDUAL_TERM_MODELS.items()
    ):
        gains[f"{term_name}_alone"] = (
            wide_scores[alone_model] - wide_scores["intercept_only"]
        )
        gains[f"{term_name}_given_others"] = (
            wide_scores["task_movement"] - wide_scores[reduced_model]
        )
    return gains.reset_index()


def calculate_saved_individual_term_predictive_gains(
    results_root,
    unit_indices=None,
    n_jobs=1,
    show_progress=True,
):
    """Calculate saved OOF D² and term-level task/movement gains."""
    required_models = (
        "intercept_only",
        "task_movement",
        *(
            model_name
            for model_pair in _INDIVIDUAL_TERM_MODELS.values()
            for model_name in model_pair
        ),
    )
    unit_d2, fold_d2 = calculate_saved_oof_deviance_explained(
        results_root,
        model_names=required_models,
        unit_indices=unit_indices,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )
    gains = calculate_individual_term_predictive_gains(unit_d2)
    return gains, unit_d2, fold_d2


def _deviance_plot_values(

    values,
    *,
    outlier_iqr_multiplier=3.0,
    set_negative_to_zero=False,
):
    """Return finite D² values as percentages for robust visualization.

    Extreme values beyond ``outlier_iqr_multiplier`` times the IQR are omitted
    from the plot only. The source table is never modified.
    """
    values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return values, 0
    if outlier_iqr_multiplier is not None:
        if outlier_iqr_multiplier <= 0:
            raise ValueError("outlier_iqr_multiplier must be positive or None")
        lower_quartile, upper_quartile = np.quantile(values, (0.25, 0.75))
        interquartile_range = upper_quartile - lower_quartile
        if np.isfinite(interquartile_range) and interquartile_range > 0:
            lower_limit = lower_quartile - outlier_iqr_multiplier * interquartile_range
            upper_limit = upper_quartile + outlier_iqr_multiplier * interquartile_range
            retained = (values >= lower_limit) & (values <= upper_limit)
            n_omitted = int((~retained).sum())
            values = values[retained]
        else:
            n_omitted = 0
    else:
        n_omitted = 0
    if set_negative_to_zero:
        values = np.maximum(values, 0)
    return 100.0 * values, n_omitted


def plot_task_movement_predictive_gains(

    gains,
    set_negative_to_zero=False,
    outlier_iqr_multiplier=3.0,
    annotate_pairwise=True,
    return_pairwise_tests=False,
):
    """Plot offset total and unique task/movement D² gains by variable group.

    When ``set_negative_to_zero`` is true, negative values are clipped to zero
    for this visualization only; the input gain table is not modified.
    Pairwise annotations compare Task with Movement for the Total and Unique
    gains using matched-unit Wilcoxon signed-rank tests.
    """
    column_pairs = {
        "Task": ("task_alone", "task_given_movement"),
        "Movement": ("movement_alone", "movement_given_task"),
    }
    required_columns = {
        column_name
        for pair in column_pairs.values()
        for column_name in pair
    }
    missing_columns = required_columns.difference(gains.columns)
    if missing_columns:
        raise ValueError(
            "Task/movement gain table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    alone_values = []
    conditional_values = []
    n_extreme_omitted = 0
    for alone_column, conditional_column in column_pairs.values():
        alone, n_alone_omitted = _deviance_plot_values(
            gains[alone_column],
            outlier_iqr_multiplier=outlier_iqr_multiplier,
            set_negative_to_zero=set_negative_to_zero,
        )
        conditional, n_conditional_omitted = _deviance_plot_values(
            gains[conditional_column],
            outlier_iqr_multiplier=outlier_iqr_multiplier,
            set_negative_to_zero=set_negative_to_zero,
        )
        alone_values.append(alone)
        conditional_values.append(conditional)
        n_extreme_omitted += n_alone_omitted + n_conditional_omitted
    if any(len(values) == 0 for values in alone_values + conditional_values):
        raise ValueError("Each task/movement gain must contain a finite value")

    figure, axis = plt.subplots(figsize=(6.4, 5.8))
    group_positions = np.arange(1, len(column_pairs) + 1)
    within_group_offset = 0.18
    alone_positions = group_positions - within_group_offset
    unique_positions = group_positions + within_group_offset
    alone_color = "#0072B2"
    conditional_color = "#B2182B"
    alone_violins = axis.violinplot(
        alone_values,
        positions=alone_positions,
        widths=0.30,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    conditional_display_values = [-values for values in conditional_values]
    conditional_violins = axis.violinplot(
        conditional_display_values,
        positions=unique_positions,
        widths=0.30,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body in alone_violins["bodies"]:
        body.set_facecolor(alone_color)
        body.set_edgecolor("black")
        body.set_alpha(0.30)
    for body in conditional_violins["bodies"]:
        body.set_facecolor(conditional_color)
        body.set_edgecolor("black")
        body.set_alpha(0.30)
    for violins in (alone_violins, conditional_violins):
        violins["cmedians"].set_color("black")
        violins["cmedians"].set_linewidth(1.8)

    random_generator = np.random.default_rng(42)
    for alone_position, unique_position, alone, conditional_display in zip(
        alone_positions,
        unique_positions,
        alone_values,
        conditional_display_values,
    ):
        axis.scatter(
            alone_position + random_generator.uniform(-0.055, 0.055, len(alone)),
            alone,
            s=11,
            color=alone_color,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.38,
            zorder=3,
        )
        axis.scatter(
            unique_position
            + random_generator.uniform(-0.055, 0.055, len(conditional_display)),
            conditional_display,
            s=11,
            color=conditional_color,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.38,
            zorder=3,
        )

    axis.axhline(0, color="black", linewidth=1.4)
    axis.set(
        xticks=group_positions,
        xticklabels=list(column_pairs),
    )
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{abs(value):.2g}")
    )
    # axis.legend(
    #     handles=[
    #         Patch(facecolor=alone_color, edgecolor="black", label="Alone"),
    #         Patch(
    #             facecolor=conditional_color,
    #             edgecolor="black",
    #             label="Unique",
    #         ),
    #     ],
    #     frameon=False,
    #     loc="best",
    # )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="0.9", linewidth=0.8)
    axis.set_axisbelow(True)

    pairwise_tests = summarize_task_movement_pairwise_comparisons(gains)
    if annotate_pairwise:
        total_test = pairwise_tests.loc[
            pairwise_tests["comparison"].eq("Total")
        ].iloc[0]
        unique_test = pairwise_tests.loc[
            pairwise_tests["comparison"].eq("Unique")
        ].iloc[0]
        plotted_min = min(
            float(np.min(values)) for values in conditional_display_values
        )
        plotted_max = max(float(np.max(values)) for values in alone_values)
        plotted_span = max(plotted_max - plotted_min, 1.0)
        bracket_height = 0.018 * plotted_span
        annotation_gap = 0.05 * plotted_span

        total_base = max(plotted_max, 0.0) + annotation_gap
        axis.plot(
            [
                alone_positions[0],
                alone_positions[0],
                alone_positions[1],
                alone_positions[1],
            ],
            [
                total_base,
                total_base + bracket_height,
                total_base + bracket_height,
                total_base,
            ],
            color="black",
            linewidth=1.1,
            clip_on=False,
        )
        total_p_value = float(total_test["p_value"])
        axis.text(
            float(np.mean(alone_positions)),
            total_base + bracket_height,
            (
                f"{_significance_stars(total_p_value)}\n"
                f"p = {total_p_value:.2g}"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

        unique_base = min(plotted_min, 0.0) - annotation_gap
        axis.plot(
            [
                unique_positions[0],
                unique_positions[0],
                unique_positions[1],
                unique_positions[1],
            ],
            [
                unique_base,
                unique_base - bracket_height,
                unique_base - bracket_height,
                unique_base,
            ],
            color="black",
            linewidth=1.1,
            clip_on=False,
        )
        unique_p_value = float(unique_test["p_value"])
        axis.text(
            float(np.mean(unique_positions)),
            unique_base - bracket_height,
            (
                f"{_significance_stars(unique_p_value)}\n"
                f"p = {unique_p_value:.2g}"
            ),
            ha="center",
            va="top",
            fontsize=9,
        )
        axis.set_ylim(
            unique_base - bracket_height - 0.13 * plotted_span,
            total_base + bracket_height + 0.13 * plotted_span,
        )

    lower_limit, upper_limit = axis.get_ylim()
    zero_axis_fraction = (0 - lower_limit) / (upper_limit - lower_limit)
    axis.text(
        -0.13,
        (1 + zero_axis_fraction) / 2,
        "Total\nD² gain (%)",
        transform=axis.transAxes,
        ha="right",
        va="center",
        color=alone_color,
        fontsize=11,
    )
    axis.text(
        -0.13,
        zero_axis_fraction / 2,
        "Unique\nD² gain (%)",
        transform=axis.transAxes,
        ha="right",
        va="center",
        color=conditional_color,
        fontsize=11,
    )
    figure.subplots_adjust(left=0.29, right=0.97, bottom=0.12, top=0.98)
    if return_pairwise_tests:
        return figure, axis, pairwise_tests
    return figure, axis


def plot_task_movement_pairwise_comparisons(
    gains,
    set_negative_to_zero=False,
    outlier_iqr_multiplier=3.0,
):
    """Plot matched task-versus-movement gains with plot-only outlier removal."""
    if (
        outlier_iqr_multiplier is not None
        and outlier_iqr_multiplier <= 0
    ):
        raise ValueError("outlier_iqr_multiplier must be positive or None")

    comparisons = {
        "total": ("task_alone", "movement_alone", "#0072B2"),
        "unique": ("task_given_movement", "movement_given_task", "#B2182B"),
    }
    required_columns = {
        column
        for task_column, movement_column, _ in comparisons.values()
        for column in (task_column, movement_column)
    }
    missing_columns = required_columns.difference(gains.columns)
    if missing_columns:
        raise ValueError(
            "Task/movement gain table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    plotted_data = {}
    for axis, (comparison, spec) in zip(axes, comparisons.items()):
        task_column, movement_column, color = spec
        paired = gains.loc[:, [task_column, movement_column]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if paired.empty:
            raise ValueError(f"No matched values are available for {comparison}")

        n_matched = len(paired)
        retained = np.ones(n_matched, dtype=bool)
        if outlier_iqr_multiplier is not None:
            for column_name in (task_column, movement_column):
                column_values = paired[column_name].to_numpy(dtype=float)
                lower_quartile, upper_quartile = np.quantile(
                    column_values,
                    (0.25, 0.75),
                )
                interquartile_range = upper_quartile - lower_quartile
                if (
                    np.isfinite(interquartile_range)
                    and interquartile_range > 0
                ):
                    lower_fence = (
                        lower_quartile
                        - outlier_iqr_multiplier * interquartile_range
                    )
                    upper_fence = (
                        upper_quartile
                        + outlier_iqr_multiplier * interquartile_range
                    )
                    retained &= (
                        (column_values >= lower_fence)
                        & (column_values <= upper_fence)
                    )
        paired = paired.loc[retained].copy()
        if paired.empty:
            raise ValueError(
                f"Outlier filtering removed every matched {comparison} value"
            )
        if set_negative_to_zero:
            paired = paired.clip(lower=0.0)
        paired.attrs["n_matched_before_outlier_filter"] = n_matched
        paired.attrs["n_extreme_omitted"] = int(n_matched - len(paired))
        paired.attrs["outlier_iqr_multiplier"] = outlier_iqr_multiplier

        task_values = 100.0 * paired[task_column].to_numpy(dtype=float)
        movement_values = 100.0 * paired[movement_column].to_numpy(dtype=float)
        lower = float(min(task_values.min(), movement_values.min()))
        upper = float(max(task_values.max(), movement_values.max()))
        span = upper - lower
        padding = 0.04 * span if span > 0 else 1.0
        limits = (lower - padding, upper + padding)

        axis.scatter(
            task_values,
            movement_values,
            s=22,
            color=color,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.42,
            rasterized=True,
            zorder=3,
        )
        axis.plot(limits, limits, color="black", linestyle="--", linewidth=1.2)
        if limits[0] < 0 < limits[1]:
            axis.axhline(0, color="0.75", linewidth=0.8, zorder=0)
            axis.axvline(0, color="0.75", linewidth=0.8, zorder=0)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(f"Task {comparison} gain (%)")
        axis.set_ylabel(f"Movement {comparison} gain (%)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="0.92", linewidth=0.7)
        axis.set_axisbelow(True)
        plotted_data[comparison] = paired

    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.98, wspace=0.30)
    return figure, axes, plotted_data


def summarize_task_movement_pairwise_comparisons(
    gains,
    n_boot=10_000,
    random_state=42,
):
    """Summarize paired task-minus-movement gains across matched neurons."""
    comparisons = {
        "Total": ("task_alone", "movement_alone"),
        "Unique": ("task_given_movement", "movement_given_task"),
    }
    required_columns = {
        column
        for pair in comparisons.values()
        for column in pair
    }
    missing_columns = required_columns.difference(gains.columns)
    if missing_columns:
        raise ValueError(
            "Task/movement gain table is missing columns: "
            f"{sorted(missing_columns)}"
        )
    if int(n_boot) < 1:
        raise ValueError("n_boot must be at least 1")

    random_generator = np.random.default_rng(random_state)
    rows = []
    for comparison, (task_column, movement_column) in comparisons.items():
        paired = gains.loc[:, [task_column, movement_column]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        difference = 100.0 * (
            paired[task_column].to_numpy(dtype=float)
            - paired[movement_column].to_numpy(dtype=float)
        )
        if not len(difference):
            raise ValueError(f"No matched values are available for {comparison}")

        rounded_difference = np.round(difference, decimals=10)
        if np.all(rounded_difference == 0):
            statistic, p_value = 0.0, 1.0
            rank_biserial = 0.0
        else:
            test = wilcoxon(
                rounded_difference,
                zero_method="pratt",
                alternative="two-sided",
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
            absolute_ranks = rankdata(np.abs(rounded_difference), method="average")
            positive_rank_sum = float(
                absolute_ranks[rounded_difference > 0].sum()
            )
            negative_rank_sum = float(
                absolute_ranks[rounded_difference < 0].sum()
            )
            ranked_total = positive_rank_sum + negative_rank_sum
            rank_biserial = (
                (positive_rank_sum - negative_rank_sum) / ranked_total
                if ranked_total > 0
                else 0.0
            )

        bootstrap_indices = random_generator.integers(
            0,
            len(difference),
            size=(int(n_boot), len(difference)),
        )
        bootstrap_medians = np.median(
            difference[bootstrap_indices],
            axis=1,
        )
        ci_low, ci_high = np.quantile(bootstrap_medians, [0.025, 0.975])
        rows.append({
            "comparison": comparison,
            "n_units": len(difference),
            "median_task_minus_movement_pct": float(np.median(difference)),
            "median_difference_ci_low_pct": float(ci_low),
            "median_difference_ci_high_pct": float(ci_high),
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
            "rank_biserial": rank_biserial,
        })

    summary = pd.DataFrame(rows)
    p_values = summary["p_value"].to_numpy(dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    summary["p_value_holm"] = adjusted
    return summary


def plot_individual_term_predictive_gains(
    gains,
    set_negative_to_zero=False,
    outlier_iqr_multiplier=3.0,
    term_names=None,
):
    """Plot offset, mirrored task/movement D² gains for individual terms.

    The upper half shows each term's total gain, ``(I + term) - I``, in blue.
    The lower half shows its unique gain, ``full - (full - term)``, in red.
    """
    term_labels = {
        "trial_type": "Trial type",
        "position": "Position",
        "progress": "Progress",
        "translation": "Translation",
        "turning": "Turning",
        "limb": "Limb",
    }
    default_term_names = tuple(term_labels)
    term_names = tuple(dict.fromkeys(term_names or default_term_names))
    if not term_names:
        raise ValueError("term_names must contain at least one term")
    task_terms = {"trial_type", "position", "progress"}
    movement_terms = {"translation", "turning", "limb"}
    grouped_names = (
        ("Task variables", tuple(term for term in term_names if term in task_terms)),
        (
            "Movement variables",
            tuple(term for term in term_names if term in movement_terms),
        ),
        (
            "Other variables",
            tuple(
                term
                for term in term_names
                if term not in task_terms and term not in movement_terms
            ),
        ),
    )
    term_groups = {
        group_label: {
            term_labels.get(term, term.replace("_", " ").title()): term
            for term in group_terms
        }
        for group_label, group_terms in grouped_names
        if group_terms
    }
    group_colors = {
        group_label: {
            "alone": "#56B4E9",
            "given_others": "#E76F51",
        }
        for group_label in term_groups
    }
    required_columns = {
        f"{term_name}_{gain_name}"
        for terms in term_groups.values()
        for term_name in terms.values()
        for gain_name in ("alone", "given_others")
    }
    missing_columns = required_columns.difference(gains.columns)
    if missing_columns:
        raise ValueError(
            "Individual-term gain table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    plot_values = {}
    n_extreme_omitted = 0
    for terms in term_groups.values():
        for term_name in terms.values():
            for gain_name in ("alone", "given_others"):
                column_name = f"{term_name}_{gain_name}"
                values, n_omitted = _deviance_plot_values(
                    gains[column_name],
                    outlier_iqr_multiplier=outlier_iqr_multiplier,
                    set_negative_to_zero=set_negative_to_zero,
                )
                if not len(values):
                    raise ValueError(
                        f"{column_name!r} must contain a finite value"
                    )
                plot_values[column_name] = values
                n_extreme_omitted += n_omitted

    figure, axes = plt.subplots(
        1,
        len(term_groups),
        figsize=(6.2 * len(term_groups), 6.2),
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    random_generator = np.random.default_rng(42)
    for axis, (group_label, terms) in zip(axes, term_groups.items()):
        alone_color = group_colors[group_label]["alone"]
        unique_color = group_colors[group_label]["given_others"]
        group_positions = np.arange(1, len(terms) + 1)
        within_group_offset = 0.18
        alone_positions = group_positions - within_group_offset
        unique_positions = group_positions + within_group_offset
        alone_values = [
            plot_values[f"{term_name}_alone"]
            for term_name in terms.values()
        ]
        conditional_display_values = [
            -plot_values[f"{term_name}_given_others"]
            for term_name in terms.values()
        ]
        alone_violins = axis.violinplot(
            alone_values,
            positions=alone_positions,
            widths=0.30,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        conditional_violins = axis.violinplot(
            conditional_display_values,
            positions=unique_positions,
            widths=0.30,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in alone_violins["bodies"]:
            body.set_facecolor(alone_color)
            body.set_edgecolor(alone_color)
            body.set_alpha(0.55)
        for body in conditional_violins["bodies"]:
            body.set_facecolor(unique_color)
            body.set_edgecolor(unique_color)
            body.set_alpha(0.55)
        for violins in (alone_violins, conditional_violins):
            violins["cmedians"].set_color("black")
            violins["cmedians"].set_linewidth(1.8)

        for alone_position, unique_position, alone, conditional_display in zip(
            alone_positions,
            unique_positions,
            alone_values,
            conditional_display_values,
        ):
            axis.scatter(
                alone_position
                + random_generator.uniform(-0.055, 0.055, len(alone)),
                alone,
                s=11,
                color=alone_color,
                edgecolors="black",
                linewidths=0.25,
                alpha=0.62,
                zorder=3,
            )
            axis.scatter(
                unique_position
                + random_generator.uniform(
                    -0.055,
                    0.055,
                    len(conditional_display),
                ),
                conditional_display,
                s=11,
                color=unique_color,
                edgecolors="black",
                linewidths=0.25,
                alpha=0.62,
                zorder=3,
            )

        axis.axhline(0, color="black", linewidth=1.4)
        axis.set(
            xticks=group_positions,
            xticklabels=list(terms),
        )
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{abs(value):.2g}")
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.8)
        axis.set_axisbelow(True)

    lower_limit, upper_limit = axes[0].get_ylim()
    zero_axis_fraction = (0 - lower_limit) / (upper_limit - lower_limit)
    axes[0].text(
        -0.11,
        (1 + zero_axis_fraction) / 2,
        "Total\nD² gain (%)",
        transform=axes[0].transAxes,
        ha="right",
        va="center",
        color="#0072B2",
        fontsize=11,
    )
    axes[0].text(
        -0.11,
        zero_axis_fraction / 2,
        "Unique\nD² gain (%)",
        transform=axes[0].transAxes,
        ha="right",
        va="center",
        color="#B2182B",
        fontsize=11,
    )
    # figure.legend(
    #     handles=[
    #         Patch(
    #             facecolor=group_colors["Task variables"]["alone"],
    #             edgecolor="black",
    #             alpha=0.30,
    #             label="Task alone",
    #         ),
    #         Patch(
    #             facecolor=group_colors["Task variables"]["given_others"],
    #             edgecolor="black",
    #             alpha=0.30,
    #             label="Task unique",
    #         ),
    #         Patch(
    #             facecolor=group_colors["Movement variables"]["alone"],
    #             edgecolor="black",
    #             alpha=0.30,
    #             label="Movement alone",
    #         ),
    #         Patch(
    #             facecolor=group_colors["Movement variables"]["given_others"],
    #             edgecolor="black",
    #             alpha=0.30,
    #             label="Movement unique",
    #         ),
    #     ],
    #     frameon=False,
    #     ncol=4,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 0.99),
    # )
    figure.subplots_adjust(
        left=0.14,
        top=0.98,
        bottom=0.13,
        wspace=0.12,
    )
    return figure, axes


def plot_task_movement_redundancy_scatter(redundancy):
    """Plot summed-alone versus joint task/movement gain for every unit."""
    required_columns = {
        "combined_alone_gain",
        "joint_gain",
        "redundancy_index",
        "interaction_type",
    }
    missing_columns = required_columns.difference(redundancy.columns)
    if missing_columns:
        raise ValueError(
            "Redundancy table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    plot_data = redundancy.loc[:, sorted(required_columns)].copy()
    numeric_columns = [
        "combined_alone_gain",
        "joint_gain",
        "redundancy_index",
    ]
    finite_rows = np.isfinite(
        plot_data[numeric_columns].to_numpy(dtype=float)
    ).all(axis=1)
    plot_data = plot_data.loc[finite_rows].copy()
    if plot_data.empty:
        raise ValueError("Redundancy table must contain a finite unit")
    plot_data[numeric_columns] *= 100.0

    category_colors = {
        "Redundancy": "#E69F00",
        "Complementarity / suppression": "#CC79A7",
        "Additive": "#7F7F7F",
    }
    figure, axis = plt.subplots(figsize=(6.2, 6.0))
    for category, color in category_colors.items():
        category_rows = plot_data.loc[
            plot_data["interaction_type"].eq(category)
        ]
        if category_rows.empty:
            continue
        axis.scatter(
            category_rows["combined_alone_gain"],
            category_rows["joint_gain"],
            s=30,
            color=color,
            edgecolors="black",
            linewidths=0.35,
            alpha=0.60,
            label=f"{category} (n = {len(category_rows)})",
            zorder=3,
        )

    all_values = np.concatenate(
        [
            plot_data["combined_alone_gain"].to_numpy(dtype=float),
            plot_data["joint_gain"].to_numpy(dtype=float),
        ]
    )
    lower_limit = min(0.0, float(all_values.min()))
    upper_limit = max(0.0, float(all_values.max()))
    value_range = upper_limit - lower_limit
    margin = 0.05 * value_range if value_range > 0 else 1.0
    limits = (lower_limit - margin, upper_limit + margin)
    axis.plot(
        limits,
        limits,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="Additive identity (y = x)",
        zorder=2,
    )
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.axhline(0, color="0.75", linewidth=0.8, zorder=1)
    axis.axvline(0, color="0.75", linewidth=0.8, zorder=1)
    axis.set(
        xlabel="Summed alone gain, $A_T + A_M$ (%)",
        ylabel="Joint gain, $G_{TM}$ (%)",
    )
    axis.legend(frameon=False, loc="upper left", fontsize=8.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(color="0.92", linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()
    return figure, axis


def plot_alone_minus_unique_distributions(redundancy):
    """Plot the shared component and test whether it is greater than zero.

    ``A_T - U_T`` and ``A_M - U_M`` are algebraically identical for every
    unit, so only one distribution is shown. The annotation reports a
    one-sided Wilcoxon signed-rank test against zero.
    """
    columns = {
        "task_alone_minus_unique",
        "movement_alone_minus_unique",
    }
    missing_columns = columns.difference(redundancy.columns)
    if missing_columns:
        raise ValueError(
            "Redundancy table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    paired_values = redundancy.loc[:, sorted(columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    paired_values = paired_values.replace([np.inf, -np.inf], np.nan).dropna()
    if paired_values.empty:
        raise ValueError("Shared-component columns must contain finite values")

    task_values = paired_values["task_alone_minus_unique"].to_numpy(
        dtype=float
    )
    movement_values = paired_values["movement_alone_minus_unique"].to_numpy(
        dtype=float
    )
    if not np.allclose(
        task_values,
        movement_values,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Task and movement alone-minus-unique values must match per unit"
        )

    shared_values = 100.0 * task_values
    rounded_values = np.round(shared_values, decimals=10)
    if np.all(rounded_values == 0):
        p_value = 1.0
    else:
        test = wilcoxon(
            rounded_values,
            zero_method="pratt",
            alternative="greater",
        )
        p_value = float(test.pvalue)

    color = "#56B4E9"
    position = 1.0
    figure, axis = plt.subplots(figsize=(4.8, 5.2))
    violins = axis.violinplot(
        [shared_values],
        positions=[position],
        widths=0.65,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body in violins["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.30)
    violins["cmedians"].set_color("black")
    violins["cmedians"].set_linewidth(1.8)

    random_generator = np.random.default_rng(42)
    axis.scatter(
        position
        + random_generator.uniform(-0.11, 0.11, len(shared_values)),
        shared_values,
        s=14,
        color=color,
        edgecolors="black",
        linewidths=0.25,
        alpha=0.42,
        zorder=3,
    )

    axis.axhline(0, color="black", linewidth=1.3)
    data_min = float(np.min(shared_values))
    data_max = float(np.max(shared_values))
    data_span = max(data_max - data_min, 1.0)
    bracket_base = max(data_max, 0.0) + 0.05 * data_span
    bracket_height = 0.018 * data_span
    axis.plot(
        [position - 0.13, position - 0.13, position + 0.13, position + 0.13],
        [
            bracket_base,
            bracket_base + bracket_height,
            bracket_base + bracket_height,
            bracket_base,
        ],
        color="black",
        linewidth=1.1,
        clip_on=False,
    )
    axis.text(
        position,
        bracket_base + bracket_height,
        f"{_significance_stars(p_value)}\nWilcoxon > 0, p = {p_value:.2g}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    axis.set(
        xticks=[position],
        xticklabels=["Shared component\n$A_T-U_T=A_M-U_M$"],
        ylabel="Shared D² gain (%)",
    )
    lower_limit = axis.get_ylim()[0]
    axis.set_ylim(
        lower_limit,
        bracket_base + bracket_height + 0.13 * data_span,
    )
    axis.set_xlim(0.55, 1.45)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="0.9", linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()
    return figure, axis


def summarize_movement_model_results(results: MovementModelResults):
    """Return one compact completion and held-out-score summary per model."""
    if results.unit_scores.empty:
        return pd.DataFrame()
    score_column = "mean_fold_log_likelihood_per_bin"
    rows = []
    for model_name, model_scores in results.unit_scores.groupby("model_name", sort=False):
        completed = model_scores.loc[model_scores["status"].eq("completed")]
        row = {
            "model_name": model_name,
            "n_completed_units": len(completed),
            "n_failed_units": int(model_scores["status"].eq("error").sum()),
            "mean_cv_log_likelihood_per_bin": completed[score_column].mean(),
            "median_cv_log_likelihood_per_bin": completed[score_column].median(),
            "mean_held_out_bins_per_unit": completed["n_held_out_bins"].mean(),
        }
        if "pooled_deviance_explained" in completed:
            row["mean_pooled_deviance_explained"] = completed[
                "pooled_deviance_explained"
            ].mean()
            row["median_pooled_deviance_explained"] = completed[
                "pooled_deviance_explained"
            ].median()
        rows.append(row)
    return pd.DataFrame(rows)


def completed_unit_scores(results: MovementModelResults):
    """Return sortable completed unit-by-model primary score rows."""
    if results.unit_scores.empty:
        return pd.DataFrame()
    return (
        results.unit_scores.loc[results.unit_scores["status"].eq("completed")]
        .sort_values(["model_name", "mean_fold_log_likelihood_per_bin"], ascending=[True, False])
        .reset_index(drop=True)
    )


def shared_completed_unit_scores(results: MovementModelResults, model_names=None):
    """Return completed scores for units available in every requested model."""
    if results.unit_scores.empty:
        return pd.DataFrame()
    if model_names is None:
        model_names = tuple(results.manifests)
    model_names = tuple(model_names)
    if not model_names:
        return pd.DataFrame()

    completed = results.unit_scores.loc[
        results.unit_scores["status"].eq("completed")
        & results.unit_scores["model_name"].isin(model_names)
    ].copy()
    completed_sets = [
        set(
            completed.loc[
                completed["model_name"].eq(model_name), "unit_index"
            ]
        )
        for model_name in model_names
    ]
    if not completed_sets or any(not unit_set for unit_set in completed_sets):
        return pd.DataFrame(columns=completed.columns)
    shared_unit_indices = set.intersection(*completed_sets)
    return (
        completed.loc[completed["unit_index"].isin(shared_unit_indices)]
        .assign(model_name=lambda table: pd.Categorical(
            table["model_name"], categories=model_names, ordered=True
        ))
        .sort_values(["model_name", "unit_index"])
        .reset_index(drop=True)
    )


def summarize_population_cv_log_likelihood(shared_unit_scores):
    """Summarize matched-unit outer-CV log likelihoods by model."""
    score_column = "mean_fold_log_likelihood_per_bin"
    required_columns = {"model_name", "unit_index", score_column}
    missing_columns = required_columns.difference(shared_unit_scores.columns)
    if missing_columns:
        raise ValueError(
            "Shared unit scores are missing columns: " f"{sorted(missing_columns)}"
        )
    return (
        shared_unit_scores.groupby("model_name", observed=True)[score_column]
        .agg(
            n_shared_units="count",
            mean="mean",
            median="median",
            standard_deviation="std",
            lower_quartile=lambda values: values.quantile(0.25),
            upper_quartile=lambda values: values.quantile(0.75),
        )
        .reset_index()
        .rename(
            columns={
                "mean": "mean_cv_log_likelihood_per_bin",
                "median": "median_cv_log_likelihood_per_bin",
                "standard_deviation": "sd_cv_log_likelihood_per_bin",
            }
        )
    )


def plot_population_cv_log_likelihood(shared_unit_scores):
    """Plot matched-unit outer-CV likelihood distributions by model."""
    score_column = "mean_fold_log_likelihood_per_bin"
    if shared_unit_scores.empty:
        raise ValueError("No units were completed for every displayed model")
    model_names = list(shared_unit_scores["model_name"].cat.categories) if isinstance(
        shared_unit_scores["model_name"].dtype, pd.CategoricalDtype
    ) else list(pd.unique(shared_unit_scores["model_name"]))
    score_groups = [
        shared_unit_scores.loc[
            shared_unit_scores["model_name"].eq(model_name), score_column
        ].to_numpy(dtype=float)
        for model_name in model_names
    ]
    if any(len(scores) == 0 for scores in score_groups):
        raise ValueError("Each displayed model must have at least one shared unit")

    figure, axis = plt.subplots(figsize=(max(6.5, 1.8 * len(model_names)), 4.5))
    positions = np.arange(1, len(model_names) + 1)
    violins = axis.violinplot(
        score_groups,
        positions=positions,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    colors = plt.get_cmap("tab10")(np.arange(len(model_names)) % 10)
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
    violins["cmedians"].set_color("black")
    violins["cmedians"].set_linewidth(1.5)

    random_generator = np.random.default_rng(42)
    for position, scores, color in zip(positions, score_groups, colors):
        jitter = random_generator.uniform(-0.075, 0.075, size=len(scores))
        axis.scatter(
            position + jitter,
            scores,
            s=10,
            color=color,
            alpha=0.5,
            linewidths=0,
            zorder=3,
        )
    n_shared_units = len(score_groups[0])
    axis.set(
        xticks=positions,
        xticklabels=model_names,
        ylabel="Mean outer-CV log likelihood per bin (higher is better)",
        title=f"Population model comparison (n = {n_shared_units} matched units)",
    )
    axis.tick_params(axis="x", rotation=20)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure, axis


def summarize_population_deviance_explained(
    unit_scores,
    model_names,
    score_column="pooled_deviance_explained",
):
    """Summarize outer-CV deviance explained by model, one row per model.

    Unlike :func:`summarize_population_cv_log_likelihood`, units are not
    restricted to those completed for every model in ``model_names`` — each
    row summarizes whichever units completed for that model alone.
    """
    required_columns = {"model_name", "unit_index", score_column}
    missing_columns = required_columns.difference(unit_scores.columns)
    if missing_columns:
        raise ValueError(
            "Unit scores are missing columns: " f"{sorted(missing_columns)}"
        )
    model_names = tuple(model_names)
    selected = unit_scores.loc[unit_scores["model_name"].isin(model_names)]
    return (
        selected.groupby("model_name", observed=True)[score_column]
        .agg(
            n_units="count",
            mean="mean",
            median="median",
            standard_deviation="std",
            lower_quartile=lambda values: values.quantile(0.25),
            upper_quartile=lambda values: values.quantile(0.75),
        )
        .reindex(model_names)
        .reset_index()
        .rename(
            columns={
                "mean": f"mean_{score_column}",
                "median": f"median_{score_column}",
                "standard_deviation": f"sd_{score_column}",
            }
        )
    )


def compare_adjacent_population_models(
    unit_scores,
    model_names,
    score_column="pooled_deviance_explained",
):
    """Run matched-unit Wilcoxon tests for adjacent models in display order."""
    model_names = tuple(model_names)
    if len(model_names) < 2:
        raise ValueError("At least two model names are required")
    required_columns = {"model_name", "unit_index", score_column}
    missing_columns = required_columns.difference(unit_scores.columns)
    if missing_columns:
        raise ValueError(
            "Unit scores are missing columns: " f"{sorted(missing_columns)}"
        )

    selected = unit_scores.loc[
        unit_scores["model_name"].isin(model_names),
        ["model_name", "unit_index", score_column],
    ].copy()
    selected[score_column] = pd.to_numeric(
        selected[score_column], errors="coerce"
    )
    selected = selected.dropna(subset=["unit_index", score_column])
    duplicated = selected.duplicated(["model_name", "unit_index"], keep=False)
    if duplicated.any():
        raise ValueError(
            "Unit scores contain duplicate model/unit rows; deduplicate before "
            "running paired comparisons"
        )

    rows = []
    for first_model, second_model in zip(model_names[:-1], model_names[1:]):
        pair = selected.loc[
            selected["model_name"].isin([first_model, second_model])
        ].pivot(index="unit_index", columns="model_name", values=score_column)
        pair = pair.reindex(columns=[first_model, second_model]).dropna()
        if pair.empty:
            raise ValueError(
                f"No matched finite units for {first_model!r} and {second_model!r}"
            )

        difference = 100.0 * (
            pair[second_model].to_numpy(dtype=float)
            - pair[first_model].to_numpy(dtype=float)
        )
        rounded_difference = np.round(difference, decimals=10)
        if np.all(rounded_difference == 0):
            statistic, p_value = 0.0, 1.0
            rank_biserial = 0.0
        else:
            test = wilcoxon(
                rounded_difference,
                zero_method="pratt",
                alternative="two-sided",
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
            absolute_ranks = rankdata(np.abs(rounded_difference), method="average")
            positive_rank_sum = float(
                absolute_ranks[rounded_difference > 0].sum()
            )
            negative_rank_sum = float(
                absolute_ranks[rounded_difference < 0].sum()
            )
            ranked_total = positive_rank_sum + negative_rank_sum
            rank_biserial = (
                (positive_rank_sum - negative_rank_sum) / ranked_total
                if ranked_total > 0
                else 0.0
            )

        rows.append({
            "first_model": first_model,
            "second_model": second_model,
            "n_matched_units": len(pair),
            "median_second_minus_first_pct": float(np.median(difference)),
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
            "rank_biserial": rank_biserial,
        })

    summary = pd.DataFrame(rows)
    p_values = summary["p_value"].to_numpy(dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    summary["p_value_holm"] = adjusted
    return summary


def _significance_stars(p_value):
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def plot_population_deviance_explained(
    unit_scores,
    model_names,
    score_column="pooled_deviance_explained",
    outlier_iqr_multiplier=3.0,
    annotate_adjacent=True,
    return_pairwise_tests=False,
):
    """Plot each model's own completed-unit deviance-explained distribution.

    Unlike :func:`plot_population_cv_log_likelihood`, models are not
    restricted to a shared, intersected unit population — each violin uses
    whichever units are completed for that model, and per-model counts may
    differ.
    """
    model_names = tuple(model_names)
    prepared_groups = [
        _deviance_plot_values(
            unit_scores.loc[
                unit_scores["model_name"].eq(model_name), score_column
            ],
            outlier_iqr_multiplier=outlier_iqr_multiplier,
        )
        for model_name in model_names
    ]
    score_groups = [values for values, _ in prepared_groups]
    if any(len(scores) == 0 for scores in score_groups):
        empty_models = [
            model_name
            for model_name, scores in zip(model_names, score_groups)
            if len(scores) == 0
        ]
        raise ValueError(
            f"No completed units with a finite {score_column!r} for: {empty_models}"
        )

    figure, axis = plt.subplots(
        figsize=(max(6.5, 1.8 * len(model_names)), 4.8),
        constrained_layout=True,
    )
    positions = np.arange(1, len(model_names) + 1)
    violins = axis.violinplot(
        score_groups,
        positions=positions,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    colors = plt.get_cmap("tab10")(np.arange(len(model_names)) % 10)
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
    violins["cmedians"].set_color("black")
    violins["cmedians"].set_linewidth(1.5)

    random_generator = np.random.default_rng(42)
    for position, scores, color in zip(positions, score_groups, colors):
        jitter = random_generator.uniform(-0.075, 0.075, size=len(scores))
        axis.scatter(
            position + jitter,
            scores,
            s=10,
            color=color,
            alpha=0.5,
            linewidths=0,
            zorder=3,
        )
    tick_labels = [
        f"{model_name}\n(n = {len(scores)})"
        for model_name, scores in zip(model_names, score_groups)
    ]
    axis.set(
        xticks=positions,
        xticklabels=tick_labels,
        ylabel="CV deviance explained (%)",
    )
    axis.tick_params(axis="x", rotation=0)
    axis.spines[["top", "right"]].set_visible(False)

    pairwise_tests = compare_adjacent_population_models(
        unit_scores,
        model_names,
        score_column=score_column,
    )
    if annotate_adjacent:
        data_min = min(float(np.min(scores)) for scores in score_groups)
        data_max = max(float(np.max(scores)) for scores in score_groups)
        data_span = max(data_max - data_min, 1.0)
        bracket_height = 0.018 * data_span
        level_step = 0.075 * data_span
        bracket_base = data_max + 0.055 * data_span
        for pair_index, test_row in pairwise_tests.iterrows():
            left = positions[pair_index]
            right = positions[pair_index + 1]
            height = bracket_base + pair_index * level_step
            axis.plot(
                [left, left, right, right],
                [height, height + bracket_height, height + bracket_height, height],
                color="black",
                linewidth=1.1,
                clip_on=False,
            )
            axis.text(
                (left + right) / 2,
                height + bracket_height,
                _significance_stars(test_row["p_value_holm"]),
                ha="center",
                va="bottom",
                fontsize=11,
            )
        lower_limit = axis.get_ylim()[0]
        upper_limit = (
            bracket_base
            + (len(pairwise_tests) - 1) * level_step
            + bracket_height
            + 0.07 * data_span
        )
        axis.set_ylim(lower_limit, upper_limit)

    if return_pairwise_tests:
        return figure, axis, pairwise_tests
    return figure, axis


def plot_oof_smoothed_rate(
    results: MovementModelResults,
    model_name,
    unit_index,
    trial_number,
    bin_size_s,
    smoothing_s=0.05,
    oof_predictions=None,
):
    """Plot observed and expected held-out firing rates for one complete trial.

    Observed spike counts are shown at their original bin resolution. Gaussian
    smoothing is applied only to the expected rate, within the selected trial,
    preventing visual blending across trial boundaries.
    """
    required_columns = {
        "model_name",
        "unit_index",
        "trial_number",
        "observed_spike_count",
        "predicted_spike_count",
    }
    prediction_data = (
        results.out_of_fold_predictions
        if oof_predictions is None
        else oof_predictions
    )
    missing_columns = required_columns.difference(prediction_data)
    if missing_columns:
        raise ValueError(
            "Out-of-fold predictions are missing columns: " f"{sorted(missing_columns)}"
        )
    selected = prediction_data.loc[
        prediction_data["model_name"].eq(model_name)
        & prediction_data["unit_index"].eq(unit_index)
        & prediction_data["trial_number"].eq(trial_number)
    ].copy()
    if selected.empty:
        raise ValueError(
            "No held-out predictions for the selected model, unit, and trial"
        )
    time_column = "time_bin_center" if "time_bin_center" in selected else "data_index"
    if time_column not in selected:
        raise ValueError("Out-of-fold predictions have no time coordinate")
    selected = selected.sort_values(time_column)
    sigma_bins = smoothing_s / bin_size_s
    if sigma_bins <= 0:
        raise ValueError("smoothing_s and bin_size_s must be positive")
    observed_rate = (
        selected["observed_spike_count"].to_numpy(dtype=float) / bin_size_s
    )
    predicted_rate = gaussian_filter1d(
        selected["predicted_spike_count"].to_numpy(dtype=float) / bin_size_s,
        sigma=sigma_bins,
        mode="nearest",
    )
    time_s = selected[time_column].to_numpy(dtype=float)
    time_s = time_s - time_s[0]
    figure, axis = plt.subplots(figsize=(10, 3.5))
    axis.plot(time_s, observed_rate, color="black", lw=1.5, label="Observed")
    axis.plot(time_s, predicted_rate, color="#0072B2", lw=1.5, label="Predicted")
    axis.set(
        xlabel="Time within trial (s)",
        ylabel="Firing rate (Hz)",
        title=(
            f"{model_name} — unit {unit_index}, trial {trial_number} "
            f"(observed: raw; predicted: {smoothing_s * 1000:g}-ms Gaussian)"
        ),
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure, axis


_DEFAULT_BEHAVIORAL_COVARIATES = {
    "continuous": ("linear_position", "speed", "trial_progress_distance"),
    "categorical": ("trial_type",),
}

_BEHAVIORAL_LABELS = {
    "linear_position": "Linear position",
    "speed": "Speed (cm/s)",
    "trial_progress_distance": "Trial progress distance",
    "body_angular_velocity_rad_s": "Body angular velocity\n(rad/s)",
    "forepaw_r_ap_displacement_cm": (
        "Right forepaw AP displacement\nfrom median (cm)"
    ),
    "trial_type": "Trial type",
}


def _oof_behavior_frame(oof_predictions, covariates, covariates_to_plot):
    """Align one unit's OOF bins to behavioral values in the frozen input table."""
    required_prediction_columns = {
        "data_index",
        "observed_spike_count",
        "predicted_spike_count",
    }
    missing_prediction_columns = required_prediction_columns.difference(
        oof_predictions.columns
    )
    if missing_prediction_columns:
        raise ValueError(
            "OOF predictions are missing columns: "
            f"{sorted(missing_prediction_columns)}"
        )

    missing_covariates = set(covariates_to_plot).difference(covariates.columns)
    if missing_covariates:
        raise ValueError(
            "The frozen covariate table is missing columns: "
            f"{sorted(missing_covariates)}"
        )

    data_index = pd.Index(oof_predictions["data_index"])
    row_positions = covariates.index.get_indexer(data_index)
    if (row_positions < 0).any():
        raise ValueError("OOF data_index is absent from the frozen covariate table")

    behavior_frame = covariates.iloc[row_positions][list(covariates_to_plot)].copy()
    behavior_frame.reset_index(drop=True, inplace=True)
    behavior_frame["observed_spike_count"] = oof_predictions[
        "observed_spike_count"
    ].to_numpy(dtype=float)
    behavior_frame["predicted_spike_count"] = oof_predictions[
        "predicted_spike_count"
    ].to_numpy(dtype=float)
    return behavior_frame


def _smooth_binned_curve_nan(curve, sigma_bins=None):
    """Gaussian-smooth a binned curve while preserving missing-bin gaps."""
    curve = np.asarray(curve, dtype=float)
    if sigma_bins is None or sigma_bins == 0:
        return curve
    if sigma_bins < 0:
        raise ValueError("empirical_smooth_sigma_bins must be non-negative")
    valid = np.isfinite(curve).astype(float)
    filled = np.where(np.isfinite(curve), curve, 0.0)
    numerator = gaussian_filter1d(filled, sigma_bins, mode="nearest")
    denominator = gaussian_filter1d(valid, sigma_bins, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def _plot_oof_continuous_tuning(
    axis,
    behavior_frame,
    covariate,
    bin_size_s,
    n_bins,
    empirical_smooth_sigma_bins=None,
    use_quantile_bins=False,
):
    values = pd.to_numeric(behavior_frame[covariate], errors="coerce")
    valid = (
        values.notna()
        & np.isfinite(values)
        & behavior_frame["observed_spike_count"].notna()
        & behavior_frame["predicted_spike_count"].notna()
    )
    plot_frame = behavior_frame.loc[valid, ["observed_spike_count", "predicted_spike_count"]]
    values = values.loc[valid].to_numpy(dtype=float)
    if len(values) == 0 or np.nanmin(values) == np.nanmax(values):
        axis.text(0.5, 0.5, "No variable data", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return

    if use_quantile_bins:
        edges = np.unique(
            np.nanquantile(values, np.linspace(0.0, 1.0, n_bins + 1))
        )
        if len(edges) < 2:
            axis.text(
                0.5, 0.5, "No variable data", ha="center", va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            return
    else:
        _, edges = np.histogram(values, bins=n_bins)
    bin_index = np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)
    centers = (edges[:-1] + edges[1:]) / 2
    observed_rate = np.full(len(centers), np.nan)
    predicted_rate = np.full(len(centers), np.nan)
    observed = plot_frame["observed_spike_count"].to_numpy(dtype=float)
    predicted = plot_frame["predicted_spike_count"].to_numpy(dtype=float)
    for bin_number in range(len(centers)):
        in_bin = bin_index == bin_number
        if in_bin.any():
            if use_quantile_bins:
                centers[bin_number] = np.nanmedian(values[in_bin])
            observed_rate[bin_number] = observed[in_bin].mean() / bin_size_s
            predicted_rate[bin_number] = predicted[in_bin].mean() / bin_size_s

    observed_plot_rate = _smooth_binned_curve_nan(
        observed_rate, empirical_smooth_sigma_bins
    )
    axis.plot(centers, observed_plot_rate, color="black", lw=1.5, label="Observed")
    axis.plot(centers, predicted_rate, color="#0072B2", lw=2.0, label="Predicted")
    axis.set(xlabel=_BEHAVIORAL_LABELS.get(covariate, covariate), ylabel="Firing rate (Hz)")


def _plot_oof_categorical_tuning(axis, behavior_frame, covariate, bin_size_s):
    valid = (
        behavior_frame[covariate].notna()
        & behavior_frame["observed_spike_count"].notna()
        & behavior_frame["predicted_spike_count"].notna()
    )
    plot_frame = behavior_frame.loc[
        valid,
        [covariate, "observed_spike_count", "predicted_spike_count"],
    ]
    if plot_frame.empty:
        axis.text(0.5, 0.5, "No variable data", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return

    grouped = plot_frame.groupby(covariate, sort=True, observed=True)[
        ["observed_spike_count", "predicted_spike_count"]
    ].mean() / bin_size_s
    x = np.arange(len(grouped))
    width = 0.38
    axis.bar(x - width / 2, grouped["observed_spike_count"], width, color="black", label="Observed")
    axis.bar(x + width / 2, grouped["predicted_spike_count"], width, color="#0072B2", label="Predicted")
    axis.set(
        xticks=x,
        xticklabels=grouped.index.astype(str),
        xlabel=_BEHAVIORAL_LABELS.get(covariate, covariate),
        ylabel="Firing rate (Hz)",
    )


def plot_oof_conditional_pred_vs_observed(
    oof_predictions,
    covariates,
    bin_size_s,
    model_name,
    unit_index,
    covariates_to_plot=None,
    n_bins=30,
    empirical_smooth_sigma_bins=None,
    quantile_binned_covariates=None,
):
    """Plot raw empirical and OOF-predicted rates against behavioral variables.

    Each curve averages bins with similar behavioral values across all held-out
    trials. As in ``encoding.ipynb``, optional Gaussian smoothing is applied
    only to the binned observed curve; predictions are never Gaussian-smoothed.
    The predictions remain genuine outer-test predictions.
    """
    if bin_size_s <= 0:
        raise ValueError("bin_size_s must be positive")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if (
        empirical_smooth_sigma_bins is not None
        and empirical_smooth_sigma_bins < 0
    ):
        raise ValueError("empirical_smooth_sigma_bins must be non-negative")
    covariates_to_plot = (
        _DEFAULT_BEHAVIORAL_COVARIATES
        if covariates_to_plot is None
        else covariates_to_plot
    )
    continuous = tuple(covariates_to_plot.get("continuous", ()))
    categorical = tuple(covariates_to_plot.get("categorical", ()))
    requested_covariates = continuous + categorical
    if not requested_covariates:
        raise ValueError("At least one behavioral covariate must be requested")
    quantile_binned_covariates = set(quantile_binned_covariates or ())
    unknown_quantile_covariates = quantile_binned_covariates.difference(continuous)
    if unknown_quantile_covariates:
        raise ValueError(
            "Quantile-binned covariates must be continuous and plotted: "
            f"{sorted(unknown_quantile_covariates)}"
        )

    behavior_frame = _oof_behavior_frame(
        oof_predictions,
        covariates,
        requested_covariates,
    )
    if len(requested_covariates) == 1:
        figure, single_axis = plt.subplots(figsize=(14, 4.8))
        axes = [single_axis]
    else:
        n_bottom_axes = len(requested_covariates) - 1
        figure_width = max(14, 4.2 * n_bottom_axes)
        figure = plt.figure(figsize=(figure_width, 7.8))
        grid = figure.add_gridspec(
            2, n_bottom_axes, hspace=0.46, wspace=0.36
        )
        axes = [figure.add_subplot(grid[0, :])]
        axes.extend(
            figure.add_subplot(grid[1, column])
            for column in range(n_bottom_axes)
        )
    requested_axes = list(zip(requested_covariates, axes))
    for covariate, axis in requested_axes:
        if covariate in continuous:
            _plot_oof_continuous_tuning(
                axis,
                behavior_frame,
                covariate,
                bin_size_s,
                n_bins,
                empirical_smooth_sigma_bins=empirical_smooth_sigma_bins,
                use_quantile_bins=covariate in quantile_binned_covariates,
            )
        else:
            _plot_oof_categorical_tuning(axis, behavior_frame, covariate, bin_size_s)
        axis.set_title(_BEHAVIORAL_LABELS.get(covariate, covariate), fontsize=12)
    for axis in axes[len(requested_axes):]:
        axis.set_axis_off()

    legend_axis = next(
        (axis for axis in axes if axis.get_legend_handles_labels()[0]),
        None,
    )
    if legend_axis is not None:
        handles, labels = legend_axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            frameon=False,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            fontsize=11,
        )
    figure.suptitle(
        f"{model_name} — unit {unit_index} "
        f"(n={len(behavior_frame):,} held-out bins)",
        fontsize=15,
        y=0.99,
    )
    figure.subplots_adjust(top=0.82)
    return figure, axes
