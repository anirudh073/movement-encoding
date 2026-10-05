"""Persistent batch orchestration for nested movement-encoding models.

The tracking-derived covariate bundle is exported once from the notebook. This
module only loads that frozen bundle and fits registry-defined models; it never
recomputes behavioral variables.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from multiprocessing import Manager
from pathlib import Path
import queue
import re
import shutil
import threading
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from movement_encoding import (
    build_movement_model_registry,
    build_movement_penalty_bands,
    build_movement_predictor_groups,
    make_contiguous_outer_trial_folds,
    movement_spline_config,
    run_nested_outer_cv_poisson_ridge,
)
from movement_features import DEFAULT_PAW_PREFIXES
from movement_model_artifacts import pack_fold_coefficients


BUNDLE_VERSION = 3
RUN_MANIFEST_VERSION = 5
SHUFFLE_SEED = 42
_BUNDLE_FILES = {
    "covariates": "covariates.pkl",
    "spike_counts": "spike_counts.npy",
    "unit_ids": "unit_ids.npy",
}
_MODEL_OUTPUT_FILES = {
    "unit_scores": "unit_scores.csv",
    "fold_scores": "fold_scores.csv",
    "oof_predictions": "oof_predictions.csv.gz",
    "clipping_summary": "clipping_summary.csv",
    "clipping_detail": "clipping_detail.csv",
    "zero_variance": "zero_variance_audit.csv",
    "penalty_search": "penalty_search.csv",
}
_OOF_SHARD_DIRECTORY = "oof_prediction_shards"
_OOF_COLUMNS_FILENAME = "oof_prediction_columns.json"
_PERMUTATION_AUDIT_FILENAME = "permutation_audit.csv"


def _json_ready(value):
    """Convert supported values to deterministic JSON-compatible objects."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot place {type(value).__name__} in a run manifest")


def _canonical_json(value):
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(value, path: Path):
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True))
    temporary_path.replace(path)


def _atomic_write_dataframe(frame: pd.DataFrame, path: Path):
    temporary_path = path.with_name(f".{path.name}.tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(temporary_path, index=False, compression=compression)
    temporary_path.replace(path)


def _atomic_write_pickle(frame: pd.DataFrame, path: Path):
    temporary_path = path.with_name(f".{path.name}.tmp")
    frame.to_pickle(temporary_path)
    temporary_path.replace(path)


def _atomic_write_npy(values, path: Path):
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("wb") as stream:
        np.save(stream, values, allow_pickle=True)
    temporary_path.replace(path)


def _require_bundle_fields(data: Mapping):
    required = {
        "cov_df_common",
        "spike_counts_common",
        "unit_ids",
        "predictor_groups",
        "spline_config",
        "inner_cv_config",
        "penalty_bands",
        "banded_optimizer_config",
        "outer_trial_folds",
    }
    missing = required.difference(data)
    if missing:
        raise KeyError(
            "Prepared movement data is missing required fields: " f"{sorted(missing)}"
        )


def export_movement_encoding_input_bundle(data: Mapping, bundle_dir):
    """Persist the notebook's already-derived model inputs for batch fitting."""
    _require_bundle_fields(data)
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    covariates = data["cov_df_common"]
    spike_counts = np.asarray(data["spike_counts_common"])
    unit_ids = np.asarray(data["unit_ids"], dtype=object)
    if not isinstance(covariates, pd.DataFrame):
        raise TypeError("cov_df_common must be a pandas DataFrame")
    if spike_counts.ndim != 2 or spike_counts.shape[1] != len(covariates):
        raise ValueError(
            "spike_counts_common must have shape (n_units, n_covariate_bins)"
        )
    if len(unit_ids) != spike_counts.shape[0]:
        raise ValueError("unit_ids length does not match spike_counts_common rows")
    trial_type_selection = data.get("trial_type_selection")
    if trial_type_selection not in {None, "both", "outbound", "inbound"}:
        raise ValueError("Invalid trial_type_selection for movement input bundle")
    if trial_type_selection in {"outbound", "inbound"}:
        if "trial_type" not in covariates or not covariates["trial_type"].eq(
            trial_type_selection
        ).all():
            raise ValueError(
                f"Input bundle contains bins outside {trial_type_selection} trials"
            )
        if "trial_type" in (data.get("model_group") or {}).get("terms", ()):
            raise ValueError(
                "A single-direction model group cannot include trial_type"
            )

    covariate_path = bundle_dir / _BUNDLE_FILES["covariates"]
    spike_path = bundle_dir / _BUNDLE_FILES["spike_counts"]
    unit_id_path = bundle_dir / _BUNDLE_FILES["unit_ids"]
    _atomic_write_pickle(covariates, covariate_path)
    _atomic_write_npy(spike_counts, spike_path)
    _atomic_write_npy(unit_ids, unit_id_path)
    bundle_manifest = {
        "bundle_version": BUNDLE_VERSION,
        "files": {
            name: {
                "filename": filename,
                "sha256": _sha256_file(bundle_dir / filename),
            }
            for name, filename in _BUNDLE_FILES.items()
        },
        "n_bins": len(covariates),
        "n_units": spike_counts.shape[0],
        "covariate_columns": covariates.columns.tolist(),
        "predictor_groups": data["predictor_groups"],
        "spline_config": data["spline_config"],
        "inner_cv_config": data["inner_cv_config"],
        "penalty_bands": data["penalty_bands"],
        "banded_optimizer_config": data["banded_optimizer_config"],
        "spike_history_enabled": False,
        "outer_trial_folds": data["outer_trial_folds"],
    }
    if data.get("sample_mode") is not None:
        bundle_manifest["sample_mode"] = data["sample_mode"]
    if data.get("trial_type_selection") is not None:
        bundle_manifest["trial_type_selection"] = data["trial_type_selection"]
    if data.get("model_group") is not None:
        bundle_manifest["model_group"] = data["model_group"]
    progress_method = data.get("distance_progress_method", covariates.attrs.get("distance_progress_method"))
    if progress_method is not None:
        bundle_manifest["distance_progress_method"] = progress_method
    _atomic_write_json(bundle_manifest, bundle_dir / "bundle_manifest.json")
    return bundle_manifest


def load_movement_encoding_input_bundle(bundle_dir):
    """Load and integrity-check a notebook-exported movement input bundle."""
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing input-bundle manifest: {manifest_path}")
    bundle_manifest = json.loads(manifest_path.read_text())
    if bundle_manifest.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(
            "Unsupported movement input-bundle version: "
            f"{bundle_manifest.get('bundle_version')!r}"
        )
    if bundle_manifest.get("spike_history_enabled") is not False:
        raise ValueError("Movement encoding requires spike_history_enabled=false")
    for name, file_spec in bundle_manifest.get("files", {}).items():
        path = bundle_dir / file_spec["filename"]
        if not path.exists():
            raise FileNotFoundError(f"Input bundle is missing {name}: {path}")
        actual_hash = _sha256_file(path)
        if actual_hash != file_spec["sha256"]:
            raise ValueError(
                f"Input bundle file checksum mismatch for {name}: {path}"
            )
    covariates = pd.read_pickle(bundle_dir / _BUNDLE_FILES["covariates"])
    spike_counts = np.load(
        bundle_dir / _BUNDLE_FILES["spike_counts"], allow_pickle=True
    )
    unit_ids = np.load(bundle_dir / _BUNDLE_FILES["unit_ids"], allow_pickle=True)
    if spike_counts.ndim != 2 or spike_counts.shape[1] != len(covariates):
        raise ValueError("Input bundle has incompatible covariate/spike-count shapes")
    if len(unit_ids) != spike_counts.shape[0]:
        raise ValueError("Input bundle has incompatible unit IDs/spike-count rows")
    return {
        "covariates": covariates,
        "spike_counts": spike_counts,
        "unit_ids": unit_ids,
        "manifest": bundle_manifest,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def make_default_movement_model_inputs():
    """Return the approved group, spline, and inner-CV definitions."""
    predictor_groups = build_movement_predictor_groups(DEFAULT_PAW_PREFIXES)
    return {
        "predictor_groups": predictor_groups,
        "spline_config": movement_spline_config(DEFAULT_PAW_PREFIXES),
        "inner_cv_config": {
            "n_splits": 3,
            "seed": 42,
            "lower_log10": -8.0,
            "upper_log10": 2.0,
            "log10_step": 2.0,
            "edge_expansion_log10": 2.0,
            "max_grid_expansions": 10,
        },
        "penalty_bands": build_movement_penalty_bands(predictor_groups),
        "banded_optimizer_config": {
            "max_iter": 20,
            "max_evaluations": 30,
            "gtol": 1e-4,
            "ftol": 1e-9,
        },
        "spike_history_enabled": False,
    }


def _safe_model_name(model_name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model_name):
        raise ValueError(f"Unsafe model name for an output path: {model_name!r}")


def _read_output_table(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _model_predictor_groups(predictor_groups, model_spec, ablation_mode=None):
    """Apply a model's optional predictor exclusions without mutating inputs."""
    excluded_predictors = set(model_spec.get("excluded_predictors", ()))
    if ablation_mode == "drop":
        excluded_predictors.update(model_spec.get("ablated_predictors", ()))
    included_groups = tuple(model_spec["included_groups"])
    selected_predictors = {
        predictor
        for group_name in included_groups
        for predictor in predictor_groups[group_name]
    }
    unavailable_exclusions = excluded_predictors.difference(selected_predictors)
    if unavailable_exclusions:
        raise ValueError(
            "Model excludes predictors not present in its included groups: "
            f"{sorted(unavailable_exclusions)}"
        )
    return {
        group_name: tuple(
            predictor
            for predictor in group_predictors
            if predictor not in excluded_predictors
        )
        for group_name, group_predictors in predictor_groups.items()
    }


def _replace_unit_rows(existing, replacement, unit_index):
    """Replace all rows for one unit, including stale success artifacts."""
    if not existing.empty and "unit_index" in existing:
        existing = existing.loc[existing["unit_index"] != unit_index]
    if replacement is None or replacement.empty:
        return existing.reset_index(drop=True)
    return pd.concat((existing, replacement), ignore_index=True, sort=False)


def _model_run_spec(
    bundle,
    model_name,
    model_spec,
    ablation_mode=None,
    penalty_source_manifest_sha256=None,
):
    run_spec = {
        "run_manifest_version": RUN_MANIFEST_VERSION,
        "input_bundle_manifest_sha256": bundle["manifest_sha256"],
        "model_name": model_name,
        "model_spec": model_spec,
        "predictor_groups": bundle["manifest"]["predictor_groups"],
        "spline_config": bundle["manifest"]["spline_config"],
        "inner_cv_config": bundle["manifest"]["inner_cv_config"],
        "penalty_bands": bundle["manifest"]["penalty_bands"],
        "banded_optimizer_config": bundle["manifest"]["banded_optimizer_config"],
        "spike_history_enabled": False,
        "outer_trial_folds": bundle["manifest"]["outer_trial_folds"],
        "unit_ids": bundle["unit_ids"],
    }
    if model_spec.get("penalty_source_model") is not None:
        if penalty_source_manifest_sha256 is None:
            raise ValueError("Inherited penalties require a source manifest hash")
        run_spec["penalty_source_run_manifest_sha256"] = (
            penalty_source_manifest_sha256
        )
    if model_spec.get("ablated_predictors"):
        run_spec["ablation_mode"] = ablation_mode
        if ablation_mode == "shuffle":
            run_spec["shuffle_seed"] = SHUFFLE_SEED
    return run_spec


def _run_spec_without_worker_count(run_spec):
    """Return a compatibility spec that ignores parallel worker count only."""
    comparable_spec = dict(run_spec or {})
    execution_config = dict(comparable_spec.get("execution_config", {}))
    execution_config.pop("n_jobs", None)
    if execution_config:
        comparable_spec["execution_config"] = execution_config
    else:
        comparable_spec.pop("execution_config", None)
    return comparable_spec


def _prepare_model_output_dir(output_dir: Path, run_spec, n_jobs):
    manifest_path = output_dir / "run_manifest.json"
    if output_dir.exists() and not manifest_path.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Refusing to use existing output directory without a manifest: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        stored_manifest = json.loads(manifest_path.read_text())
        stored_run_spec = _run_spec_without_worker_count(
            stored_manifest.get("run_spec")
        )
        requested_run_spec = _run_spec_without_worker_count(run_spec)
        if _canonical_json(stored_run_spec) != _canonical_json(requested_run_spec):
            raise RuntimeError(
                f"Saved run manifest does not exactly match requested parameters: "
                f"{output_dir}"
            )
        return
    _atomic_write_json(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution_config": {"n_jobs": int(n_jobs)},
            "run_spec": run_spec,
        },
        manifest_path,
    )


def _unit_artifacts(
    cv_result,
    unit_index,
    unit_id,
    model_name,
    ablation_mode=None,
):
    """Attach unit identity to every persisted CV artifact."""
    fold_scores = cv_result.fold_scores.copy()
    fold_scores.insert(0, "model_name", model_name)
    fold_scores.insert(1, "unit_id", str(unit_id))
    fold_scores.insert(2, "unit_index", unit_index)
    if ablation_mode is not None:
        fold_scores.insert(1, "ablation_mode", ablation_mode)

    oof_predictions = cv_result.out_of_fold_predictions.copy()
    oof_predictions.insert(0, "model_name", model_name)
    oof_predictions.insert(1, "unit_id", str(unit_id))
    oof_predictions.insert(2, "unit_index", unit_index)
    oof_predictions.insert(3, "data_index", oof_predictions.index.to_numpy())
    if ablation_mode is not None:
        oof_predictions.insert(1, "ablation_mode", ablation_mode)
    oof_predictions = oof_predictions.reset_index(drop=True)

    clipping_summaries = [cv_result.clipping_summary.assign(stage="outer_test")]
    clipping_details = [cv_result.clipping_detail.assign(stage="outer_test")]
    zero_variance_reports = []
    for tuning_result in cv_result.tuning_results.values():
        clipping_summaries.append(
            tuning_result.clipping_summary.assign(stage="inner_validation")
        )
        clipping_details.append(
            tuning_result.clipping_detail.assign(stage="inner_validation")
        )
        zero_variance_reports.append(
            tuning_result.zero_variance_report.assign(stage="inner_training")
        )
    for outer_fit in cv_result.outer_fits.values():
        zero_variance_reports.append(
            outer_fit.transformer.zero_variance_report.assign(
                stage="outer_training"
            )
        )

    def add_unit_columns(frames):
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame(
                columns=["model_name", "unit_id", "unit_index"]
            )
        frame = pd.concat(frames, ignore_index=True, sort=False)
        frame.insert(0, "model_name", model_name)
        frame.insert(1, "unit_id", str(unit_id))
        frame.insert(2, "unit_index", unit_index)
        if ablation_mode is not None:
            frame.insert(1, "ablation_mode", ablation_mode)
        return frame

    return {
        "fold_scores": fold_scores,
        "oof_predictions": oof_predictions,
        "clipping_summary": add_unit_columns(clipping_summaries),
        "clipping_detail": add_unit_columns(clipping_details),
        "zero_variance": add_unit_columns(zero_variance_reports),
        "penalty_search": add_unit_columns([cv_result.penalty_search]),
    }


def _write_model_outputs(output_dir: Path, tables):

    """Write only compact per-unit/fold/audit tables during model fitting."""
    for table_name, table in tables.items():
        _atomic_write_dataframe(table, output_dir / _MODEL_OUTPUT_FILES[table_name])


def _merge_permutation_audit(
    existing,
    current,
    *,
    model_name,
    ablation_mode,
    unit_index,
    unit_id,
):
    """Store one common audit and only unit-specific mismatches thereafter."""
    if current is None or current.empty:
        return existing
    if "unit_index" in existing and "audit_scope" in existing:
        existing = existing.loc[
            ~(
                existing["audit_scope"].eq("unit_exception")
                & existing["unit_index"].eq(unit_index)
            )
        ].copy()
    identity_columns = current.columns.tolist()
    common = existing.loc[existing.get("audit_scope", pd.Series(dtype=str)).eq("common")]
    if common.empty:
        common = current.copy()
        common.insert(0, "model_name", model_name)
        common.insert(1, "ablation_mode", ablation_mode)
        common.insert(2, "audit_scope", "common")
        common.insert(3, "unit_id", pd.NA)
        common.insert(4, "unit_index", pd.NA)
        return pd.concat((existing, common), ignore_index=True, sort=False)
    reference = (
        common.loc[:, identity_columns].reset_index(drop=True).fillna("").astype(str)
    )
    candidate = current.loc[:, identity_columns].reset_index(drop=True).fillna("").astype(str)
    if reference.equals(candidate):
        return existing
    exception = current.copy()
    exception.insert(0, "model_name", model_name)
    exception.insert(1, "ablation_mode", ablation_mode)
    exception.insert(2, "audit_scope", "unit_exception")
    exception.insert(3, "unit_id", str(unit_id))
    exception.insert(4, "unit_index", unit_index)
    return pd.concat((existing, exception), ignore_index=True, sort=False)


def _oof_shard_path(output_dir: Path, unit_index):
    return output_dir / _OOF_SHARD_DIRECTORY / f"unit_{unit_index:04d}.csv.gz"


def _write_oof_prediction_shard(output_dir: Path, unit_index, prediction_table):
    """Persist one unit's OOF predictions without rewriting other units' data."""
    shard_directory = output_dir / _OOF_SHARD_DIRECTORY
    shard_directory.mkdir(exist_ok=True)
    columns_path = output_dir / _OOF_COLUMNS_FILENAME
    columns = prediction_table.columns.tolist()
    if columns_path.exists():
        stored_columns = json.loads(columns_path.read_text())
        if stored_columns != columns:
            raise ValueError(
                "OOF prediction columns differ from existing shards; refusing "
                "to create an incompatible combined file"
            )
    else:
        _atomic_write_json(columns, columns_path)
    shard_path = _oof_shard_path(output_dir, unit_index)
    temporary_path = shard_path.with_name(f".{shard_path.name}.tmp")
    prediction_table.to_csv(
        temporary_path,
        index=False,
        header=False,
        compression="gzip",
    )
    temporary_path.replace(shard_path)


def _remove_oof_prediction_shard(output_dir: Path, unit_index):
    """Remove stale predictions for a refit unit that did not complete."""
    shard_path = _oof_shard_path(output_dir, unit_index)
    if shard_path.exists():
        shard_path.unlink()


def _finalize_oof_predictions(output_dir: Path):
    """Combine gzip member shards once, after all scheduled units are done."""
    shard_directory = output_dir / _OOF_SHARD_DIRECTORY
    columns_path = output_dir / _OOF_COLUMNS_FILENAME
    final_path = output_dir / _MODEL_OUTPUT_FILES["oof_predictions"]
    shard_paths = sorted(shard_directory.glob("unit_*.csv.gz")) if shard_directory.exists() else []
    if not shard_paths:
        if final_path.exists():
            final_path.unlink()
        return
    if not columns_path.exists():
        raise RuntimeError("OOF shards exist but their column manifest is missing")
    columns = json.loads(columns_path.read_text())
    temporary_path = final_path.with_name(f".{final_path.name}.tmp")
    with gzip.open(temporary_path, "wt", newline="") as stream:
        csv.writer(stream).writerow(columns)
    with temporary_path.open("ab") as destination:
        for shard_path in shard_paths:
            with shard_path.open("rb") as source:
                shutil.copyfileobj(source, destination)
    temporary_path.replace(final_path)


def _fit_movement_unit_task(
    unit_index,
    unit_id,
    covariates,
    spike_counts_unit,
    predictor_groups,
    model_name,
    model_spec,
    spline_config,
    outer_folds,
    inner_cv_config,
    penalty_bands,
    banded_optimizer_config,
    fixed_penalty_plan=None,
    penalty_source_model=None,
    ablation_mode=None,
    progress_queue=None,
    save_coefficients=False,
):
    """Fit one unit without writing shared artifacts; safe for joblib workers."""
    try:
        def progress_callback(event):
            if progress_queue is not None:
                progress_queue.put(
                    {
                        "model_name": model_name,
                        "unit_index": unit_index,
                        "unit_id": str(unit_id),
                        **event,
                    }
                )

        if penalty_source_model is not None and fixed_penalty_plan is None:
            raise RuntimeError(
                f"Penalty source {penalty_source_model!r} is incomplete for "
                f"unit {unit_index}"
            )
        unit_covariates = covariates
        unit_spike_counts = np.asarray(spike_counts_unit)
        if len(unit_covariates) != len(unit_spike_counts):
            raise ValueError("covariates and spike_counts_unit must have equal length")
        model_predictor_groups = _model_predictor_groups(
            predictor_groups,
            model_spec,
            ablation_mode=ablation_mode,
        )
        shuffled_predictors = (
            tuple(model_spec.get("ablated_predictors", ()))
            if ablation_mode == "shuffle"
            else ()
        )
        # Clipping is expected under training-derived spline bounds. It is
        # retained in the returned audit tables, but warning per clip would
        # make the multi-worker terminal display unreadable.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*clipped .*held-out samples.*",
                category=UserWarning,
                module=r"movement_encoding",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*dropped .*zero-variance training design columns.*",
                category=UserWarning,
                module=r"movement_encoding",
            )
            cv_result = run_nested_outer_cv_poisson_ridge(
                unit_covariates,
                unit_spike_counts,
                predictor_groups=model_predictor_groups,
                included_groups=model_spec["included_groups"],
                spline_config=spline_config,
                outer_folds=outer_folds,
                inner_cv_config=inner_cv_config,
                penalty_bands=penalty_bands,
                banded_optimizer_config=banded_optimizer_config,
                fixed_penalty_plan=fixed_penalty_plan,
                penalty_source_model=penalty_source_model,
                progress_callback=progress_callback,
                ablated_predictors=shuffled_predictors,
                shuffle_seed=SHUFFLE_SEED,
            )
        unit_score = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "unit_id": str(unit_id),
                    "unit_index": unit_index,
                    "status": "completed",
                    "error": None,
                    "mean_fold_log_likelihood_per_bin": (
                        cv_result.mean_fold_log_likelihood_per_bin
                    ),
                    "pooled_log_likelihood_per_bin": (
                        cv_result.pooled_log_likelihood_per_bin
                    ),
                    "total_log_likelihood": cv_result.total_log_likelihood,
                    "n_held_out_bins": cv_result.n_held_out_bins,
                    "mean_fold_deviance_explained": (
                        cv_result.mean_fold_deviance_explained
                    ),
                    "pooled_deviance_explained": (
                        cv_result.pooled_deviance_explained
                    ),
                    "n_input_bins": len(covariates),
                }
            ]
        )
        required_finite_fields = (
            "mean_fold_log_likelihood_per_bin",
            "pooled_log_likelihood_per_bin",
            "total_log_likelihood",
        )
        non_finite_fields = [
            field
            for field in required_finite_fields
            if not np.isfinite(unit_score.loc[0, field])
        ]
        if non_finite_fields:
            raise ValueError(
                "Non-finite score(s) after fitting "
                f"{sorted(non_finite_fields)}; likely optimizer divergence "
                "rather than a valid fit"
            )
        if ablation_mode is not None:
            unit_score.insert(1, "ablation_mode", ablation_mode)
        return {
            "unit_index": unit_index,
            "unit_id": unit_id,
            "unit_score": unit_score,
            "artifacts": _unit_artifacts(
                cv_result,
                unit_index,
                unit_id,
                model_name,
                ablation_mode=ablation_mode,
            ),
            "permutation_audit": cv_result.permutation_audit,
            "coefficients": pack_fold_coefficients(cv_result.outer_fits) if save_coefficients else None,
        }
    except Exception as error:
        if progress_queue is not None:
            progress_queue.put(
                {
                    "model_name": model_name,
                    "unit_index": unit_index,
                    "unit_id": str(unit_id),
                    "event": "unit_failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        error_score = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "unit_id": str(unit_id),
                    "unit_index": unit_index,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            ]
        )
        if ablation_mode is not None:
            error_score.insert(1, "ablation_mode", ablation_mode)
        return {
            "unit_index": unit_index,
            "unit_id": unit_id,
            "unit_score": error_score,
            "artifacts": {},
            "permutation_audit": pd.DataFrame(),
        }


def _monitor_outer_fold_progress(
    progress_queue,
    stop_event,
    fold_progress,
    n_outer_folds,
):
    """Update the parent-owned fold bar from worker progress messages."""
    while not stop_event.is_set() or not progress_queue.empty():
        try:
            event = progress_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        event_name = event["event"]
        unit_label = f"unit {event['unit_index']} ({event['unit_id']})"
        if event_name == "outer_fold_started":
            fold_progress.set_postfix_str(
                f"{unit_label}, fold {event['outer_fold'] + 1}/{n_outer_folds}"
            )
        elif event_name == "outer_fold_completed":
            fold_progress.update(1)
            fold_progress.set_postfix_str(
                f"{unit_label}, completed fold {event['outer_fold'] + 1}/{n_outer_folds}"
            )
        elif event_name == "unit_failed":
            fold_progress.set_postfix_str(f"{unit_label}, failed")


def _ordered_model_names(model_names, registry):
    """Place requested penalty-source models before their dependents."""
    requested = list(dict.fromkeys(model_names))
    ordered = []
    remaining = list(requested)
    while remaining:
        progressed = False
        for model_name in list(remaining):
            source = registry[model_name].get("penalty_source_model")
            if source not in remaining or source in ordered:
                ordered.append(model_name)
                remaining.remove(model_name)
                progressed = True
        if not progressed:
            raise ValueError(
                "Penalty-source dependencies contain a cycle: "
                f"{remaining}"
            )
    return ordered



def _load_fixed_penalty_plans(output_root, source_model_name, bundle):
    """Load fold-specific penalties from one completed source-model run."""
    source_dir = Path(output_root) / source_model_name
    manifest_path = source_dir / "run_manifest.json"
    fold_scores_path = source_dir / _MODEL_OUTPUT_FILES["fold_scores"]
    unit_scores_path = source_dir / _MODEL_OUTPUT_FILES["unit_scores"]
    for required_path in (manifest_path, fold_scores_path, unit_scores_path):
        if not required_path.exists():
            raise FileNotFoundError(
                "Inherited penalties require a completed source-model run; "
                f"missing {required_path}"
            )
    source_manifest = json.loads(manifest_path.read_text())
    source_spec = source_manifest.get("run_spec", {})
    if source_spec.get("model_name") != source_model_name:
        raise ValueError("Penalty-source manifest has the wrong model name")
    if source_spec.get("input_bundle_manifest_sha256") != bundle["manifest_sha256"]:
        raise ValueError("Penalty-source run used a different input bundle manifest")
    expected_bands = tuple(bundle["manifest"]["penalty_bands"])
    penalty_columns = [f"penalty_{name}" for name in expected_bands]
    fold_scores = _read_output_table(fold_scores_path)
    unit_scores = _read_output_table(unit_scores_path)
    required_columns = {
        "unit_index", "outer_fold", "selected_null_model", *penalty_columns
    }
    missing_columns = required_columns.difference(fold_scores.columns)
    if missing_columns:
        raise ValueError(
            "Penalty-source fold scores are missing required columns: "
            f"{sorted(missing_columns)}"
        )
    completed_units = set(
        unit_scores.loc[
            unit_scores["status"].eq("completed"), "unit_index"
        ].astype(int)
    )
    expected_folds = {
        int(fold["fold"])
        for fold in bundle["manifest"]["outer_trial_folds"]
    }
    plans = {}
    for unit_index in range(len(bundle["unit_ids"])):
        if unit_index not in completed_units:
            continue
        unit_rows = fold_scores.loc[
            fold_scores["unit_index"].astype(int).eq(unit_index)
        ]
        if unit_rows["outer_fold"].duplicated().any():
            raise ValueError(
                f"Penalty source has duplicate folds for unit {unit_index}"
            )
        found_folds = set(unit_rows["outer_fold"].astype(int))
        if found_folds != expected_folds:
            raise ValueError(
                f"Penalty source has incomplete folds for unit {unit_index}; "
                f"found={sorted(found_folds)}, expected={sorted(expected_folds)}"
            )
        null_values = unit_rows["selected_null_model"]
        if null_values.dtype == object:
            null_flags = null_values.astype(str).str.lower().eq("true")
        else:
            null_flags = null_values.fillna(False).astype(bool)
        if null_flags.any():
            raise ValueError(
                f"Penalty source contains a null-selected fold for unit "
                f"{unit_index}; finite banded fits are required"
            )
        plans[unit_index] = {
            int(row["outer_fold"]): {
                "penalties": {
                    band_name: float(row[f"penalty_{band_name}"])
                    for band_name in expected_bands
                },
                "selected_null_model": False,
            }
            for _, row in unit_rows.iterrows()
        }
    return plans, _sha256_file(manifest_path)



def fit_movement_models_from_bundle(
    bundle,
    output_root,
    model_names: Sequence,
    refit: bool = False,
    retry_failed: bool = False,
    registry: Mapping | None = None,
    n_jobs: int = -1,
    show_progress: bool = True,
    ablation_mode: str = "shuffle",
    save_coefficients: bool = False,
):
    """Fit requested registry models across all units of one frozen bundle.

    A normal restart skips every unit with a saved terminal result (completed
    or failed), unless ``refit`` is true. Set ``retry_failed`` to rerun only
    previously failed units; completed units still require ``refit`` to rerun.
    Workers fit independent units in parallel, while this parent process alone
    writes shared output files. Existing outputs are used only when their full
    saved run specification exactly matches the requested settings.

    ``save_coefficients`` persists numerical full/task_movement fold weights
    and feature metadata. Existing completed units without these artifacts
    require an explicit refit or a new output root; they are never silently fit.
    """
    predictor_groups = bundle["manifest"]["predictor_groups"]
    registry = dict(registry or build_movement_model_registry(predictor_groups))
    unknown_models = set(model_names).difference(registry)
    if unknown_models:
        raise KeyError(f"Unknown movement model names: {sorted(unknown_models)}")
    if not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer")
    if ablation_mode not in {"drop", "shuffle"}:
        raise ValueError("ablation_mode must be either 'drop' or 'shuffle'")
    output_root = Path(output_root)
    model_names = _ordered_model_names(model_names, registry)
    results = []
    for model_name in model_names:
        _safe_model_name(model_name)
        model_spec = registry[model_name]
        penalty_source_model = model_spec.get("penalty_source_model")
        fixed_penalty_plans = None
        penalty_source_manifest_sha256 = None
        if penalty_source_model is not None:
            fixed_penalty_plans, penalty_source_manifest_sha256 = (
                _load_fixed_penalty_plans(
                    output_root, penalty_source_model, bundle
                )
            )
        model_ablation_mode = (
            ablation_mode if model_spec.get("ablated_predictors") else None
        )
        output_name = (
            f"{model_name}__{model_ablation_mode}"
            if model_ablation_mode is not None else model_name
        )
        output_dir = output_root / output_name
        save_model_coefficients = save_coefficients and model_name in ("full", "task_movement") and model_ablation_mode is None
        run_spec = _model_run_spec(
            bundle,
            model_name,
            model_spec,
            ablation_mode=model_ablation_mode,
            penalty_source_manifest_sha256=(
                penalty_source_manifest_sha256
            ),
        )
        _prepare_model_output_dir(output_dir, run_spec, n_jobs=n_jobs)
        tables = {
            table_name: _read_output_table(output_dir / filename)
            for table_name, filename in _MODEL_OUTPUT_FILES.items()
            if table_name != "oof_predictions"
        }
        permutation_audit = (
            _read_output_table(output_dir / _PERMUTATION_AUDIT_FILENAME)
            if model_ablation_mode == "shuffle" else pd.DataFrame()
        )
        skipped_unit_indices = set()
        if not refit and not tables["unit_scores"].empty:
            saved_unit_indices = set(
                tables["unit_scores"]["unit_index"].astype(int)
            )
            completed_unit_indices = set(
                tables["unit_scores"].loc[
                    tables["unit_scores"]["status"].eq("completed"), "unit_index"
                ].astype(int)
            )
            completed_with_shards = {
                unit_index
                for unit_index in completed_unit_indices
                if _oof_shard_path(output_dir, unit_index).exists()
            }
            if retry_failed:
                skipped_unit_indices = completed_with_shards
            else:
                # A completed unit without its OOF shard is incomplete and
                # must be rerun; a saved error is a terminal result by default.
                skipped_unit_indices = saved_unit_indices.difference(
                    completed_unit_indices.difference(completed_with_shards)
                )

        if save_model_coefficients and not refit and not tables["unit_scores"].empty:
            missing_coefficients = [
                int(u) for u in tables["unit_scores"].loc[
                    tables["unit_scores"].status.eq("completed"), "unit_index"]
                if int(u) in skipped_unit_indices and not (
                    output_dir / "coefficients" / f"unit_{int(u):04d}.json").exists()
            ]
            if missing_coefficients:
                raise ValueError(
                    f"{model_name}: {len(missing_coefficients)} completed units have no saved coefficients. "
                    "Use a new --output-root or explicitly --refit with --save-coefficients.")
        pending_units = [
            (unit_index, unit_id)
            for unit_index, unit_id in enumerate(bundle["unit_ids"])
            if unit_index not in skipped_unit_indices
        ]
        n_outer_folds = len(bundle["manifest"]["outer_trial_folds"])
        saved_fold_count = min(
            len(tables["fold_scores"]),
            len(bundle["unit_ids"]) * n_outer_folds,
        )
        unit_progress = tqdm(
            total=len(bundle["unit_ids"]),
            initial=len(skipped_unit_indices),
            desc=f"{model_name}: units",
            unit="unit",
            disable=not show_progress,
        )
        fold_progress = tqdm(
            total=saved_fold_count + len(pending_units) * n_outer_folds,
            initial=saved_fold_count,
            desc=f"{model_name}: outer folds",
            unit="fold",
            leave=False,
            disable=not show_progress,
        )
        manager = None
        progress_queue = None
        progress_stop_event = None
        monitor = None
        if show_progress and pending_units:
            try:
                manager = Manager()
                progress_queue = manager.Queue()
            except (EOFError, OSError, PermissionError) as error:
                warnings.warn(
                    "Could not create the live worker-progress queue; "
                    "continuing with completed-unit progress only. "
                    f"({type(error).__name__}: {error})"
                )
        if progress_queue is not None:
            progress_stop_event = threading.Event()
            monitor = threading.Thread(
                target=_monitor_outer_fold_progress,
                args=(
                    progress_queue,
                    progress_stop_event,
                    fold_progress,
                    n_outer_folds,
                ),
                daemon=True,
            )
            monitor.start()
        try:
            worker_results = Parallel(
                n_jobs=n_jobs,
                return_as="generator_unordered",
            )(
                delayed(_fit_movement_unit_task)(
                    unit_index,
                    unit_id,
                    bundle["covariates"],
                    bundle["spike_counts"][unit_index],
                    predictor_groups,
                    model_name,
                    model_spec,
                    bundle["manifest"]["spline_config"],
                    bundle["manifest"]["outer_trial_folds"],
                    bundle["manifest"]["inner_cv_config"],
                    bundle["manifest"]["penalty_bands"],
                    bundle["manifest"]["banded_optimizer_config"],
                    (
                        None
                        if fixed_penalty_plans is None
                        else fixed_penalty_plans.get(unit_index)
                    ),
                    penalty_source_model,
                    model_ablation_mode,
                    progress_queue,
                    save_coefficients=save_model_coefficients,
                )
                for unit_index, unit_id in pending_units
            )
            for worker_result in worker_results:
                unit_index = worker_result["unit_index"]
                unit_id = worker_result["unit_id"]
                coefficient_path = output_dir / "coefficients" / f"unit_{unit_index:04d}.json"
                # A rerun must never leave stale weights paired with new OOF rows.
                coefficient_path.unlink(missing_ok=True)
                _remove_oof_prediction_shard(output_dir, unit_index)
                for table_name in tables:
                    tables[table_name] = _replace_unit_rows(
                        tables[table_name], None, unit_index
                    )
                tables["unit_scores"] = _replace_unit_rows(
                    tables["unit_scores"], worker_result["unit_score"], unit_index
                )
                artifacts = worker_result["artifacts"]
                if "oof_predictions" in artifacts:
                    _write_oof_prediction_shard(
                        output_dir,
                        unit_index,
                        artifacts["oof_predictions"],
                    )
                if worker_result.get("coefficients") is not None:
                    coefficient_path.parent.mkdir(exist_ok=True)
                    _atomic_write_json(dict(
                        format_version=1, unit_index=int(unit_index), unit_id=str(unit_id),
                        input_bundle_manifest_sha256=bundle["manifest_sha256"],
                        run_spec=run_spec, folds=worker_result["coefficients"]), coefficient_path)
                for table_name, artifact in artifacts.items():
                    if table_name == "oof_predictions":
                        continue
                    tables[table_name] = _replace_unit_rows(
                        tables[table_name], artifact, unit_index
                    )
                if model_ablation_mode == "shuffle":
                    permutation_audit = _merge_permutation_audit(
                        permutation_audit,
                        worker_result["permutation_audit"],
                        model_name=model_name,
                        ablation_mode=model_ablation_mode,
                        unit_index=unit_index,
                        unit_id=unit_id,
                    )
                    _atomic_write_dataframe(
                        permutation_audit,
                        output_dir / _PERMUTATION_AUDIT_FILENAME,
                    )
                status = worker_result["unit_score"].iloc[0]["status"]
                unit_progress.update(1)
                if progress_queue is None:
                    fold_progress.update(n_outer_folds)
                    fold_progress.set_postfix_str(
                        f"unit {unit_index} ({unit_id}), {status}"
                    )
                if status == "completed":
                    tqdm.write(f"{model_name}: unit {unit_index} ({unit_id}) completed")
                else:
                    error = worker_result["unit_score"].iloc[0]["error"]
                    tqdm.write(
                        f"{model_name}: unit {unit_index} ({unit_id}) failed: {error}"
                    )
                _write_model_outputs(output_dir, tables)
        finally:
            if progress_stop_event is not None:
                progress_stop_event.set()
            if monitor is not None:
                monitor.join(timeout=2)
            if manager is not None:
                manager.shutdown()
            unit_progress.close()
            fold_progress.close()

        _finalize_oof_predictions(output_dir)

        completed = 0
        if not tables["unit_scores"].empty:
            completed = int(tables["unit_scores"]["status"].eq("completed").sum())
        results.append(
            {
                "model_name": model_name,
                "output_dir": str(output_dir),
                "ablation_mode": model_ablation_mode,
                "n_completed_units": completed,
                "n_total_units": len(bundle["unit_ids"]),
            }
        )
    return pd.DataFrame(results)


def default_outer_folds(covariates, n_splits=5, seed=42):
    """Create the approved outer folds when constructing an input bundle."""
    return make_contiguous_outer_trial_folds(
        covariates["trial_number"], n_splits=n_splits, seed=seed
    )
