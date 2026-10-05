"""Portable full-model coefficients with frozen-bundle provenance.

Patsy DesignInfo cannot be pickled. Save numerical transformation metadata and
recreate the feature bases from the exact frozen outer-training samples on
load. This never estimates regression weights or selects penalties.
"""
from dataclasses import asdict, dataclass
from importlib import import_module
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd

__all__ = ["pack_fold_coefficients", "load_atlas_coefficients"]


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _transform_metadata(transformer):
    return _plain(dict(
        included_groups=transformer.included_groups,
        direct_columns=transformer.direct_columns,
        categorical_columns=transformer.categorical_columns,
        categorical_levels=transformer.categorical_levels,
        categorical_column_names=transformer.categorical_column_names,
        retained_columns=transformer.retained_columns,
        means=transformer.means.to_dict(), scales=transformer.scales.to_dict(),
        splines={p: dict(spec=asdict(b.spec), columns=b.column_names)
                 for p, b in transformer.spline_bases.items()},
        branch_splines={p: dict(spec=asdict(b.spec), columns=b.column_names)
                        for p, b in transformer.branch_spline_bases.items()}))


def pack_fold_coefficients(fits):
    """Convert completed full fits to JSON data before crossing worker boundaries."""
    rows = []
    for fold, fit in fits.items():
        model = fit.model
        rows.append(dict(
            fold=_plain(fold), train_trial_ids=_plain(fit.train_trial_ids),
            feature_columns=list(model.feature_columns),
            coefficients=None if model.estimator is None else model.estimator.coef_.tolist(),
            intercept=None if model.estimator is None else float(model.estimator.intercept_),
            intercept_only_mean=model.intercept_only_mean,
            feature_scales=None if model.feature_scales is None else model.feature_scales.to_dict(),
            transformer=_transform_metadata(fit.transformer)))
    return _plain(rows)


@dataclass
class _SavedEstimator:
    coef_: np.ndarray
    intercept_: float

    def predict(self, features):
        return np.exp(self.intercept_ + np.asarray(features) @ self.coef_)


def load_atlas_coefficients(atlas, unit, folder=None):
    """Load a unit JSON, rebuild only feature bases, and verify training metadata.

    ``folder`` accepts model_fits, its full-model subfolder, or the coefficients
    directory itself. Prediction verification follows in full_model_components.
    """
    unit = int(unit)
    folder = Path(folder) if folder else atlas.results_root / atlas.full_name
    filename = f"unit_{unit:04d}.json"
    candidates = [folder / filename, folder / "coefficients" / filename,
                  folder / atlas.full_name / "coefficients" / filename,
                  folder / "model_fits" / atlas.full_name / "coefficients" / filename]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No saved coefficients for cell {unit} under {folder}. Run fitting with --save-coefficients.")
    artifact = json.loads(path.read_text())
    if (artifact.get("format_version") != 1 or artifact.get("unit_index") != unit
            or str(artifact.get("unit_id")) != str(atlas.unit_ids[unit])
            or artifact.get("input_bundle_manifest_sha256") != atlas.bundle_hash):
        raise ValueError("Coefficient artifact does not match this unit/frozen bundle.")
    spec = atlas.specs[atlas.full_name]
    if artifact["run_spec"] != spec:
        raise ValueError("Coefficient artifact full-model specification differs from the atlas.")
    enc = import_module(".movement_encoding", __package__) if __package__ else import_module("movement_encoding")
    excluded = set(spec["model_spec"].get("excluded_predictors", []))
    groups = {g: tuple(p for p in ps if p not in excluded) for g, ps in spec["predictor_groups"].items()}
    fits = {}
    for row in artifact["folds"]:
        fold = row["fold"]
        if fold not in atlas.fold_masks or fold in fits:
            raise ValueError("Invalid or duplicate coefficient fold.")
        train, _ = atlas.fold_masks[fold]
        expected_trials = set(atlas.covariates.loc[train, "trial_number"].unique())
        if set(row["train_trial_ids"]) != expected_trials:
            raise ValueError("Coefficient training trials differ from atlas folds.")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Outer fold .*: dropped", category=UserWarning)
            _, transformer = enc.fit_design_matrix_transformer(
                atlas.covariates.loc[train], predictor_groups=groups,
                included_groups=spec["model_spec"]["included_groups"],
                spline_config=spec["spline_config"], fold=fold)
        actual, expected = _transform_metadata(transformer), row["transformer"]
        for key in ("means", "scales"):
            a, b = actual.pop(key), expected[key]
            if set(a) != set(b) or not np.allclose([a[k] for k in b], list(b.values()), rtol=1e-10, atol=1e-12):
                raise ValueError(f"Fold {fold}: saved transformation {key} cannot be reproduced.")
        if actual != {k: v for k, v in expected.items() if k not in ("means", "scales")}:
            raise ValueError(f"Fold {fold}: saved feature basis cannot be reproduced.")
        transformer.means = pd.Series(expected["means"], dtype=float)
        transformer.scales = pd.Series(expected["scales"], dtype=float)
        model = enc.FittedPoissonRidgeModel(
            estimator=None if row["coefficients"] is None else _SavedEstimator(
                np.asarray(row["coefficients"], float), float(row["intercept"])),
            feature_columns=tuple(row["feature_columns"]),
            intercept_only_mean=row["intercept_only_mean"],
            feature_scales=None if row["feature_scales"] is None else pd.Series(row["feature_scales"], dtype=float))
        fits[fold] = SimpleNamespace(model=model, transformer=transformer,
                                     train_trial_ids=tuple(row["train_trial_ids"]))
    if set(fits) != set(atlas.fold_masks):
        raise ValueError("Coefficient artifact is missing outer folds.")
    return fits
