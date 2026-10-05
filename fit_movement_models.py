#!/usr/bin/env python3
"""Fit selected nested movement-encoding models from a frozen input bundle.

First export the notebook's already-derived data with
``export_movement_encoding_input_bundle``. Then run, for example:

    conda run -n spyglass python notebooks/fit_movement_models.py \
        --input-dir /path/to/movement_encoding_input \
        --models intercept_only task_only movement_only task_movement

For a bundle exported for a selected term group, run the complete matched
family with ``--term-model-group``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from movement_encoding import (
    build_movement_model_registry,
    build_movement_term_model_group,
)
from movement_model_runner import (
    fit_movement_models_from_bundle,
    load_movement_encoding_input_bundle,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit nested movement-encoding models from an input bundle."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory created by export_movement_encoding_input_bundle().",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Result directory (default: <input-dir>/model_fits).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Registry model keys to fit. Defaults to the four initial models, "
            "or to the complete selected-term family with --term-model-group."
        ),
    )
    parser.add_argument(
        "--term-model-group",
        action="store_true",
        help=(
            "Fit a self-contained full/intercept/alone/reduced family using "
            "the term bands recorded in the input bundle."
        ),
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help=(
            "Refit every unit only if the existing run manifest exactly "
            "matches the requested parameters."
        ),
    )
    parser.add_argument(
        "--save-coefficients",
        action="store_true",
        help=("Save full/task_movement outer-fold weights and transformation metadata "
              "for analysis without regression refitting. Existing completed fits "
              "without coefficients require --refit or a new --output-root."),
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Retry units with a previously saved error. Ordinary restarts skip "
            "both completed and failed units."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel unit-fitting workers (default: -1, all CPU cores; 1 is sequential).",
    )
    parser.add_argument(
        "--custom-model",
        action="append",
        default=[],
        metavar="NAME=GROUP,GROUP",
        help=(
            "Add a registry model without editing code; repeat as needed. "
            "Example: --custom-model task_without_limb=task,translation,turning"
        ),
    )
    parser.add_argument(
        "--custom-ablation",
        action="append",
        default=[],
        metavar="NAME=PREDICTOR_OR_GROUP,...",
        help=(
            "Define predictors or whole groups ablated from a custom model; "
            "repeat as needed. Example: --custom-ablation no_speed=speed"
        ),
    )
    parser.add_argument(
        "--custom-penalty-source",
        action="append",
        default=[],
        metavar="NAME=SOURCE_MODEL",
        help=(
            "Reuse fold-wise penalties from an existing model for a custom "
            "model; repeat as needed. Example: "
            "--custom-penalty-source limb_alone=task_movement"
        ),
    )
    parser.add_argument(
        "--ablation-mode",
        choices=("drop", "shuffle"),
        default="shuffle",
        help=(
            "How declared ablations are fit (default: shuffle). Shuffling uses "
            "the fixed internal seed 42."
        ),
    )
    return parser.parse_args()


def parse_custom_models(custom_model_args):
    """Parse repeated NAME=GROUP,GROUP command-line model definitions."""
    custom_models = {}
    for definition in custom_model_args:
        if definition.count("=") != 1:
            raise ValueError(
                f"Custom model must be NAME=GROUP,GROUP, not {definition!r}"
            )
        name, group_text = definition.split("=", 1)
        groups = tuple(
            group.strip() for group in group_text.split(",") if group.strip()
        )
        if not name or not groups:
            raise ValueError(
                f"Custom model must have a name and at least one group: "
                f"{definition!r}"
            )
        if name in custom_models:
            raise ValueError(
                f"Custom model was specified more than once: {name!r}"
            )
        custom_models[name] = groups
    return custom_models


def attach_custom_ablations(
    custom_models, custom_ablation_args, predictor_groups
):
    """Attach validated predictor/group ablations to custom model specs."""
    custom_specs = {
        name: {"included_groups": tuple(groups), "label": name}
        for name, groups in custom_models.items()
    }
    seen_names = set()
    for definition in custom_ablation_args:
        if definition.count("=") != 1:
            raise ValueError(
                "Custom ablation must be NAME=PREDICTOR_OR_GROUP,..., "
                f"not {definition!r}"
            )
        name, target_text = definition.split("=", 1)
        targets = tuple(
            target.strip() for target in target_text.split(",") if target.strip()
        )
        if not name or not targets:
            raise ValueError(
                f"Custom ablation must have a name and target: {definition!r}"
            )
        if name in seen_names:
            raise ValueError(
                f"Custom ablation was specified more than once: {name!r}"
            )
        seen_names.add(name)
        if name not in custom_specs:
            raise ValueError(
                f"Custom ablation {name!r} has no matching --custom-model"
            )
        included_groups = custom_specs[name]["included_groups"]
        unknown_groups = set(included_groups).difference(predictor_groups)
        if unknown_groups:
            raise ValueError(
                f"Custom model {name!r} uses unavailable groups: "
                f"{sorted(unknown_groups)}"
            )
        selected_predictors = {
            predictor
            for group in included_groups
            for predictor in predictor_groups[group]
        }
        ablated = []
        for target in targets:
            if target in predictor_groups:
                target_predictors = predictor_groups[target]
            elif target in selected_predictors:
                target_predictors = (target,)
            else:
                raise ValueError(
                    f"Custom ablation {name!r} has unavailable target "
                    f"{target!r}"
                )
            for predictor in target_predictors:
                if predictor == "Intercept":
                    raise ValueError("Intercept cannot be ablated")
                if predictor not in selected_predictors:
                    raise ValueError(
                        f"Custom ablation {name!r} targets {predictor!r}, "
                        "which is not in its included groups"
                    )
                if predictor not in ablated:
                    ablated.append(predictor)
        if "branch_id" in ablated and "branch_pos_scaled" not in ablated:
            if "branch_pos_scaled" not in selected_predictors:
                raise ValueError(
                    "A branch_id ablation also requires branch_pos_scaled in "
                    "the custom model"
                )
            ablated.append("branch_pos_scaled")
        custom_specs[name]["ablated_predictors"] = tuple(ablated)
    return custom_specs


def attach_custom_penalty_sources(custom_specs, custom_penalty_source_args):
    """Attach explicitly requested inherited-penalty sources to custom specs."""
    custom_specs = {name: dict(spec) for name, spec in custom_specs.items()}
    seen_names = set()
    for definition in custom_penalty_source_args:
        if definition.count("=") != 1:
            raise ValueError(
                "Custom penalty source must be NAME=SOURCE_MODEL, "
                f"not {definition!r}"
            )
        name, source_model = (
            value.strip() for value in definition.split("=", 1)
        )
        if not name or not source_model:
            raise ValueError(
                "Custom penalty source must have a model name and source: "
                f"{definition!r}"
            )
        if name in seen_names:
            raise ValueError(
                f"Custom penalty source was specified more than once: {name!r}"
            )
        seen_names.add(name)
        if name not in custom_specs:
            raise ValueError(
                f"Custom penalty source {name!r} has no matching --custom-model"
            )
        if source_model == name:
            raise ValueError(
                f"Custom model {name!r} cannot inherit penalties from itself"
            )
        custom_specs[name]["penalty_source_model"] = source_model
    return custom_specs


def main():
    args = parse_args()
    bundle = load_movement_encoding_input_bundle(args.input_dir)
    if args.term_model_group:
        if args.custom_model or args.custom_ablation or args.custom_penalty_source:
            raise ValueError(
                "--term-model-group cannot be combined with custom model options"
            )
        term_group = build_movement_term_model_group(
            bundle["manifest"]["predictor_groups"],
            terms=tuple(bundle["manifest"]["penalty_bands"]),
        )
        recorded_penalty_bands = {
            name: tuple(predictors)
            for name, predictors in bundle["manifest"]["penalty_bands"].items()
        }
        if term_group["penalty_bands"] != recorded_penalty_bands:
            raise ValueError(
                "The input bundle's penalty bands do not match its selected "
                "term definitions"
            )
        registry = term_group["registry"]
        model_names = args.models or term_group["model_names"]
    else:
        custom_models = parse_custom_models(args.custom_model)
        custom_specs = attach_custom_ablations(
            custom_models,
            args.custom_ablation,
            bundle["manifest"]["predictor_groups"],
        )
        custom_specs = attach_custom_penalty_sources(
            custom_specs,
            args.custom_penalty_source,
        )
        registry = build_movement_model_registry(
            bundle["manifest"]["predictor_groups"],
            additional_models=custom_specs,
        )
        model_names = args.models or [
            "task_movement",
            "intercept_only",
            "task_only",
            "movement_only",
        ]
    output_root = args.output_root or args.input_dir / "model_fits"
    n_bins = len(bundle["covariates"])
    n_units = len(bundle["unit_ids"])
    model_list = ", ".join(model_names)
    print(f"Input: {n_bins:,} bins × {n_units} units")
    print(f"Models: {model_list}")
    print(f"Output: {output_root}")
    print(f"Parallel workers: {args.n_jobs}")
    print(f"Ablation mode: {args.ablation_mode}")
    if args.refit:
        print("Resume policy: refit all units")
    elif args.retry_failed:
        print("Resume policy: retry failed units; keep completed units")
    else:
        print("Resume policy: keep all saved terminal unit results")
    summary = fit_movement_models_from_bundle(
        bundle,
        output_root=output_root,
        model_names=model_names,
        refit=args.refit,
        retry_failed=args.retry_failed,
        registry=registry,
        n_jobs=args.n_jobs,
        ablation_mode=args.ablation_mode,
        save_coefficients=args.save_coefficients,
    )
    print("\nRun summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
