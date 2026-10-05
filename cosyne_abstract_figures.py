"""Compact, axis-aware figures for the COSYNE movement-encoding abstract.

The notebook owns data preparation. This module owns visual styling, panel
drawing, composite layout, and deterministic export. See ``figures_plan.md``
before changing figure content or semantics.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import textwrap
from typing import Callable, Mapping, Sequence
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from movement_visualization import plot_progress_with_ci


COLORS = {
    "task": "#3B6FB6",
    "movement": "#D55E00",
    "shared": "#7A5195",
    "translation": "#009E73",
    "turning": "#E69F00",
    "limb": "#CC79A7",
    "full": "#30343B",
    "intercept": "#9AA0A6",
    "grid": "#D9DDE3",
    "paper": "#FFFFFF",
}

MODEL_COLORS = {
    "intercept_only": COLORS["intercept"],
    "task_only": COLORS["task"],
    "movement_only": COLORS["movement"],
    "task_movement": COLORS["full"],
}

MODEL_LABELS = {
    "intercept_only": "Intercept",
    "task_only": "Task",
    "movement_only": "Movement",
    "task_movement": "Full",
}

TERM_GROUPS = {
    "task": (
        ("Trial type", "trial_type"),
        ("Position", "position"),
        ("Progress", "progress"),
    ),
    "movement": (
        ("Translation", "translation"),
        ("Turning", "turning"),
        ("Limb", "limb"),
    ),
}

DEFAULT_PROGRESS_SPECS = (
    ("speed", "Speed (cm/s)", COLORS["translation"], {}),
    (
        "tangential_acceleration_cm_s2",
        "Acceleration (cm/s²)",
        COLORS["translation"],
        {},
    ),
    (
        "abs_body_angular_velocity_rad_s",
        "Angular speed (rad/s)",
        COLORS["turning"],
        {"window_size": 0.05, "step_size": 0.01},
    ),
    (
        "mean_fore_hind_spacing",
        "Fore–hind spacing (cm)",
        COLORS["limb"],
        {},
    ),
)

DEFAULT_FIGURE1_BUNDLE = Path(
    "/media/labuser/NA_1_2025/spyglass/wilbur/analysis/"
    "movement_encoding_banded/covariates.pkl"
)
DEFAULT_FIGURE1_MULTI_ANIMAL_R2 = (
    Path(__file__).resolve().parent
    / "abstract_figures/figure_1C_multi_animal_progress_r2_matched.csv"
)

FIGURE1_ANIMALS = ("Wilbur", "Senor", "J")
FIGURE1_ANIMAL_COLORS = {
    "Wilbur": "#3B6FB6",
    "Senor": "#D55E00",
    "J": "#009E73",
}
FIGURE1_ANIMAL_LABELS = {"Wilbur": "Rat W", "Senor": "Rat S", "J": "Rat J"}

FIGURE1_PREDICTABILITY_SPECS = (
    ("speed", "Speed"),
    ("tangential_acceleration_cm_s2", "Acceleration"),
    ("abs_body_angular_velocity_rad_s", "Angular speed"),
    ("gait_phase_sin", "Gait phase (sin)"),
    ("gait_phase_cos", "Gait phase (cos)"),
    ("lateral_phase_sin", "Lateral phase (sin)"),
    ("lateral_phase_cos", "Lateral phase (cos)"),
    ("relative_paw_vigor", "Relative paw vigor"),
    ("fore_minus_hind_vigor", "Fore − hind vigor"),
    ("mean_stance_width", "Mean stance width"),
    ("fore_minus_hind_width", "Fore − hind width"),
    ("mean_fore_hind_spacing", "Fore–hind spacing"),
    ("fore_hind_spacing_asymmetry", "Spacing asymmetry"),
)

FIGURE2_MODEL_PAIRS = {
    "trial_type": ("trial_type_alone", "task_movement_minus_trial_type"),
    "position": ("position_alone", "task_movement_minus_position"),
    "progress": ("progress_alone", "task_movement_minus_progress"),
    "translation": ("translation_alone", "task_movement_minus_translation"),
    "turning": ("turning_alone", "task_movement_minus_turning"),
    "limb": ("limb_alone", "task_movement_minus_limb"),
}
FIGURE2_REQUIRED_MODELS = (
    *MODEL_COLORS,
    *(model for pair in FIGURE2_MODEL_PAIRS.values() for model in pair),
)


def _style_parameters(font_size: float = 10.0) -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size - 0.5,
        "ytick.labelsize": font_size - 0.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.fontsize": font_size - 0.5,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


@contextmanager
def publication_style(font_size: float = 10.0):
    """Temporarily apply the abstract's final-size plotting style."""
    with mpl.rc_context(_style_parameters(font_size)):
        yield


def add_panel_label(
    ax: Axes,
    label: str,
    *,
    x: float = -0.16,
    y: float = 1.08,
) -> None:
    """Place a consistent panel label relative to an axis."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def _despine_and_grid(ax: Axes, *, axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def _finite_percent(values) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    return 100.0 * array[np.isfinite(array)]


def _validate_columns(table: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns).difference(table.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _feature_color(label: str) -> str:
    lower = label.lower()
    if any(word in lower for word in ("turn", "angular")):
        return COLORS["turning"]
    if any(word in lower for word in ("speed", "acceleration", "translation")):
        return COLORS["translation"]
    return COLORS["limb"]


def load_multi_animal_progress_r2(
    path: str | Path = DEFAULT_FIGURE1_MULTI_ANIMAL_R2,
) -> pd.DataFrame:
    """Load the completed animal-by-variable Figure 1C score table."""
    scores = pd.read_csv(path)
    _validate_columns(
        scores, ["animal", "variable", "combined_r2"], "Figure 1C scores"
    )
    expected = {
        (animal, variable)
        for animal in FIGURE1_ANIMALS
        for variable, _ in FIGURE1_PREDICTABILITY_SPECS
    }
    observed = set(zip(scores["animal"], scores["variable"]))
    if len(scores) != len(expected) or observed != expected:
        raise ValueError("Figure 1C requires one score for every animal and variable")
    if not np.isfinite(scores["combined_r2"]).all():
        raise ValueError("Figure 1C contains non-finite R² values")
    return scores


def load_multi_animal_progress_profiles() -> pd.DataFrame:
    """Load the saved Figure 1B progress profiles for all three animals."""
    root = Path(__file__).resolve().parent / "abstract_figures"
    profiles = pd.concat(
        [
            pd.read_csv(root / f"figure_1B_{animal.lower()}_progress_profiles.csv")
            for animal in FIGURE1_ANIMALS
        ],
        ignore_index=True,
    )
    _validate_columns(
        profiles, ["animal", "variable", "progress", "mean", "n_trials"],
        "Figure 1B profiles",
    )
    expected = {
        (animal, variable)
        for animal in FIGURE1_ANIMALS
        for variable, _, _, _ in DEFAULT_PROGRESS_SPECS
    }
    observed = set(zip(profiles["animal"], profiles["variable"]))
    if observed != expected:
        raise ValueError("Figure 1B profiles must cover all animals and variables")
    return profiles


def mean_animal_progress_r2(scores: pd.DataFrame) -> pd.Series:
    """Average each variable's held-out R² equally across the three animals."""
    _validate_columns(
        scores, ["animal", "variable", "combined_r2"], "Figure 1C scores"
    )
    table = scores.pivot(index="variable", columns="animal", values="combined_r2")
    table = table.loc[[column for column, _ in FIGURE1_PREDICTABILITY_SPECS], FIGURE1_ANIMALS]
    result = table.mean(axis=1)
    result.index = [label for _, label in FIGURE1_PREDICTABILITY_SPECS]
    return result


def prepare_real_figure1_data(
    covariate_path: str | Path = DEFAULT_FIGURE1_BUNDLE,
    *,
    compute_predictability: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Load saved real covariates for Figure 1B, optionally scoring Wilbur.

    The bundle is the run-only, outbound/inbound, complete-case covariate table
    exported by ``movement_groups_encoding.ipynb``. Progress predictability is
    recomputed with the same trial-grouped five-fold CV helper used there.
    """
    from movement_features import add_paw_position_displacements

    covariate_path = Path(covariate_path)
    if not covariate_path.exists():
        raise FileNotFoundError(f"Missing movement covariates: {covariate_path}")
    behavior_data = pd.read_pickle(covariate_path).copy()
    required = {
        "trial_number",
        "trial_type",
        "trial_progress_distance",
        "position_x",
        "position_y",
        "zone",
        "body_angular_velocity_rad_s",
        *(column for column, _ in FIGURE1_PREDICTABILITY_SPECS),
    }
    required.discard("abs_body_angular_velocity_rad_s")
    _validate_columns(behavior_data, sorted(required), "saved Figure 1 covariates")
    if not behavior_data["zone"].eq("run").all():
        raise ValueError("Figure 1 covariates must contain run samples only")
    unexpected_trial_types = set(behavior_data["trial_type"].dropna().unique()).difference(
        {"outbound", "inbound"}
    )
    if unexpected_trial_types:
        raise ValueError(
            "Unexpected Figure 1 trial types: "
            f"{sorted(unexpected_trial_types)}"
        )

    behavior_data["abs_body_angular_velocity_rad_s"] = behavior_data[
        "body_angular_velocity_rad_s"
    ].abs()
    behavior_data, _ = add_paw_position_displacements(behavior_data)
    if not compute_predictability:
        return behavior_data, None, None

    from movement_encoding import summarize_progress_predictability
    value_columns = [column for column, _ in FIGURE1_PREDICTABILITY_SPECS]
    labels = [label for _, label in FIGURE1_PREDICTABILITY_SPECS]
    predictability = summarize_progress_predictability(
        behavior_data,
        value_cols=value_columns,
        labels=labels,
        progress_col="trial_progress_distance",
        split_by_trial_type=False,
    )
    r2_columns = [
        column for column in predictability.columns if column.endswith("_r2")
    ]
    progress_r2 = (
        predictability.set_index("label")[r2_columns]
        .sort_values(r2_columns[0], ascending=False)
    )
    return behavior_data, progress_r2, predictability


def wilbur_track_geometry() -> tuple[dict[int, tuple[float, float]], tuple]:
    """Return the stored Wilbur W-track node positions and edge order."""
    # encoding_utils configures a legacy Seaborn context at import time. Keep
    # that side effect inside a temporary context so publication typography is
    # unchanged when this function is first called during figure assembly.
    with mpl.rc_context():
        from encoding_utils import (
            WILBUR_WTRACK_CENTER0_LINEAR_EDGE_ORDER,
            WILBUR_WTRACK_CENTER0_NODE_POSITIONS,
        )

    node_positions = {
        int(node): tuple(map(float, position))
        for node, position in WILBUR_WTRACK_CENTER0_NODE_POSITIONS.items()
    }
    edges = tuple(
        tuple(map(int, edge))
        for edge in WILBUR_WTRACK_CENTER0_LINEAR_EDGE_ORDER
    )
    return node_positions, edges


def _representative_progress_trial(
    data: pd.DataFrame,
    *,
    preferred_trial_type: str = "outbound",
) -> pd.DataFrame:
    coordinate_columns = (
        ("projected_x_position", "projected_y_position")
        if {"projected_x_position", "projected_y_position"}.issubset(data.columns)
        else ("position_x", "position_y")
    )
    required = {
        *coordinate_columns,
        "trial_number",
        "trial_progress_distance",
    }
    _validate_columns(data, sorted(required), "trajectory data")
    work = data.copy()
    if "trial_type" in work and work["trial_type"].eq(preferred_trial_type).any():
        work = work.loc[work["trial_type"].eq(preferred_trial_type)]
    work = work.loc[:, sorted(required)].replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        raise ValueError("No finite trajectory is available for the track schematic")
    trial_stats = work.groupby("trial_number").agg(
        start=("trial_progress_distance", "min"),
        end=("trial_progress_distance", "max"),
        n_samples=("trial_progress_distance", "size"),
    )
    trial_stats["coverage"] = trial_stats["end"] - trial_stats["start"]
    maximum_gap = work.groupby("trial_number")["trial_progress_distance"].apply(
        lambda values: values.sort_values().diff().max()
    )
    trial_stats["maximum_gap"] = maximum_gap
    maximum_coverage = float(trial_stats["coverage"].max())
    candidates = trial_stats.loc[
        trial_stats["coverage"] >= max(0.95, 0.98 * maximum_coverage)
    ]
    if candidates.empty:
        candidates = trial_stats.nlargest(1, "coverage")
    selected_trial = candidates["maximum_gap"].sort_values().index[0]
    return (
        work.loc[work["trial_number"].eq(selected_trial)]
        .sort_values("trial_progress_distance")
        .reset_index(drop=True)
    )


def plot_task_progress_schematic(
    ax: Axes,
    *,
    trajectory_data: pd.DataFrame | None = None,
    node_positions: Mapping[int, Sequence[float]] | None = None,
    edges: Sequence[Sequence[int]] | None = None,
) -> Axes:
    """Draw the real Wilbur track graph and one progress-colored traversal."""
    if node_positions is None or edges is None:
        stored_nodes, stored_edges = wilbur_track_geometry()
        node_positions = stored_nodes if node_positions is None else node_positions
        edges = stored_edges if edges is None else edges
    node_positions = {
        int(node): np.asarray(position, dtype=float)
        for node, position in node_positions.items()
    }
    edges = tuple(tuple(map(int, edge)) for edge in edges)

    for start_node, end_node in edges:
        segment = np.vstack(
            [node_positions[start_node], node_positions[end_node]]
        )
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color="#C6CBD2",
            linewidth=7.0,
            solid_capstyle="round",
            zorder=1,
        )

    degrees = {node: 0 for node in node_positions}
    for start_node, end_node in edges:
        degrees[start_node] += 1
        degrees[end_node] += 1
    well_nodes = [node for node, degree in degrees.items() if degree == 1]
    wells = np.vstack([node_positions[node] for node in well_nodes])
    ax.scatter(
        wells[:, 0],
        wells[:, 1],
        s=48,
        facecolor="white",
        edgecolor=COLORS["full"],
        linewidth=1.0,
        zorder=5,
    )

    if trajectory_data is not None:
        full_route = _representative_progress_trial(trajectory_data)
        stride = max(1, len(full_route) // 600)
        route = full_route.iloc[::stride].copy()
        if route.index[-1] != full_route.index[-1]:
            route = pd.concat([route, full_route.iloc[[-1]]])
        coordinate_columns = (
            ["projected_x_position", "projected_y_position"]
            if {"projected_x_position", "projected_y_position"}.issubset(
                route.columns
            )
            else ["position_x", "position_y"]
        )
        points = route[coordinate_columns].to_numpy(dtype=float)
        progress = route["trial_progress_distance"].to_numpy(dtype=float)
    else:
        route_nodes = (0, 1, 2, 3)
        points = np.vstack([node_positions[node] for node in route_nodes])
        progress = np.linspace(0.0, 1.0, len(points))

    segments = np.stack([points[:-1], points[1:]], axis=1)
    progress_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "task_progress",
        ("#9ECAE1", COLORS["task"]),
    )
    progress_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    trajectory = mpl.collections.LineCollection(
        segments,
        cmap=progress_cmap,
        norm=progress_norm,
        linewidth=2.0,
        zorder=3,
    )
    trajectory.set_array((progress[:-1] + progress[1:]) / 2.0)
    ax.add_collection(trajectory)
    for value, point in ((0.0, points[0]), (1.0, points[-1])):
        ax.scatter(
            *point,
            s=30,
            facecolor=progress_cmap(value),
            edgecolor="white",
            linewidth=0.8,
            zorder=6,
        )
        ax.annotate(
            f"{int(value)}",
            xy=point,
            xytext=(5, 5),
            textcoords="offset points",
            color=progress_cmap(value),
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    all_nodes = np.vstack(list(node_positions.values()))
    span = np.ptp(all_nodes, axis=0)
    margin = np.maximum(0.08 * span, 3.0)
    ax.set_xlim(all_nodes[:, 0].min() - margin[0], all_nodes[:, 0].max() + margin[0])
    ax.set_ylim(all_nodes[:, 1].min() - margin[1], all_nodes[:, 1].max() + margin[1])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return ax


def plot_progress_profile_grid(
    container: Figure | SubFigure,
    subplot_spec,
    data: pd.DataFrame,
    *,
    specs: Sequence[tuple[str, str, str, Mapping]] = DEFAULT_PROGRESS_SPECS,
    progress_col: str = "trial_progress_distance",
    n_columns: int = 2,
) -> tuple[list[Axes], dict[str, pd.DataFrame]]:
    """Draw compact movement-versus-progress small multiples."""
    if not specs:
        raise ValueError("At least one progress-profile specification is required")
    _validate_columns(
        data,
        [progress_col, *(spec[0] for spec in specs)],
        "progress-profile data",
    )
    n_rows = int(np.ceil(len(specs) / n_columns))
    grid = subplot_spec.subgridspec(
        n_rows,
        n_columns,
        wspace=0.42,
        hspace=0.38,
    )
    axes: list[Axes] = []
    summaries = {}
    for index, (column, label, color, summary_kwargs) in enumerate(specs):
        ax = container.add_subplot(grid[index // n_columns, index % n_columns])
        ax, summary = plot_progress_with_ci(
            data,
            value_col=column,
            ax=ax,
            label=label,
            progress_col=progress_col,
            color=color,
            **dict(summary_kwargs),
        )
        ax.set_xticks((0.0, 0.5, 1.0))
        if index < (n_rows - 1) * n_columns:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        _despine_and_grid(ax)
        axes.append(ax)
        summaries[column] = summary

    for index in range(len(specs), n_rows * n_columns):
        empty_ax = container.add_subplot(grid[index // n_columns, index % n_columns])
        empty_ax.axis("off")
    return axes, summaries


def plot_multi_animal_progress_profile_grid(
    container: Figure | SubFigure,
    subplot_spec,
    profiles: pd.DataFrame,
    *,
    specs: Sequence[tuple[str, str, str, Mapping]] = DEFAULT_PROGRESS_SPECS,
) -> tuple[list[Axes], dict[str, pd.DataFrame]]:
    """Draw the same progress variables as three animal mean curves."""
    grid = subplot_spec.subgridspec(2, 2, wspace=0.42, hspace=0.38)
    compact_labels = {
        "speed": "Speed (cm/s)",
        "tangential_acceleration_cm_s2": "Accel. (cm/s²)",
        "abs_body_angular_velocity_rad_s": "Angular (rad/s)",
        "mean_fore_hind_spacing": "F–H spacing (cm)",
    }
    axes = []
    summaries = {}
    for index, (column, label, _, _) in enumerate(specs):
        ax = container.add_subplot(grid[index // 2, index % 2])
        variable_profiles = profiles.loc[profiles["variable"].eq(column)]
        for animal in FIGURE1_ANIMALS:
            summary = variable_profiles.loc[variable_profiles["animal"].eq(animal)]
            valid = summary["mean"].notna()
            ax.plot(
                summary.loc[valid, "progress"],
                summary.loc[valid, "mean"],
                color=FIGURE1_ANIMAL_COLORS[animal],
                linewidth=1.55,
                label=FIGURE1_ANIMAL_LABELS[animal],
            )
        ax.set_xlim(0, 1)
        ax.set_xticks((0.0, 0.5, 1.0))
        ax.set_ylabel(compact_labels.get(column, label), fontsize=7.5, labelpad=1)
        if index < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Trial progress")
        _despine_and_grid(ax)
        axes.append(ax)
        summaries[column] = variable_profiles
    return axes, summaries


def plot_ranked_progress_r2(
    ax: Axes,
    progress_r2: pd.Series | pd.DataFrame,
    *,
    column: str | None = None,
    max_variables: int | None = None,
    xlabel: str = "Cross-validated R²",
) -> Axes:
    """Plot ranked cross-validated R-squared values as horizontal bars."""
    if isinstance(progress_r2, pd.DataFrame):
        if progress_r2.empty:
            raise ValueError("progress_r2 must not be empty")
        column = column or str(progress_r2.columns[0])
        if column not in progress_r2:
            raise ValueError(f"Unknown progress R-squared column: {column!r}")
        values = progress_r2[column]
    else:
        values = progress_r2
    values = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if max_variables is not None:
        values = values.iloc[-max_variables:]
    if values.empty:
        raise ValueError("No finite progress R-squared values are available")

    labels = [textwrap.fill(str(label), width=20) for label in values.index]
    colors = [_feature_color(label) for label in labels]
    positions = np.arange(len(values))
    ax.barh(positions, values.to_numpy(float), color=colors, height=0.72, zorder=2)
    ax.set_yticks(positions, labels)
    ax.set_xlabel(xlabel)
    ax.set_xlim(left=min(0.0, float(values.min())))
    y_tick_size = 8.0 if len(values) > 10 else 9.5
    ax.tick_params(axis="y", length=0, labelsize=y_tick_size)
    _despine_and_grid(ax, axis="x")
    return ax


def plot_multi_animal_progress_r2(
    axes: Sequence[Axes], scores: pd.DataFrame
) -> Sequence[Axes]:
    """Show aligned animal-specific R² columns ordered by Wilbur R²."""
    _validate_columns(
        scores, ["animal", "variable", "combined_r2"], "Figure 1C scores"
    )
    if len(axes) != len(FIGURE1_ANIMALS):
        raise ValueError("Figure 1C requires one axis per animal")
    variable_labels = dict(FIGURE1_PREDICTABILITY_SPECS)
    table = scores.pivot(index="variable", columns="animal", values="combined_r2")
    table = table.loc[:, FIGURE1_ANIMALS]
    order = table["Wilbur"].sort_values(ascending=False).index
    positions = np.arange(len(order))
    for index, (ax, animal) in enumerate(zip(axes, FIGURE1_ANIMALS)):
        values = table.loc[order, animal].to_numpy(float)
        ax.axvline(0, color=COLORS["intercept"], linewidth=0.75, zorder=1)
        ax.hlines(positions, 0, values, color=COLORS["task"], linewidth=1.0, zorder=2)
        ax.scatter(values, positions, s=13, color=COLORS["task"], zorder=3)
        ax.set_xlim(-0.22, 0.63)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xticks((-0.2, 0.0, 0.3, 0.6))
        ax.set_xlabel("CV R²", fontsize=8, labelpad=2)
        ax.tick_params(axis="x", labelsize=6.5, length=2, pad=2)
        ax.text(
            0.5, 1.035, FIGURE1_ANIMAL_LABELS[animal],
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
        )
        ax.set_yticks(positions)
        if index == 0:
            ax.set_yticklabels(
                [variable_labels[variable] for variable in order], fontsize=7.5
            )
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0, pad=2)
        ax.spines[["top", "right", "left"]].set_visible(False)
    return axes


def plot_encoding_schematic(ax: Axes) -> Axes:
    """Draw the predictor-to-neural-model and variance-partition schematic."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def rounded_box(x, y, width, height, color, text):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            facecolor=mpl.colors.to_rgba(color, 0.13),
            edgecolor=color,
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center")

    rounded_box(0.02, 0.76, 0.27, 0.14, COLORS["task"], "Task")
    rounded_box(0.02, 0.54, 0.27, 0.14, COLORS["movement"], "Movement")
    rounded_box(0.59, 0.65, 0.37, 0.17, COLORS["full"], "Neural activity")
    for y in (0.61, 0.83):
        ax.add_patch(
            FancyArrowPatch(
                (0.31, y),
                (0.58, 0.735),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=1.0,
                color=COLORS["full"],
            )
        )

    task_circle = Circle(
        (0.41, 0.25),
        0.19,
        facecolor=mpl.colors.to_rgba(COLORS["task"], 0.28),
        edgecolor=COLORS["task"],
        linewidth=1.2,
    )
    movement_circle = Circle(
        (0.59, 0.25),
        0.19,
        facecolor=mpl.colors.to_rgba(COLORS["movement"], 0.25),
        edgecolor=COLORS["movement"],
        linewidth=1.2,
    )
    ax.add_patch(task_circle)
    ax.add_patch(movement_circle)
    ax.text(0.31, 0.25, "Task\nunique", ha="center", va="center", fontsize=7.5)
    ax.text(0.50, 0.25, "Shared", ha="center", va="center", fontsize=7.5)
    ax.text(0.69, 0.25, "Movement\nunique", ha="center", va="center", fontsize=7.5)
    ax.text(0.25, 0.04, "Total", color=COLORS["task"], ha="center", va="center")
    ax.text(0.75, 0.04, "Total", color=COLORS["movement"], ha="center", va="center")
    return ax


def prepare_figure_2_tables(
    unit_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive all Figure 2 panels from matched cell-level outer-CV scores."""
    score_column = "pooled_deviance_explained"
    _validate_columns(
        unit_scores, ["model_name", "unit_index", score_column], "cell-level scores"
    )
    unit_columns = ["unit_id", "unit_index"] if "unit_id" in unit_scores else ["unit_index"]
    required_models = FIGURE2_REQUIRED_MODELS
    missing = set(required_models).difference(unit_scores["model_name"].unique())
    if missing:
        raise ValueError(f"Figure 2 is missing fitted models: {sorted(missing)}")
    selected = unit_scores.loc[
        unit_scores["model_name"].isin(required_models),
        [*unit_columns, "model_name", score_column],
    ].copy()
    selected[score_column] = pd.to_numeric(selected[score_column], errors="coerce")
    if selected.duplicated([*unit_columns, "model_name"]).any():
        raise ValueError("Figure 2 scores contain duplicate model/cell rows")
    wide = selected.pivot(
        index=unit_columns, columns="model_name", values=score_column
    ).reindex(columns=required_models).dropna()
    if wide.empty:
        raise ValueError("No cells have complete Figure 2 model scores")

    intercept = wide["intercept_only"]
    full = wide["task_movement"]
    group_gains = pd.DataFrame(index=wide.index)
    group_gains["task_alone"] = wide["task_only"] - intercept
    group_gains["task_given_movement"] = full - wide["movement_only"]
    group_gains["movement_alone"] = wide["movement_only"] - intercept
    group_gains["movement_given_task"] = full - wide["task_only"]

    subgroup_gains = pd.DataFrame(index=wide.index)
    for term, (alone_model, without_model) in FIGURE2_MODEL_PAIRS.items():
        subgroup_gains[f"{term}_alone"] = wide[alone_model] - intercept
        subgroup_gains[f"{term}_given_others"] = full - wide[without_model]

    population = selected.loc[
        selected["model_name"].isin(MODEL_COLORS)
    ].merge(wide.reset_index()[unit_columns], on=unit_columns, how="inner")
    return population, subgroup_gains.reset_index(), group_gains.reset_index()


def _matched_population_scores(
    unit_scores: pd.DataFrame,
    model_names: Sequence[str],
    score_column: str,
) -> pd.DataFrame:
    unit_columns = ["unit_id", "unit_index"] if "unit_id" in unit_scores else ["unit_index"]
    selected = unit_scores.loc[
        unit_scores["model_name"].isin(model_names),
        ["model_name", *unit_columns, score_column],
    ].copy()
    selected[score_column] = pd.to_numeric(selected[score_column], errors="coerce")
    selected = selected.dropna(subset=[*unit_columns, score_column])
    duplicated = selected.duplicated(["model_name", *unit_columns], keep=False)
    if duplicated.any():
        raise ValueError("Population scores contain duplicate model/unit rows")
    wide = selected.pivot(index=unit_columns, columns="model_name", values=score_column)
    wide = wide.reindex(columns=model_names).dropna()
    return wide.stack().rename(score_column).reset_index()


def plot_population_performance(
    ax: Axes,
    unit_scores: pd.DataFrame,
    *,
    model_names: Sequence[str] = (
        "intercept_only",
        "task_only",
        "movement_only",
        "task_movement",
    ),
    score_column: str = "pooled_deviance_explained",
    matched_units: bool = True,
    show_n: bool = True,
) -> pd.DataFrame:
    """Plot model-wise population outer-CV deviance explained."""
    _validate_columns(
        unit_scores,
        ["model_name", "unit_index", score_column],
        "population scores",
    )
    model_names = tuple(model_names)
    if matched_units:
        plot_data = _matched_population_scores(unit_scores, model_names, score_column)
    else:
        plot_data = unit_scores.loc[
            unit_scores["model_name"].isin(model_names),
            ["model_name", "unit_index", score_column],
        ].copy()
        plot_data[score_column] = pd.to_numeric(plot_data[score_column], errors="coerce")
        plot_data = plot_data.dropna(subset=[score_column])

    groups = [
        _finite_percent(
            plot_data.loc[plot_data["model_name"].eq(model), score_column]
        )
        for model in model_names
    ]
    if any(not len(group) for group in groups):
        missing = [model for model, group in zip(model_names, groups) if not len(group)]
        raise ValueError(f"No finite population scores for: {missing}")

    positions = np.arange(1, len(model_names) + 1)
    colors = [MODEL_COLORS.get(model, COLORS["full"]) for model in model_names]
    variable_indices = [
        index for index, values in enumerate(groups) if np.ptp(values) > 1e-8
    ]
    if variable_indices:
        violins = ax.violinplot(
            [groups[index] for index in variable_indices],
            positions=positions[variable_indices],
            widths=0.72,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body, index in zip(violins["bodies"], variable_indices):
            color = colors[index]
            body.set_facecolor(mpl.colors.to_rgba(color, 0.42))
            body.set_edgecolor(color)
            body.set_linewidth(0.9)
        violins["cmedians"].set_color(COLORS["full"])
        violins["cmedians"].set_linewidth(1.2)
    for index, values in enumerate(groups):
        if index not in variable_indices:
            ax.hlines(
                float(np.median(values)), positions[index] - 0.34,
                positions[index] + 0.34, color=colors[index], linewidth=1.5,
            )

    random_generator = np.random.default_rng(42)
    for position, values, color in zip(positions, groups, colors):
        jitter = random_generator.uniform(-0.085, 0.085, len(values))
        ax.scatter(
            position + jitter,
            values,
            s=7,
            color=color,
            alpha=0.38,
            linewidth=0,
            zorder=3,
            rasterized=len(values) > 500,
        )
    labels = [
        f"{MODEL_LABELS.get(model, model)}\n$n$={len(values)}"
        if show_n else MODEL_LABELS.get(model, model)
        for model, values in zip(model_names, groups)
    ]
    ax.set_xticks(positions, labels)
    ax.set_ylabel("CV deviance explained (%)")
    _despine_and_grid(ax)
    return plot_data


def _draw_total_unique_violins(
    ax: Axes,
    total_values: Sequence[np.ndarray],
    unique_values: Sequence[np.ndarray],
    labels: Sequence[str],
    colors: Sequence[str],
    *,
    ylabel: str = "Contribution to CV deviance explained (%)",
    mirror_unique: bool = False,
    outlier_iqr_multiplier: float | None = None,
) -> None:
    if not labels or len(labels) != len(total_values) or len(labels) != len(unique_values):
        raise ValueError("labels, total_values, and unique_values must have equal length")
    if outlier_iqr_multiplier is not None and outlier_iqr_multiplier <= 0:
        raise ValueError("outlier_iqr_multiplier must be positive or None")

    def display_values(values: np.ndarray) -> np.ndarray:
        if outlier_iqr_multiplier is None:
            return values
        lower_quartile, upper_quartile = np.quantile(values, (0.25, 0.75))
        interquartile_range = upper_quartile - lower_quartile
        if not np.isfinite(interquartile_range) or interquartile_range <= 0:
            return values
        lower_limit = lower_quartile - outlier_iqr_multiplier * interquartile_range
        upper_limit = upper_quartile + outlier_iqr_multiplier * interquartile_range
        return values[(values >= lower_limit) & (values <= upper_limit)]

    positions = np.arange(1, len(labels) + 1)
    offset = 0.17
    total_positions = positions - offset
    unique_positions = positions + offset
    display_total_values = [display_values(values) for values in total_values]
    display_unique_values = (
        [(-1 if mirror_unique else 1) * display_values(values)
         for values in unique_values]
    )
    total_violins = ax.violinplot(
        display_total_values,
        positions=total_positions,
        widths=0.27,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    unique_violins = ax.violinplot(
        display_unique_values,
        positions=unique_positions,
        widths=0.27,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, color in zip(total_violins["bodies"], colors):
        body.set_facecolor(mpl.colors.to_rgba(color, 0.50))
        body.set_edgecolor(color)
        body.set_linewidth(0.9)
    for body, color in zip(unique_violins["bodies"], colors):
        body.set_facecolor(COLORS["paper"])
        body.set_edgecolor(color)
        body.set_linewidth(1.1)
    for collection in (total_violins, unique_violins):
        collection["cmedians"].set_color(COLORS["full"])
        collection["cmedians"].set_linewidth(1.1)

    random_generator = np.random.default_rng(42)
    for total_position, unique_position, total, unique, color in zip(
        total_positions,
        unique_positions,
        display_total_values,
        display_unique_values,
        colors,
    ):
        ax.scatter(
            total_position + random_generator.uniform(-0.04, 0.04, len(total)),
            total,
            s=5,
            color=color,
            alpha=0.28,
            linewidth=0,
            zorder=3,
            rasterized=len(total) > 500,
        )
        ax.scatter(
            unique_position + random_generator.uniform(-0.04, 0.04, len(unique)),
            unique,
            s=5,
            facecolor="white",
            edgecolor=color,
            alpha=0.45,
            linewidth=0.45,
            zorder=3,
            rasterized=len(unique) > 500,
        )
    ax.axhline(0, color=COLORS["full"], linewidth=0.7, zorder=1)
    ax.set_xticks(positions, labels)
    ax.set_ylabel(ylabel)
    _despine_and_grid(ax)
    if mirror_unique:
        ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda value, _: f"{abs(value):g}")
        )


def plot_subgroup_contributions(
    axes: Sequence[Axes],
    gains: pd.DataFrame,
) -> Sequence[Axes]:
    """Plot total and unique task/movement subgroup contributions."""
    if len(axes) != 2:
        raise ValueError("Two axes are required for task and movement subgroups")
    required = [
        f"{term}_{suffix}"
        for groups in TERM_GROUPS.values()
        for _, term in groups
        for suffix in ("alone", "given_others")
    ]
    _validate_columns(gains, required, "subgroup gains")
    for axis, (family, terms) in zip(axes, TERM_GROUPS.items()):
        labels = [label for label, _ in terms]
        total = [_finite_percent(gains[f"{term}_alone"]) for _, term in terms]
        unique = [
            _finite_percent(gains[f"{term}_given_others"]) for _, term in terms
        ]
        colors = [
            COLORS["task"] if family == "task" else COLORS[term]
            for _, term in terms
        ]
        _draw_total_unique_violins(axis, total, unique, labels, colors)
    axes[1].set_ylabel("")
    return axes


def plot_combined_subgroup_contributions(
    ax: Axes,
    gains: pd.DataFrame,
    *,
    mirror_unique: bool = False,
    outlier_iqr_multiplier: float | None = None,
) -> Axes:
    """Plot all six predictor subgroups on one space-efficient axis."""
    ordered_terms = (*TERM_GROUPS["task"], *TERM_GROUPS["movement"])
    required = [
        f"{term}_{suffix}"
        for _, term in ordered_terms
        for suffix in ("alone", "given_others")
    ]
    _validate_columns(gains, required, "subgroup gains")
    labels = [label for label, _ in ordered_terms]
    total = [_finite_percent(gains[f"{term}_alone"]) for _, term in ordered_terms]
    unique = [
        _finite_percent(gains[f"{term}_given_others"]) for _, term in ordered_terms
    ]
    colors = [
        COLORS["task"],
        COLORS["task"],
        COLORS["task"],
        COLORS["translation"],
        COLORS["turning"],
        COLORS["limb"],
    ]
    _draw_total_unique_violins(
        ax, total, unique, labels, colors,
        mirror_unique=mirror_unique,
        outlier_iqr_multiplier=outlier_iqr_multiplier,
    )
    ax.axvline(3.5, color=COLORS["grid"], linewidth=0.8, zorder=1)
    ax.tick_params(axis="x", labelrotation=25)
    for tick_label in ax.get_xticklabels():
        tick_label.set_horizontalalignment("right")
    return ax


def plot_total_unique_distributions(
    ax: Axes,
    gains: pd.DataFrame,
    *,
    mirror_unique: bool = False,
    outlier_iqr_multiplier: float | None = None,
) -> Axes:
    """Plot Figure 2D option 1: task/movement total and unique violins."""
    columns = {
        "task_alone": ("Task", COLORS["task"]),
        "task_given_movement": ("Task", COLORS["task"]),
        "movement_alone": ("Movement", COLORS["movement"]),
        "movement_given_task": ("Movement", COLORS["movement"]),
    }
    _validate_columns(gains, list(columns), "task/movement gains")
    _draw_total_unique_violins(
        ax,
        [_finite_percent(gains["task_alone"]), _finite_percent(gains["movement_alone"])],
        [
            _finite_percent(gains["task_given_movement"]),
            _finite_percent(gains["movement_given_task"]),
        ],
        ["Task", "Movement"],
        [COLORS["task"], COLORS["movement"]],
        mirror_unique=mirror_unique,
        outlier_iqr_multiplier=outlier_iqr_multiplier,
    )
    return ax


def plot_population_performance_compact(
    ax: Axes, unit_scores: pd.DataFrame
) -> pd.DataFrame:
    """Space-efficient four-model population boxplots for Figure 2B."""
    model_names = tuple(MODEL_COLORS)
    plot_data = _matched_population_scores(
        unit_scores, model_names, "pooled_deviance_explained"
    )
    groups = [
        _finite_percent(
            plot_data.loc[
                plot_data["model_name"].eq(model), "pooled_deviance_explained"
            ]
        )
        for model in model_names
    ]
    if any(len(values) == 0 for values in groups):
        raise ValueError("Figure 2B has no matched cells for one or more models")
    boxes = ax.boxplot(
        groups,
        vert=False,
        positions=np.arange(1, 5),
        widths=0.57,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        medianprops={"color": COLORS["full"], "linewidth": 1.35},
        whiskerprops={"color": COLORS["full"], "linewidth": 0.8},
        capprops={"color": COLORS["full"], "linewidth": 0.8},
    )
    for box, model in zip(boxes["boxes"], model_names):
        color = MODEL_COLORS[model]
        box.set_facecolor(mpl.colors.to_rgba(color, 0.28))
        box.set_edgecolor(color)
        box.set_linewidth(1.0)
    ax.set_yticks(np.arange(1, 5), [MODEL_LABELS[model] for model in model_names])
    ax.invert_yaxis()
    ax.axvline(0, color=COLORS["full"], linewidth=0.8, zorder=0)
    ax.set_xlabel("CV deviance explained (%)")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.65)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    return plot_data


def plot_total_unique_compact(
    ax: Axes,
    gains: pd.DataFrame,
    entries: Sequence[tuple[str, str, str, str]],
    *,
    show_legend: bool = False,
) -> Axes:
    """Paired median dots and interquartile ranges for total and unique gains."""
    required = [column for _, total, unique, _ in entries for column in (total, unique)]
    _validate_columns(gains, required, "contribution gains")
    positions = np.arange(len(entries), 0, -1)
    for y, (label, total_column, unique_column, color) in zip(positions, entries):
        pair = gains[[total_column, unique_column]].apply(pd.to_numeric, errors="coerce")
        pair = 100 * pair.replace([np.inf, -np.inf], np.nan).dropna()
        if pair.empty:
            raise ValueError(f"No finite total/unique pairs for {label}")
        total = pair[total_column].to_numpy()
        unique = pair[unique_column].to_numpy()
        total_median, unique_median = np.median(total), np.median(unique)
        ax.plot(
            [total_median, unique_median], [y + 0.13, y - 0.13],
            color=color, alpha=0.45, linewidth=0.8, zorder=2,
        )
        for values, median, offset, filled in (
            (total, total_median, 0.13, True),
            (unique, unique_median, -0.13, False),
        ):
            lower, upper = np.quantile(values, [0.25, 0.75])
            ax.plot([lower, upper], [y + offset, y + offset],
                    color=color, linewidth=2.2, alpha=0.70, zorder=3)
            ax.scatter(
                [median], [y + offset], s=26,
                facecolor=color if filled else "white",
                edgecolor=color, linewidth=1.0, zorder=4,
            )
    ax.set_yticks(positions, [entry[0] for entry in entries])
    ax.set_ylim(0.45, len(entries) + (1.05 if show_legend else 0.55))
    ax.axvline(0, color=COLORS["full"], linewidth=0.75, zorder=0)
    ax.set_xlabel("Contribution (%)")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.65)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    if show_legend:
        handles = [
            mpl.lines.Line2D([], [], marker="o", linestyle="none", markersize=5,
                             markerfacecolor=COLORS["full"], markeredgecolor=COLORS["full"],
                             label="Total"),
            mpl.lines.Line2D([], [], marker="o", linestyle="none", markersize=5,
                             markerfacecolor="white", markeredgecolor=COLORS["full"],
                             label="Unique"),
        ]
        ax.legend(handles=handles, loc="upper right", ncol=2, frameon=False,
                  fontsize=7, handletextpad=0.3, columnspacing=0.7)
    return ax


def plot_total_unique_identity(
    axes: Sequence[Axes],
    gains: pd.DataFrame,
) -> Sequence[Axes]:
    """Plot Figure 2D option 2: total versus unique contribution per neuron."""
    if len(axes) != 2:
        raise ValueError("Two axes are required for the identity-line version")
    pairs = (
        ("task_alone", "task_given_movement", "Task", COLORS["task"]),
        ("movement_alone", "movement_given_task", "Movement", COLORS["movement"]),
    )
    _validate_columns(
        gains,
        [column for total, unique, _, _ in pairs for column in (total, unique)],
        "task/movement gains",
    )
    prepared = []
    all_values = []
    for total_column, unique_column, label, color in pairs:
        pair = gains.loc[:, [total_column, unique_column]].apply(
            pd.to_numeric, errors="coerce"
        )
        pair = 100.0 * pair.replace([np.inf, -np.inf], np.nan).dropna()
        if pair.empty:
            raise ValueError(f"No finite {label.lower()} total/unique pairs")
        prepared.append((pair, total_column, unique_column, label, color))
        all_values.append(pair.to_numpy().ravel())
    concatenated = np.concatenate(all_values)
    lower = min(0.0, float(np.nanmin(concatenated)))
    upper = max(0.0, float(np.nanmax(concatenated)))
    span = upper - lower
    margin = 0.06 * span if span > 0 else 1.0
    limits = (lower - margin, upper + margin)

    for axis, (pair, total_column, unique_column, label, color) in zip(axes, prepared):
        axis.scatter(
            pair[total_column],
            pair[unique_column],
            s=10,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            alpha=0.55,
            zorder=3,
            rasterized=len(pair) > 500,
        )
        axis.plot(limits, limits, color=COLORS["full"], linestyle="--", linewidth=0.9)
        axis.axhline(0, color=COLORS["grid"], linewidth=0.6)
        axis.axvline(0, color=COLORS["grid"], linewidth=0.6)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Total contribution (%)")
        axis.text(
            0.05,
            0.95,
            label,
            color=color,
            fontweight="bold",
            transform=axis.transAxes,
            ha="left",
            va="top",
        )
        _despine_and_grid(axis)
    axes[0].set_ylabel("Unique contribution (%)")
    axes[1].set_ylabel("")
    return axes


def _pending_generalization_panel(ax: Axes) -> None:
    """Reserve panel space for an across-task result supplied later."""
    ax.axis("off")


def _load_svg_panel(svg_path: str | Path) -> Path:
    """Resolve the vector source, which is read again when the figure is saved."""
    source = Path(svg_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Panel A SVG not found: {source}")
    return source


def _insert_svg_panel(
    output_path: Path,
    source_path: Path,
    ax: Axes,
    source_viewbox: tuple[float, float, float, float] | None = None,
) -> None:
    """Replace the preview with the original editable SVG in the export."""
    namespace = "http://www.w3.org/2000/svg"
    xlink_namespace = "http://www.w3.org/1999/xlink"
    ET.register_namespace("", namespace)
    ET.register_namespace("xlink", xlink_namespace)
    composite = ET.parse(output_path)
    root = composite.getroot()
    source = ET.parse(source_path).getroot()
    if source_viewbox is not None:
        source.set("viewBox", " ".join(str(value) for value in source_viewbox))
    # Matplotlib and the imported drawing both define glyph and group IDs.
    # Prefix the drawing's IDs so its clipping paths and text stay intact.
    id_map = {
        element.get("id"): f"panel_a_{element.get('id')}"
        for element in source.iter()
        if element.get("id")
    }
    for element in source.iter():
        old_id = element.get("id")
        if old_id is not None:
            element.set("id", id_map[old_id])
        if element.tag == f"{{{namespace}}}image":
            image_href = element.get(f"{{{xlink_namespace}}}href")
            if image_href is not None:
                image_href = image_href.replace("base64, ", "base64,", 1)
                element.set(f"{{{xlink_namespace}}}href", image_href)
                element.set("href", image_href)
        for key, value in list(element.attrib.items()):
            if key.endswith("}href") or key == "href":
                if value.startswith("#") and value[1:] in id_map:
                    value = f"#{id_map[value[1:]]}"
            for original, replacement in id_map.items():
                value = value.replace(f"url(#{original})", f"url(#{replacement})")
            element.set(key, value)
    _, _, canvas_width, canvas_height = map(float, root.attrib["viewBox"].split())
    left, bottom, width, height = ax.get_position().bounds
    source.attrib.update({
        "x": str(left * canvas_width),
        "y": str((1 - bottom - height) * canvas_height),
        "width": str(width * canvas_width),
        "height": str(height * canvas_height),
        "preserveAspectRatio": "xMidYMid meet",
    })
    figure_group = next(
        (element for element in root.iter() if element.get("id") == "figure_1"),
        root,
    )
    axes_index = next(
        (index for index, child in enumerate(figure_group)
         if child.get("id") == "axes_1"),
        len(figure_group),
    )
    figure_group.insert(axes_index, source)
    composite.write(output_path, encoding="utf-8", xml_declaration=True)


def assemble_figure_1(
    behavior_data: pd.DataFrame,
    progress_r2: pd.Series | pd.DataFrame,
    *,
    progress_specs: Sequence[tuple[str, str, str, Mapping]] = DEFAULT_PROGRESS_SPECS,
    progress_r2_column: str | None = None,
    progress_r2_xlabel: str = "Cross-validated R²",
    generalization_plotter: Callable[[Axes], object] | None = None,
    track_node_positions: Mapping[int, Sequence[float]] | None = None,
    track_edges: Sequence[Sequence[int]] | None = None,
    schematic_svg_path: str | Path | None = None,
    schematic_svg_viewbox: tuple[float, float, float, float] | None = None,
    figsize: tuple[float, float] = (7.2, 5.2),
) -> tuple[Figure, dict[str, object]]:
    """Assemble the first abstract figure at its intended physical size."""
    multi_animal_r2 = isinstance(progress_r2, pd.DataFrame) and {
        "animal", "variable", "combined_r2"
    }.issubset(progress_r2.columns)
    multi_animal_profiles = {
        "animal", "variable", "progress", "mean"
    }.issubset(behavior_data.columns)
    with publication_style():
        figure = plt.figure(figsize=figsize, facecolor="white")
        top = figure.add_gridspec(
            1,
            2,
            width_ratios=(0.34, 0.66),
            left=0.075,
            right=0.985,
            bottom=0.58,
            top=0.93 if multi_animal_profiles else 0.965,
            wspace=0.34,
        )
        bottom = figure.add_gridspec(
            1,
            2,
            width_ratios=(0.70, 0.30) if multi_animal_r2 else (0.58, 0.42),
            left=0.24 if multi_animal_r2 else 0.18,
            right=0.985,
            bottom=0.09,
            top=0.46 if multi_animal_r2 else 0.49,
            wspace=0.44 if multi_animal_r2 else 0.40,
        )
        schematic_ax = figure.add_subplot(top[0, 0])
        schematic_ax.axis("off")
        if schematic_svg_path is not None:
            source = _load_svg_panel(schematic_svg_path)
            figure._editable_svg_panel = (source, schematic_ax, schematic_svg_viewbox)
        add_panel_label(schematic_ax, "A", x=-0.10, y=1.04)

        if multi_animal_profiles:
            profile_axes, profile_summaries = plot_multi_animal_progress_profile_grid(
                figure, top[0, 1], behavior_data, specs=progress_specs
            )
            handles, labels = profile_axes[0].get_legend_handles_labels()
            figure.legend(
                handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.998),
                ncol=3, fontsize=8, frameon=False, columnspacing=1.0,
                handlelength=1.6,
            )
        else:
            profile_axes, profile_summaries = plot_progress_profile_grid(
                figure, top[0, 1], behavior_data, specs=progress_specs
            )
        add_panel_label(profile_axes[0], "B", x=-0.30, y=1.18)

        if multi_animal_r2:
            r2_grid = bottom[0, 0].subgridspec(1, 3, wspace=0.16)
            r2_axes = [figure.add_subplot(r2_grid[0, index]) for index in range(3)]
            plot_multi_animal_progress_r2(r2_axes, progress_r2)
            r2_ax = r2_axes[0]
        else:
            r2_ax = figure.add_subplot(bottom[0, 0])
            plot_ranked_progress_r2(
                r2_ax,
                progress_r2,
                column=progress_r2_column,
                xlabel=progress_r2_xlabel,
            )
            r2_axes = [r2_ax]
        add_panel_label(r2_ax, "C", x=-0.55 if len(r2_axes) == 3 else -0.18, y=1.08)

        generalization_ax = figure.add_subplot(bottom[0, 1])
        if generalization_plotter is None:
            _pending_generalization_panel(generalization_ax)
        else:
            generalization_plotter(generalization_ax)
        add_panel_label(generalization_ax, "D", x=-0.10, y=1.08)
    return figure, {
        "schematic": schematic_ax,
        "profiles": profile_axes,
        "profile_summaries": profile_summaries,
        "progress_r2": r2_ax,
        "progress_r2_axes": r2_axes,
        "generalization": generalization_ax,
    }


def assemble_figure_2(
    population_scores: pd.DataFrame,
    subgroup_gains: pd.DataFrame,
    task_movement_gains: pd.DataFrame,
    *,
    option_2d: str = "violin",
    matched_population_units: bool = True,
    figsize: tuple[float, float] = (7.2, 4.9),
) -> tuple[Figure, dict[str, object]]:
    """Assemble the second abstract figure with either Figure 2D prototype."""
    if option_2d not in {"violin", "identity"}:
        raise ValueError("option_2d must be 'violin' or 'identity'")
    with publication_style():
        figure = plt.figure(figsize=figsize, facecolor="white")
        top = figure.add_gridspec(
            1,
            2,
            width_ratios=(0.37, 0.63),
            left=0.075,
            right=0.985,
            bottom=0.55,
            top=0.965,
            wspace=0.32,
        )
        bottom_width_ratios = (
            (0.62, 0.38) if option_2d == "violin" else (0.45, 0.55)
        )
        bottom = figure.add_gridspec(
            1,
            2,
            width_ratios=bottom_width_ratios,
            left=0.09,
            right=0.985,
            bottom=0.11,
            top=0.46,
            wspace=0.34,
        )

        schematic_ax = figure.add_subplot(top[0, 0])
        plot_encoding_schematic(schematic_ax)
        add_panel_label(schematic_ax, "A", x=-0.10, y=1.04)

        population_ax = figure.add_subplot(top[0, 1])
        population_plot_data = plot_population_performance(
            population_ax,
            population_scores,
            matched_units=matched_population_units,
        )
        add_panel_label(population_ax, "B", x=-0.12, y=1.08)

        subgroup_axes = [figure.add_subplot(bottom[0, 0])]
        plot_combined_subgroup_contributions(subgroup_axes[0], subgroup_gains)
        add_panel_label(subgroup_axes[0], "C", x=-0.16, y=1.08)

        if option_2d == "violin":
            d_axes = [figure.add_subplot(bottom[0, 1])]
            plot_total_unique_distributions(d_axes[0], task_movement_gains)
        else:
            identity_grid = bottom[0, 1].subgridspec(1, 2, wspace=0.20)
            d_axes = [
                figure.add_subplot(identity_grid[0, 0]),
                figure.add_subplot(identity_grid[0, 1]),
            ]
            plot_total_unique_identity(d_axes, task_movement_gains)
        add_panel_label(d_axes[0], "D", x=-0.34, y=1.10)
    return figure, {
        "schematic": schematic_ax,
        "population": population_ax,
        "population_plot_data": population_plot_data,
        "subgroups": subgroup_axes,
        "comparison": d_axes,
    }


def assemble_figure_2_grid(
    population_scores: pd.DataFrame,
    subgroup_gains: pd.DataFrame,
    task_movement_gains: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.2, 6.0),
) -> tuple[Figure, dict[str, object]]:
    """Arrange the unchanged Figure 2 violin panels in a roomy 2-by-2 grid."""
    with publication_style():
        figure = plt.figure(figsize=figsize, facecolor="white")
        top = figure.add_gridspec(
            1, 2, width_ratios=(0.40, 0.60),
            left=0.055, right=0.985, bottom=0.58, top=0.965, wspace=0.18,
        )
        bottom = figure.add_gridspec(
            1, 2, width_ratios=(0.38, 0.62),
            left=0.18, right=0.985, bottom=0.13, top=0.51, wspace=0.13,
        )
        schematic_ax = figure.add_subplot(top[0, 0])
        schematic_ax.axis("off")
        add_panel_label(schematic_ax, "A", x=-0.03, y=1.02)

        population_ax = figure.add_subplot(top[0, 1])
        population_plot_data = plot_population_performance(
            population_ax, population_scores, matched_units=True, show_n=False
        )
        population_ax.set_ylabel("Deviance explained (%)")
        add_panel_label(population_ax, "B", x=0.00, y=1.06)

        group_ax = figure.add_subplot(bottom[0, 0])
        plot_total_unique_distributions(
            group_ax, task_movement_gains, mirror_unique=True,
            outlier_iqr_multiplier=3.0,
        )
        group_ax.set_ylabel("")
        lower, upper = group_ax.get_ylim()
        zero_fraction = -lower / (upper - lower)
        group_ax.text(
            -0.56, (1 + zero_fraction) / 2, "Total\nD² gain (%)",
            transform=group_ax.transAxes,
            ha="left", va="center", fontsize=9, color=COLORS["full"],
        )
        group_ax.text(
            -0.56, zero_fraction / 2, "Unique\nD² gain (%)",
            transform=group_ax.transAxes,
            ha="left", va="center", fontsize=9, color=COLORS["full"],
        )
        add_panel_label(group_ax, "C", x=0.00, y=1.06)

        subgroup_ax = figure.add_subplot(bottom[0, 1])
        plot_combined_subgroup_contributions(
            subgroup_ax, subgroup_gains, mirror_unique=True,
            outlier_iqr_multiplier=3.0,
        )
        subgroup_ax.set_ylabel("")
        add_panel_label(subgroup_ax, "D", x=0.00, y=1.06)
    return figure, {
        "schematic": schematic_ax,
        "population": population_ax,
        "population_plot_data": population_plot_data,
        "task_movement": group_ax,
        "subgroups": subgroup_ax,
    }


def save_composite(
    figure: Figure,
    filename_stem: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "svg"),
) -> tuple[Path, ...]:
    """Export without changing the composite's physical canvas size."""
    stem = Path(filename_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    output_paths = []
    with mpl.rc_context(_style_parameters()):
        for file_format in formats:
            normalized = file_format.lower().lstrip(".")
            output_path = stem.with_suffix(f".{normalized}")
            svg_panel = getattr(figure, "_editable_svg_panel", None)
            figure.savefig(
                output_path,
                format=normalized,
                facecolor="white",
                metadata={"Creator": "Matplotlib"} if normalized == "pdf" else None,
            )
            if normalized == "svg" and svg_panel is not None:
                _insert_svg_panel(output_path, svg_panel[0], svg_panel[1], svg_panel[2])
            output_paths.append(output_path)
    return tuple(output_paths)
