"""Visualization helpers for grouped movement variables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


def exclude_reward_port_ranges(
    data: pd.DataFrame,
    reward_ranges=None,
    position_col: str = "linear_position",
) -> pd.DataFrame:
    """Return samples outside the configured linearized reward-port ranges."""
    if reward_ranges is None:
        return data
    if position_col not in data:
        raise ValueError(f"data is missing {position_col!r}")

    in_reward_range = pd.Series(False, index=data.index)
    for start_pos, end_pos in reward_ranges:
        in_reward_range |= data[position_col].between(
            start_pos,
            end_pos,
            inclusive="both",
        )
    return data.loc[~in_reward_range].copy()


def _wrap_angle(angle, period=2 * np.pi):
    angle = np.asarray(angle, dtype=float)
    return (angle + period / 2.0) % period - period / 2.0


def _circular_mean(values, period=2 * np.pi):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    phases = values * (2 * np.pi / period)
    mean_phase = np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases)))
    return mean_phase * (period / (2 * np.pi))


def compute_progress_summary(
    data: pd.DataFrame,
    value_col: str,
    progress_col: str = "trial_progress_distance",
    trial_col: str = "trial_number",
    window_size: float = 0.12,
    step_size: float = 0.025,
    confidence: float = 0.95,
    min_trials: int = 5,
    circular: bool = False,
    circular_period: float = 2 * np.pi,
    n_boot: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Summarize a variable over progress with trial-level confidence bands."""
    required = {value_col, progress_col, trial_col}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {sorted(missing)}")

    work = data.loc[:, [trial_col, progress_col, value_col]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    work = work[work[progress_col].between(0.0, 1.0)]
    if work.empty:
        raise ValueError(f"No finite samples are available for {value_col!r}")

    centers = np.arange(0.0, 1.0 + step_size * 0.5, step_size)
    half_window = window_size / 2.0
    alpha = 1.0 - confidence
    rng = np.random.default_rng(random_state)
    rows = []

    for center in centers:
        in_window = work[progress_col].between(
            max(0.0, center - half_window),
            min(1.0, center + half_window),
        )
        window_df = work.loc[in_window]
        if circular:
            trial_values = (
                window_df.groupby(trial_col, observed=True)[value_col]
                .apply(lambda values: _circular_mean(values, circular_period))
                .dropna()
                .to_numpy(dtype=float)
            )
        else:
            trial_values = (
                window_df.groupby(trial_col, observed=True)[value_col]
                .mean()
                .dropna()
                .to_numpy(dtype=float)
            )

        n_trials = len(trial_values)
        if n_trials < min_trials:
            rows.append((center, np.nan, np.nan, np.nan, n_trials))
            continue

        mean_value = (
            _circular_mean(trial_values, circular_period)
            if circular
            else float(np.mean(trial_values))
        )
        sample_indices = rng.integers(0, n_trials, size=(int(n_boot), n_trials))
        boot_samples = trial_values[sample_indices]
        if circular:
            boot_phases = boot_samples * (2 * np.pi / circular_period)
            boot_means = np.arctan2(
                np.mean(np.sin(boot_phases), axis=1),
                np.mean(np.cos(boot_phases), axis=1),
            )
            boot_means *= circular_period / (2 * np.pi)
            centered_boot = _wrap_angle(
                boot_means - mean_value,
                period=circular_period,
            )
        else:
            boot_means = np.mean(boot_samples, axis=1)
            centered_boot = boot_means - mean_value
        lower_delta, upper_delta = np.quantile(
            centered_boot, [alpha / 2.0, 1.0 - alpha / 2.0]
        )
        lower = mean_value + lower_delta
        upper = mean_value + upper_delta

        rows.append((center, mean_value, lower, upper, n_trials))

    summary = pd.DataFrame(
        rows,
        columns=["progress", "mean", "lower", "upper", "n_trials"],
    )
    if circular:
        finite = summary["mean"].notna().to_numpy()
        if finite.any():
            wrapped_mean = summary.loc[finite, "mean"].to_numpy()
            unwrapped_mean = np.unwrap(
                wrapped_mean,
                period=circular_period,
            )
            unwrapped_mean -= circular_period * np.round(
                np.nanmedian(unwrapped_mean) / circular_period
            )
            offsets = unwrapped_mean - wrapped_mean
            summary.loc[finite, "mean"] = unwrapped_mean
            summary.loc[finite, "lower"] += offsets
            summary.loc[finite, "upper"] += offsets
    return summary


def plot_progress_with_ci(
    data: pd.DataFrame,
    value_col: str,
    ax=None,
    label: str | None = None,
    progress_col: str = "trial_progress_distance",
    circular: bool = False,
    circular_period: float = 2 * np.pi,
    color: str = "#0072B2",
    **summary_kwargs,
):
    """Plot a trial-level mean and 95% CI over normalized trial progress."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.5), layout="constrained")
    summary = compute_progress_summary(
        data,
        value_col=value_col,
        progress_col=progress_col,
        circular=circular,
        circular_period=circular_period,
        **summary_kwargs,
    )
    valid = summary[["mean", "lower", "upper"]].notna().all(axis=1)
    x = summary.loc[valid, "progress"].to_numpy(dtype=float)
    mean = summary.loc[valid, "mean"].to_numpy(dtype=float)
    lower = summary.loc[valid, "lower"].to_numpy(dtype=float)
    upper = summary.loc[valid, "upper"].to_numpy(dtype=float)
    if not len(x):
        raise ValueError(f"Too few trials to summarize {value_col!r}")
    ax.fill_between(x, lower, upper, color=color, alpha=0.22, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=2)
    if not circular and np.nanmin(lower) < 0 < np.nanmax(upper):
        ax.axhline(0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Trial progress", fontsize=9)
    ax.set_ylabel(label or value_col, fontsize=9)
    ax.tick_params(labelsize=8)
    return ax, summary


def _color_scale(values, kind):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("Cannot construct a color scale without finite values")

    if kind == "circular":
        return "twilight", Normalize(vmin=-np.pi, vmax=np.pi), _circular_mean
    if kind == "circular_signed":
        return (
            "coolwarm",
            TwoSlopeNorm(vmin=-np.pi, vcenter=0.0, vmax=np.pi),
            _circular_mean,
        )
    if kind == "circular_degrees":
        return (
            "twilight",
            Normalize(vmin=-180.0, vmax=180.0),
            lambda values: _circular_mean(values, 360.0),
        )
    if kind == "signed":
        limit = float(np.nanquantile(np.abs(finite), 0.98))
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0
        return "coolwarm", TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), np.mean
    if kind == "positive":
        upper = float(np.nanquantile(finite, 0.98))
        if not np.isfinite(upper) or upper <= 0:
            upper = float(np.nanmax(finite)) or 1.0
        return "viridis", Normalize(vmin=0.0, vmax=upper), np.mean
    raise ValueError(
        "kind must be 'circular', 'circular_signed', 'circular_degrees', "
        "'signed', or 'positive'"
    )


def plot_wtrack_variable_map(
    data: pd.DataFrame,
    value_col: str,
    ax=None,
    label: str | None = None,
    kind: str = "signed",
    marker_size: float = 5.0,
    marker_alpha: float = 0.65,
    color_scale=None,
    add_colorbar: bool = True,
):
    """Plot every valid position sample as a square colored by its value."""
    required = {"position_x", "position_y", value_col}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {sorted(missing)}")
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.5), layout="constrained")

    work = data.loc[:, ["position_x", "position_y", value_col]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if work.empty:
        raise ValueError(f"No finite spatial samples are available for {value_col!r}")

    if color_scale is None:
        cmap, norm, _ = _color_scale(work[value_col].to_numpy(), kind)
    else:
        cmap, norm, _ = color_scale
    spatial = ax.scatter(
        work["position_x"],
        work["position_y"],
        c=work[value_col],
        marker="s",
        s=marker_size,
        alpha=marker_alpha,
        cmap=cmap,
        norm=norm,
        linewidths=0,
        rasterized=True,
    )
    if add_colorbar:
        colorbar = ax.figure.colorbar(spatial, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(label or value_col, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x position (cm)", fontsize=9)
    ax.set_ylabel("y position (cm)", fontsize=9)
    ax.tick_params(labelsize=8)
    return ax, spatial


def compute_transition_progress(
    data: pd.DataFrame,
    transitions_by_condition,
    progress_col: str = "trial_progress_distance",
    segment_col: str = "track_segment_id",
    trial_col: str = "trial_number",
    condition_col: str = "trial_type",
) -> pd.DataFrame:
    """Estimate median progress of directed edge transitions across trials."""
    required = {progress_col, segment_col, trial_col, condition_col}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {sorted(missing)}")

    trial_records = []
    for (condition, trial), trial_df in data.groupby(
        [condition_col, trial_col],
        observed=True,
        sort=False,
    ):
        event_transitions = transitions_by_condition.get(condition, {})
        if not event_transitions:
            continue
        trial_df = trial_df.sort_index()
        segments = trial_df[segment_col].to_numpy(dtype=float)
        progress = trial_df[progress_col].to_numpy(dtype=float)
        valid = np.isfinite(segments) & np.isfinite(progress)
        segments = segments[valid].astype(int)
        progress = progress[valid]
        if len(segments) < 2:
            continue

        observed_transitions = list(zip(segments[:-1], segments[1:]))
        transition_progress = 0.5 * (progress[:-1] + progress[1:])
        for event, transitions in event_transitions.items():
            transitions = {tuple(pair) for pair in transitions}
            matches = np.array(
                [pair in transitions for pair in observed_transitions],
                dtype=bool,
            )
            if matches.any():
                trial_records.append(
                    {
                        condition_col: condition,
                        trial_col: trial,
                        "event": event,
                        "progress": float(np.median(transition_progress[matches])),
                    }
                )

    if not trial_records:
        raise ValueError("No requested track-segment transitions were found")
    trial_events = pd.DataFrame.from_records(trial_records)
    summary = (
        trial_events.groupby([condition_col, "event"], observed=True)["progress"]
        .agg(progress="median", n_trials="count")
        .reset_index()
    )
    return summary


def compute_progress_heatmap(
    data: pd.DataFrame,
    value_cols,
    progress_col: str = "trial_progress_distance",
    trial_col: str = "trial_number",
    condition_col: str = "trial_type",
    conditions=("outbound", "inbound"),
    n_bins: int = 40,
    min_trials: int = 5,
):
    """Return trial-weighted, jointly row-z-scored progress matrices."""
    value_cols = list(value_cols)
    required = {progress_col, trial_col, condition_col, *value_cols}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {sorted(missing)}")
    if not value_cols:
        raise ValueError("value_cols must contain at least one variable")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    work = data.loc[
        data[condition_col].isin(conditions),
        [condition_col, trial_col, progress_col, *value_cols],
    ].copy()
    work = work.dropna(subset=[condition_col, trial_col, progress_col])
    work = work[work[progress_col].between(0.0, 1.0)]
    if work.empty:
        raise ValueError("No samples are available for the requested conditions")

    work["_progress_bin"] = np.minimum(
        np.floor(work[progress_col].to_numpy(dtype=float) * n_bins).astype(int),
        n_bins - 1,
    )
    trial_means = work.groupby(
        [condition_col, trial_col, "_progress_bin"],
        observed=True,
    )[value_cols].mean()
    condition_means = trial_means.groupby(
        [condition_col, "_progress_bin"],
        observed=True,
    ).mean()
    trial_counts = trial_means.groupby(
        [condition_col, "_progress_bin"],
        observed=True,
    ).count()

    full_index = pd.MultiIndex.from_product(
        [conditions, range(n_bins)],
        names=[condition_col, "_progress_bin"],
    )
    condition_means = condition_means.reindex(full_index)
    trial_counts = trial_counts.reindex(full_index, fill_value=0)

    mean_matrices = {}
    for condition in conditions:
        matrix = condition_means.loc[condition, value_cols].to_numpy().T
        counts = trial_counts.loc[condition, value_cols].to_numpy().T
        matrix[counts < min_trials] = np.nan
        mean_matrices[condition] = matrix

    combined = np.concatenate(
        [mean_matrices[condition] for condition in conditions],
        axis=1,
    )
    if np.any(~np.isfinite(combined).any(axis=1)):
        unavailable = [
            column
            for column, row in zip(value_cols, combined)
            if not np.isfinite(row).any()
        ]
        raise ValueError(f"No progress-bin means are available for {unavailable}")

    row_mean = np.nanmean(combined, axis=1, keepdims=True)
    row_std = np.nanstd(combined, axis=1, keepdims=True)
    row_std[~np.isfinite(row_std) | (row_std == 0)] = 1.0
    z_scores = {
        condition: (matrix - row_mean) / row_std
        for condition, matrix in mean_matrices.items()
    }
    progress_centers = (np.arange(n_bins) + 0.5) / n_bins
    return progress_centers, mean_matrices, z_scores


def plot_progress_heatmap(
    data: pd.DataFrame,
    value_cols,
    labels=None,
    progress_col: str = "trial_progress_distance",
    conditions=("outbound", "inbound"),
    n_bins: int = 40,
    min_trials: int = 5,
    row_boundaries=None,
    event_progress: pd.DataFrame | None = None,
    z_limit: float | None = 2.5,
    reward_ranges=None,
    figsize=None,
):
    """Plot outbound and inbound variable-by-progress summary heatmaps."""
    value_cols = list(value_cols)
    labels = list(labels) if labels is not None else value_cols
    if len(labels) != len(value_cols):
        raise ValueError("labels must have one entry per value column")

    data = exclude_reward_port_ranges(data, reward_ranges)
    progress, means, z_scores = compute_progress_heatmap(
        data,
        value_cols=value_cols,
        progress_col=progress_col,
        conditions=conditions,
        n_bins=n_bins,
        min_trials=min_trials,
    )
    combined_z = np.concatenate(
        [z_scores[condition] for condition in conditions],
        axis=1,
    )
    if z_limit is None:
        finite_z = np.abs(combined_z[np.isfinite(combined_z)])
        color_limit = max(1.0, float(np.max(finite_z)))
    else:
        color_limit = float(z_limit)
        if not np.isfinite(color_limit) or color_limit <= 0:
            raise ValueError("z_limit must be positive and finite, or None")
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")

    if figsize is None:
        figsize = (12.0, max(5.0, 0.38 * len(value_cols) + 1.5))
    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=figsize,
        sharey=True,
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    image = None
    tick_positions = np.linspace(-0.5, n_bins - 0.5, 6)
    tick_labels = [f"{value:.1f}" for value in np.linspace(0.0, 1.0, 6)]

    for index, (ax, condition) in enumerate(zip(axes, conditions)):
        image = ax.imshow(
            z_scores[condition],
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap,
            norm=norm,
        )
        ax.set_xlabel(f"{condition.title()} trial progress", fontsize=9)
        ax.set_xticks(tick_positions, tick_labels)
        ax.set_yticks(np.arange(len(labels)))
        if index == 0:
            ax.set_yticklabels(labels)
        ax.tick_params(labelsize=8)
        for boundary in row_boundaries or []:
            ax.axhline(boundary - 0.5, color="white", linewidth=1.5)
        if event_progress is not None:
            condition_events = event_progress.loc[
                event_progress["trial_type"].eq(condition)
            ]
            for event in condition_events.itertuples(index=False):
                marker_x = event.progress * n_bins - 0.5
                ax.scatter(
                    marker_x,
                    1.015,
                    marker="v",
                    s=48,
                    color="black",
                    transform=ax.get_xaxis_transform(),
                    clip_on=False,
                    zorder=5,
                )
                ax.text(
                    marker_x,
                    1.045,
                    event.event,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    transform=ax.get_xaxis_transform(),
                    clip_on=False,
                )

    colorbar = fig.colorbar(
        image,
        ax=axes,
        fraction=0.03,
        pad=0.02,
        extend="both",
    )
    colorbar.set_label("Row-wise z-scored mean", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    return fig, axes, {
        "progress": progress,
        "mean": means,
        "z_score": z_scores,
    }


def plot_variable_pair(
    data: pd.DataFrame,
    value_col: str,
    label: str,
    kind: str,
    progress_col: str = "trial_progress_distance",
    progress_value_col: str | None = None,
    progress_label: str | None = None,
    progress_summary_kwargs: dict | None = None,
    reward_ranges=None,
    figsize=(14, 4.2),
):
    """Plot progress tuning and outbound/inbound W-track maps.

    By default the trial-progress line and the W-track maps both use
    `value_col`. Pass `progress_value_col`/`progress_label` to plot a
    different column on the progress axis (e.g. an absolute value) while the
    W-track maps stay on `value_col`.
    """
    data = exclude_reward_port_ranges(data, reward_ranges)
    if "trial_type" not in data:
        raise ValueError("data is missing 'trial_type'")
    fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")
    plot_progress_with_ci(
        data,
        value_col=progress_value_col or value_col,
        label=progress_label or label,
        progress_col=progress_col,
        circular=kind in {"circular", "circular_signed", "circular_degrees"},
        circular_period=360.0 if kind == "circular_degrees" else 2 * np.pi,
        ax=axes[0],
        **(progress_summary_kwargs or {}),
    )
    color_scale = _color_scale(data[value_col].to_numpy(dtype=float), kind)
    spatial = None
    for ax, trial_type in [
        (axes[1], "outbound"),
        (axes[2], "inbound"),
    ]:
        _, spatial = plot_wtrack_variable_map(
            data.loc[data["trial_type"].eq(trial_type)],
            value_col=value_col,
            label=label,
            kind=kind,
            ax=ax,
            color_scale=color_scale,
            add_colorbar=False,
        )
        ax.set_title(trial_type.title(), fontsize=10)
        ax.set_xlabel("x position", fontsize=9)
    colorbar = fig.colorbar(spatial, ax=axes[1:], fraction=0.035, pad=0.03)
    colorbar.set_label(label, fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    return fig, axes


def plot_variable_pair_by_trial_type(
    data: pd.DataFrame,
    value_col: str,
    label: str,
    kind: str,
    progress_col: str = "trial_progress_distance",
    reward_ranges=None,
    figsize=(9.5, 8.0),
):
    """Plot progress tuning and W-track maps, each split by trial type.

    Top row is the outbound/inbound trial-progress profiles; bottom row is
    the matching outbound/inbound W-track spatial maps.
    """
    data = exclude_reward_port_ranges(data, reward_ranges)
    if "trial_type" not in data:
        raise ValueError("data is missing 'trial_type'")
    fig, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained")
    circular = kind in {"circular", "circular_signed", "circular_degrees"}
    circular_period = 360.0 if kind == "circular_degrees" else 2 * np.pi
    color_scale = _color_scale(data[value_col].to_numpy(dtype=float), kind)
    spatial = None
    for col, (trial_type, title) in enumerate(
        [("outbound", "Outbound"), ("inbound", "Inbound")]
    ):
        subset = data.loc[data["trial_type"].eq(trial_type)]
        plot_progress_with_ci(
            subset,
            value_col=value_col,
            label=label,
            progress_col=progress_col,
            circular=circular,
            circular_period=circular_period,
            ax=axes[0, col],
        )
        axes[0, col].set_xlabel("Trial progress", fontsize=9)
        axes[0, col].set_title(title, fontsize=10)
        _, spatial = plot_wtrack_variable_map(
            subset,
            value_col=value_col,
            label=label,
            kind=kind,
            ax=axes[1, col],
            color_scale=color_scale,
            add_colorbar=False,
        )
        axes[1, col].set_xlabel("x position", fontsize=9)
    colorbar = fig.colorbar(spatial, ax=axes[1, :], fraction=0.035, pad=0.03)
    colorbar.set_label(label, fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    return fig, axes


def plot_paw_position_panel(
    data: pd.DataFrame,
    paw_prefix: str,
    paw_label: str,
    progress_col: str = "trial_progress_distance",
    progress_window_size: float = 0.02,
    progress_step_size: float = 0.005,
    reward_ranges=None,
):
    """Plot median-centered AP and lateral position for one selected paw.

    Paw-position profiles use narrow, densely sampled progress windows so the
    within-trial stepping structure is not attenuated by broad averaging.
    """
    specs = [
        (
            f"{paw_prefix}_ap_displacement_cm",
            "Anterior–posterior displacement (cm)",
        ),
        (
            f"{paw_prefix}_lateral_displacement_cm",
            "Lateral displacement (cm)",
        ),
    ]
    data = exclude_reward_port_ranges(data, reward_ranges)
    if "trial_type" not in data:
        raise ValueError("data is missing 'trial_type'")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for row, (column, label) in enumerate(specs):
        plot_progress_with_ci(
            data,
            value_col=column,
            label=label,
            progress_col=progress_col,
            window_size=progress_window_size,
            step_size=progress_step_size,
            ax=axes[row, 0],
        )
        color_scale = _color_scale(data[column].to_numpy(dtype=float), "signed")
        spatial = None
        for ax, trial_type in [
            (axes[row, 1], "outbound"),
            (axes[row, 2], "inbound"),
        ]:
            _, spatial = plot_wtrack_variable_map(
                data.loc[data["trial_type"].eq(trial_type)],
                value_col=column,
                label=label,
                kind="signed",
                ax=ax,
                color_scale=color_scale,
                add_colorbar=False,
            )
            ax.set_xlabel("x position", fontsize=9)
            if row == 0:
                ax.set_title(trial_type.title(), fontsize=10)
        colorbar = fig.colorbar(
            spatial,
            ax=axes[row, 1:],
            fraction=0.035,
            pad=0.03,
        )
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    return fig, axes


def plot_paw_velocity_panel(
    data: pd.DataFrame,
    paw_prefix: str,
    paw_label: str,
    progress_col: str = "trial_progress_distance",
    progress_window_size: float = 0.02,
    progress_step_size: float = 0.005,
    reward_ranges=None,
):
    """Plot AP velocity, lateral velocity, and their speed magnitude.

    Profiles use narrow, densely sampled progress windows so limb-cycle
    structure is not attenuated by broad averaging.
    """
    specs = [
        (f"{paw_prefix}_ap_velocity_cm_s", "AP velocity (cm/s)", "signed"),
        (
            f"{paw_prefix}_lateral_velocity_cm_s",
            "Lateral velocity (cm/s)",
            "signed",
        ),
        (f"{paw_prefix}_speed_cm_s", "Relative speed (cm/s)", "positive"),
    ]
    data = exclude_reward_port_ranges(data, reward_ranges)
    if "trial_type" not in data:
        raise ValueError("data is missing 'trial_type'")
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), layout="constrained")
    for row, (column, label, kind) in enumerate(specs):
        plot_progress_with_ci(
            data,
            value_col=column,
            label=label,
            progress_col=progress_col,
            window_size=progress_window_size,
            step_size=progress_step_size,
            ax=axes[row, 0],
        )
        color_scale = _color_scale(data[column].to_numpy(dtype=float), kind)
        spatial = None
        for ax, trial_type in [
            (axes[row, 1], "outbound"),
            (axes[row, 2], "inbound"),
        ]:
            _, spatial = plot_wtrack_variable_map(
                data.loc[data["trial_type"].eq(trial_type)],
                value_col=column,
                label=label,
                kind=kind,
                ax=ax,
                color_scale=color_scale,
                add_colorbar=False,
            )
            ax.set_xlabel("x position", fontsize=9)
            if row == 0:
                ax.set_title(trial_type.title(), fontsize=10)
        colorbar = fig.colorbar(
            spatial,
            ax=axes[row, 1:],
            fraction=0.035,
            pad=0.03,
        )
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    return fig, axes
