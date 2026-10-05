"""Small, standalone plots for the COSYNE abstract figure notebook v2."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from cosyne_abstract_figures import COLORS, publication_style


__all__ = [
    "V2_FIGURE_STYLE",
    "load_wtrack_trials",
    "plot_wtrack_trial",
    "plot_wtrack_angular_velocity_trial",
    "save_wtrack_trial_svg",
    "save_wtrack_angular_velocity_trial_svg",
    "show_wtrack_trial_selector",
    "show_wtrack_angular_velocity_trial_selector",
    "build_gait_example_cache",
    "load_gait_example",
    "plot_wtrack_gait_phase_trial",
    "save_wtrack_gait_phase_trial_svg",
    "show_wtrack_gait_phase_trial_selector",
    "plot_progress_speed_profile",
    "save_progress_speed_profile_svg",
    "show_wtrack_progress_speed_selector",
    "NORMALIZED_PROFILE_SPECS",
    "plot_progress_fit_grid",
    "save_progress_fit_grid_svg",
    "plot_progress_predictability",
    "save_progress_predictability_svg",
    "plot_normalized_progress_profile",
    "save_normalized_progress_profile_svg",
]


BASE_FONT_SIZE = 10.0
SMALL_FONT_SIZE = BASE_FONT_SIZE / 2.0
V2_FIGURE_STYLE = {
    "axes.labelsize": SMALL_FONT_SIZE,
    "xtick.labelsize": SMALL_FONT_SIZE,
    "ytick.labelsize": SMALL_FONT_SIZE,
}


@contextmanager
def _v2_publication_style():
    """Apply the shared style, with labels and tick labels at half size."""
    with publication_style(font_size=BASE_FONT_SIZE):
        with mpl.rc_context(V2_FIGURE_STYLE):
            yield


WTRACK_SVG_DIR = Path("/home/labuser/Documents/lab-notebook/presentations/Cosyne")


WTRACK_POSITION_FILES = {
    "Wilbur": Path(
        "/media/labuser/NA_1_2025/spyglass/wilbur/analysis/position/"
        "trialized_position_center0.csv"
    ),
    "Senor": Path(
        "/media/labuser/NA_1_2025/spyglass/Senor20201220/analysis/position/"
        "trialized_position_Senor20201220_center0.csv"
    ),
    # Repaired, validated export combining epochs 2, 4, 6 and 8.
    "J": Path(
        "/media/labuser/NA_1_2025/spyglass_common/notebooks/repaired_position/"
        "J1620210722/trialized_position_J1620210722_center0_epoch2_filter_validated.csv"
    ),
}


@lru_cache(maxsize=3)
def load_wtrack_trials(animal: str) -> pd.DataFrame:
    """Load only the columns needed for the projected trial trajectory."""
    if animal not in WTRACK_POSITION_FILES:
        raise ValueError(f"Choose one of {tuple(WTRACK_POSITION_FILES)}")
    path = WTRACK_POSITION_FILES[animal]
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = (
        "time",
        "epoch",
        "trial_number",
        "trial_type",
        "speed",
        "orientation",
        "trial_progress_distance",
        "projected_x_position",
        "projected_y_position",
    )
    data = pd.read_csv(path, usecols=columns)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=(
            "epoch",
            "trial_number",
            "speed",
            "projected_x_position",
            "projected_y_position",
        )
    )
    data["epoch"] = data["epoch"].astype(int)
    data["trial_number"] = data["trial_number"].astype(int)
    return data


@lru_cache(maxsize=3)
def _wtrack_geometry(animal: str):
    # encoding_utils sets a Seaborn context on import; retain the figure style.
    with mpl.rc_context():
        from encoding_utils import (
            SENOR_WTRACK_CENTER0_LINEAR_EDGE_ORDER,
            SENOR_WTRACK_CENTER0_NODE_POSITIONS,
            WILBUR_WTRACK_CENTER0_LINEAR_EDGE_ORDER,
            WILBUR_WTRACK_CENTER0_NODE_POSITIONS,
        )

    if animal == "Wilbur":
        return WILBUR_WTRACK_CENTER0_NODE_POSITIONS, WILBUR_WTRACK_CENTER0_LINEAR_EDGE_ORDER
    if animal == "Senor":
        return SENOR_WTRACK_CENTER0_NODE_POSITIONS, SENOR_WTRACK_CENTER0_LINEAR_EDGE_ORDER
    if animal == "J":
        return _j16_wtrack_geometry()
    raise ValueError(f"Choose one of {tuple(WTRACK_POSITION_FILES)}")


def _j16_wtrack_geometry(cache_dir: str | Path | None = None):
    """J16 node positions and edge order; fetched once from the track graph, then cached."""
    import json

    cache_path = Path(cache_dir or Path(__file__).resolve().parent / "cosyne_cache") / "j16_wtrack_geometry.json"
    if not cache_path.is_file():
        import spyglass.common  # noqa: F401  (loads the DataJoint config)
        import spyglass.linearization.v1 as sgpl

        graph = (sgpl.TrackGraph & {"track_graph_name": "Wtrack_J1620210722"}).fetch1()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "node_positions": {
                str(i): [float(x) for x in position]
                for i, position in enumerate(graph["node_positions"])
            },
            "linear_edge_order": [[int(a), int(b)] for a, b in graph["linear_edge_order"]],
        }))
    stored = json.loads(cache_path.read_text())
    nodes = {int(i): tuple(position) for i, position in stored["node_positions"].items()}
    return nodes, tuple(tuple(edge) for edge in stored["linear_edge_order"])


def _trial_options(data: pd.DataFrame):
    options = []
    for (epoch, trial), group in data.groupby(["epoch", "trial_number"], sort=True):
        if len(group) < 2:
            continue
        trial_types = group["trial_type"].dropna()
        kind = str(trial_types.iloc[0]) if not trial_types.empty else "unlabeled"
        options.append((f"epoch {epoch} · trial {trial} ({kind})", (epoch, trial)))
    if not options:
        raise ValueError("No trials with at least two valid position and speed samples")
    return options


def plot_wtrack_trial(
    animal: str,
    epoch: int,
    trial_number: int,
    *,
    data: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (3.25, 2.25),
    metric: str = "speed",
    progress_arrow: bool = False,
):
    """Draw a W-track graph with a speed- or angular-velocity-colored path.

    ``progress_arrow`` adds an arrow beside the route, labeled 0 to 1, for
    trial progress.
    """
    if data is None:
        data = load_wtrack_trials(animal)
    trial = data.loc[
        data["epoch"].eq(epoch) & data["trial_number"].eq(trial_number)
    ].sort_values("time")
    if trial.empty:
        raise ValueError(f"No valid samples for {animal}, epoch {epoch}, trial {trial_number}")

    nodes, edges = _wtrack_geometry(animal)
    raw_nodes = np.asarray(list(nodes.values()), dtype=float)
    x_origin = float(raw_nodes[:, 1].min())
    y_reference = float(raw_nodes[:, 0].max())
    graph_points = {
        node: (position[1] - x_origin, y_reference - position[0])
        for node, position in nodes.items()
    }
    trial_x = trial["projected_y_position"].to_numpy(float) - x_origin
    trial_y = y_reference - trial["projected_x_position"].to_numpy(float)
    if metric == "speed":
        values = trial["speed"].to_numpy(float)
        cmap = "turbo"
        # Round the top of the scale up to a multiple of 10 so its label is clean.
        top = max(10.0, float(np.ceil(values.max() / 10.0) * 10.0))
        norm = mpl.colors.Normalize(vmin=0, vmax=top)
        colorbar_label = "Speed (cm/s)"
    elif metric == "angular_velocity":
        trial = trial.dropna(subset=["orientation"]).copy()
        if len(trial) < 3:
            raise ValueError(
                f"Too few orientation samples for {animal}, epoch {epoch}, "
                f"trial {trial_number}"
            )
        trial_x = trial["projected_y_position"].to_numpy(float) - x_origin
        trial_y = y_reference - trial["projected_x_position"].to_numpy(float)
        time = trial["time"].to_numpy(float)
        orientation = np.unwrap(trial["orientation"].to_numpy(float))
        median_dt = float(np.median(np.diff(time)))
        sigma_samples = max(0.0, 0.1 / median_dt)
        smoothed_orientation = gaussian_filter1d(
            orientation, sigma=sigma_samples, mode="nearest"
        )
        values = np.gradient(smoothed_orientation, time)
        finite_abs = np.abs(values[np.isfinite(values)])
        limit = max(1.0, float(np.quantile(finite_abs, 0.98)))
        cmap = "coolwarm"
        norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        colorbar_label = "Angular velocity (rad/s)"
    elif metric == "gait_phase":
        if animal != "Wilbur":
            raise ValueError("Gait phase is only cached for Wilbur")
        gait_features, _ = load_gait_example()
        sin_phase = gait_features["gait_phase_sin"].reindex(trial["time"]).to_numpy(float)
        cos_phase = gait_features["gait_phase_cos"].reindex(trial["time"]).to_numpy(float)
        # NaN outside accepted stride intervals, where phase is undefined.
        values = (np.arctan2(sin_phase, cos_phase) % (2 * np.pi)) / (2 * np.pi)
        # coolwarm run out and back, so phase 0 and 1 share a color.
        coolwarm = plt.get_cmap("coolwarm")
        cmap = mpl.colors.ListedColormap(
            np.vstack((
                coolwarm(np.linspace(0, 1, 128)),
                coolwarm(np.linspace(1, 0, 128)),
            )),
            name="coolwarm_cyclic",
        )
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        colorbar_label = "Gait phase (cycles)"
    else:
        raise ValueError(
            "metric must be 'speed', 'angular_velocity' or 'gait_phase'"
        )

    points = np.column_stack((trial_x, trial_y))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    # Skip isolated tracking jumps rather than drawing lines across the graph.
    continuous = np.linalg.norm(np.diff(points, axis=0), axis=1) <= 15.0
    if metric == "gait_phase":
        # Phase is circular: take each segment's starting phase, not a mean.
        segment_values = values[:-1]
        undefined = continuous & ~np.isfinite(segment_values)
        continuous = continuous & np.isfinite(segment_values)
    else:
        segment_values = (values[:-1] + values[1:]) / 2.0
        undefined = None

    with _v2_publication_style():
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        for start, end in edges:
            edge_points = np.asarray([graph_points[start], graph_points[end]])
            ax.plot(
                edge_points[:, 0], edge_points[:, 1],
                color=COLORS["intercept"], linewidth=1.3, zorder=1,
            )
        if undefined is not None and undefined.any():
            ax.add_collection(LineCollection(
                segments[undefined], colors=COLORS["grid"],
                linewidth=2.5, rasterized=True, zorder=2,
            ))
        overlay_path = LineCollection(
            segments[continuous], cmap=cmap, norm=norm,
            linewidth=2.5, rasterized=True, zorder=3,
        )
        overlay_path.set_array(segment_values[continuous])
        ax.add_collection(overlay_path)
        graph_array = np.asarray(list(graph_points.values()))
        ax.scatter(
            graph_array[:, 0], graph_array[:, 1], s=24,
            facecolor="white", edgecolor=COLORS["full"],
            linewidth=0.8, zorder=4,
        )
        padding = 8.0
        ax.set_xlim(graph_array[:, 0].min() - padding, graph_array[:, 0].max() + padding)
        ax.set_ylim(graph_array[:, 1].min() - padding, graph_array[:, 1].max() + padding)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_title(f"{animal} · epoch {epoch}, trial {trial_number}")
        if progress_arrow:
            _add_progress_arrow(ax, trial, x_origin, y_reference, graph_array)
        colorbar = fig.colorbar(overlay_path, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(colorbar_label)
        if metric == "speed":
            # Label only the ends of the scale.
            top = float(norm.vmax)
            colorbar.set_ticks([0.0, top])
            colorbar.set_ticklabels(["0", f"{top:.0f}"])
        # Horizontal extent of the track nodes (figure fractions), so a plot
        # placed beneath can share the same width.
        fig.canvas.draw()
        node_x = ax.transData.transform([[graph_array[:, 0].min(), 0], [graph_array[:, 0].max(), 0]])
        fig.track_x_fraction = tuple(fig.transFigure.inverted().transform(node_x)[:, 0])
    return fig


def _add_progress_arrow(ax, trial, x_origin, y_reference, graph_array):
    """Draw a 0 -> 1 arrow alongside the trial's route through the track nodes.

    The route is the first sample, the track nodes the trial passes, then the
    last sample. Each leg is offset to the inside of its nearest turn.
    """
    color = COLORS["task"]
    xy = np.column_stack((
        trial["projected_y_position"].to_numpy(float) - x_origin,
        y_reference - trial["projected_x_position"].to_numpy(float),
    ))
    span = np.ptp(graph_array, axis=0)
    radius = 0.12 * float(span.max())  # paths can cut corners well inside the nodes
    first, last = xy[0], xy[-1]
    visited = []
    for node in graph_array:
        near = np.flatnonzero(np.linalg.norm(xy - node, axis=1) < radius)
        if len(near) and np.linalg.norm(first - node) > radius and np.linalg.norm(last - node) > radius:
            visited.append((near[0], node))
    route = np.vstack([first, *[n for _, n in sorted(visited, key=lambda v: v[0])], last])
    if len(route) < 2:
        return
    # Snap the end samples onto the axis of their leg so the arrow is straight.
    for end, nxt in ((0, 1), (-1, -2)):
        delta = route[nxt] - route[end]
        if len(route) > 2:
            if abs(delta[0]) < abs(delta[1]):
                route[end, 0] = route[nxt, 0]
            else:
                route[end, 1] = route[nxt, 1]
    legs = np.diff(route, axis=0)
    legs = legs / np.linalg.norm(legs, axis=1, keepdims=True)
    cross = legs[:-1, 0] * legs[1:, 1] - legs[:-1, 1] * legs[1:, 0]
    offset = 0.05 * float(span.max())
    normals = []
    for k, leg in enumerate(legs):
        turn = cross[k] if k < len(cross) else cross[-1] if len(cross) else 1.0
        side = 1.0 if turn >= 0 else -1.0  # left of travel for a left turn
        normals.append(side * np.array([-leg[1], leg[0]]))
    shifted = [route[0] + offset * normals[0]]
    for k in range(1, len(route) - 1):
        a, b = normals[k - 1], normals[k]
        shifted.append(route[k] + offset * (a + b) if abs(a @ b) < 1 - 1e-9 else route[k] + offset * a)
    shifted.append(route[-1] + offset * normals[-1])
    shifted = np.asarray(shifted)
    ax.plot(shifted[:-1, 0], shifted[:-1, 1], color=color, linewidth=0.9, zorder=5, solid_capstyle="butt")
    ax.annotate(
        "", xy=shifted[-1], xytext=shifted[-2],
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=0.9, shrinkA=0, shrinkB=0),
        zorder=5,
    )
    for label, point, step in (("0", shifted[0], -legs[0]), ("1", shifted[-1], legs[-1])):
        ax.annotate(
            label, xy=point, xytext=0.35 * offset * step + point, color=color,
            fontweight="bold", fontsize=SMALL_FONT_SIZE + 1,
            ha="center", va="center", zorder=6,
        )
    centre = 0.5 * (shifted.min(axis=0) + shifted.max(axis=0))
    ax.text(
        centre[0], centre[1], "trial progress", color=color,
        fontsize=SMALL_FONT_SIZE + 1, ha="center", va="center", zorder=6,
    )


def plot_wtrack_angular_velocity_trial(*args, **kwargs):
    """Draw the selected W-track trial colored by signed angular velocity."""
    return plot_wtrack_trial(*args, metric="angular_velocity", **kwargs)


def save_wtrack_trial_svg(
    animal: str,
    epoch: int,
    trial_number: int,
    *,
    data: pd.DataFrame | None = None,
    directory: str | Path = WTRACK_SVG_DIR,
    metric: str = "speed",
    progress_arrow: bool = False,
) -> Path:
    """Save the currently selected standalone plot as an editable SVG."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    metric_name = {
        "speed": "speed",
        "gait_phase": "gait_phase",
    }.get(metric, "angular_velocity")
    path = directory / (
        f"cosyne_wtrack_{metric_name}_{animal.lower()}_e{epoch}_trial{trial_number}.svg"
    )
    fig = plot_wtrack_trial(
        animal, epoch, trial_number, data=data, metric=metric,
        progress_arrow=progress_arrow,
    )
    try:
        with mpl.rc_context({"svg.fonttype": "none"}):
            fig.savefig(path, format="svg")
    finally:
        plt.close(fig)
    return path


def save_wtrack_angular_velocity_trial_svg(*args, **kwargs):
    """Save the selected angular-velocity W-track plot as an SVG."""
    return save_wtrack_trial_svg(*args, metric="angular_velocity", **kwargs)


def show_wtrack_trial_selector(animal: str, *, metric: str = "speed"):
    """Display an epoch-aware trial dropdown, plot, and Save SVG button."""
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    data = load_wtrack_trials(animal)
    options = _trial_options(data)
    preferred = next((value for label, value in options if "(outbound)" in label), options[0][1])
    dropdown = widgets.Dropdown(
        options=options, value=preferred, description="Trial:",
        layout=widgets.Layout(width="440px"),
    )
    save_button = widgets.Button(description="Save SVG")
    output = widgets.Output()
    save_status = widgets.Output()

    def draw(_=None):
        epoch, trial_number = dropdown.value
        with output:
            clear_output(wait=True)
            fig = plot_wtrack_trial(
                animal, epoch, trial_number, data=data, metric=metric
            )
            display(fig)
            plt.close(fig)

    dropdown.observe(draw, names="value")

    def save_selected(_):
        epoch, trial_number = dropdown.value
        with save_status:
            clear_output(wait=True)
            try:
                path = save_wtrack_trial_svg(
                    animal, epoch, trial_number, data=data, metric=metric,
                )
            except OSError as error:
                print(f"Could not save SVG: {error}")
            else:
                print(f"Saved SVG: {path}")

    save_button.on_click(save_selected)
    draw()
    display(widgets.VBox([widgets.HBox([dropdown, save_button]), output, save_status]))
    return dropdown


def show_wtrack_angular_velocity_trial_selector(animal: str):
    """Display the trial selector for the angular-velocity overlay."""
    return show_wtrack_trial_selector(animal, metric="angular_velocity")


# --- Gait phase example ------------------------------------------------------

GAIT_CACHE_DIR = Path(__file__).resolve().parent / "cosyne_cache"
GAIT_SESSION_NAME = "wilbur20210512"
GAIT_FEATURE_COLUMNS = (
    "epoch",
    "trial_number",
    "speed",
    "reference_paw_world_speed_cm_s",
    "gait_phase_sin",
    "gait_phase_cos",
)


def build_gait_example_cache(cache_dir: str | Path = GAIT_CACHE_DIR) -> Path:
    """Derive the validated gait phase once (read-only DB fetch) and cache it.

    Mirrors the pose fetch and ``derive_movement_features`` call of
    ``movement_groups_encoding.ipynb`` with its visually checked gait settings.
    """
    import spyglass.common  # noqa: F401  (loads dj config via common_imports)
    import spyglass.linearization.v1 as sgpl
    import spyglass.position.v1 as sgp
    from spyglass.position import PositionOutput
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    with mpl.rc_context():
        import encoding_utils as eu
        from movement_features import DEFAULT_PAW_PREFIXES, derive_movement_features

    nwb_file = get_nwb_copy_filename(f"{GAIT_SESSION_NAME}.nwb")
    raw = pd.read_csv(eu.CONFIG["trialized_position"], index_col="time")
    epochs = sorted(raw["epoch"].dropna().astype(int).unique().tolist())
    pos_rows = (
        PositionOutput.DLCPosV1
        & {"nwb_file_name": nwb_file}
        & [{"epoch": e} for e in epochs]
    ).fetch(as_dict=True)
    key_names = (
        "nwb_file_name", "epoch", "video_file_num", "project_name",
        "dlc_model_name", "dlc_model_params_name",
    )
    pose_by_epoch = {}
    for epoch in epochs:
        rows = [r for r in pos_rows if int(r["epoch"]) == epoch]
        if len(rows) != 1:
            raise ValueError(f"Expected one DLCPosV1 row for epoch {epoch}; found {len(rows)}")
        row = rows[0]
        base = {k: row[k] for k in key_names}
        params = (
            sgp.DLCSmoothInterpCohortSelection
            & {**base, "dlc_si_cohort_selection_name": row["dlc_si_cohort_centroid"]}
        ).fetch1("bodyparts_params_dict")
        pose_by_epoch[epoch] = {
            part: (
                sgp.DLCSmoothInterp
                & {**base, "bodypart": part, "dlc_si_params_name": params[part]}
            ).fetch1_dataframe()
            for part in DEFAULT_PAW_PREFIXES
        }
    track_graph = (
        sgpl.TrackGraph & {"track_graph_name": eu.CONFIG["wtrack_name"]}
    ).fetch1()

    features, gait_events = derive_movement_features(
        raw,
        pose_by_epoch=pose_by_epoch,
        node_positions=track_graph["node_positions"],
        linear_edge_order=track_graph["linear_edge_order"],
        paw_prefixes=DEFAULT_PAW_PREFIXES.copy(),
        center_well_node=0,
        track_segment_order=(3, 1, 0, 2, 4),
        gait_reference_paw="forepawR",
        gait_smoothing_s=0.1,
        gait_peak_prominence_cm_s=10.0,
        gait_peak_distance_samples=2,
        gait_swing_boundary_cm_s=15.0,
        gait_min_body_speed_cm_s=5.0,
        gait_min_stance_samples=2,
        gait_min_stride_s=0.18,
        gait_max_stride_s=0.80,
        return_gait_events=True,
    )
    out = raw[["epoch", "trial_number", "speed"]].join(
        features[["reference_paw_world_speed_cm_s", "gait_phase_sin", "gait_phase_cos"]]
    )
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_dir / "gait_features.csv")
    gait_events.to_csv(cache_dir / "gait_events.csv")
    return cache_dir


@lru_cache(maxsize=1)
def load_gait_example(cache_dir: str | Path = GAIT_CACHE_DIR):
    """Return cached (features, touchdown-event) tables; build them if absent."""
    cache_dir = Path(cache_dir)
    if not (cache_dir / "gait_features.csv").is_file():
        build_gait_example_cache(cache_dir)
    features = pd.read_csv(cache_dir / "gait_features.csv", index_col=0)
    events = pd.read_csv(cache_dir / "gait_events.csv", index_col=0)
    return features, events


def plot_wtrack_gait_phase_trial(*args, **kwargs):
    """Draw the selected W-track trial colored by gait phase (cycles)."""
    return plot_wtrack_trial(*args, metric="gait_phase", **kwargs)


def save_wtrack_gait_phase_trial_svg(*args, **kwargs):
    """Save the selected gait-phase W-track plot as an SVG."""
    return save_wtrack_trial_svg(*args, metric="gait_phase", **kwargs)


def show_wtrack_gait_phase_trial_selector(animal: str = "Wilbur"):
    """Display the trial selector for the gait-phase overlay."""
    return show_wtrack_trial_selector(animal, metric="gait_phase")


def plot_progress_speed_profile(
    animal: str,
    epoch: int,
    trial_number: int,
    *,
    data: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (3.25, 1.2),
    smoothing_sigma: float = 0.0,
    align_to=None,
    height_in: float | None = None,
):
    """Speed against trial progress for one trial (x is progress, y is speed).

    ``height_in`` sets the figure height in inches (default: ``figsize[1]``).

    With ``align_to`` (a W-track figure from ``plot_wtrack_trial``) the axes span
    the same horizontal extent as that figure's track nodes.

    ``smoothing_sigma`` is the Gaussian width in samples; 0 plots raw speed.
    """
    if data is None:
        data = load_wtrack_trials(animal)
    trial = data.loc[
        data["epoch"].eq(epoch) & data["trial_number"].eq(trial_number)
    ].dropna(subset=["trial_progress_distance", "speed"])
    if trial.empty:
        raise ValueError(f"No valid samples for {animal}, epoch {epoch}, trial {trial_number}")
    trial = trial.sort_values("trial_progress_distance")

    speed = trial["speed"].to_numpy(float)
    if smoothing_sigma > 0:
        speed = gaussian_filter1d(speed, sigma=smoothing_sigma, mode="nearest")

    if height_in is not None:
        figsize = (figsize[0], float(height_in))
    with _v2_publication_style():
        if align_to is not None:
            x0, x1 = align_to.track_x_fraction
            height = figsize[1]
            bottom_in, top_in = 0.36, 0.1  # room for the tick labels and x label
            fig = plt.figure(figsize=(align_to.get_size_inches()[0], height))
            ax = fig.add_axes([
                x0, bottom_in / height, x1 - x0,
                max(0.05, height - bottom_in - top_in) / height,
            ])
        else:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.plot(
            trial["trial_progress_distance"], speed,
            color=COLORS["translation"], linewidth=1.2,
        )
        # A short plot has no room for a one-line rotated y label.
        ylabel = "Speed\n(cm/s)" if figsize[1] < 1.0 else "Speed (cm/s)"
        ax.set(xlabel="Trial progress", ylabel=ylabel, xlim=(0, 1), ylim=(0, None))
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def save_progress_speed_profile_svg(
    animal: str,
    epoch: int,
    trial_number: int,
    *,
    data: pd.DataFrame | None = None,
    directory: str | Path = WTRACK_SVG_DIR,
    smoothing_sigma: float = 0.0,
    align_to=None,
    height_in: float | None = None,
) -> Path:
    """Save the speed-vs-progress plot as an editable SVG."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    path = directory / (
        f"cosyne_speed_vs_progress_{animal.lower()}_e{epoch}_trial{trial_number}.svg"
    )
    fig = plot_progress_speed_profile(
        animal, epoch, trial_number, data=data, smoothing_sigma=smoothing_sigma,
        align_to=align_to, height_in=height_in,
    )
    try:
        with mpl.rc_context({"svg.fonttype": "none"}):
            fig.savefig(path, format="svg")
    finally:
        plt.close(fig)
    return path


def show_wtrack_progress_speed_selector(
    animal: str, *, smoothing_sigma: float = 1.0, profile_height_in: float = 0.8,
):
    """One trial dropdown driving the speed-colored W-track (with progress arrow)
    and the speed-vs-progress profile, smoothed by ``smoothing_sigma`` samples and
    ``profile_height_in`` inches tall."""
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    data = load_wtrack_trials(animal)
    options = _trial_options(data)
    preferred = next((value for label, value in options if "(outbound)" in label), options[0][1])
    dropdown = widgets.Dropdown(
        options=options, value=preferred, description="Trial:",
        layout=widgets.Layout(width="440px"),
    )
    save_button = widgets.Button(description="Save SVGs")
    output = widgets.Output()
    save_status = widgets.Output()

    def draw(_=None):
        epoch, trial_number = dropdown.value
        with output:
            clear_output(wait=True)
            track_fig = plot_wtrack_trial(
                animal, epoch, trial_number, data=data, progress_arrow=True,
            )
            profile_fig = plot_progress_speed_profile(
                animal, epoch, trial_number, data=data,
                smoothing_sigma=smoothing_sigma, align_to=track_fig,
                height_in=profile_height_in,
            )
            for fig in (track_fig, profile_fig):
                display(fig)
                plt.close(fig)

    dropdown.observe(draw, names="value")

    def save_selected(_):
        epoch, trial_number = dropdown.value
        with save_status:
            clear_output(wait=True)
            try:
                track_fig = plot_wtrack_trial(
                    animal, epoch, trial_number, data=data, progress_arrow=True,
                )
                try:
                    paths = [
                        save_wtrack_trial_svg(
                            animal, epoch, trial_number, data=data, progress_arrow=True,
                        ),
                        save_progress_speed_profile_svg(
                            animal, epoch, trial_number, data=data,
                            smoothing_sigma=smoothing_sigma, align_to=track_fig,
                            height_in=profile_height_in,
                        ),
                    ]
                finally:
                    plt.close(track_fig)
            except OSError as error:
                print(f"Could not save SVGs: {error}")
            else:
                for path in paths:
                    print(f"Saved SVG: {path}")

    save_button.on_click(save_selected)
    draw()
    display(widgets.VBox([widgets.HBox([dropdown, save_button]), output, save_status]))
    return dropdown


# --- Animal progress profiles, normalized to each animal's maximum -----------

# variable -> (normalized y label, raw y label, decimals used when printing the animal's maximum)
NORMALIZED_PROFILE_SPECS = {
    "speed": ("Speed (norm.)", "Speed (cm/s)", 0),
    "abs_body_angular_velocity_rad_s": ("Angular speed (norm.)", "Angular speed (rad/s)", 1),
    "mean_fore_hind_spacing": ("F\u2013H spacing (norm.)", "F\u2013H spacing (cm)", 1),
    "relative_paw_vigor": ("Relative paw vigor (norm.)", "Relative paw vigor (cm/s)", 0),
}
# Variables outside Figure 1B's four; their profiles live in separate *_extra files.
EXTRA_PROFILE_VARIABLES = ("relative_paw_vigor",)


# Lower y limit of the normalized plot for variables that should not start at zero.
NORMALIZED_PROFILE_YMIN = {"mean_fore_hind_spacing": 0.4}


def load_progress_profiles(run_mask: bool = False, *, extra: bool = False) -> pd.DataFrame:
    """Saved Figure 1B profiles for all animals, with or without the run-zone mask.

    Unmasked profiles keep every outbound/inbound sample and the original
    full-trial progress. Masked ones are the run-only, renormalized profiles.
    ``extra`` loads the profiles of ``EXTRA_PROFILE_VARIABLES``; any that are
    missing are generated once by running the Figure 1 script (read-only
    database fetch, as in the original notebook).
    """
    import subprocess
    import sys

    from cosyne_abstract_figures import FIGURE1_ANIMALS

    root = Path(__file__).resolve().parent / "abstract_figures"
    suffix = "" if run_mask else "_unmasked"
    name = "progress_profiles_extra" if extra else "progress_profiles"
    paths = [root / f"figure_1B_{animal.lower()}_{name}{suffix}.csv" for animal in FIGURE1_ANIMALS]
    if extra:
        script = Path(__file__).resolve().parent / "figure1_multi_animal_predictability.py"
        for animal, path in zip(FIGURE1_ANIMALS, paths):
            if not path.is_file():
                flags = ["--profiles", "--extra-profiles"] + ([] if run_mask else ["--unmasked-profiles"])
                print(f"Generating {path.name} (one-time)...", flush=True)
                subprocess.run([sys.executable, str(script), animal, *flags], check=True, cwd=script.parent)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}; run figure1_multi_animal_predictability.py <animal> --profiles"
            + ("" if run_mask else " --unmasked-profiles")
        )
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def progress_profile_trial_counts(variable: str = "speed", *, run_mask: bool = False) -> pd.DataFrame:
    """Number of trials behind each animal's profile (min and max over progress bins)."""
    from cosyne_abstract_figures import FIGURE1_ANIMALS

    profiles = load_progress_profiles(run_mask, extra=variable in EXTRA_PROFILE_VARIABLES)
    profiles = profiles.loc[profiles["variable"].eq(variable) & profiles["mean"].notna()]
    counts = profiles.groupby("animal")["n_trials"].agg(["min", "max"])
    return counts.reindex(list(FIGURE1_ANIMALS)).rename(
        columns={"min": "n_trials_min", "max": "n_trials_max"}
    )


def plot_normalized_progress_profile(
    variable: str = "speed",
    *,
    figsize: tuple[float, float] = (3.25, 1.5),
    normalize: bool = True,
    show_ci: bool = False,
    run_mask: bool = False,
):
    """One variable's progress profile for each animal.

    Uses the saved Figure 1B mean curves. With ``normalize`` each animal's curve
    is divided by its own maximum; otherwise it is in original units. The
    maximum of each animal's curve (in original units) is printed above the plot
    in that animal's color. ``show_ci`` adds the 5th-95th percentile of the
    trial bootstrap as a shaded band. By default every outbound/inbound sample is
    used (no run-zone mask); ``run_mask=True`` restricts to run-zone samples.
    """
    from cosyne_abstract_figures import FIGURE1_ANIMALS, FIGURE1_ANIMAL_COLORS

    if variable not in NORMALIZED_PROFILE_SPECS:
        raise ValueError(f"Choose one of {tuple(NORMALIZED_PROFILE_SPECS)}")
    normalized_label, raw_label, decimals = NORMALIZED_PROFILE_SPECS[variable]
    ylabel = normalized_label if normalize else raw_label
    profiles = load_progress_profiles(run_mask, extra=variable in EXTRA_PROFILE_VARIABLES)
    profiles = profiles.loc[profiles["variable"].eq(variable)]
    if show_ci and not {"lower_5", "upper_95"}.issubset(profiles.columns):
        raise ValueError(
            "Saved profiles lack the 5-95 band; rerun "
            "figure1_multi_animal_predictability.py <animal> --profiles"
        )

    with _v2_publication_style():
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        upper_limit = 1.05
        positions = np.linspace(0, 1, 2 * len(FIGURE1_ANIMALS) + 1)[1::2]
        for animal, x_text in zip(FIGURE1_ANIMALS, positions):
            curve = profiles.loc[profiles["animal"].eq(animal)].dropna(subset=["mean"])
            peak = float(curve["mean"].max())
            color = FIGURE1_ANIMAL_COLORS[animal]
            scale = peak if normalize else 1.0
            if show_ci:
                band = curve.dropna(subset=["lower_5", "upper_95"])
                ax.fill_between(
                    band["progress"], band["lower_5"] / scale, band["upper_95"] / scale,
                    color=color, alpha=0.2, linewidth=0,
                )
                upper_limit = max(upper_limit, float(band["upper_95"].max() / peak) * 1.02)
            ax.plot(curve["progress"], curve["mean"] / scale, color=color, linewidth=1.55)
            ax.text(
                x_text, 1.02, f"{peak:.{decimals}f}", color=color,
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=SMALL_FONT_SIZE + 1,
            )
        ax.set(
            xlim=(0, 1),
            ylim=(NORMALIZED_PROFILE_YMIN.get(variable, 0.0) if normalize else 0.0,
                  upper_limit if normalize else None),
            xticks=(0.0, 0.5, 1.0), xlabel="Trial progress", ylabel=ylabel,
        )
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def save_normalized_progress_profile_svg(
    variable: str = "speed",
    *,
    directory: str | Path = WTRACK_SVG_DIR,
    **kwargs,
) -> Path:
    """Save the normalized progress profile as an editable SVG."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    path = directory / f"cosyne_normalized_{variable}_vs_progress.svg"
    fig = plot_normalized_progress_profile(variable, **kwargs)
    try:
        with mpl.rc_context({"svg.fonttype": "none"}):
            fig.savefig(path, format="svg")
    finally:
        plt.close(fig)
    return path


# --- Predictability of behavioral variables from trial progress --------------


def plot_progress_predictability(
    scores: pd.Series | None = None,
    *,
    figsize: tuple[float, float] = (3.25, 2.4),
    truncate: bool = False,
    last_variable: str = "Fore \u2212 hind vigor",
):
    """Ranked bars of cross-validated R^2 for predicting each variable from progress.

    Uses the saved Figure 1C scores (Wilbur, Senor, J), each variable's R^2
    averaged equally across the three animals. With ``truncate``, only bars
    ranked above ``last_variable``, and that variable itself, are drawn.
    """
    from cosyne_abstract_figures import (
        load_multi_animal_progress_r2, mean_animal_progress_r2, plot_ranked_progress_r2,
    )

    if scores is None:
        scores = mean_animal_progress_r2(load_multi_animal_progress_r2())
    if truncate:
        def _key(label):
            return "".join(str(label).lower().replace("\u2212", "-").replace("\u2013", "-").split())

        ranked = scores.sort_values(ascending=False)
        matches = [i for i, label in enumerate(ranked.index) if _key(label) == _key(last_variable)]
        if not matches:
            raise ValueError(f"last_variable must be one of {list(ranked.index)}")
        scores = ranked.iloc[: matches[0] + 1]
    with _v2_publication_style():
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_ranked_progress_r2(ax, scores, xlabel="Mean CV R\u00b2")
        ax.tick_params(axis="y", labelsize=SMALL_FONT_SIZE)
    return fig


def save_progress_predictability_svg(
    scores: pd.Series | None = None,
    *,
    directory: str | Path = WTRACK_SVG_DIR,
    **kwargs,
) -> Path:
    """Save the progress-predictability bars as an editable SVG."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    path = directory / "cosyne_progress_predictability.svg"
    fig = plot_progress_predictability(scores, **kwargs)
    try:
        with mpl.rc_context({"svg.fonttype": "none"}):
            fig.savefig(path, format="svg")
    finally:
        plt.close(fig)
    return path


# --- Observed vs model-predicted progress profiles (Wilbur) ------------------

# variable -> (y label, color key)
PROGRESS_FIT_SPECS = (
    ("speed", "Speed (cm/s)", "translation"),
    ("abs_body_angular_velocity_rad_s", "Angular speed (rad/s)", "turning"),
    ("mean_fore_hind_spacing", "F\u2013H spacing (cm)", "limb"),
)


def _progress_spline_fit(progress, values, grid):
    """Predict ``values`` on ``grid`` from progress with the predictability model.

    Same spline + ridge pipeline and hyperparameters as the defaults of
    ``summarize_progress_predictability`` (fit here on all samples, not CV).
    """
    import inspect

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer

    from movement_encoding import summarize_progress_predictability

    defaults = inspect.signature(summarize_progress_predictability).parameters
    valid = np.isfinite(progress) & np.isfinite(values)
    model = make_pipeline(
        SplineTransformer(
            n_knots=defaults["n_knots"].default, degree=defaults["degree"].default,
        ),
        Ridge(alpha=defaults["ridge_alpha"].default),
    )
    model.fit(progress[valid].reshape(-1, 1), values[valid])
    return model.predict(grid.reshape(-1, 1))


def plot_progress_fit_grid(
    *,
    figsize: tuple[float, float] = (3.25, 4.0),
):
    """Wilbur speed, angular velocity and F-H spacing against trial progress (3 rows).

    Each panel shows the observed mean profile (saved Figure 1B curve) and the
    spline + ridge fit of that variable on progress.
    """
    from cosyne_abstract_figures import (
        load_multi_animal_progress_profiles, prepare_real_figure1_data,
    )

    data, _, _ = prepare_real_figure1_data(compute_predictability=False)
    profiles = load_multi_animal_progress_profiles()
    profiles = profiles.loc[profiles["animal"].eq("Wilbur")]
    grid = np.linspace(0.0, 1.0, 201)
    progress = data["trial_progress_distance"].to_numpy(float)

    with _v2_publication_style():
        fig, axes = plt.subplots(
            3, 1, figsize=figsize, sharex=True, constrained_layout=True,
        )
        for ax, (variable, ylabel, color_key) in zip(axes, PROGRESS_FIT_SPECS):
            observed = profiles.loc[profiles["variable"].eq(variable)].dropna(subset=["mean"])
            predicted = _progress_spline_fit(progress, data[variable].to_numpy(float), grid)
            ax.plot(observed["progress"], observed["mean"], color=COLORS[color_key], linewidth=1.55)
            ax.plot(grid, predicted, color=COLORS["full"], linewidth=1.0, linestyle=(0, (3, 2)))
            ax.set(xlim=(0, 1), xticks=(0.0, 0.5, 1.0), ylabel=ylabel)
            ax.spines[["top", "right"]].set_visible(False)
        axes[-1].set_xlabel("Trial progress")
        axes[0].legend(
            handles=[
                plt.Line2D([], [], color=COLORS["intercept"], linewidth=1.55, label="Observed"),
                plt.Line2D([], [], color=COLORS["full"], linewidth=1.0,
                           linestyle=(0, (3, 2)), label="Predicted from progress"),
            ],
            loc="upper left", frameon=False, fontsize=SMALL_FONT_SIZE + 1,
        )
    return fig


def save_progress_fit_grid_svg(*, directory: str | Path = WTRACK_SVG_DIR, **kwargs) -> Path:
    """Save the observed-vs-fit grid as an editable SVG."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    path = directory / "cosyne_progress_fit_grid_wilbur.svg"
    fig = plot_progress_fit_grid(**kwargs)
    try:
        with mpl.rc_context({"svg.fonttype": "none"}):
            fig.savefig(path, format="svg")
    finally:
        plt.close(fig)
    return path
