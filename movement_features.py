"""Derive body-centered movement features from W-track DLC trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


DEFAULT_PAW_PREFIXES = {
    "forepawL": "forepaw_l",
    "forepawR": "forepaw_r",
    "hindpawL": "hindpaw_l",
    "hindpawR": "hindpaw_r",
}


def paw_position_column_pairs(paw_prefixes=None):
    """Map raw body-frame paw positions to centered visualization columns."""
    prefixes = (paw_prefixes or DEFAULT_PAW_PREFIXES).values()
    return {
        f"{prefix}_ap_cm": f"{prefix}_ap_displacement_cm"
        for prefix in prefixes
    } | {
        f"{prefix}_lateral_cm": f"{prefix}_lateral_displacement_cm"
        for prefix in prefixes
    }


def paw_distance_column_pairs():
    """Map raw inter-paw distances to centered visualization columns."""
    return {
        "forepaw_width_cm": "forepaw_width_displacement_cm",
        "hindpaw_width_cm": "hindpaw_width_displacement_cm",
        "left_fore_hind_distance_cm": "left_fore_hind_distance_displacement_cm",
        "right_fore_hind_distance_cm": "right_fore_hind_distance_displacement_cm",
    }


def _add_baseline_displacements(data, column_pairs, baselines=None):
    """Add baseline-centered columns from a raw-to-output column mapping."""
    missing = set(column_pairs).difference(data.columns)
    if missing:
        raise ValueError(f"data is missing source columns: {sorted(missing)}")

    if baselines is None:
        baselines = data.loc[:, list(column_pairs)].median()
    else:
        baselines = pd.Series(baselines, dtype=float).reindex(column_pairs)
    if baselines.isna().any():
        missing_baselines = baselines.index[baselines.isna()].tolist()
        raise ValueError(f"Missing baselines: {missing_baselines}")

    transformed = data.copy()
    for source_col, output_col in column_pairs.items():
        transformed[output_col] = transformed[source_col] - baselines[source_col]
    return transformed, baselines


def add_paw_position_displacements(
    data: pd.DataFrame,
    paw_prefixes=None,
    baselines: pd.Series | None = None,
):
    """Add paw-position displacements using supplied or fitted medians.

    Returns a copy and the raw-column medians used. Passing baselines fitted on
    training data makes the same transform safe to apply to held-out data.
    """
    return _add_baseline_displacements(
        data,
        paw_position_column_pairs(paw_prefixes),
        baselines=baselines,
    )


def add_paw_distance_displacements(
    data: pd.DataFrame,
    baselines: pd.Series | None = None,
):
    """Add median-centered inter-paw distance columns."""
    return _add_baseline_displacements(
        data,
        paw_distance_column_pairs(),
        baselines=baselines,
    )


def prepare_movement_model_fold(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    continuous_columns,
):
    """Legacy raw-predictor standardization helper.

    The new spline encoding pipeline standardizes post-spline design columns
    in ``movement_encoding.py`` instead. Paw coordinates and distances are not
    median-centered here.
    """
    train_data = train_data.copy()
    test_data = test_data.copy()

    continuous_columns = list(continuous_columns)
    missing = set(continuous_columns).difference(train_data.columns)
    missing |= set(continuous_columns).difference(test_data.columns)
    if missing:
        raise ValueError(f"Missing continuous model columns: {sorted(missing)}")

    means = train_data[continuous_columns].mean()
    scales = train_data[continuous_columns].std(ddof=0)
    invalid_scales = ~np.isfinite(scales) | scales.eq(0)
    scales.loc[invalid_scales] = 1.0
    train_data.loc[:, continuous_columns] = (
        train_data[continuous_columns] - means
    ) / scales
    test_data.loc[:, continuous_columns] = (
        test_data[continuous_columns] - means
    ) / scales

    preprocessing = {
        "means": means,
        "scales": scales,
    }
    return train_data, test_data, preprocessing


def wrap_angle(angle):
    """Wrap angles in radians to [-pi, pi)."""
    angle = np.asarray(angle, dtype=float)
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _valid_runs(values, times, gap_factor=3.0):
    """Yield contiguous finite sample runs without crossing acquisition gaps."""
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    valid = np.isfinite(values) & np.isfinite(times)
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        return

    positive_dt = np.diff(times)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
    max_gap = gap_factor * np.median(positive_dt) if len(positive_dt) else np.inf

    split_after = np.flatnonzero(
        (np.diff(valid_indices) != 1)
        | (np.diff(times[valid_indices]) <= 0)
        | (np.diff(times[valid_indices]) > max_gap)
    )
    for run in np.split(valid_indices, split_after + 1):
        if len(run):
            yield run


def _differentiate_by_group(values, times, groups, unwrap=False):
    """Differentiate within groups and finite runs, never across missing gaps."""
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    groups = np.asarray(groups)
    derivative = np.full(values.shape, np.nan, dtype=float)

    for group in pd.unique(groups[pd.notna(groups)]):
        group_indices = np.flatnonzero(groups == group)
        group_values = values[group_indices]
        group_times = times[group_indices]
        for local_run in _valid_runs(group_values, group_times):
            if len(local_run) < 2:
                continue
            run_values = group_values[local_run]
            if unwrap:
                run_values = np.unwrap(run_values)
            run_times = group_times[local_run] - group_times[local_run[0]]
            edge_order = 2 if len(local_run) >= 3 else 1
            derivative[group_indices[local_run]] = np.gradient(
                run_values,
                run_times,
                edge_order=edge_order,
            )
    return derivative


def _signed_angular_velocity_by_group(forward_x, forward_y, times, groups):
    """Return signed rotation rate of a forward vector within valid runs.

    Positive values are counterclockwise (left turns) and negative values are
    clockwise (right turns) in the stored x-y coordinate system. The cross/dot
    construction is invariant to the vector's absolute orientation.
    """
    forward_x = np.asarray(forward_x, dtype=float)
    forward_y = np.asarray(forward_y, dtype=float)
    times = np.asarray(times, dtype=float)
    groups = np.asarray(groups)
    angular_velocity = np.full(forward_x.shape, np.nan, dtype=float)

    magnitude = np.hypot(forward_x, forward_y)
    valid_vector = (
        np.isfinite(forward_x)
        & np.isfinite(forward_y)
        & np.isfinite(magnitude)
        & (magnitude > 0)
    )
    run_marker = np.where(valid_vector, 0.0, np.nan)

    for group in pd.unique(groups[pd.notna(groups)]):
        group_indices = np.flatnonzero(groups == group)
        group_times = times[group_indices]
        for local_run in _valid_runs(run_marker[group_indices], group_times):
            if len(local_run) < 2:
                continue
            run_indices = group_indices[local_run]
            unit_x = forward_x[run_indices] / magnitude[run_indices]
            unit_y = forward_y[run_indices] / magnitude[run_indices]
            cross = unit_x[:-1] * unit_y[1:] - unit_y[:-1] * unit_x[1:]
            dot = unit_x[:-1] * unit_x[1:] + unit_y[:-1] * unit_y[1:]
            relative_angle = np.concatenate(
                ([0.0], np.cumsum(np.arctan2(cross, dot)))
            )
            run_times = times[run_indices] - times[run_indices[0]]
            edge_order = 2 if len(local_run) >= 3 else 1
            angular_velocity[run_indices] = np.gradient(
                relative_angle,
                run_times,
                edge_order=edge_order,
            )
    return angular_velocity


def _smooth_by_group(values, times, groups, window_s):
    """Centered moving-average smoothing within finite acquisition runs."""
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    groups = np.asarray(groups)
    smoothed = np.full(values.shape, np.nan, dtype=float)
    if window_s <= 0:
        return values.copy()

    for group in pd.unique(groups[pd.notna(groups)]):
        group_indices = np.flatnonzero(groups == group)
        group_values = values[group_indices]
        group_times = times[group_indices]
        positive_dt = np.diff(group_times)
        positive_dt = positive_dt[
            np.isfinite(positive_dt) & (positive_dt > 0)
        ]
        if not len(positive_dt):
            continue
        window_samples = max(1, int(round(window_s / np.median(positive_dt))))
        if window_samples > 1 and window_samples % 2 == 0:
            window_samples += 1

        for local_run in _valid_runs(group_values, group_times):
            run_size = min(window_samples, len(local_run))
            if run_size > 1 and run_size % 2 == 0:
                run_size -= 1
            if run_size <= 1:
                smoothed[group_indices[local_run]] = group_values[local_run]
            else:
                smoothed[group_indices[local_run]] = uniform_filter1d(
                    group_values[local_run],
                    size=run_size,
                    mode="nearest",
                )
    return smoothed


def detect_reference_paw_gait(
    times,
    groups,
    body_speed,
    paw_x,
    paw_y,
    smoothing_s=0.1,
    peak_prominence_cm_s=10.0,
    peak_distance_samples=2,
    swing_boundary_cm_s=15.0,
    min_body_speed_cm_s=5.0,
    min_stance_samples=2,
    min_stride_s=0.18,
    max_stride_s=0.80,
):
    """Detect reference-paw events and phase with the validated speed rule.

    Swing candidates are peaks in smoothed world-coordinate paw speed. Their
    lift-off and touchdown boundaries are the surrounding crossings of the
    larger of ``swing_boundary_cm_s`` and concurrent body speed. Consecutive
    candidates separated by fewer than ``min_stance_samples`` are merged.
    Phase is defined only between accepted consecutive touchdowns.

    Returns
    -------
    phase : ndarray
        Continuous phase in radians on half-open accepted stride intervals.
    paw_speed : ndarray
        Smoothed world-coordinate paw speed used by the detector.
    events : pandas.DataFrame
        One row per retained swing, including lift-off, touchdown, and the
        validity and duration of the stride beginning at that touchdown.
    """
    times = np.asarray(times, dtype=float)
    groups = np.asarray(groups)
    body_speed = np.asarray(body_speed, dtype=float)
    paw_x = np.asarray(paw_x, dtype=float)
    paw_y = np.asarray(paw_y, dtype=float)
    if not (
        len(times)
        == len(groups)
        == len(body_speed)
        == len(paw_x)
        == len(paw_y)
    ):
        raise ValueError("All gait-detector inputs must have equal length")
    if min_stance_samples < 0:
        raise ValueError("min_stance_samples must be nonnegative")
    if not 0 < min_stride_s < max_stride_s:
        raise ValueError("Require 0 < min_stride_s < max_stride_s")

    paw_x_smooth = _smooth_by_group(paw_x, times, groups, smoothing_s)
    paw_y_smooth = _smooth_by_group(paw_y, times, groups, smoothing_s)
    paw_speed = np.hypot(
        _differentiate_by_group(paw_x_smooth, times, groups),
        _differentiate_by_group(paw_y_smooth, times, groups),
    )

    events = []
    run_number = 0
    valid_marker = np.where(
        np.isfinite(paw_speed) & np.isfinite(body_speed),
        0.0,
        np.nan,
    )
    for group in pd.unique(groups[pd.notna(groups)]):
        group_indices = np.flatnonzero(groups == group)
        for local_run in _valid_runs(
            valid_marker[group_indices],
            times[group_indices],
        ):
            run_number += 1
            indices = group_indices[local_run]
            run_paw_speed = paw_speed[indices]
            run_body_speed = body_speed[indices]
            swing_boundary = np.maximum(
                swing_boundary_cm_s,
                run_body_speed,
            )
            peaks, properties = find_peaks(
                run_paw_speed,
                prominence=peak_prominence_cm_s,
                distance=peak_distance_samples,
            )
            candidates = []
            for peak, prominence in zip(peaks, properties["prominences"]):
                if (
                    run_body_speed[peak] < min_body_speed_cm_s
                    or run_paw_speed[peak] <= swing_boundary[peak]
                ):
                    continue

                left = int(peak)
                while (
                    left > 0
                    and run_body_speed[left] >= min_body_speed_cm_s
                    and run_paw_speed[left] > swing_boundary[left]
                ):
                    left -= 1
                lift = (
                    left + 1
                    if run_paw_speed[left] <= swing_boundary[left]
                    else left
                )

                right = int(peak)
                while (
                    right < len(indices) - 1
                    and run_body_speed[right] >= min_body_speed_cm_s
                    and run_paw_speed[right] > swing_boundary[right]
                ):
                    right += 1
                if lift < peak < right:
                    candidates.append(
                        {
                            "group": group,
                            "run": run_number,
                            "lift_off_idx": int(indices[lift]),
                            "peak_idx": int(indices[peak]),
                            "touchdown_idx": int(indices[right]),
                            "peak_speed_cm_s": float(run_paw_speed[peak]),
                            "prominence_cm_s": float(prominence),
                        }
                    )

            best_by_interval = {}
            for event in candidates:
                key = (event["lift_off_idx"], event["touchdown_idx"])
                if (
                    key not in best_by_interval
                    or event["peak_speed_cm_s"]
                    > best_by_interval[key]["peak_speed_cm_s"]
                ):
                    best_by_interval[key] = event
            deduplicated = sorted(
                best_by_interval.values(),
                key=lambda event: event["lift_off_idx"],
            )

            positive_dt = np.diff(times[indices])
            positive_dt = positive_dt[
                np.isfinite(positive_dt) & (positive_dt > 0)
            ]
            if not len(positive_dt):
                continue
            min_stance_s = min_stance_samples * float(np.median(positive_dt))
            merged = []
            for event in deduplicated:
                if (
                    merged
                    and times[event["lift_off_idx"]]
                    - times[merged[-1]["touchdown_idx"]]
                    < min_stance_s
                ):
                    previous = merged[-1]
                    previous["touchdown_idx"] = max(
                        previous["touchdown_idx"],
                        event["touchdown_idx"],
                    )
                    if (
                        event["peak_speed_cm_s"]
                        > previous["peak_speed_cm_s"]
                    ):
                        previous["peak_idx"] = event["peak_idx"]
                        previous["peak_speed_cm_s"] = event[
                            "peak_speed_cm_s"
                        ]
                        previous["prominence_cm_s"] = event[
                            "prominence_cm_s"
                        ]
                else:
                    merged.append(event.copy())
            events.extend(merged)

    events.sort(key=lambda event: event["touchdown_idx"])
    phase = np.full(len(times), np.nan, dtype=float)
    for event in events:
        event["lift_off_time"] = times[event["lift_off_idx"]]
        event["peak_time"] = times[event["peak_idx"]]
        event["touchdown_time"] = times[event["touchdown_idx"]]
        event["accepted_stride_to_next"] = False
        event["next_touchdown_idx"] = np.nan
        event["next_touchdown_time"] = np.nan
        event["stance_duration_s"] = np.nan
        event["swing_duration_s"] = (
            event["touchdown_time"] - event["lift_off_time"]
        )
        event["stride_duration_s"] = np.nan

    for current, following in zip(events[:-1], events[1:]):
        first = current["touchdown_idx"]
        second = following["touchdown_idx"]
        stride_duration = times[second] - times[first]
        stance_duration = times[following["lift_off_idx"]] - times[first]
        accepted = (
            current["run"] == following["run"]
            and min_stride_s <= stride_duration <= max_stride_s
            and np.nanmean(body_speed[first : second + 1])
            >= min_body_speed_cm_s
            and np.isfinite(paw_speed[first : second + 1]).all()
            and 0 <= stance_duration <= stride_duration
        )
        current["accepted_stride_to_next"] = bool(accepted)
        current["next_touchdown_idx"] = int(second)
        current["next_touchdown_time"] = times[second]
        current["stance_duration_s"] = stance_duration
        current["stride_duration_s"] = stride_duration
        if accepted:
            stride_indices = np.arange(first, second)
            phase[stride_indices] = (
                2
                * np.pi
                * (times[stride_indices] - times[first])
                / stride_duration
            )

    event_columns = [
        "group",
        "run",
        "lift_off_idx",
        "lift_off_time",
        "peak_idx",
        "peak_time",
        "touchdown_idx",
        "touchdown_time",
        "next_touchdown_idx",
        "next_touchdown_time",
        "peak_speed_cm_s",
        "prominence_cm_s",
        "stance_duration_s",
        "swing_duration_s",
        "stride_duration_s",
        "accepted_stride_to_next",
    ]
    return phase, paw_speed, pd.DataFrame(events, columns=event_columns)


def detect_lateral_coordination_phase(
    times,
    groups,
    body_speed,
    lateral_signal,
    smoothing_s=0.1,
    peak_prominence_cm=0.2,
    min_body_speed_cm_s=5.0,
    min_cycle_s=0.18,
    max_cycle_s=0.80,
):
    """Return peak-to-peak phase for a body-centered lateral signal.

    Each accepted cycle contains two prominent peaks and an intervening trough
    whose excursion from both peaks meets peak_prominence_cm. Phase is defined
    only on half-open intervals between accepted consecutive peaks.
    """
    times = np.asarray(times, dtype=float)
    groups = np.asarray(groups)
    body_speed = np.asarray(body_speed, dtype=float)
    lateral_signal = np.asarray(lateral_signal, dtype=float)
    if not (
        len(times)
        == len(groups)
        == len(body_speed)
        == len(lateral_signal)
    ):
        raise ValueError("All lateral-phase inputs must have equal length")
    if peak_prominence_cm <= 0:
        raise ValueError("peak_prominence_cm must be positive")
    if not 0 < min_cycle_s < max_cycle_s:
        raise ValueError("Require 0 < min_cycle_s < max_cycle_s")

    smoothed_signal = _smooth_by_group(
        lateral_signal,
        times,
        groups,
        smoothing_s,
    )
    valid_marker = np.where(
        np.isfinite(smoothed_signal) & np.isfinite(body_speed),
        0.0,
        np.nan,
    )
    peaks = []
    run_number = 0
    for group in pd.unique(groups[pd.notna(groups)]):
        group_indices = np.flatnonzero(groups == group)
        for local_run in _valid_runs(
            valid_marker[group_indices],
            times[group_indices],
        ):
            run_number += 1
            indices = group_indices[local_run]
            if len(indices) < 3:
                continue
            positive_dt = np.diff(times[indices])
            positive_dt = positive_dt[
                np.isfinite(positive_dt) & (positive_dt > 0)
            ]
            if not len(positive_dt):
                continue
            min_peak_distance = max(
                1,
                int(round(min_cycle_s / np.median(positive_dt))),
            )
            run_peaks, properties = find_peaks(
                smoothed_signal[indices],
                prominence=peak_prominence_cm,
                distance=min_peak_distance,
            )
            for peak, prominence in zip(
                run_peaks,
                properties["prominences"],
            ):
                peak_idx = int(indices[peak])
                if body_speed[peak_idx] < min_body_speed_cm_s:
                    continue
                peaks.append(
                    {
                        "group": group,
                        "run": run_number,
                        "peak_idx": peak_idx,
                        "peak_time": times[peak_idx],
                        "peak_value_cm": smoothed_signal[peak_idx],
                        "prominence_cm": float(prominence),
                        "accepted_cycle_to_next": False,
                        "next_peak_idx": np.nan,
                        "next_peak_time": np.nan,
                        "trough_idx": np.nan,
                        "trough_time": np.nan,
                        "trough_value_cm": np.nan,
                        "cycle_excursion_cm": np.nan,
                        "cycle_duration_s": np.nan,
                    }
                )

    peaks.sort(key=lambda event: event["peak_idx"])
    phase = np.full(len(times), np.nan, dtype=float)
    for current, following in zip(peaks[:-1], peaks[1:]):
        first = current["peak_idx"]
        second = following["peak_idx"]
        duration = times[second] - times[first]
        current["next_peak_idx"] = int(second)
        current["next_peak_time"] = times[second]
        current["cycle_duration_s"] = duration
        if current["run"] != following["run"] or second - first < 2:
            continue

        interior = smoothed_signal[first : second + 1]
        trough = first + int(np.argmin(interior))
        trough_value = smoothed_signal[trough]
        excursion = min(
            smoothed_signal[first] - trough_value,
            smoothed_signal[second] - trough_value,
        )
        current["trough_idx"] = int(trough)
        current["trough_time"] = times[trough]
        current["trough_value_cm"] = trough_value
        current["cycle_excursion_cm"] = excursion
        accepted = (
            min_cycle_s <= duration <= max_cycle_s
            and first < trough < second
            and excursion >= peak_prominence_cm
            and np.nanmean(body_speed[first : second + 1])
            >= min_body_speed_cm_s
            and np.isfinite(smoothed_signal[first : second + 1]).all()
        )
        current["accepted_cycle_to_next"] = bool(accepted)
        if accepted:
            cycle_indices = np.arange(first, second)
            phase[cycle_indices] = (
                2
                * np.pi
                * (times[cycle_indices] - times[first])
                / duration
            )

    event_columns = [
        "group",
        "run",
        "peak_idx",
        "peak_time",
        "peak_value_cm",
        "prominence_cm",
        "next_peak_idx",
        "next_peak_time",
        "trough_idx",
        "trough_time",
        "trough_value_cm",
        "cycle_excursion_cm",
        "cycle_duration_s",
        "accepted_cycle_to_next",
    ]
    return (
        phase,
        smoothed_signal,
        pd.DataFrame(peaks, columns=event_columns),
    )


def _align_bodypart_by_frame(position_df, pose_by_epoch, bodypart):
    """Align one DLC body part using epoch and video-frame index."""
    aligned_x = np.full(len(position_df), np.nan, dtype=float)
    aligned_y = np.full(len(position_df), np.nan, dtype=float)

    epochs = position_df["epoch"].to_numpy()
    frames = position_df["video_frame_ind"].to_numpy()
    for epoch, epoch_pose in pose_by_epoch.items():
        if bodypart not in epoch_pose:
            raise KeyError(f"Missing {bodypart!r} pose data for epoch {epoch}")
        pose_df = epoch_pose[bodypart]
        required = {"video_frame_ind", "x", "y"}
        missing = required.difference(pose_df.columns)
        if missing:
            raise ValueError(
                f"{bodypart!r} pose data for epoch {epoch} is missing {sorted(missing)}"
            )

        frame_pose = (
            pose_df.loc[:, ["video_frame_ind", "x", "y"]]
            .drop_duplicates("video_frame_ind", keep="last")
            .set_index("video_frame_ind")
        )
        epoch_mask = epochs == epoch
        epoch_frames = frames[epoch_mask]
        aligned_x[epoch_mask] = frame_pose["x"].reindex(epoch_frames).to_numpy()
        aligned_y[epoch_mask] = frame_pose["y"].reindex(epoch_frames).to_numpy()

    return aligned_x, aligned_y


def _outbound_track_axis_by_segment(
    node_positions,
    linear_edge_order,
    center_well_node,
    track_segment_order=None,
):
    """Return each edge axis directed away from the center well."""
    node_positions = np.asarray(node_positions, dtype=float)
    adjacency = {node: [] for node in range(len(node_positions))}
    for start_node, end_node in linear_edge_order:
        start_node = int(start_node)
        end_node = int(end_node)
        adjacency[start_node].append(end_node)
        adjacency[end_node].append(start_node)

    center_well_node = int(center_well_node)
    distance_from_center = {center_well_node: 0}
    queue = [center_well_node]
    for node in queue:
        for neighbor in adjacency[node]:
            if neighbor not in distance_from_center:
                distance_from_center[neighbor] = distance_from_center[node] + 1
                queue.append(neighbor)

    if track_segment_order is None:
        track_segment_order = range(len(linear_edge_order))
    if len(track_segment_order) != len(linear_edge_order):
        raise ValueError("track_segment_order must have one ID per track edge")

    axis_by_segment = {}
    for segment_id, (start_node, end_node) in zip(
        track_segment_order,
        linear_edge_order,
    ):
        start_node = int(start_node)
        end_node = int(end_node)
        start_distance = distance_from_center.get(start_node)
        end_distance = distance_from_center.get(end_node)
        if start_distance is None or end_distance is None:
            raise ValueError("Track graph contains a node disconnected from center_well_node")
        if start_distance == end_distance:
            raise ValueError("Cannot orient a track edge away from center_well_node")
        inner_node, outer_node = (
            (start_node, end_node)
            if start_distance < end_distance
            else (end_node, start_node)
        )
        delta = node_positions[outer_node] - node_positions[inner_node]
        axis_by_segment[segment_id] = np.arctan2(delta[1], delta[0])
    return axis_by_segment


def derive_movement_features(
    position_df: pd.DataFrame,
    pose_by_epoch: Mapping[int, Mapping[str, pd.DataFrame]],
    node_positions: Sequence[Sequence[float]] | None = None,
    linear_edge_order: Sequence[Sequence[int]] | None = None,
    paw_prefixes: Mapping[str, str] | None = None,
    body_frame_smoothing_s: float = 0.1,
    paw_kinematics_smoothing_s: float = 0.1,
    movement_direction_smoothing_s: float = 0.25,
    movement_direction_min_speed_cm_s: float = 5.0,
    center_well_node: int = 0,
    track_segment_order: Sequence[int] | None = None,
    gait_reference_paw: str = "forepawR",
    gait_smoothing_s: float = 0.1,
    gait_peak_prominence_cm_s: float = 10.0,
    gait_peak_distance_samples: int = 2,
    gait_swing_boundary_cm_s: float = 15.0,
    gait_min_body_speed_cm_s: float = 5.0,
    gait_min_stance_samples: int = 2,
    gait_min_stride_s: float = 0.18,
    gait_max_stride_s: float = 0.80,
    lateral_phase_smoothing_s: float = 0.1,
    lateral_phase_peak_prominence_cm: float = 0.2,
    lateral_phase_min_body_speed_cm_s: float = 5.0,
    lateral_phase_min_cycle_s: float = 0.18,
    lateral_phase_max_cycle_s: float = 0.80,
    return_gait_events: bool = False,
    return_lateral_phase_events: bool = False,
) -> (
    pd.DataFrame
    | tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """Return translation, body-axis, and body-centered paw features.

    The reference frame is estimated from smoothed paw trajectories. Its center
    is halfway between the hind-paw and forepaw midpoints, and its anterior axis
    points from the hind midpoint to the fore midpoint. Original filtered paw
    coordinates are translated and rotated into that frame, then smoothed before
    differentiation. Lateral position is positive to the animal's left. Paw
    speed is the magnitude of the AP/lateral body-frame velocity. Track-relative orientation uses the
    current edge's longitudinal axis, directed outward on outbound trials and
    inward on inbound trials. Movement-relative orientation compares the body
    axis with the smoothed centroid-velocity direction and is undefined below
    ``movement_direction_min_speed_cm_s``. If track geometry is omitted,
    track-relative orientation is undefined; the other features are unchanged.
    """
    required = {
        "position_x",
        "position_y",
        "speed",
        "track_segment_id",
        "trial_type",
        "video_frame_ind",
        "epoch",
    }
    missing = required.difference(position_df.columns)
    if missing:
        raise ValueError(f"position_df is missing required columns: {sorted(missing)}")
    if not position_df.index.is_monotonic_increasing:
        raise ValueError("position_df must be sorted by its time index")

    paw_prefixes = dict(paw_prefixes or DEFAULT_PAW_PREFIXES)
    times = position_df.index.to_numpy(dtype=float)
    epochs = position_df["epoch"].to_numpy()
    required_paws = {"forepawL", "forepawR", "hindpawL", "hindpawR"}
    missing_paws = required_paws.difference(paw_prefixes)
    if missing_paws:
        raise ValueError(
            f"paw_prefixes is missing required body-frame paws: {sorted(missing_paws)}"
        )
    if gait_reference_paw not in paw_prefixes:
        raise ValueError(
            f"Unknown gait_reference_paw {gait_reference_paw!r}; "
            f"expected one of {sorted(paw_prefixes)}"
        )

    paw_coordinates = {}
    smoothed_paw_coordinates = {}
    for bodypart in paw_prefixes:
        paw_x, paw_y = _align_bodypart_by_frame(
            position_df,
            pose_by_epoch,
            bodypart,
        )
        paw_coordinates[bodypart] = (paw_x, paw_y)
        smoothed_paw_coordinates[bodypart] = (
            _smooth_by_group(
                paw_x,
                times,
                epochs,
                body_frame_smoothing_s,
            ),
            _smooth_by_group(
                paw_y,
                times,
                epochs,
                body_frame_smoothing_s,
            ),
        )

    fore_x = 0.5 * (
        smoothed_paw_coordinates["forepawL"][0]
        + smoothed_paw_coordinates["forepawR"][0]
    )
    fore_y = 0.5 * (
        smoothed_paw_coordinates["forepawL"][1]
        + smoothed_paw_coordinates["forepawR"][1]
    )
    hind_x = 0.5 * (
        smoothed_paw_coordinates["hindpawL"][0]
        + smoothed_paw_coordinates["hindpawR"][0]
    )
    hind_y = 0.5 * (
        smoothed_paw_coordinates["hindpawL"][1]
        + smoothed_paw_coordinates["hindpawR"][1]
    )
    body_center_x = 0.5 * (fore_x + hind_x)
    body_center_y = 0.5 * (fore_y + hind_y)
    body_forward_x = fore_x - hind_x
    body_forward_y = fore_y - hind_y
    body_orientation = np.arctan2(body_forward_y, body_forward_x)
    body_angular_velocity = _signed_angular_velocity_by_group(
        body_forward_x,
        body_forward_y,
        times,
        epochs,
    )

    movement_center_x = _smooth_by_group(
        position_df["position_x"].to_numpy(dtype=float),
        times,
        epochs,
        movement_direction_smoothing_s,
    )
    movement_center_y = _smooth_by_group(
        position_df["position_y"].to_numpy(dtype=float),
        times,
        epochs,
        movement_direction_smoothing_s,
    )
    movement_velocity_x = _differentiate_by_group(
        movement_center_x,
        times,
        epochs,
    )
    movement_velocity_y = _differentiate_by_group(
        movement_center_y,
        times,
        epochs,
    )
    movement_speed = np.hypot(movement_velocity_x, movement_velocity_y)
    movement_direction = np.arctan2(movement_velocity_y, movement_velocity_x)
    movement_relative_orientation = wrap_angle(
        body_orientation - movement_direction
    )
    movement_relative_orientation[
        movement_speed < movement_direction_min_speed_cm_s
    ] = np.nan

    if (node_positions is None) != (linear_edge_order is None):
        raise ValueError("Provide both track geometry arrays or neither")
    outbound_axis_by_segment = (
        _outbound_track_axis_by_segment(
            node_positions,
            linear_edge_order,
            center_well_node,
            track_segment_order,
        )
        if node_positions is not None
        else {}
    )
    segment_ids = position_df["track_segment_id"].to_numpy()
    outbound_track_axis = np.array(
        [outbound_axis_by_segment.get(int(segment), np.nan)
         if np.isfinite(segment) else np.nan
         for segment in segment_ids],
        dtype=float,
    )
    trial_type = position_df["trial_type"].to_numpy()
    movement_track_axis = np.full(len(position_df), np.nan, dtype=float)
    outbound = trial_type == "outbound"
    inbound = trial_type == "inbound"
    movement_track_axis[outbound] = outbound_track_axis[outbound]
    movement_track_axis[inbound] = outbound_track_axis[inbound] + np.pi

    features = pd.DataFrame(index=position_df.index)
    features["tangential_acceleration_cm_s2"] = _differentiate_by_group(
        position_df["speed"].to_numpy(dtype=float),
        times,
        epochs,
    )
    features["body_orientation_rad"] = wrap_angle(body_orientation)
    features["movement_relative_body_orientation_rad"] = (
        movement_relative_orientation
    )
    track_relative_orientation = wrap_angle(
        body_orientation - movement_track_axis
    )
    features["track_relative_orientation_rad"] = track_relative_orientation
    features["track_relative_orientation_deg"] = np.rad2deg(
        track_relative_orientation
    )
    features["body_angular_velocity_rad_s"] = body_angular_velocity

    cos_orientation = np.cos(body_orientation)
    sin_orientation = np.sin(body_orientation)
    body_frame_positions = {}
    for bodypart, prefix in paw_prefixes.items():
        paw_x, paw_y = paw_coordinates[bodypart]
        delta_x = paw_x - body_center_x
        delta_y = paw_y - body_center_y
        ap_position = cos_orientation * delta_x + sin_orientation * delta_y
        lateral_position = -sin_orientation * delta_x + cos_orientation * delta_y

        ap_position = _smooth_by_group(
            ap_position,
            times,
            epochs,
            paw_kinematics_smoothing_s,
        )
        lateral_position = _smooth_by_group(
            lateral_position,
            times,
            epochs,
            paw_kinematics_smoothing_s,
        )

        ap_col = f"{prefix}_ap_cm"
        lateral_col = f"{prefix}_lateral_cm"
        ap_velocity_col = f"{prefix}_ap_velocity_cm_s"
        lateral_velocity_col = f"{prefix}_lateral_velocity_cm_s"
        speed_col = f"{prefix}_speed_cm_s"
        features[ap_col] = ap_position
        features[lateral_col] = lateral_position
        body_frame_positions[bodypart] = (ap_position, lateral_position)

        ap_velocity = _differentiate_by_group(ap_position, times, epochs)
        lateral_velocity = _differentiate_by_group(lateral_position, times, epochs)
        features[ap_velocity_col] = ap_velocity
        features[lateral_velocity_col] = lateral_velocity
        features[speed_col] = np.hypot(ap_velocity, lateral_velocity)

    distance_specs = {
        "forepaw_width_cm": ("forepawL", "forepawR"),
        "hindpaw_width_cm": ("hindpawL", "hindpawR"),
        "left_fore_hind_distance_cm": ("forepawL", "hindpawL"),
        "right_fore_hind_distance_cm": ("forepawR", "hindpawR"),
    }
    for column, (first_paw, second_paw) in distance_specs.items():
        first_ap, first_lateral = body_frame_positions[first_paw]
        second_ap, second_lateral = body_frame_positions[second_paw]
        features[column] = np.hypot(
            first_ap - second_ap,
            first_lateral - second_lateral,
        )

    body_center_velocity_x = _differentiate_by_group(
        body_center_x,
        times,
        epochs,
    )
    body_center_velocity_y = _differentiate_by_group(
        body_center_y,
        times,
        epochs,
    )
    relative_paw_speeds = {}
    for bodypart in paw_prefixes:
        paw_x, paw_y = smoothed_paw_coordinates[bodypart]
        paw_velocity_x = _differentiate_by_group(paw_x, times, epochs)
        paw_velocity_y = _differentiate_by_group(paw_y, times, epochs)
        offset_x = paw_x - body_center_x
        offset_y = paw_y - body_center_y
        relative_velocity_x = (
            paw_velocity_x
            - body_center_velocity_x
            + body_angular_velocity * offset_y
        )
        relative_velocity_y = (
            paw_velocity_y
            - body_center_velocity_y
            - body_angular_velocity * offset_x
        )
        relative_paw_speeds[bodypart] = np.hypot(
            relative_velocity_x,
            relative_velocity_y,
        )

    fore_vigor = 0.5 * (
        relative_paw_speeds["forepawL"]
        + relative_paw_speeds["forepawR"]
    )
    hind_vigor = 0.5 * (
        relative_paw_speeds["hindpawL"]
        + relative_paw_speeds["hindpawR"]
    )
    features["relative_paw_vigor"] = 0.5 * (fore_vigor + hind_vigor)
    features["fore_minus_hind_vigor"] = fore_vigor - hind_vigor

    forepaw_l_ap, forepaw_l_lateral = body_frame_positions["forepawL"]
    forepaw_r_ap, forepaw_r_lateral = body_frame_positions["forepawR"]
    hindpaw_l_ap, hindpaw_l_lateral = body_frame_positions["hindpawL"]
    hindpaw_r_ap, hindpaw_r_lateral = body_frame_positions["hindpawR"]
    fore_width = np.abs(forepaw_l_lateral - forepaw_r_lateral)
    hind_width = np.abs(hindpaw_l_lateral - hindpaw_r_lateral)
    features["mean_stance_width"] = 0.5 * (fore_width + hind_width)
    features["fore_minus_hind_width"] = fore_width - hind_width
    fore_center_ap = 0.5 * (forepaw_l_ap + forepaw_r_ap)
    hind_center_ap = 0.5 * (hindpaw_l_ap + hindpaw_r_ap)
    features["mean_fore_hind_spacing"] = fore_center_ap - hind_center_ap
    features["fore_hind_spacing_asymmetry"] = (
        (forepaw_l_ap - hindpaw_l_ap)
        - (forepaw_r_ap - hindpaw_r_ap)
    )
    features["lateral_coordination_signal_cm"] = 0.5 * (
        forepaw_l_lateral
        + hindpaw_r_lateral
        - forepaw_r_lateral
        - hindpaw_l_lateral
    )

    gait_phase, reference_paw_speed, gait_events = detect_reference_paw_gait(
        times,
        epochs,
        position_df["speed"].to_numpy(dtype=float),
        *paw_coordinates[gait_reference_paw],
        smoothing_s=gait_smoothing_s,
        peak_prominence_cm_s=gait_peak_prominence_cm_s,
        peak_distance_samples=gait_peak_distance_samples,
        swing_boundary_cm_s=gait_swing_boundary_cm_s,
        min_body_speed_cm_s=gait_min_body_speed_cm_s,
        min_stance_samples=gait_min_stance_samples,
        min_stride_s=gait_min_stride_s,
        max_stride_s=gait_max_stride_s,
    )
    features["gait_phase_sin"] = np.sin(gait_phase)
    features["gait_phase_cos"] = np.cos(gait_phase)
    features["reference_paw_world_speed_cm_s"] = reference_paw_speed

    (
        lateral_phase,
        lateral_signal_smoothed,
        lateral_phase_events,
    ) = detect_lateral_coordination_phase(
        times,
        epochs,
        position_df["speed"].to_numpy(dtype=float),
        features["lateral_coordination_signal_cm"].to_numpy(dtype=float),
        smoothing_s=lateral_phase_smoothing_s,
        peak_prominence_cm=lateral_phase_peak_prominence_cm,
        min_body_speed_cm_s=lateral_phase_min_body_speed_cm_s,
        min_cycle_s=lateral_phase_min_cycle_s,
        max_cycle_s=lateral_phase_max_cycle_s,
    )
    features["lateral_phase_sin"] = np.sin(lateral_phase)
    features["lateral_phase_cos"] = np.cos(lateral_phase)
    features["lateral_coordination_signal_smoothed_cm"] = (
        lateral_signal_smoothed
    )

    if return_gait_events and return_lateral_phase_events:
        return features, gait_events, lateral_phase_events
    if return_gait_events:
        return features, gait_events
    if return_lateral_phase_events:
        return features, lateral_phase_events
    return features


def _interpolate_contiguous(source_times, source_values, target_times):
    """Linearly interpolate without bridging NaNs or acquisition gaps."""
    source_times = np.asarray(source_times, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    target_times = np.asarray(target_times, dtype=float)
    interpolated = np.full(target_times.shape, np.nan, dtype=float)

    for run in _valid_runs(source_values, source_times):
        if len(run) < 2:
            continue
        in_run = (
            (target_times >= source_times[run[0]])
            & (target_times <= source_times[run[-1]])
        )
        if in_run.any():
            interpolated[in_run] = np.interp(
                target_times[in_run],
                source_times[run],
                source_values[run],
            )
    return interpolated


def interpolate_features_to_bins(
    feature_df: pd.DataFrame,
    source_epochs,
    target_df: pd.DataFrame,
    time_col: str = "time_bin_center",
) -> pd.DataFrame:
    """Append video-rate features to a binned covariate table by epoch."""
    if time_col not in target_df:
        raise ValueError(f"target_df is missing {time_col!r}")
    if "epoch" not in target_df:
        raise ValueError("target_df is missing 'epoch'")
    if len(source_epochs) != len(feature_df):
        raise ValueError("source_epochs must have one entry per feature_df row")

    augmented = target_df.copy()
    source_times = feature_df.index.to_numpy(dtype=float)
    source_epochs = np.asarray(source_epochs)
    target_times = augmented[time_col].to_numpy(dtype=float)
    target_epochs = augmented["epoch"].to_numpy()

    for column in feature_df.columns:
        values = np.full(len(augmented), np.nan, dtype=float)
        for epoch in pd.unique(target_epochs[pd.notna(target_epochs)]):
            source_mask = source_epochs == epoch
            target_mask = target_epochs == epoch
            if not source_mask.any() or not target_mask.any():
                continue
            values[target_mask] = _interpolate_contiguous(
                source_times[source_mask],
                feature_df.loc[source_mask, column].to_numpy(dtype=float),
                target_times[target_mask],
            )
        augmented[column] = values
    return augmented
