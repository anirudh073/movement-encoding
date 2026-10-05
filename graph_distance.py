"""Physical distance between projected positions on a connected track graph.

Use original video-rate linear positions, never scalar coordinates interpolated
across segment changes. Linear display gaps and unrelated edges are not travel.
"""
import numpy as np

__all__ = ['cumulative_graph_distance', 'recompute_graph_progress']


def cumulative_graph_distance(linear_position, track_segment_id, *,
                              node_positions, linear_edge_order,
                              edge_spacing, segment_order):
    """Accumulate shortest graph paths between consecutive observed positions.

    Same-edge movement includes reversals. Between edges, minimize over the
    endpoint-to-endpoint graph paths. Missing samples remain NaN; the path
    between surrounding valid observations is a lower bound on missing travel.
    """
    pos = np.asarray(linear_position, dtype=float)
    segments = np.asarray(track_segment_id, dtype=float)
    if pos.ndim != 1 or pos.shape != segments.shape:
        raise ValueError('Position and segment ID must be equally sized 1D arrays.')
    edges = list(linear_edge_order)
    if len(edges) != len(segment_order) or len(set(segment_order)) != len(edges):
        raise ValueError('Each edge needs a unique segment ID.')
    if len(edge_spacing) != len(edges) - 1:
        raise ValueError('Supply one display spacing between consecutive edges.')
    nodes = dict(node_positions) if hasattr(node_positions, 'keys') else dict(enumerate(node_positions))
    node_index = {key: i for i, key in enumerate(nodes)}
    adjacency = np.full((len(nodes), len(nodes)), np.inf)
    np.fill_diagonal(adjacency, 0.)
    endpoints, lengths, starts = [], [], []
    cursor = 0.
    for i, (a, b) in enumerate(edges):
        length = float(np.linalg.norm(np.asarray(nodes[b]) - np.asarray(nodes[a])))
        if not np.isfinite(length) or length <= 0:
            raise ValueError('Graph edges must have finite positive length.')
        ai, bi = node_index[a], node_index[b]
        adjacency[ai, bi] = adjacency[bi, ai] = length
        endpoints.append((ai, bi)); lengths.append(length); starts.append(cursor)
        cursor += length + (edge_spacing[i] if i < len(edge_spacing) else 0.)
    for k in range(len(nodes)):
        adjacency = np.minimum(adjacency, adjacency[:, k, None] + adjacency[None, k, :])
    if not np.isfinite(adjacency).all():
        raise ValueError('Graph must be connected.')
    valid = np.isfinite(pos) & np.isfinite(segments)
    result = np.full(pos.shape, np.nan)
    if not valid.any():
        return result
    seg_to_edge = {sid: i for i, sid in enumerate(segment_order)}
    unknown = set(segments[valid]) - set(seg_to_edge)
    if unknown:
        raise ValueError(f'Unknown segment IDs: {sorted(unknown)}')
    edge_index = np.array([seg_to_edge[sid] for sid in segments[valid]])
    lengths = np.asarray(lengths)[edge_index]
    local = pos[valid] - np.asarray(starts)[edge_index]
    if np.any((local < -1e-6) | (local > lengths + 1e-6)):
        raise ValueError('Linear position lies outside its graph edge. Check geometry and use original video samples.')
    local = np.clip(local, 0., lengths)
    endpoints = np.asarray(endpoints)[edge_index]
    to_ends = np.column_stack([local, lengths - local])
    steps = np.full(len(local) - 1, np.inf)
    for a in (0, 1):
        for b in (0, 1):
            steps = np.minimum(steps, to_ends[:-1, a] +
                               adjacency[endpoints[:-1, a], endpoints[1:, b]] + to_ends[1:, b])
    same = edge_index[:-1] == edge_index[1:]
    steps[same] = np.minimum(steps[same], np.abs(np.diff(local))[same])
    result[valid] = np.r_[0., steps.cumsum()]
    return result


def recompute_graph_progress(frame, **geometry):
    """Recompute full-trial cumulative distance before interpolation/filtering.

    Rows must be in time order. Trials remain independent. This returns a copy,
    with both cumulative cm and normalized full-trial progress for auditing.
    """
    out = frame.copy()
    distance = np.full(len(out), np.nan)
    progress = np.full(len(out), np.nan)
    for indices in out.groupby('trial_number', sort=False).indices.values():
        trial = out.iloc[indices]
        d = cumulative_graph_distance(trial.linear_position, trial.track_segment_id, **geometry)
        distance[indices] = d
        finite = d[np.isfinite(d)]
        if len(finite) and finite[-1] > 0:
            progress[indices] = d / finite[-1]
    out['graph_distance_since_trial_start_cm'] = distance
    out['trial_progress_distance'] = progress
    out.attrs['distance_progress_method'] = 'cumulative_shortest_graph_path_v1'
    return out
