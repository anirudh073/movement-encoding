"""Spike visualisation: rasters, ISI histograms, autocorrelograms, etc."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.ndimage import gaussian_filter1d
sns.set_context("talk")


def _select_units(spike_counts, units):
    """Return (selected_spike_counts, selected_unit_ids) for a unit subset."""
    all_ids = np.arange(spike_counts.shape[0])
    if units is None:
        return spike_counts, all_ids
    units = np.asarray(units)
    return spike_counts[units], units


def _grid_shape(n):
    """Return (nrows, ncols) for a roughly square grid of n panels."""
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def _compute_isi_matrix(spk, bin_size=0.002, max_isi_ms=100, n_hist_bins=100):
    """ISI histogram per unit.

    Returns
    -------
    isi_matrix        : (n_units, n_hist_bins) integer counts per ISI bin
    bin_edges_ms      : (n_hist_bins + 1,) histogram edges in ms
    n_spikes_per_unit : (n_units,) total spikes per unit
    """
    n_units = spk.shape[0]
    bin_edges_ms = np.linspace(0, max_isi_ms, n_hist_bins + 1)
    isi_matrix = np.zeros((n_units, n_hist_bins), dtype=np.float64)
    n_spikes_per_unit = np.zeros(n_units, dtype=np.int64)
    for i in range(n_units):
        spike_bins = np.where(spk[i] > 0)[0]
        n_spikes_per_unit[i] = len(spike_bins)
        if len(spike_bins) < 2:
            continue
        isis_ms = np.diff(spike_bins) * bin_size * 1000.0
        # strict <: max_isi_ms is the exclusive upper bound of the displayed
        # range. Without this, np.histogram's closed-right last bin would
        # catch both the second-to-last and the boundary quantized ISI value,
        # producing a 2x edge spike for units with broad ISI tails.
        isis_ms = isis_ms[isis_ms < max_isi_ms]
        counts, _ = np.histogram(isis_ms, bins=bin_edges_ms)
        isi_matrix[i] = counts
    return isi_matrix, bin_edges_ms, n_spikes_per_unit


def _compute_acg_matrix(spk, bin_size=0.002, max_lag_ms=50, plot_bin_ms=None):
    """Autocorrelogram per unit, rebinned to plot_bin_ms, symmetric about 0.

    Returns
    -------
    acg_matrix        : (n_units, 2*n_half + 1) coincidence counts
    lags_ms           : (2*n_half + 1,) lag axis in ms (centered on 0)
    n_spikes_per_unit : (n_units,) total spikes per unit
    plot_bin_ms       : actual plot bin width (snapped to multiple of bin_size)
    """
    spk_f = np.asarray(spk).astype(np.float32)
    n_units, n_bins_total = spk_f.shape

    data_bin_ms = bin_size * 1000.0
    if plot_bin_ms is None:
        plot_bin_ms = data_bin_ms
    rebin = max(1, round(plot_bin_ms / data_bin_ms))
    plot_bin_ms = rebin * data_bin_ms
    max_lag_bins = int(max_lag_ms / data_bin_ms)

    n_spikes_per_unit = spk_f.sum(axis=1).astype(np.int64)

    # Vectorized batch ACG at lags within ±max_lag_bins (no full np.correlate).
    acg_full = np.zeros((n_units, 2 * max_lag_bins + 1), dtype=np.float32)
    for lag in range(1, max_lag_bins + 1):
        v = (spk_f[:, :n_bins_total - lag] * spk_f[:, lag:]).sum(axis=1)
        acg_full[:, max_lag_bins + lag] = v
        acg_full[:, max_lag_bins - lag] = v  # ACG is symmetric

    if rebin > 1:
        n_trim = (max_lag_bins // rebin) * rebin
        neg = acg_full[:, max_lag_bins - n_trim:max_lag_bins]
        pos = acg_full[:, max_lag_bins + 1:max_lag_bins + 1 + n_trim]
        neg = neg.reshape(n_units, -1, rebin).sum(axis=2)
        pos = pos.reshape(n_units, -1, rebin).sum(axis=2)
        zero_col = np.zeros((n_units, 1), dtype=acg_full.dtype)
        acg_matrix = np.concatenate([neg, zero_col, pos], axis=1)
    else:
        acg_matrix = acg_full

    n_half = acg_matrix.shape[1] // 2
    lags_ms = np.arange(-n_half, n_half + 1) * plot_bin_ms
    return acg_matrix, lags_ms, n_spikes_per_unit, plot_bin_ms


def _get_bouts(mask, min_len):
    """Return contiguous True intervals as (start, stop) bin pairs."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(padded.astype(int))
    starts = np.where(d == 1)[0]
    stops = np.where(d == -1)[0]
    valid = (stops - starts) >= min_len
    return list(zip(starts[valid], stops[valid]))


def _normalize_bout_rate(spks_row, bouts, n_bins, n_pad, pad_frac, bin_dur_s,
                         sigma=0, min_bouts=3):
    """Return mean/SEM firing rate over a fixed normalized bout timeline."""
    n_total = n_bins + 2 * n_pad
    snippets = []

    for start, stop in bouts:
        bout_len = stop - start
        pad = int(round(pad_frac * bout_len))
        start_pad, stop_pad = start - pad, stop + pad
        if start_pad < 0 or stop_pad > len(spks_row):
            continue

        raw = np.asarray(spks_row[start_pad:stop_pad], dtype=float)
        if len(raw) < 2:
            continue

        # Interpolate cumulative counts onto normalized bin edges so the
        # normalized counts preserve total spike count within each snippet.
        cum = np.concatenate([[0.0], np.cumsum(raw)])
        x_orig = np.linspace(0, n_total, len(cum))
        cum_interp = np.interp(np.arange(n_total + 1), x_orig, cum)
        counts_norm = np.diff(cum_interp)
        norm_bin_s = (stop_pad - start_pad) * bin_dur_s / n_total
        snippets.append(counts_norm / norm_bin_s)

    n_used = len(snippets)
    mean = np.full(n_total, np.nan, dtype=float)
    sem = np.full(n_total, np.nan, dtype=float)
    if n_used < min_bouts:
        return mean, sem, n_used

    arr = np.stack(snippets)
    mean = arr.mean(axis=0)
    if sigma is not None and sigma > 0:
        mean = gaussian_filter1d(mean, sigma=sigma)
    sem = arr.std(axis=0) / np.sqrt(n_used)
    return mean, sem, n_used


def _normalize_bout_signal(signal, bouts, n_bins, n_pad, pad_frac):
    """Return the average normalized profile for a scalar bout-aligned signal."""
    n_total = n_bins + 2 * n_pad
    snippets = []

    for start, stop in bouts:
        bout_len = stop - start
        pad = int(round(pad_frac * bout_len))
        start_pad, stop_pad = start - pad, stop + pad
        if start_pad < 0 or stop_pad > len(signal):
            continue

        raw = np.asarray(signal[start_pad:stop_pad], dtype=float)
        if len(raw) < 2:
            continue

        x_orig = np.linspace(0, n_total, len(raw))
        snippets.append(np.interp(np.arange(n_total), x_orig, raw))

    if not snippets:
        return np.full(n_total, np.nan, dtype=float), 0

    arr = np.stack(snippets)
    return arr.mean(axis=0), len(snippets)


def _nan_gaussian_filter1d(matrix, sigma, axis=1):
    """Gaussian smooth while ignoring NaNs instead of spreading them."""
    finite = np.isfinite(matrix)
    values = np.where(finite, matrix, 0.0)
    weights = finite.astype(float)
    smooth_values = gaussian_filter1d(values, sigma=sigma, axis=axis, mode="nearest")
    smooth_weights = gaussian_filter1d(weights, sigma=sigma, axis=axis, mode="nearest")

    out = np.full_like(matrix, np.nan, dtype=float)
    valid = smooth_weights > 0
    out[valid] = smooth_values[valid] / smooth_weights[valid]
    return out


def _bout_firing_rates(spks_row, bouts, bin_dur):
    """Return one firing rate per bout for a single unit."""
    rates = []
    for start, stop in bouts:
        duration_s = (stop - start) * bin_dur
        if duration_s <= 0:
            continue
        rates.append(np.nansum(spks_row[start:stop]) / duration_s)
    return np.asarray(rates, dtype=float)


def compute_run_rest_firing_rates(
    spike_counts,
    speed,
    units=None,
    speed_threshold=10.0,
    min_bout_bins=20,
    bin_dur=0.025,
    min_bouts=1,
):
    """Compute per-unit run/rest firing rates from real, unnormalized bouts.

    Each unit-condition value is the mean firing rate across bouts. The returned
    SEM is computed across bout firing rates.
    """
    spk = np.asarray(spike_counts)
    spk, uids = _select_units(spk, units)
    uids = np.asarray(uids)
    speed = np.asarray(speed, dtype=float)

    if spk.ndim != 2:
        raise ValueError("spike_counts must be a 2-D array (n_units x n_time_bins)")
    if speed.ndim != 1:
        raise ValueError("speed must be a 1-D array aligned to spike_counts")
    if spk.shape[1] != len(speed):
        raise ValueError(
            "spike_counts and speed must have the same number of time bins; "
            f"got {spk.shape[1]} and {len(speed)}"
        )

    run_bouts = _get_bouts(speed > speed_threshold, min_bout_bins)
    rest_bouts = _get_bouts(speed <= speed_threshold, min_bout_bins)

    conditions = {}
    for label, bouts in [("rest", rest_bouts), ("run", run_bouts)]:
        mean = np.full(len(uids), np.nan, dtype=float)
        std = np.full(len(uids), np.nan, dtype=float)
        sem = np.full(len(uids), np.nan, dtype=float)
        n_bouts = np.zeros(len(uids), dtype=int)
        per_bout_rates = []

        for idx, row in enumerate(spk):
            rates = _bout_firing_rates(row, bouts, bin_dur=bin_dur)
            per_bout_rates.append(rates)
            n_bouts[idx] = len(rates)
            if len(rates) < min_bouts:
                continue
            mean[idx] = np.mean(rates)
            std[idx] = np.std(rates, ddof=1) if len(rates) > 1 else 0.0
            sem[idx] = std[idx] / np.sqrt(len(rates))

        conditions[label] = {
            "mean": mean,
            "std": std,
            "sem": sem,
            "n_bouts": n_bouts,
            "per_bout_rates": per_bout_rates,
            "bouts": bouts,
        }

    return {
        "units": uids,
        "conditions": conditions,
        "params": {
            "speed_threshold": speed_threshold,
            "min_bout_bins": min_bout_bins,
            "bin_dur": bin_dur,
            "min_bouts": min_bouts,
        },
    }


def compute_bout_normalized_firing_rates(
    spike_counts,
    speed,
    units=None,
    speed_threshold=10.0,
    min_bout_bins=20,
    bin_dur=0.025,
    n_norm=100,
    n_pad_norm=20,
    pad_frac=0.5,
    smooth_sigma=2,
    min_bouts=3,
):
    """Compute normalized run/rest firing-rate curves for a set of units.

    Parameters
    ----------
    spike_counts    : 2-D array (n_units x n_time_bins)
    speed           : 1-D array of speed values aligned to spike-count bins
    units           : list of unit indices to compute (default: all)
    speed_threshold : threshold in cm/s; `speed > threshold` defines run
    min_bout_bins   : minimum number of bins required for a run/rest bout
    bin_dur         : duration of each time bin in seconds
    n_norm          : normalized bins spanning the bout itself
    n_pad_norm      : normalized bins on each side for pre/post padding
    pad_frac        : pad each bout by this fraction of its duration
    smooth_sigma    : Gaussian sigma, in normalized bins, applied to the mean
    min_bouts       : minimum valid bouts required to keep a unit-condition

    Returns
    -------
    result : dict
        Keys:
        - `units`: selected unit IDs
        - `t_norm`: normalized time axis in percent of bout duration
        - `conditions`: per-condition dict with keys
          `mean`, `sem`, `n_bouts`, `speed_profile`, `n_speed_bouts`, `bouts`
        - `params`: the parameters used for the calculation
    """
    spk = np.asarray(spike_counts)
    spk, uids = _select_units(spk, units)
    uids = np.asarray(uids)
    speed = np.asarray(speed, dtype=float)

    if spk.ndim != 2:
        raise ValueError("spike_counts must be a 2-D array (n_units x n_time_bins)")
    if speed.ndim != 1:
        raise ValueError("speed must be a 1-D array aligned to spike_counts")
    if spk.shape[1] != len(speed):
        raise ValueError(
            "spike_counts and speed must have the same number of time bins; "
            f"got {spk.shape[1]} and {len(speed)}"
        )

    n_total = n_norm + 2 * n_pad_norm
    t_norm = np.linspace(-pad_frac * 100, (1 + pad_frac) * 100, n_total)

    run_bouts = _get_bouts(speed > speed_threshold, min_bout_bins)
    rest_bouts = _get_bouts(speed <= speed_threshold, min_bout_bins)

    conditions = {}
    for label, bouts in [("rest", rest_bouts), ("run", run_bouts)]:
        mean = np.full((len(uids), n_total), np.nan, dtype=float)
        sem = np.full((len(uids), n_total), np.nan, dtype=float)
        n_bouts = np.zeros(len(uids), dtype=int)

        for idx, row in enumerate(spk):
            mean[idx], sem[idx], n_bouts[idx] = _normalize_bout_rate(
                row,
                bouts,
                n_bins=n_norm,
                n_pad=n_pad_norm,
                pad_frac=pad_frac,
                bin_dur_s=bin_dur,
                sigma=smooth_sigma,
                min_bouts=min_bouts,
            )

        speed_profile, n_speed_bouts = _normalize_bout_signal(
            speed,
            bouts,
            n_bins=n_norm,
            n_pad=n_pad_norm,
            pad_frac=pad_frac,
        )

        conditions[label] = {
            "mean": mean,
            "sem": sem,
            "n_bouts": n_bouts,
            "speed_profile": speed_profile,
            "n_speed_bouts": n_speed_bouts,
            "bouts": bouts,
        }

    return {
        "units": uids,
        "t_norm": t_norm,
        "conditions": conditions,
        "params": {
            "speed_threshold": speed_threshold,
            "min_bout_bins": min_bout_bins,
            "bin_dur": bin_dur,
            "n_norm": n_norm,
            "n_pad_norm": n_pad_norm,
            "pad_frac": pad_frac,
            "smooth_sigma": smooth_sigma,
            "min_bouts": min_bouts,
        },
    }


#  1. Raster plot

def plot_raster(spike_counts, cov_df,
                epoch=None, units=None, plot_position=False,
                figsize=None):
    """Raster plot for a set of units. Use %matplotlib qt for zoom/pan.

    Parameters
    ----------
    spike_counts  : 2-D array (n_units x n_bins)
    cov_df        : DataFrame with 'time_bin_center' column (and 'epoch')
    epoch         : if given, restrict to bins where cov_df["epoch"] == epoch
    units         : list of unit indices to plot (default: all)
    plot_position : if True, add a panel below showing linear position
                    coloured by speed. Requires 'linear_position' and
                    'speed' columns in cov_df.
    figsize       : figure size tuple
    """
    times = cov_df["time_bin_center"].values.copy()
    spk = np.asarray(spike_counts)

    if epoch is not None:
        mask = cov_df["epoch"].values == epoch
        times = times[mask]
        spk = spk[:, mask]
        cov_sub = cov_df.iloc[np.where(mask)[0]]
    else:
        cov_sub = cov_df

    # show time relative to start of the displayed window
    t0 = times[0]
    times = times - t0

    spk, uids = _select_units(spk, units)

    spike_times = [times[spk[i] > 0] for i in range(len(uids))]

    if figsize is None:
        h = max(3, 0.4 * len(uids)) + (2.5 if plot_position else 0)
        figsize = (14, h)

    if plot_position:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[max(3, 0.4 * len(uids)), 2.5],
            width_ratios=[40, 1]
        )
        ax_raster = fig.add_subplot(gs[0, 0])
        ax_pos = fig.add_subplot(gs[1, 0], sharex=ax_raster)
        cax = fig.add_subplot(gs[:, 1])
    else:
        fig, ax_raster = plt.subplots(figsize=figsize, constrained_layout=True)

    ax_raster.eventplot(spike_times, lineoffsets=np.arange(len(uids)),
                        linelengths=0.8, linewidths=0.4, color="k")
    ax_raster.set_yticks(np.arange(len(uids)))
    ax_raster.set_yticklabels(uids, fontsize=7)
    ax_raster.set_ylabel("Unit")
    title = "Raster"
    if epoch is not None:
        title += f" — epoch {epoch}"
    ax_raster.set_title(title)
    sns.despine(ax=ax_raster)

    if plot_position:
        pos = cov_sub["linear_position"].values
        spd = cov_sub["speed"].values
        pos_times = cov_sub["time_bin_center"].values - t0
        sc = ax_pos.scatter(pos_times, pos, c=spd, s=0.3, cmap="viridis",
                            rasterized=True)
        ax_pos.set_ylabel("Linear position")
        ax_pos.set_xlabel("Time (s)")
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label("Speed (cm/s)", fontsize=8)
        sns.despine(ax=ax_pos)
    else:
        ax_raster.set_xlabel("Time (s)")

    if plot_position:
        return fig, (ax_raster, ax_pos)
    return fig, ax_raster


#  2. ISI distribution grid 

def plot_isi_grid(spike_counts, bin_size=0.002, cov_df=None, epoch=None,
                  units=None, max_isi_ms=100, n_hist_bins=100, figsize=None):
    """Plot ISI distributions for a set of units in an auto-sized grid.

    Parameters
    ----------
    spike_counts  : 2-D array (n_units x n_bins)
    bin_size      : bin width in seconds (default 0.002)
    cov_df        : DataFrame with 'epoch' column (needed if epoch is set)
    epoch         : if given, restrict to bins where cov_df["epoch"] == epoch
    units         : list of unit indices to plot (default: all)
    max_isi_ms    : upper ISI limit in ms for the histogram (default 100)
    n_hist_bins   : number of histogram bins (default 100)
    figsize       : figure size tuple
    """
    spk = np.asarray(spike_counts)
    if epoch is not None:
        mask = cov_df["epoch"].values == epoch
        spk = spk[:, mask]
    spk, uids = _select_units(spk, units)
    n = len(uids)
    nrows, ncols = _grid_shape(n)

    isi_matrix, bin_edges_ms, n_spikes_per_unit = _compute_isi_matrix(
        spk, bin_size=bin_size, max_isi_ms=max_isi_ms, n_hist_bins=n_hist_bins
    )
    bin_centers_ms = 0.5 * (bin_edges_ms[:-1] + bin_edges_ms[1:])
    bin_width_ms = bin_edges_ms[1] - bin_edges_ms[0]

    if figsize is None:
        figsize = (3.5 * ncols, 2.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for idx in range(n):
        ax = axes.flat[idx]
        if n_spikes_per_unit[idx] < 2:
            ax.text(0.5, 0.5, "< 2 spikes", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(f"Unit {uids[idx]}", fontsize=8)
            continue

        ax.bar(bin_centers_ms, isi_matrix[idx],
               width=bin_width_ms, color="steelblue", edgecolor="none",
               align="center")
        ax.axvline(bin_size * 1000, color="red", lw=0.8, ls="--",
                   label=f"1 bin ({bin_size*1000:.0f} ms)")
        ax.set_title(f"Unit {uids[idx]}", fontsize=8)
        ax.set_xlabel("ISI (ms)", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=6)
        sns.despine(ax=ax)

    for idx in range(n, nrows * ncols):
        axes.flat[idx].set_visible(False)

    title = f"Inter-spike interval distributions (bin = {bin_size*1000:.0f} ms)"
    if epoch is not None:
        title += f" — epoch {epoch}"
    fig.suptitle(title, fontsize=11)
    return fig, axes


#  3. Autocorrelogram grid 

def plot_acg_grid(spike_counts, bin_size=0.002, cov_df=None, epoch=None,
                  units=None, max_lag_ms=50, plot_bin_ms=None, figsize=None):
    """Plot autocorrelograms for a set of units in an auto-sized grid.

    Parameters
    ----------
    spike_counts : 2-D array (n_units x n_bins)
    bin_size     : spike-count bin width in seconds (default 0.002)
    cov_df       : DataFrame with 'epoch' column (needed if epoch is set)
    epoch        : if given, restrict to bins where cov_df["epoch"] == epoch
    units        : list of unit indices to plot (default: all)
    max_lag_ms   : maximum lag in ms (default 50)
    plot_bin_ms  : display bin width in ms (default: same as bin_size).
                   Must be a multiple of bin_size. Bins are summed.
    figsize      : figure size tuple
    """
    spk = np.asarray(spike_counts)
    if epoch is not None:
        mask = cov_df["epoch"].values == epoch
        spk = spk[:, mask]
    spk, uids = _select_units(spk, units)
    n = len(uids)
    nrows, ncols = _grid_shape(n)

    acg_matrix, lags_ms, n_spikes_per_unit, plot_bin_ms = _compute_acg_matrix(
        spk, bin_size=bin_size, max_lag_ms=max_lag_ms, plot_bin_ms=plot_bin_ms
    )

    if figsize is None:
        figsize = (3.5 * ncols, 2.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for idx in range(n):
        ax = axes.flat[idx]
        if n_spikes_per_unit[idx] < 2:
            ax.text(0.5, 0.5, "< 2 spikes", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(f"Unit {uids[idx]}", fontsize=8)
            continue

        ax.bar(lags_ms, acg_matrix[idx],
               width=plot_bin_ms * 0.9, color="steelblue", edgecolor="none")
        ax.set_title(f"Unit {uids[idx]}", fontsize=8)
        ax.set_xlabel("Lag (ms)", fontsize=7)
        ax.set_ylabel("Coincidences", fontsize=7)
        ax.tick_params(labelsize=6)
        sns.despine(ax=ax)

    for idx in range(n, nrows * ncols):
        axes.flat[idx].set_visible(False)

    title = f"Autocorrelograms (±{max_lag_ms} ms, {plot_bin_ms:.0f} ms bins)"
    if epoch is not None:
        title += f" — epoch {epoch}"
    fig.suptitle(title, fontsize=11)
    return fig, axes


#  4. Waveform template grid

def plot_waveform_grid(waveform_extractor, unit_ids=None, figsize=None):
    """Plot mean waveform templates (peak channel) for a set of units in an auto-sized grid.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
    unit_ids           : list of unit IDs to plot (default: all in extractor)
    figsize            : figure size tuple
    """
    all_ids = list(waveform_extractor.sorting.get_unit_ids())
    uids = all_ids if unit_ids is None else list(unit_ids)

    n = len(uids)
    nrows, ncols = _grid_shape(n)
    fs = waveform_extractor.recording.sampling_frequency

    if figsize is None:
        figsize = (3.5 * ncols, 2.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for idx, uid in enumerate(uids):
        ax = axes.flat[idx]
        try:
            template = waveform_extractor.get_template(uid, mode="average", force_dense=True)
        except Exception:
            ax.text(0.5, 0.5, "no template", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(f"Unit {uid}", fontsize=8)
            continue

        peak_ch = np.argmax(np.ptp(template, axis=0))
        t_ms = (np.arange(template.shape[0]) / fs) * 1000.0

        ax.plot(t_ms, template[:, peak_ch], color="C2", lw=1.2)
        ax.set_title(f"Unit {uid}", fontsize=8)
        ax.set_xlabel("Time (ms)", fontsize=7)
        ax.tick_params(labelsize=6)
        sns.despine(ax=ax)

    for idx in range(n, nrows * ncols):
        axes.flat[idx].set_visible(False)

    fig.suptitle("Waveform templates (peak channel)", fontsize=11)
    return fig, axes


#  5. ISI heatmap (rows = units, x = ISI ms)

def plot_isi_heatmap(spike_counts, bin_size=0.002, cov_df=None, epoch=None,
                     units=None, max_isi_ms=100, n_hist_bins=100,
                     cmap="Blues", normalize=False, sort_by=None,
                     smooth_sigma_ms=None, figsize=None):
    """ISI distributions as a heatmap (one row per unit).

    Parameters
    ----------
    spike_counts    : 2-D array (n_units x n_bins)
    bin_size        : spike-count bin width in seconds (default 0.002)
    cov_df          : DataFrame with 'epoch' column (needed if epoch is set)
    epoch           : if given, restrict to bins where cov_df["epoch"] == epoch
    units           : list of unit indices to plot (default: all)
    max_isi_ms      : upper ISI limit in ms (default 100)
    n_hist_bins     : number of ISI histogram bins (default 100)
    cmap            : matplotlib colormap (default "Blues")
    normalize       : if True, divide each row by its peak so every row spans [0, 1]
    sort_by         : None (input order) or "peak" (sort rows by argmax ISI bin)
    smooth_sigma_ms : if not None, apply a Gaussian filter of this sigma (ms)
                      along each row before normalize/sort. Smoothing in real
                      ms is invariant to n_hist_bins / max_isi_ms changes.
    figsize         : figure size tuple
    """
    spk = np.asarray(spike_counts)
    if epoch is not None:
        mask = cov_df["epoch"].values == epoch
        spk = spk[:, mask]
    spk, uids = _select_units(spk, units)
    uids = np.asarray(uids)
    n = len(uids)

    isi_matrix, bin_edges_ms, _ = _compute_isi_matrix(
        spk, bin_size=bin_size, max_isi_ms=max_isi_ms, n_hist_bins=n_hist_bins
    )

    if smooth_sigma_ms is not None and smooth_sigma_ms > 0:
        bin_w_ms = bin_edges_ms[1] - bin_edges_ms[0]
        sigma_bins = smooth_sigma_ms / bin_w_ms
        isi_matrix = gaussian_filter1d(isi_matrix, sigma=sigma_bins, axis=1)

    if normalize:
        row_max = isi_matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        isi_matrix = isi_matrix / row_max

    if sort_by == "peak":
        order = np.argsort(np.argmax(isi_matrix, axis=1))
        isi_matrix = isi_matrix[order]
        uids = uids[order]
    elif sort_by is not None:
        raise ValueError(f"sort_by must be None or 'peak', got {sort_by!r}")

    if figsize is None:
        figsize = (8, max(3, 0.25 * n + 2))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    extent = [bin_edges_ms[0], bin_edges_ms[-1], n - 0.5, -0.5]
    im = ax.imshow(isi_matrix, aspect="auto", cmap=cmap,
                   extent=extent, interpolation="nearest")

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(uids, fontsize=7)
    ax.set_xlabel("ISI (ms)")
    ax.set_ylabel("Unit")
    title = f"ISI distributions (bin = {bin_size*1000:.0f} ms)"
    if epoch is not None:
        title += f" — epoch {epoch}"
    ax.set_title(title, fontsize=11)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized" if normalize else "Count", fontsize=8)
    return fig, ax


#  6. ACG heatmap (rows = units, x = lag ms)

def plot_acg_heatmap(spike_counts, bin_size=0.002, cov_df=None, epoch=None,
                     units=None, max_lag_ms=50, plot_bin_ms=None,
                     cmap="Blues", normalize=False, sort_by=None,
                     smooth_sigma_ms=None, figsize=None):
    """Autocorrelograms as a heatmap (one row per unit).

    Parameters
    ----------
    spike_counts    : 2-D array (n_units x n_bins)
    bin_size        : spike-count bin width in seconds (default 0.002)
    cov_df          : DataFrame with 'epoch' column (needed if epoch is set)
    epoch           : if given, restrict to bins where cov_df["epoch"] == epoch
    units           : list of unit indices to plot (default: all)
    max_lag_ms      : maximum lag in ms (default 50)
    plot_bin_ms     : display bin width in ms (default: same as bin_size).
                      Snapped to a multiple of bin_size; bins are summed.
    cmap            : matplotlib colormap (default "Blues")
    normalize       : if True, divide each row by its max so colors are 0–1
    sort_by         : None (input order) or "peak" (sort rows by argmax in the
                      positive-lag half — ACG is symmetric, so argmax on the
                      full row always lands on the negative side and is
                      therefore not used).
    smooth_sigma_ms : if not None, apply a Gaussian filter of this sigma (ms)
                      along each row before normalize/sort.
    figsize         : figure size tuple
    """
    spk = np.asarray(spike_counts)
    if epoch is not None:
        mask = cov_df["epoch"].values == epoch
        spk = spk[:, mask]
    spk, uids = _select_units(spk, units)
    uids = np.asarray(uids)
    n = len(uids)

    acg_matrix, lags_ms, _, plot_bin_ms = _compute_acg_matrix(
        spk, bin_size=bin_size, max_lag_ms=max_lag_ms, plot_bin_ms=plot_bin_ms
    )

    if smooth_sigma_ms is not None and smooth_sigma_ms > 0:
        sigma_bins = smooth_sigma_ms / plot_bin_ms
        acg_matrix = gaussian_filter1d(acg_matrix, sigma=sigma_bins, axis=1)

    if normalize:
        row_max = acg_matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        acg_matrix = acg_matrix / row_max

    if sort_by == "peak":
        # ACG is symmetric: argmax on the full row would tie-break to the
        # negative side. Use the positive-lag half so peak position has the
        # natural interpretation "peak ACG at +X ms".
        center = acg_matrix.shape[1] // 2
        peak_idx = np.argmax(acg_matrix[:, center + 1:], axis=1)
        order = np.argsort(peak_idx)
        acg_matrix = acg_matrix[order]
        uids = uids[order]
    elif sort_by is not None:
        raise ValueError(f"sort_by must be None or 'peak', got {sort_by!r}")

    if figsize is None:
        figsize = (8, max(3, 0.25 * n + 2))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    half_bin = plot_bin_ms / 2.0
    extent = [lags_ms[0] - half_bin, lags_ms[-1] + half_bin, n - 0.5, -0.5]
    im = ax.imshow(acg_matrix, aspect="auto", cmap=cmap,
                   extent=extent, interpolation="nearest")

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(uids, fontsize=7)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Unit")
    title = f"Autocorrelograms (±{max_lag_ms} ms, {plot_bin_ms:.0f} ms bins)"
    if epoch is not None:
        title += f" — epoch {epoch}"
    ax.set_title(title, fontsize=11)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized" if normalize else "Coincidences", fontsize=8)
    return fig, ax


#  7. Bout-normalized firing-rate plots

def plot_bout_normalized_firing_rates(
    result,
    conditions=("rest", "run"),
    colors=None,
    plot_speed_profile=True,
    speed_colors=None,
    speed_alpha=0.35,
    speed_lw=0.9,
    show_sem=True,
    figsize=None,
    boundary_color="k",
):
    """Plot normalized run/rest firing-rate curves for each unit in a grid.

    Parameters
    ----------
    result             : output of `compute_bout_normalized_firing_rates`
    conditions         : iterable of condition names to plot
    colors             : dict mapping condition name to line color
    plot_speed_profile : if True, overlay average speed on a secondary axis
    speed_colors       : dict mapping condition name to background speed color.
                         Defaults to `colors`.
    speed_alpha        : alpha for the background speed trace
    speed_lw           : linewidth for the background speed trace
    show_sem           : if True, fill mean +/- SEM
    figsize            : figure size tuple
    """
    units = np.asarray(result["units"])
    t_norm = np.asarray(result["t_norm"])
    params = result["params"]
    cond_data = result["conditions"]

    n = len(units)
    nrows, ncols = _grid_shape(n)

    if figsize is None:
        figsize = (3.5 * ncols, 2.5 * nrows)

    if colors is None:
        colors = {"rest": "steelblue", "run": "darkorange"}
    if speed_colors is None:
        speed_colors = colors

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
    )
    axes = np.atleast_2d(axes)

    for idx, uid in enumerate(units):
        ax = axes.flat[idx]
        plotted_labels = []

        for label in conditions:
            if label not in cond_data:
                continue
            mean = cond_data[label]["mean"][idx]
            sem = cond_data[label]["sem"][idx]
            n_bouts = cond_data[label]["n_bouts"][idx]
            if np.all(np.isnan(mean)):
                continue

            color = colors.get(label, None)
            ax.plot(t_norm, mean, color=color, lw=1.2, label=f"{label} (n={n_bouts})")
            plotted_labels.append(label)
            if show_sem and not np.all(np.isnan(sem)):
                ax.fill_between(t_norm, mean - sem, mean + sem, color=color, alpha=0.25)

        if plot_speed_profile:
            ax2 = ax.twinx()
            for label in conditions:
                if label not in cond_data:
                    continue
                speed_profile = cond_data[label]["speed_profile"]
                if np.all(np.isnan(speed_profile)):
                    continue
                ax2.plot(
                    t_norm,
                    speed_profile,
                    color=speed_colors.get(label, colors.get(label, "k")),
                    lw=speed_lw,
                    alpha=speed_alpha,
                )
            ax2.set_yticks([])
            ax2.set_ylabel("")

        ax.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)
        ax.axvline(100, color="k", lw=1.0, ls="--", alpha=0.7)
        ax.set_xlim(-params["pad_frac"] * 100 - 5, (1 + params["pad_frac"]) * 100 + 5)
        ax.set_title(f"Unit {uid}", fontsize=8)
        ax.tick_params(labelsize=6)
        if idx == 0 and plotted_labels:
            ax.legend(fontsize=5)
        if idx % ncols == 0:
            ax.set_ylabel("FR (Hz)", fontsize=7)

    for idx in range(n, nrows * ncols):
        axes.flat[idx].set_visible(False)

    fig.suptitle(
        f"Time-normalized PSTH (speed > {params['speed_threshold']} cm/s = run)",
        fontsize=11,
    )
    fig.supxlabel("Normalized time within bout (%)", fontsize=9)
    return fig, axes


def plot_bout_normalized_fr_heatmaps(
    result,
    conditions=("run", "rest"),
    cmap="viridis",
    normalize_rows=False,
    smooth_sigma=None,
    sort_by=None,
    vmin=None,
    vmax=None,
    share_color_scale=True,
    figsize=None,
    boundary_color="k",
):
    """Plot bout-normalized firing rates as side-by-side heatmaps.

    Parameters
    ----------
    result            : output of `compute_bout_normalized_firing_rates`
    conditions        : pair of condition names to plot. Defaults to run/rest.
    cmap              : matplotlib colormap
    normalize_rows    : if True, divide each unit row by its own peak
    smooth_sigma      : optional Gaussian sigma in normalized time bins
    sort_by           : None, "run", "rest", "peak_run", or "peak_rest"
    vmin, vmax        : color scale bounds. If omitted, inferred from data
    share_color_scale : if True, both panels use the same color scale
    figsize           : figure size tuple
    boundary_color    : color for the 0% and 100% bout-boundary lines
    """
    if len(conditions) != 2:
        raise ValueError("conditions must contain exactly two condition names")

    units = np.asarray(result["units"])
    t_norm = np.asarray(result["t_norm"])
    cond_data = result["conditions"]
    params = result["params"]

    missing = [label for label in conditions if label not in cond_data]
    if missing:
        raise ValueError(f"unknown condition(s): {missing}")

    matrices = {
        label: np.asarray(cond_data[label]["mean"], dtype=float).copy()
        for label in conditions
    }

    if smooth_sigma is not None and smooth_sigma > 0:
        matrices = {
            label: _nan_gaussian_filter1d(matrix, sigma=smooth_sigma, axis=1)
            for label, matrix in matrices.items()
        }

    if normalize_rows:
        for label, matrix in matrices.items():
            row_max = np.ones((matrix.shape[0], 1), dtype=float)
            valid_rows = np.any(np.isfinite(matrix), axis=1)
            if np.any(valid_rows):
                row_max[valid_rows, 0] = np.nanmax(matrix[valid_rows], axis=1)
            row_max[~np.isfinite(row_max) | (row_max == 0)] = 1.0
            matrices[label] = matrix / row_max

    if sort_by is None:
        order = np.arange(len(units))
    elif sort_by in conditions:
        matrix = matrices[sort_by]
        values = np.full(matrix.shape[0], -np.inf, dtype=float)
        valid_rows = np.any(np.isfinite(matrix), axis=1)
        if np.any(valid_rows):
            values[valid_rows] = np.nanmean(matrix[valid_rows], axis=1)
        order = np.argsort(values)
    elif sort_by.startswith("peak_") and sort_by[5:] in conditions:
        matrix = matrices[sort_by[5:]]
        peak_idx = np.argmax(np.where(np.isfinite(matrix), matrix, -np.inf), axis=1)
        order = np.argsort(peak_idx)
    else:
        valid_sort = [None, *conditions, *(f"peak_{label}" for label in conditions)]
        raise ValueError(f"sort_by must be one of {valid_sort}, got {sort_by!r}")

    units = units[order]
    matrices = {label: matrix[order] for label, matrix in matrices.items()}

    if share_color_scale:
        all_data = np.concatenate([matrices[label].ravel() for label in conditions])
        all_data = all_data[np.isfinite(all_data)]
        if vmin is None:
            vmin = np.min(all_data) if len(all_data) else 0.0
        if vmax is None:
            vmax = np.max(all_data) if len(all_data) else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0

    if figsize is None:
        figsize = (12, max(3, 0.25 * len(units) + 2))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    extent = [t_norm[0], t_norm[-1], len(units) - 0.5, -0.5]
    images = []
    for ax, label in zip(axes, conditions):
        matrix = matrices[label]
        im_vmin = vmin
        im_vmax = vmax
        if not share_color_scale:
            finite = matrix[np.isfinite(matrix)]
            if im_vmin is None:
                im_vmin = np.min(finite) if len(finite) else 0.0
            if im_vmax is None:
                im_vmax = np.max(finite) if len(finite) else 1.0
            if im_vmin == im_vmax:
                im_vmax = im_vmin + 1.0

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            extent=extent,
            interpolation="nearest",
            vmin=im_vmin,
            vmax=im_vmax,
        )
        images.append(im)

        ax.axvline(0, color=boundary_color, lw=1.0, ls="--", alpha=0.9)
        ax.axvline(100, color=boundary_color, lw=1.0, ls="--", alpha=0.9)
        ax.set_xlim(-params["pad_frac"] * 100 - 5, (1 + params["pad_frac"]) * 100 + 5)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Normalized time within bout (%)")

    axes[0].set_ylabel("Unit")
    axes[0].set_yticks(np.arange(len(units)))
    axes[0].set_yticklabels(units, fontsize=7)

    cbar = fig.colorbar(images[0], ax=axes, shrink=0.9)
    cbar.set_label("Row-normalized FR" if normalize_rows else "FR (Hz)", fontsize=8)

    fig.suptitle(
        f"Bout-normalized firing rates (speed > {params['speed_threshold']} cm/s = run)",
        fontsize=11,
    )
    return fig, axes


def plot_run_rest_firing_rate_bars(
    result,
    conditions=("rest", "run"),
    colors=None,
    figsize=None,
    bar_width=0.38,
    capsize=3,
    rotate_xticks=None,
):
    """Plot per-unit run/rest firing rates with standard error bars.

    Parameters
    ----------
    result        : output of `compute_run_rest_firing_rates`
    conditions    : conditions to plot, default rest/run
    colors        : dict mapping condition name to bar color
    figsize       : figure size tuple
    bar_width     : width of each condition bar
    capsize       : error-bar cap size
    rotate_xticks : bool. Defaults to rotating labels when many units are shown
    """
    units = np.asarray(result["units"])
    cond_data = result["conditions"]
    params = result["params"]

    missing = [label for label in conditions if label not in cond_data]
    if missing:
        raise ValueError(f"unknown condition(s): {missing}")

    if colors is None:
        colors = {"rest": "steelblue", "run": "darkorange"}
    if figsize is None:
        figsize = (max(8, 0.45 * len(units) + 3), 4)
    if rotate_xticks is None:
        rotate_xticks = len(units) > 12

    x = np.arange(len(units))
    offsets = (np.arange(len(conditions)) - (len(conditions) - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for offset, label in zip(offsets, conditions):
        mean = np.asarray(cond_data[label]["mean"], dtype=float)
        sem = np.asarray(cond_data[label]["sem"], dtype=float)
        n_bouts = cond_data[label]["n_bouts"]
        bar_labels = [f"n={n}" if np.isfinite(m) else "" for m, n in zip(mean, n_bouts)]

        bars = ax.bar(
            x + offset,
            mean,
            width=bar_width,
            yerr=sem,
            capsize=capsize,
            color=colors.get(label, None),
            edgecolor="none",
            label=label,
            alpha=0.9,
        )
        ax.bar_label(bars, labels=bar_labels, fontsize=6, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(
        units,
        rotation=90 if rotate_xticks else 0,
        fontsize=7 if rotate_xticks else 8,
    )
    ax.set_xlabel("Unit")
    ax.set_ylabel("FR (Hz)")
    ax.set_title(
        f"Run vs rest firing rates (speed > {params['speed_threshold']} cm/s = run)",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    sns.despine(ax=ax)
    return fig, ax
