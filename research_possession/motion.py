"""Player-only, identity-free motion summaries and offline temporal features.

Every input observation is a point and a provider-reported speed in metres per
second. No ball geometry, target label, or stable player/track identity is used.
Team labels only partition the points; swapping the teams leaves every feature
unchanged. Temporal features use future observations and are for offline use.
"""

import numpy as np


PLAYER_SUMMARY_NAMES = (
    "speed_q10_mps", "speed_q50_mps", "speed_q90_mps", "speed_mean_mps",
    "speed_std_mps", "stationary_fraction", "running_fraction",
    "visible_location_count", "team_location_count_min", "team_location_count_max",
    "known_speed_fraction", "team_x_mad_m", "team_y_mad_m",
    "team_median_separation_m", "x_range80_fraction", "y_range80_fraction",
    "touchline_distance_q50_fraction", "duplicate_fraction",
)

MOTION_WINDOWS_S = (0.5, 2.0, 5.0)


def _quantiles(values, quantiles):
    """Row quantiles, with unknown all-missing rows and no NumPy warnings."""
    result = np.full((len(values), len(quantiles)), np.nan)
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    observed = counts > 0
    if observed.any():
        # Sorting short player rows once avoids nanquantile's per-row Python
        # overhead on full-match arrays. This is the same linear interpolation.
        ordered = np.sort(np.where(finite[observed], values[observed], np.inf), axis=1)
        position = (counts[observed, None] - 1) * np.asarray(quantiles)[None, :]
        lower = np.floor(position).astype(int)
        upper = np.ceil(position).astype(int)
        fraction = position - lower
        row = np.arange(observed.sum())[:, None]
        result[observed] = ordered[row, lower] * (1 - fraction) + ordered[row, upper] * fraction
    return result


def _divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float),
                     where=denominator > 0)


def _summarize_batch(xy, speeds, teams, pitch):
    visible = np.isfinite(xy).all(axis=2)
    raw_count = visible.sum(axis=1)
    speeds = np.where(visible & np.isfinite(speeds) & (speeds >= 0), speeds, np.nan)

    # Exact colocated records on the same team are duplicate observations, not
    # extra players. Median speed within a group makes its representative
    # independent of slot order, including duplicates with missing speed.
    same = ((teams[:, None] == teams[None, :])[None, :, :] &
            (xy[:, :, None, 0] == xy[:, None, :, 0]) &
            (xy[:, :, None, 1] == xy[:, None, :, 1]) &
            visible[:, :, None] & visible[:, None, :])
    duplicate = (same & np.tril(np.ones((len(teams), len(teams)), dtype=bool), -1)).any(axis=2)
    repeated_rows = duplicate.any(axis=1)
    if repeated_rows.any():
        grouped = np.where(same[repeated_rows], speeds[repeated_rows, None, :], np.nan)
        grouped = _quantiles(grouped.reshape(-1, len(teams)), [0.5])[:, 0]
        speeds[repeated_rows] = grouped.reshape(repeated_rows.sum(), len(teams))
    visible &= ~duplicate
    speeds = np.where(visible, speeds, np.nan)
    xy = np.where(visible[:, :, None], xy, np.nan)
    count = visible.sum(axis=1)
    speed_known = np.isfinite(speeds)
    speed_count = speed_known.sum(axis=1)
    speed_mean = _divide(np.nansum(speeds, axis=1), speed_count)
    speed_variance = _divide(np.nansum((speeds - speed_mean[:, None]) ** 2, axis=1), speed_count)
    speed_quantiles = _quantiles(speeds, [0.1, 0.5, 0.9])
    team_counts = np.column_stack([(visible & (teams == side)).sum(axis=1) for side in (0, 1)])
    centers = np.full((len(xy), 2, 2), np.nan)
    centered = np.full_like(xy, np.nan)
    for side in (0, 1):
        selected = teams == side
        for dimension in (0, 1):
            centers[:, side, dimension] = _quantiles(xy[:, selected, dimension], [0.5])[:, 0]
        centered[:, selected] = xy[:, selected] - centers[:, side, None, :]
    spreads = np.column_stack([_quantiles(np.abs(centered[:, :, d]), [0.5])[:, 0]
                               for d in (0, 1)])
    separation = np.linalg.norm(centers[:, 0] - centers[:, 1], axis=1)
    ranges = np.column_stack([np.diff(_quantiles(xy[:, :, d], [0.1, 0.9]), axis=1)[:, 0] / pitch[d]
                              for d in (0, 1)])
    touchline = np.minimum(np.abs(xy[:, :, 1]), np.abs(pitch[1] - xy[:, :, 1])) / pitch[1]
    duplicate_fraction = np.divide(raw_count - count, raw_count, out=np.zeros(len(xy)), where=raw_count > 0)
    return np.column_stack((
        speed_quantiles, speed_mean, np.sqrt(speed_variance),
        _divide(((speeds < 1.0) & speed_known).sum(axis=1), speed_count),
        _divide(((speeds > 3.0) & speed_known).sum(axis=1), speed_count),
        count, team_counts.min(axis=1), team_counts.max(axis=1),
        _divide(speed_count, count), spreads, separation, ranges,
        _quantiles(touchline, [0.5])[:, 0], duplicate_fraction,
    ))


def summarize_player_frames(xy_m, speed_mps, teams, pitch):
    """Return ``(N, 18)`` summaries from frame/slot arrays in positive metres.

    ``xy_m`` is ``(N, P, 2)``, ``speed_mps`` is ``(N, P)``, and ``teams`` is
    ``(P,)`` with values 0/1. A slot is used only within its own frame; it may
    change identity at any time. Missing geometry excludes the entire point;
    invalid or negative speed excludes only that speed. Coverage counts are
    observed distinct locations, not an estimate of the number of players.
    Exact same-team coordinate duplicates collapse to their median known speed;
    nearby observations are retained because they may be distinct players.

    Speeds use m/s, spread and team-center separation use metres, and occupancy
    widths and touchline distance are divided by pitch dimensions. The team
    centers are coordinate medians and the spreads are pooled absolute median
    deviations from those centers. Unknown quantities remain NaN; a frame with
    no players has zero counts and zero duplicate fraction, not zero speed.
    """
    xy = np.asarray(xy_m, dtype=float)
    speed = np.asarray(speed_mps, dtype=float)
    team = np.asarray(teams)
    dimensions = np.asarray(pitch, dtype=float)
    if xy.ndim != 3 or xy.shape[2] != 2 or speed.shape != xy.shape[:2]:
        raise ValueError("Player arrays must have shapes (N, P, 2) and (N, P)")
    if team.shape != (xy.shape[1],) or not np.isin(team, [0, 1]).all():
        raise ValueError("Each player slot must have team 0 or 1")
    if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or (dimensions <= 0).any():
        raise ValueError("pitch must contain two finite positive dimensions")
    result = np.full((len(xy), len(PLAYER_SUMMARY_NAMES)), np.nan)
    if not xy.shape[1]:
        result[:, [7, 8, 9, 17]] = 0
        return result
    # Bound duplicate-comparison memory even when track fragmentation creates
    # many slots. Most of the remaining computation is vectorized by frame.
    batch_size = max(1, min(2048, 2_000_000 // (xy.shape[1] ** 2)))
    for start in range(0, len(xy), batch_size):
        end = min(len(xy), start + batch_size)
        result[start:end] = _summarize_batch(xy[start:end], speed[start:end], team, dimensions)
    return result


def summarize_players(xy_m, speed_mps, teams, pitch):
    """One-frame convenience interface returning ``(18,)``; see above."""
    return summarize_player_frames(np.asarray(xy_m)[None, ...], np.asarray(speed_mps)[None, ...],
                                   teams, pitch)[0]


def _base_summary(features):
    n = len(features)
    if features.player_summary is not None:
        result = np.asarray(features.player_summary, dtype=float)
        if result.shape != (n, len(PLAYER_SUMMARY_NAMES)):
            raise ValueError("player_summary must have shape (N, len(PLAYER_SUMMARY_NAMES))")
        return result
    # Keep the six-argument v1 Features constructor useful. Only quantities
    # actually available in v1 are populated; the rest get missing indicators.
    speed = np.asarray(features.player_speed_mps, dtype=float)
    counts = np.asarray(features.player_count, dtype=float)
    if speed.shape != (n,) or counts.shape != (n, 2):
        raise ValueError("Legacy player speed/count arrays have inconsistent shapes")
    result = np.full((n, len(PLAYER_SUMMARY_NAMES)), np.nan)
    result[:, 1] = np.where(np.isfinite(speed) & (speed >= 0), speed, np.nan)
    known = np.isfinite(counts).all(axis=1) & (counts >= 0).all(axis=1)
    result[known, 7] = counts[known].sum(axis=1)
    result[known, 8] = counts[known].min(axis=1)
    result[known, 9] = counts[known].max(axis=1)
    return result


def _window_mean(cumulative, support, starts, ends):
    total = cumulative[ends] - cumulative[starts]
    count = support[ends] - support[starts]
    return _divide(total, count)


def motion_design(features, blocks):
    """Return finite ``float32[N, 252]`` predictors and their ordered names.

    Raw player summaries are followed by a missing flag for every quantity.
    For each 0.5, 2 and 5 second window, append the centered mean and the
    future-minus-past mean contrast, each with its own missing flags. The window
    radius is ``max(1, round(seconds * fps / 2))`` frames; contrast halves omit
    the current frame. Missing inputs never count as zero observations. Zero
    imputation occurs only in the final design, alongside explicit missing flags.

    ``blocks`` contains ordered, disjoint half-open period slices. Every window
    is clipped to its own block, and temporal quantities outside blocks are
    missing. Partial boundary windows are allowed; a contrast needs observations
    on both sides. This deliberately uses future *player observations* and is
    suitable only for offline inference. No ball or target values are inspected.
    """
    base = _base_summary(features)
    n, width = base.shape
    fps = float(features.fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    periods = []
    previous_end = 0
    for block in blocks:
        if len(block) != 2:
            raise ValueError("Motion blocks must be (start, end) pairs")
        start, end = block
        if (not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)) or
                start < previous_end or end <= start or end > n):
            raise ValueError("Motion blocks must be ordered, disjoint, and within the feature rows")
        periods.append((int(start), int(end)))
        previous_end = end
    names = []
    design = np.empty((n, width * 2 * (1 + 2 * len(MOTION_WINDOWS_S))), dtype=np.float32)
    column = 0

    def append(values, prefix):
        nonlocal column
        known = np.isfinite(values)
        limit = np.finfo(np.float32).max
        design[:, column:column + width] = np.clip(np.where(known, values, 0), -limit, limit)
        design[:, column + width:column + 2 * width] = ~known
        names.extend(prefix + name for name in PLAYER_SUMMARY_NAMES)
        names.extend(prefix + name + "_missing" for name in PLAYER_SUMMARY_NAMES)
        column += 2 * width

    append(base, "")
    for seconds in MOTION_WINDOWS_S:
        mean = np.full_like(base, np.nan)
        contrast = np.full_like(base, np.nan)
        radius = max(1, round(seconds * fps / 2))
        for start, end in periods:
            values = base[start:end]
            known = np.isfinite(values)
            sums = np.vstack((np.zeros(width), np.cumsum(np.where(known, values, 0), axis=0)))
            support = np.vstack((np.zeros(width, dtype=np.int64), np.cumsum(known, axis=0)))
            current = np.arange(end - start)
            left = np.maximum(0, current - radius)
            right = np.minimum(end - start, current + radius + 1)
            mean[start:end] = _window_mean(sums, support, left, right)
            past = _window_mean(sums, support, left, current)
            future = _window_mean(sums, support, current + 1, right)
            contrast[start:end] = future - past
        append(mean, f"mean_{seconds:g}s_")
        append(contrast, f"contrast_{seconds:g}s_")
    return design, names
