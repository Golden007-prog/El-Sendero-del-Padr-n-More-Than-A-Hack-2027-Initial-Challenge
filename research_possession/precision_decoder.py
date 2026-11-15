"""Binary duration-regularized status and ownership-preserving timing refinement.

Unary costs are rates integrated as u/fps, exactly as in frozen v2; transition
costs are charged once and remain unscaled. The soft one-second run penalty is
paid only when an interior run ends. This is min-cost inference, not a posterior
or a calibrated confidence estimate. All audit records are returned locally.
"""

from dataclasses import dataclass

import numpy as np

from .boundary_decoder import BoundaryConfig, decode_boundary
from .decoder import STATES, _validate
from .motion import motion_design
from .timing_model import _blocks, _runs, motion_support


@dataclass(frozen=True)
class PrecisionConfig:
    short_run_cost: float = 0.
    refine: bool = False

    def __post_init__(self):
        if (not np.isscalar(self.short_run_cost) or isinstance(self.short_run_cost, (bool, np.bool_)) or
                not np.isfinite(self.short_run_cost) or self.short_run_cost < 0):
            raise ValueError("short_run_cost must be finite and nonnegative")
        if not isinstance(self.refine, (bool, np.bool_)):
            raise ValueError("refine must be boolean")


def _duration_path(emission, switch_cost, fps, kappa):
    """Capped-age binary DP; vectorized over ages, deterministic traceback.

Age K (round(fps), at least one) represents all one-second-or-longer runs.
The first run starts at that censored age and the final run has no exit cost.
Only two predecessor ages per frame/state need storage: every other age has a
unique continuation. Space is O(N + K), work O(NK); no Python age loop is used.
    """
    emission = np.asarray(emission, dtype=float)
    switch_cost = np.asarray(switch_cost, dtype=float)
    n = len(emission)
    if (emission.shape != (n, 2) or switch_cost.shape != (n,) or
            not np.isfinite(emission).all() or not np.isfinite(switch_cost).all() or
            (switch_cost < 0).any() or not np.isfinite(fps) or fps <= 0 or
            not np.isfinite(kappa) or kappa < 0):
        raise ValueError("Invalid binary duration costs or frame rate")
    if not n:
        return np.empty(0, dtype=np.int8)
    age_count = max(1, round(float(fps)))
    unary = emission / fps
    if age_count == 1:
        previous = unary[0].copy()
        back = np.zeros((n, 2), dtype=np.int8)
        for frame in range(1, n):
            changed = previous[::-1] + switch_cost[frame]
            take_switch = changed < previous
            back[frame] = np.where(take_switch, [1, 0], [0, 1])
            previous = np.minimum(previous, changed) + unary[frame]
            previous -= previous.min()
        state = int(np.argmin(previous))
        path = np.empty(n, dtype=np.int8)
        for frame in range(n - 1, -1, -1):
            path[frame] = state
            state = int(back[frame, state])
        return path
    penalties = kappa * np.maximum(0., 1. - np.arange(1, age_count + 1) / fps)
    penalties[-1] = 0.
    previous = np.full((2, age_count), np.inf)
    previous[:, -1] = unary[0]
    switch_from = np.empty((n, 2), dtype=np.int32)
    mature_from = np.empty((n, 2), dtype=bool)
    states = np.arange(2)
    for frame in range(1, n):
        exit_cost = previous + penalties
        best_age = np.argmin(exit_cost, axis=1)
        current = np.empty_like(previous)
        current[:, 1:] = previous[:, :-1]
        # Prefer the existing mature run on ties; switches use the earliest age.
        keep_mature = previous[:, -1] <= previous[:, -2]
        current[:, -1] = np.minimum(previous[:, -1], previous[:, -2])
        current[:, 0] = exit_cost[states[::-1], best_age[::-1]] + switch_cost[frame]
        current += unary[frame, :, None]
        current -= current.min()
        switch_from[frame] = best_age[::-1]
        mature_from[frame] = keep_mature
        previous = current
    state, age = np.unravel_index(np.argmin(previous), previous.shape)
    path = np.empty(n, dtype=np.int8)
    for frame in range(n - 1, -1, -1):
        path[frame] = state
        if frame == 0:
            break
        if age == 0:
            age = int(switch_from[frame, state])
            state = 1 - state
        elif age == age_count - 1:
            age -= not mature_from[frame, state]
        else:
            age -= 1
    return path


def _project_boundaries(proposals, lower, upper):
    """Exact integer L1 projection with strict order and per-boundary bounds.

Dynamic programming uses only each boundary's bounded movement neighborhood.
Tied solutions choose the earliest final boundary, recursively earliest prior
boundaries. Fixed unsupported anchors use equal lower/upper bounds.
    """
    proposals = np.asarray(proposals, dtype=float)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if (proposals.ndim != 1 or lower.shape != proposals.shape or upper.shape != proposals.shape or
            not np.isfinite(proposals).all() or not np.isfinite(lower).all() or
            not np.isfinite(upper).all() or (lower != np.floor(lower)).any() or
            (upper != np.floor(upper)).any() or (lower > upper).any()):
        raise ValueError("Invalid boundary projection proposals or bounds")
    if not len(proposals):
        return np.empty(0, dtype=np.int64)
    grids, backs = [], []
    previous_grid = np.arange(int(lower[0]), int(upper[0]) + 1)
    cost = np.abs(previous_grid - proposals[0])
    grids.append(previous_grid)
    for index in range(1, len(proposals)):
        prefix_cost = np.minimum.accumulate(cost)
        improved = np.r_[True, cost[1:] < prefix_cost[:-1]]
        prefix_arg = np.maximum.accumulate(np.where(improved, np.arange(len(cost)), 0))
        grid = np.arange(int(lower[index]), int(upper[index]) + 1)
        prior = np.searchsorted(previous_grid, grid, side="left") - 1
        feasible = prior >= 0
        back = np.zeros(len(grid), dtype=np.int64)
        back[feasible] = prefix_arg[prior[feasible]]
        current = np.full(len(grid), np.inf)
        current[feasible] = prefix_cost[prior[feasible]] + np.abs(grid[feasible] - proposals[index])
        grids.append(grid)
        backs.append(back)
        previous_grid, cost = grid, current
    if not np.isfinite(cost).any():
        raise ValueError("Boundary constraints have no ordered solution")
    output = np.empty(len(proposals), dtype=np.int64)
    chosen = int(np.argmin(cost))
    for index in range(len(proposals) - 1, -1, -1):
        output[index] = grids[index][chosen]
        if index:
            chosen = int(backs[index - 1][chosen])
    return output


def _retime(status, features, periods, supported_blocks, evidence, model):
    design, names = motion_design(features, periods)
    if hasattr(model, "feature_names") and names != model.feature_names:
        raise ValueError("Timing feature schema differs from the trained model")
    support = motion_support(features) & (evidence["status_quality"] > 0)
    groups, all_frames, all_directions = [], [], []
    for start, end in supported_blocks:
        boundaries = np.flatnonzero(np.diff(status[start:end])) + start + 1
        if len(boundaries):
            directions = status[boundaries].astype(int) - status[boundaries - 1].astype(int)
            groups.append((start, end, boundaries, directions))
            all_frames.extend(boundaries)
            all_directions.extend(directions)
    if not all_frames:
        return status.copy(), []
    frames = np.asarray(all_frames, dtype=np.int64)
    directions = np.asarray(all_directions, dtype=np.int8)
    supported = support[frames]
    offsets = np.zeros(len(frames), dtype=float)
    if supported.any():
        predicted = np.asarray(model.predict_offsets(design[frames[supported]], directions[supported]), dtype=float)
        if predicted.shape != (int(supported.sum()),):
            raise ValueError("Timing model must return one offset per boundary")
        offsets[supported] = predicted
    finite = np.isfinite(offsets)
    movable = supported & finite
    bounded = np.clip(np.where(finite, offsets, 0.), -1.5, 1.5)
    proposals = frames + bounded * features.fps
    result, audit = status.copy(), []
    cursor = 0
    radius = int(np.floor(1.5 * features.fps + 1e-9))
    for start, end, boundaries, direction in groups:
        count = len(boundaries)
        chosen = slice(cursor, cursor + count)
        lower = np.maximum(boundaries - radius, start + 1)
        upper = np.minimum(boundaries + radius, end - 1)
        can_move = movable[chosen]
        lower = np.where(can_move, lower, boundaries)
        upper = np.where(can_move, upper, boundaries)
        final = _project_boundaries(proposals[chosen], lower, upper)
        position = start
        state = int(status[start])
        for original, proposed, actual, sign, index in zip(
                boundaries, proposals[chosen], final, direction, range(cursor, cursor + count)):
            result[position:actual] = state
            position, state = int(actual), 1 - state
            reasons = []
            if not supported[index]:
                reasons.append("unsupported_motion")
            elif not finite[index]:
                reasons.append("invalid_prediction")
            if finite[index] and abs(offsets[index]) > 1.5:
                reasons.append("offset_clipped")
            local = index - cursor
            nearest = int(np.clip(np.ceil(proposed - .5), lower[local], upper[local]))
            if actual != nearest:
                reasons.append("order_projection")
            if proposed < start + 1 or proposed > end - 1:
                reasons.append("support_block_limit")
            if abs(proposed - original) > radius + 1e-9:
                reasons.append("frame_move_limit")
            audit.append({"block_start": start, "block_end": end,
                          "direction": "LD" if sign == 1 else "DL",
                          "original_frame": int(original), "proposed_frame": float(proposed),
                          "final_frame": int(actual), "original_time_s": float(original / features.fps),
                          "proposed_time_s": float(proposed / features.fps),
                          "final_time_s": float(actual / features.fps),
                          "predicted_offset_s": float(offsets[index]) if finite[index] else None,
                          "bounded_offset_s": float(bounded[index]),
                          "player_support": bool(supported[index]),
                          "status_quality": float(evidence["status_quality"][original]),
                          "reasons": reasons})
        result[position:end] = state
        cursor += count
    return result, audit


def _owner_hole(control, switch_cost, freshness, start, end, left, right, fps):
    output = np.full(end - start, "X", dtype="U1")
    for a, b in _runs(freshness[start:end] > 0, start):
        home = np.clip(1 / (1 + np.exp(-3. * control[a:b])), .01, .99)
        emission = -np.column_stack((np.log(home), np.log1p(-home))) / fps
        forward, backward = np.empty_like(emission), np.zeros_like(emission)
        forward[0] = emission[0]
        if a == start and left in ("H", "A"):
            forward[0, 1 if left == "H" else 0] += switch_cost[a]
        forward[0] -= forward[0].min()
        for i in range(1, b - a):
            forward[i] = np.minimum(forward[i - 1], forward[i - 1, ::-1] + switch_cost[a + i]) + emission[i]
            forward[i] -= forward[i].min()
        if b == end and right in ("H", "A"):
            backward[-1, 1 if right == "H" else 0] += switch_cost[b]
        for i in range(b - a - 2, -1, -1):
            future = emission[i + 1] + backward[i + 1]
            backward[i] = np.minimum(future, future[::-1] + switch_cost[a + i + 1])
            backward[i] -= backward[i].min()
        marginal = forward + backward
        owner = np.array(["H", "A"])[np.argmin(marginal, axis=1)]
        owner[np.abs(marginal[:, 0] - marginal[:, 1]) <= 1e-9] = "X"
        output[a - start:b - start] = owner
    return output


def _ownership(status, baseline, evidence, supported_blocks, fps):
    labels = np.full(len(status), "X", dtype="U1")
    labels[status == 1] = "D"
    retained = (status == 0) & (baseline != "D")
    labels[retained] = baseline[retained]
    for block_start, block_end in supported_blocks:
        for start, end in _runs(status[block_start:block_end] == 0, block_start):
            for a, b in _runs(baseline[start:end] == "D", start):
                left = labels[a - 1] if a > start else "X"
                right = labels[b] if b < end else "X"
                labels[a:b] = _owner_hole(evidence["control_evidence"], evidence["team_switch_cost"],
                                           evidence["support_freshness"], a, b, left, right, fps)
    return labels


def decode_precision_periods(features, blocks, pdead, pboundary, timing_model=None,
                             config=PrecisionConfig(), baseline_labels=None, baseline_evidence=None):
    """Return H/A/D/X labels, separate -1/0/1 status, and local evidence/audit.

Frozen v2 is called once per period with its selected status_weight=1. Passing
both baseline_labels and its full-length baseline_evidence avoids that call.
Owner labels are preserved wherever both methods are live; only newly-live
holes are inferred again after final retiming, with same-run retained anchors.
    """
    if not isinstance(config, PrecisionConfig):
        raise TypeError("config must be a PrecisionConfig instance")
    _validate(features)
    n, fps = len(features), float(features.fps)
    periods = _blocks(blocks, n)
    for name, score in (("pdead", pdead), ("pboundary", pboundary)):
        values = np.asarray(score, dtype=float)
        if values.shape != (n,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{name} must be a finite [0,1] vector of length N")
    pdead, pboundary = np.asarray(pdead, dtype=float), np.asarray(pboundary, dtype=float)
    if config.refine and timing_model is None:
        raise ValueError("Directional timing model is required when refinement is enabled")
    if baseline_labels is not None:
        baseline = np.asarray(baseline_labels)
        if baseline.shape != (n,) or not np.isin(baseline, STATES).all():
            raise ValueError("Baseline labels must be a H/A/D/X vector of length N")
        baseline = baseline.copy()
    else:
        baseline = np.full(n, "X", dtype="U1")
    if baseline_evidence is None:
        evidence = {}
        for start, end in periods:
            labels, local = decode_boundary(features.sliced(start, end), pdead[start:end], pboundary[start:end],
                                             BoundaryConfig(status_weight=1.))
            if baseline_labels is None:
                baseline[start:end] = labels
            for key, value in local.items():
                value = np.asarray(value)
                if key not in evidence:
                    shape = (n,) + value.shape[1:]
                    evidence[key] = np.ones(shape, dtype=bool) if key == "unsupported" else np.zeros(shape, dtype=value.dtype)
                evidence[key][start:end] = value
    else:
        if baseline_labels is None:
            raise ValueError("Frozen evidence reuse also requires frozen baseline labels")
        evidence = dict(baseline_evidence)
    required = ("fused_dead_score", "live_dead_switch_cost", "status_quality", "unsupported",
                "control_evidence", "team_switch_cost", "support_freshness", "dead_evidence")
    if not periods and baseline_evidence is None:
        evidence = {key: np.zeros(n) for key in required}
    for key in required:
        if key not in evidence or np.asarray(evidence[key]).shape != (n,):
            raise ValueError(f"Frozen v2 evidence requires a length-N {key} vector")
        evidence[key] = np.asarray(evidence[key])
        if not np.isfinite(evidence[key]).all():
            raise ValueError("Frozen v2 evidence must be finite")
    status = np.full(n, -1, dtype=np.int8)
    supported_blocks = []
    for start, end in periods:
        for a, b in _runs(~evidence["unsupported"][start:end].astype(bool), start):
            supported_blocks.append((a, b))
            dead = np.clip(evidence["fused_dead_score"][a:b], .01, .99)
            unary = -np.column_stack((np.log1p(-dead), np.log(dead)))
            status[a:b] = _duration_path(unary, evidence["live_dead_switch_cost"][a:b], fps, config.short_run_cost)
    evidence["baseline_labels"] = baseline
    evidence["status_before_refine"] = status.copy()
    evidence["supported_blocks"] = supported_blocks
    evidence["retiming_audit"] = []
    if config.refine:
        status, evidence["retiming_audit"] = _retime(status, features, periods, supported_blocks,
                                                     evidence, timing_model)
    labels = _ownership(status, baseline, evidence, supported_blocks, fps)
    player_dead = .5 + evidence["status_quality"] * (pdead - .5)
    geometry_led = .95 * evidence["dead_evidence"] > player_dead
    evidence["geometry_led_boundary_frames"] = np.array(
        [int(frame) for a, b in supported_blocks
         for frame in np.flatnonzero(np.diff(status[a:b])) + a + 1 if geometry_led[frame]], dtype=np.int64)
    return labels, status, evidence
