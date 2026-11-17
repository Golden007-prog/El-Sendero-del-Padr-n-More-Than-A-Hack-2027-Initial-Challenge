"""Externally supervised, symmetry-preserving H/A ownership with frozen D/X."""
from collections.abc import Mapping
import copy
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


TRAIN_IDS = ("J03WOH", "J03WMX", "J03WN1", "J03WOY")
SWITCH_COSTS = (0.6, 1.2, 2.4)
WINDOW_SECONDS = (0.5, 2.0)
THREAD_LIMIT = 4
EVIDENCE_KEYS = (
    "direct_support", "control_advantage", "ball_quality", "control_evidence",
    "support_age_s", "support_freshness", "ball_speed_mps", "ball_speed_observed",
    "proximity_confidence", "both_teams_visible",
)
MOTION_NAMES = (
    "speed_q10_mps", "speed_q50_mps", "speed_q90_mps", "speed_mean_mps",
    "speed_std_mps", "stationary_fraction", "running_fraction",
    "visible_location_count", "team_location_count_min", "team_location_count_max",
    "known_speed_fraction", "team_x_mad_m", "team_y_mad_m",
    "team_median_separation_m", "x_range80_fraction", "y_range80_fraction",
    "touchline_distance_q50_fraction", "duplicate_fraction",
)
RAW_CHANNEL_NAMES = (
    "nearest_home_m", "nearest_away_m", "home_visible_count", "away_visible_count",
    "median_player_speed_mps", *MOTION_NAMES, *EVIDENCE_KEYS,
)
MODEL_SETTINGS = {
    "max_iter": 800, "learning_rate": 0.07, "max_leaf_nodes": 31,
    "min_samples_leaf": 60, "l2_regularization": 2.0,
    "random_state": 2027, "early_stopping": False,
}


def _positive_fps(fps):
    try:
        valid = (np.isscalar(fps) and not isinstance(fps, (bool, np.bool_))
                 and np.isfinite(fps) and fps > 0)
    except TypeError:
        valid = False
    if not valid:
        raise ValueError("fps must be finite and positive")
    return float(fps)


def _periods(blocks, n):
    periods, previous = [], 0
    for block in blocks:
        if not isinstance(block, (tuple, list, np.ndarray)) or len(block) != 2:
            raise ValueError("periods must be ordered half-open integer pairs")
        start, end = block
        if any(not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_))
               for x in (start, end)):
            raise ValueError("period endpoints must be integers")
        if not previous <= start < end <= n:
            raise ValueError("periods must be disjoint, ordered and in bounds")
        periods.append((int(start), int(end)))
        previous = int(end)
    return periods


def _raw_channels(features, evidence):
    fps = _positive_fps(features.fps)
    nearest = np.asarray(features.nearest_m, dtype=float)
    if nearest.ndim != 2 or nearest.shape[1] != 2 or (nearest < 0).any():
        raise ValueError("nearest_m must be a nonnegative N-by-2 distance array")
    n = len(nearest)
    counts = np.asarray(features.player_count, dtype=float)
    speed = np.asarray(features.player_speed_mps, dtype=float)
    summary = np.asarray(features.player_summary, dtype=float)
    if (counts.shape != (n, 2) or not np.isfinite(counts).all()
            or (counts < 0).any() or (counts != np.floor(counts)).any()):
        raise ValueError("player_count must contain nonnegative N-by-2 integers")
    if speed.shape != (n,) or np.isinf(speed).any() or (speed < 0).any():
        raise ValueError("player speed must be length N, nonnegative or missing")
    if summary.shape != (n, len(MOTION_NAMES)) or np.isinf(summary).any():
        raise ValueError("player_summary must use the frozen N-by-18 pooled schema")
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must map frozen predictor names to vectors")
    values = []
    for key in EVIDENCE_KEYS:
        if key not in evidence:
            raise ValueError(f"missing predictor vector: {key}")
        value = np.asarray(evidence[key], dtype=float)
        if value.shape != (n,) or not np.isfinite(value).all():
            raise ValueError(f"{key} must be a finite length-N predictor")
        lower = -1 if key in ("control_advantage", "control_evidence") else 0
        if (value < lower).any():
            raise ValueError(f"{key} is outside its frozen predictor range")
        if key not in ("support_age_s", "ball_speed_mps") and (value > 1).any():
            raise ValueError(f"{key} is outside its frozen predictor range")
        if key in ("both_teams_visible", "ball_speed_observed") and not np.isin(value, [0, 1]).all():
            raise ValueError(f"{key} must contain zero-one values")
        values.append(value)
    nearest = np.where(np.isfinite(nearest), nearest, np.nan)
    return np.column_stack((nearest, counts, speed, summary, *values)), fps


def _schema():
    width = len(RAW_CHANNEL_NAMES)
    raw_index = np.arange(width)
    raw_index[:4] = [1, 0, 3, 2]
    raw_sign = np.ones(width, dtype=np.float32)
    for name in ("control_advantage", "control_evidence"):
        raw_sign[RAW_CHANNEL_NAMES.index(name)] = -1
    prefixes = [""] + [prefix for seconds in WINDOW_SECONDS
                        for prefix in (f"mean_{seconds:g}s_", f"contrast_{seconds:g}s_")]
    names, indices, signs = [], [], []
    for prefix in prefixes:
        offset = len(names)
        names.extend(prefix + name for name in RAW_CHANNEL_NAMES)
        names.extend(prefix + name + "_missing" for name in RAW_CHANNEL_NAMES)
        indices.extend(offset + raw_index)
        indices.extend(offset + width + raw_index)
        signs.extend(raw_sign)
        signs.extend(np.ones(width, dtype=np.float32))
    return tuple(names), np.array(indices, dtype=np.int64), np.array(signs, dtype=np.float32)


FEATURE_NAMES, SWAP_INDEX, SWAP_SIGN = _schema()


def _window_mean(sums, support, left, right):
    count = support[right] - support[left]
    total = sums[right] - sums[left]
    result = np.full_like(total, np.nan)
    np.divide(total, count, out=result, where=count > 0)
    return result


def feature_design(features, evidence, blocks):
    """Build predictor-only features; no target/status mask is accepted."""
    base, fps = _raw_channels(features, evidence)
    n, width = base.shape
    periods = _periods(blocks, n)
    design = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    column = 0

    def append(values):
        nonlocal column
        known = np.isfinite(values)
        limit = np.finfo(np.float32).max
        design[:, column:column + width] = np.clip(np.where(known, values, 0), -limit, limit)
        design[:, column + width:column + 2 * width] = ~known
        column += 2 * width

    append(base)
    for seconds in WINDOW_SECONDS:
        mean, contrast = np.full_like(base, np.nan), np.full_like(base, np.nan)
        radius = max(1, round(seconds * fps / 2))
        for start, end in periods:
            values = base[start:end]
            known = np.isfinite(values)
            sums = np.vstack((np.zeros(width), np.cumsum(np.where(known, values, 0), axis=0)))
            support = np.vstack((np.zeros(width, dtype=np.int64), np.cumsum(known, axis=0)))
            current = np.arange(end - start)
            left, right = np.maximum(0, current - radius), np.minimum(end - start, current + radius + 1)
            mean[start:end] = _window_mean(sums, support, left, right)
            contrast[start:end] = (_window_mean(sums, support, current + 1, right)
                                   - _window_mean(sums, support, left, current))
        append(mean)
        append(contrast)
    return design, FEATURE_NAMES


def swap_design(design):
    design = np.asarray(design)
    if design.ndim != 2 or design.shape[1] != len(FEATURE_NAMES) or not np.isfinite(design).all():
        raise ValueError("design must use the finite frozen feature schema")
    return design[:, SWAP_INDEX] * SWAP_SIGN


def swap_predictors(features, evidence):
    """Swap only predictor team axes; return required evidence keys, never targets."""
    _raw_channels(features, evidence)
    swapped = copy.copy(features)
    swapped.nearest_m = np.asarray(features.nearest_m)[:, ::-1].copy()
    swapped.player_count = np.asarray(features.player_count)[:, ::-1].copy()
    changed = {key: np.asarray(evidence[key]).copy() for key in EVIDENCE_KEYS}
    for key in ("control_advantage", "control_evidence"):
        changed[key] = -changed[key]
    return swapped, changed


@dataclass
class OwnershipModel:
    classifier: object
    feature_names: tuple[str, ...]
    train_ids: list[str]
    training_counts: dict
    settings: dict

    def predict_proba(self, features, evidence, blocks):
        design, names = feature_design(features, evidence, blocks)
        if tuple(self.feature_names) != names:
            raise ValueError("owner feature schema differs from the fitted model")
        if not len(design):
            return np.empty((0, 2), dtype=float)
        classes = list(self.classifier.classes_)
        if set(classes) != {0, 1}:
            raise ValueError("owner model must contain both binary owner classes")
        with threadpool_limits(limits=THREAD_LIMIT):
            home = self.classifier.predict_proba(design)[:, classes.index(1)]
            swapped_home = self.classifier.predict_proba(swap_design(design))[:, classes.index(1)]
        if (not np.isfinite(home).all() or not np.isfinite(swapped_home).all()
                or ((home < 0) | (home > 1)).any()
                or ((swapped_home < 0) | (swapped_home > 1)).any()):
            raise ValueError("owner classifier returned invalid scores")
        signed = np.clip(0.5 * (home - swapped_home), -0.49, 0.49)
        return np.column_stack((0.5 + signed, 0.5 - signed))


def fit_owner(training_matches):
    """Each item is (match_id, features, H/A/D/X targets, periods, evidence)."""
    xs, ys, ws, ids, counts = [], [], [], [], {}
    for position, match in enumerate(training_matches):
        if not isinstance(match, (tuple, list)) or len(match) != 5:
            raise ValueError("training items must be five-field tuples")
        match_id, features, labels, blocks, evidence = match
        if position >= len(TRAIN_IDS) or match_id != TRAIN_IDS[position]:
            raise ValueError("training must use exactly the original four matches in fixed order")
        design, names = feature_design(features, evidence, blocks)
        labels = np.asarray(labels)
        if labels.shape != (len(design),) or not np.isin(labels, list("HADX")).all():
            raise ValueError("external targets must be a length-N H/A/D/X vector")
        periods = _periods(blocks, len(design))
        if not periods:
            raise ValueError("training match must have at least one period")
        stride = max(1, round(_positive_fps(features.fps) / 5))
        indices = np.concatenate([np.arange(start, end, stride) for start, end in periods])
        indices = indices[np.isin(labels[indices], ["H", "A"])]
        if not len(indices):
            raise ValueError("training match has no sampled H/A targets")
        target = (labels[indices] == "H").astype(np.int8)
        classes, class_counts = np.unique(target, return_counts=True)
        weight = np.empty(len(target), dtype=float)
        for value, count in zip(classes, class_counts):
            weight[target == value] = 1.0 / (len(classes) * count)
        observed = design[indices]
        xs.extend((observed, swap_design(observed)))
        ys.extend((target, 1 - target))
        ws.extend((weight / 2, weight / 2))
        ids.append(match_id)
        counts[match_id] = {
            "sample_stride_frames": stride,
            "sampled_H": int((target == 1).sum()), "sampled_A": int((target == 0).sum()),
            "sampled_original_rows": len(target), "augmented_rows": 2 * len(target),
            "present_original_classes": len(classes),
            "match_weight_before_global_normalization": float(weight.sum()),
            "original_class_weight_before_augmentation": {
                "H" if value == 1 else "A": float(weight[target == value].sum()) for value in classes},
        }
    if tuple(ids) != TRAIN_IDS:
        raise ValueError("training must contain all four fixed match identities")
    x, y, weights = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    global_scale = len(weights) / weights.sum()
    weights *= global_scale
    classifier = HistGradientBoostingClassifier(**MODEL_SETTINGS)
    with threadpool_limits(limits=THREAD_LIMIT):
        classifier.fit(x, y, sample_weight=weights)
    settings = {
        **MODEL_SETTINGS, "thread_limit": THREAD_LIMIT, "sample_target_hz": 5,
        "augmentation": "exact team swap and opposite target; half weight per copy",
        "global_weight_scale": float(global_scale),
        "total_augmented_rows": len(y), "feature_count": len(names),
        "probabilities_calibrated": False,
    }
    return OwnershipModel(classifier, names, ids, counts, settings)


def predict_owner(model, features, evidence, blocks):
    if not isinstance(model, OwnershipModel):
        raise TypeError("model must be an OwnershipModel")
    return model.predict_proba(features, evidence, blocks)


def _binary_path(probabilities, anchor, fps, cost):
    unary = -np.log(probabilities) / fps
    n = len(unary)
    anchor_code = (anchor == "A").astype(np.int8)
    previous = unary[0].copy()
    previous -= previous.min()
    disagreements = np.array([int(anchor_code[0] != state) for state in (0, 1)])
    back = np.zeros((n, 2), dtype=np.int8)
    ties = 0

    def choose(costs, errors, preferred):
        nonlocal ties
        if costs[0] != costs[1]:
            return int(costs[1] < costs[0])
        ties += 1
        if errors[0] != errors[1]:
            return int(errors[1] < errors[0])
        return int(preferred)

    for frame in range(1, n):
        current = np.empty(2)
        current_errors = np.empty(2, dtype=np.int64)
        for state in (0, 1):
            alternatives = previous + np.array([cost if state != predecessor else 0 for predecessor in (0, 1)])
            predecessor = choose(alternatives, disagreements, anchor_code[frame - 1])
            back[frame, state] = predecessor
            current[state] = alternatives[predecessor] + unary[frame, state]
            current_errors[state] = disagreements[predecessor] + int(state != anchor_code[frame])
        current -= current.min()
        previous, disagreements = current, current_errors
    state = choose(previous, disagreements, anchor_code[-1])
    path = np.empty(n, dtype=np.int8)
    for frame in range(n - 1, -1, -1):
        path[frame] = state
        state = int(back[frame, state])
    return np.array(["H", "A"])[path], ties


def decode_owner(anchor_v4, probabilities, blocks, fps, switch_cost):
    """Decode each frozen live span; path-consistent ties prefer anchor agreement."""
    fps = _positive_fps(fps)
    try:
        valid = (np.isscalar(switch_cost) and not isinstance(switch_cost, (bool, np.bool_))
                 and np.isfinite(switch_cost) and switch_cost in SWITCH_COSTS)
    except TypeError:
        valid = False
    if not valid:
        raise ValueError("switch_cost must be one of the fixed values 0.6, 1.2, 2.4")
    anchor = np.asarray(anchor_v4)
    if anchor.ndim != 1 or not np.isin(anchor, list("HADX")).all():
        raise ValueError("anchor_v4 must be a one-dimensional H/A/D/X vector")
    n = len(anchor)
    probs = np.asarray(probabilities, dtype=float)
    if (probs.shape != (n, 2) or not np.isfinite(probs).all()
            or ((probs <= 0) | (probs > 1)).any()
            or not np.allclose(probs.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("probabilities must be positive normalized N-by-2 H/A scores")
    periods = _periods(blocks, n)
    output = anchor.astype("U1", copy=True)
    processed = np.zeros(n, dtype=bool)
    spans, tied = 0, 0
    for start, end in periods:
        processed[start:end] = True
        live = np.isin(anchor[start:end], ["H", "A"])
        edges = np.flatnonzero(np.diff(np.r_[False, live, False]))
        for left, right in edges.reshape(-1, 2):
            a, b = start + int(left), start + int(right)
            output[a:b], local_ties = _binary_path(probs[a:b], anchor[a:b], fps, float(switch_cost))
            spans += 1
            tied += local_ties
    audit = {
        "switch_cost": float(switch_cost), "live_spans_decoded": spans,
        "tie_break_decisions": tied, "changed_live_frames": int((output != anchor).sum()),
        "home_to_away_frames": int(((anchor == "H") & (output == "A")).sum()),
        "away_to_home_frames": int(((anchor == "A") & (output == "H")).sum()),
        "dead_mask_unchanged": bool(np.array_equal(output == "D", anchor == "D")),
        "unknown_mask_unchanged": bool(np.array_equal(output == "X", anchor == "X")),
        "outside_periods_unchanged": bool(np.array_equal(output[~processed], anchor[~processed])),
        "probabilities_calibrated": False,
    }
    if not all(audit[key] for key in ("dead_mask_unchanged", "unknown_mask_unchanged", "outside_periods_unchanged")):
        raise AssertionError("owner decoding changed protected support")
    return output, audit
