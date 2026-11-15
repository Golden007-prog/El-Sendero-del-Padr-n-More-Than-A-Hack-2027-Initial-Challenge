"""Externally fitted turnover evidence and variable-cost H/A decoding.

The runner supplies the sealed parent owner_model module on its import path.
Only reference-target construction accepts labels; feature windows never do.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from owner_model import (
    EVIDENCE_KEYS, FEATURE_NAMES, MODEL_SETTINGS, THREAD_LIMIT, TRAIN_IDS,
    OwnershipModel, _periods, _positive_fps, feature_design, predict_owner,
    swap_design, swap_predictors,
)


VARIANTS = ("adaptive_full", "adaptive_no_memory")
MEMORY_CHANNELS = ("control_evidence", "support_age_s", "support_freshness")
TARGET_RADIUS_SECONDS = 0.2
BASE_SWITCH_COST = 0.6
BOUNDARY_ATTENUATION = 0.8
MIN_SWITCH_COST = 0.12


def _memory_column(name):
    base = name.removesuffix("_missing")
    return any(base == channel or base.endswith("_" + channel)
               for channel in MEMORY_CHANNELS)


MEMORY_COLUMNS = tuple(i for i, name in enumerate(FEATURE_NAMES) if _memory_column(name))
NO_MEMORY_COLUMNS = tuple(i for i in range(len(FEATURE_NAMES)) if i not in MEMORY_COLUMNS)


def _feature_indices(use_memory):
    if not isinstance(use_memory, bool):
        raise ValueError("use_memory must be an explicit boolean")
    return tuple(range(len(FEATURE_NAMES))) if use_memory else NO_MEMORY_COLUMNS


def _labels(labels, n=None):
    values = np.asarray(labels)
    if (values.ndim != 1 or (n is not None and len(values) != n)
            or not np.isin(values, list("HADX")).all()):
        raise ValueError("labels must be a length-N H/A/D/X vector")
    return values


def _live_spans(labels, periods):
    for start, end in periods:
        live = np.isin(labels[start:end], ["H", "A"])
        edges = np.flatnonzero(np.diff(np.r_[False, live, False]))
        for left, right in edges.reshape(-1, 2):
            yield start + int(left), start + int(right)


def boundary_targets(labels, blocks, fps):
    """Return -1 excluded, 0 nonboundary, 1 within 0.2 s of a direct takeover.

    A takeover at t means a reference H/A change from t-1 to t. Both endpoints
    must lie in one period and one uninterrupted reference H/A span.
    """
    fps = _positive_fps(fps)
    labels = _labels(labels)
    periods = _periods(blocks, len(labels))
    target = np.full(len(labels), -1, dtype=np.int8)
    radius = int(np.floor(TARGET_RADIUS_SECONDS * fps))
    for start, end in _live_spans(labels, periods):
        target[start:end] = 0
        changes = start + 1 + np.flatnonzero(labels[start + 1:end] != labels[start:end - 1])
        for frame in changes:
            left, right = max(start, int(frame) - radius), min(end, int(frame) + radius + 1)
            candidate = np.arange(left, right)
            within = np.abs(candidate - frame) / fps <= TARGET_RADIUS_SECONDS
            target[candidate[within]] = 1
    return target


def _training_items(training_matches):
    try:
        items = list(training_matches)
    except TypeError as error:
        raise ValueError("training_matches must contain three or four five-field tuples") from error
    if len(items) not in (3, 4):
        raise ValueError("fit must use exactly three original training matches or all four")
    if any(not isinstance(item, (tuple, list)) or len(item) != 5 for item in items):
        raise ValueError("training items must be (id, features, labels, blocks, evidence)")
    ids = [item[0] for item in items]
    if (any(not isinstance(mid, str) or mid not in TRAIN_IDS for mid in ids)
            or len(set(ids)) != len(ids)
            or ids != [mid for mid in TRAIN_IDS if mid in ids]):
        raise ValueError("fit identities must be a unique canonical subset of the original four")
    return items, ids


def _sample_indices(features, blocks, n):
    periods = _periods(blocks, n)
    if not periods:
        raise ValueError("training match must have at least one period")
    stride = max(1, round(_positive_fps(features.fps) / 5))
    indices = np.concatenate([np.arange(start, end, stride) for start, end in periods])
    return indices, stride, periods


def _balanced_weight(target):
    classes, counts = np.unique(target, return_counts=True)
    weight = np.empty(len(target), dtype=float)
    for value, count in zip(classes, counts):
        weight[target == value] = 1.0 / (len(classes) * count)
    return weight, classes


def _fit(training_matches, owner, use_memory=True):
    items, ids = _training_items(training_matches)
    columns = _feature_indices(use_memory)
    if owner and not use_memory:
        raise ValueError("the ownership head always uses the complete frozen schema")
    names = tuple(FEATURE_NAMES[i] for i in columns)
    xs, ys, ws, counts = [], [], [], {}
    for mid, features, labels, blocks, evidence in items:
        # Only predictor fields and period boundaries enter feature_design.
        design, design_names = feature_design(features, evidence, blocks)
        if design_names != FEATURE_NAMES:
            raise ValueError("parent owner feature schema drifted")
        labels = _labels(labels, len(design))
        sampled, stride, periods = _sample_indices(features, blocks, len(design))
        if owner:
            indices = sampled[np.isin(labels[sampled], ["H", "A"])]
            target = (labels[indices] == "H").astype(np.int8)
        else:
            targets = boundary_targets(labels, periods, features.fps)
            indices = sampled[targets[sampled] >= 0]
            target = targets[indices]
        if not len(indices):
            raise ValueError("training match has no sampled known H/A targets")
        weight, classes = _balanced_weight(target)
        observed = design[indices]
        xs.extend((observed[:, columns], swap_design(observed)[:, columns]))
        ys.extend((target, 1 - target if owner else target.copy()))
        ws.extend((weight / 2, weight / 2))
        class_names = {0: "A", 1: "H"} if owner else {0: "nonboundary", 1: "boundary"}
        count = {
            "sample_stride_frames": int(stride),
            "sampled_original_rows": int(len(target)), "augmented_rows": int(2 * len(target)),
            "present_original_classes": int(len(classes)),
            "match_weight_before_global_normalization": float(weight.sum()),
            "original_class_weight_before_augmentation": {
                class_names[int(value)]: float(weight[target == value].sum()) for value in classes
            },
        }
        for value, name in class_names.items():
            count["sampled_" + name] = int((target == value).sum())
        counts[mid] = count
    x, y, weights = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("the fitted classifier requires both binary classes")
    global_scale = len(weights) / weights.sum()
    weights *= global_scale
    classifier = HistGradientBoostingClassifier(**MODEL_SETTINGS)
    with threadpool_limits(limits=THREAD_LIMIT):
        classifier.fit(x, y, sample_weight=weights)
    settings = {
        **MODEL_SETTINGS, "thread_limit": THREAD_LIMIT, "sample_target_hz": 5,
        "augmentation": ("exact team swap and opposite target; half weight per copy" if owner
                         else "exact team swap and same boundary target; half weight per copy"),
        "global_weight_scale": float(global_scale),
        "total_augmented_rows": int(len(y)), "feature_count": int(len(names)),
        "probabilities_calibrated": False,
    }
    if owner:
        return OwnershipModel(classifier, names, ids, counts, settings)
    settings.update({
        "variant": VARIANTS[0 if use_memory else 1],
        "target_radius_seconds": TARGET_RADIUS_SECONDS,
        "edge_alignment": "score[t] prices the incoming edge t-1 to t",
        "dropped_feature_names": [FEATURE_NAMES[i] for i in MEMORY_COLUMNS] if not use_memory else [],
    })
    return BoundaryModel(classifier, names, ids, counts, settings, use_memory)


def fit_fold_owner(training_matches):
    """Fit the identical frozen owner recipe on canonical three/four train IDs."""
    return _fit(training_matches, owner=True)


def fit_boundary(training_matches, use_memory):
    """Fit a symmetric takeover-evidence head; three/four original train IDs only."""
    return _fit(training_matches, owner=False, use_memory=use_memory)


@dataclass
class BoundaryModel:
    classifier: object
    feature_names: tuple[str, ...]
    train_ids: list[str]
    training_counts: dict
    settings: dict
    use_memory: bool

    def predict_proba(self, features, evidence, blocks):
        columns = _feature_indices(self.use_memory)
        expected = tuple(FEATURE_NAMES[i] for i in columns)
        if tuple(self.feature_names) != expected:
            raise ValueError("boundary model feature schema differs from the frozen variant")
        design, names = feature_design(features, evidence, blocks)
        if names != FEATURE_NAMES:
            raise ValueError("parent owner feature schema drifted")
        if not len(design):
            return np.empty(0, dtype=float)
        classes = list(self.classifier.classes_)
        if set(classes) != {0, 1}:
            raise ValueError("boundary classifier must contain both binary classes")
        with threadpool_limits(limits=THREAD_LIMIT):
            original = self.classifier.predict_proba(design[:, columns])[:, classes.index(1)]
            mirrored = self.classifier.predict_proba(swap_design(design)[:, columns])[:, classes.index(1)]
        if (original.shape != (len(design),) or mirrored.shape != (len(design),)
                or not np.isfinite(original).all() or not np.isfinite(mirrored).all()
                or ((original < 0) | (original > 1)).any()
                or ((mirrored < 0) | (mirrored > 1)).any()):
            raise ValueError("boundary classifier returned invalid decision evidence")
        return 0.5 * (original + mirrored)


def predict_boundary(model, features, evidence, blocks):
    if not isinstance(model, BoundaryModel):
        raise TypeError("model must be a BoundaryModel")
    return model.predict_proba(features, evidence, blocks)


def transition_costs(boundary_scores):
    """One bounded cost per frame, interpreted on its incoming temporal edge."""
    scores = np.asarray(boundary_scores, dtype=float)
    if (scores.ndim != 1 or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("boundary_scores must be finite length-N values in [0, 1]")
    return np.clip(BASE_SWITCH_COST * (1 - BOUNDARY_ATTENUATION * scores),
                   MIN_SWITCH_COST, BASE_SWITCH_COST)


def cost_audit(anchor, boundary_scores, blocks):
    """Aggregate only costs on eligible within-span edges; reset edges are unused."""
    anchor = _labels(anchor)
    costs = transition_costs(boundary_scores)
    if costs.shape != anchor.shape:
        raise ValueError("boundary_scores and anchor must have the same length")
    periods = _periods(blocks, len(anchor))
    used = np.zeros(len(anchor), dtype=bool)
    for start, end in _live_spans(anchor, periods):
        used[start + 1:end] = True
    selected = costs[used]
    return {
        "base_switch_cost": BASE_SWITCH_COST,
        "boundary_attenuation": BOUNDARY_ATTENUATION,
        "allowed_minimum_cost": MIN_SWITCH_COST,
        "eligible_transition_edges": int(len(selected)),
        "minimum_edge_cost": float(selected.min()) if len(selected) else None,
        "maximum_edge_cost": float(selected.max()) if len(selected) else None,
        "mean_edge_cost": float(selected.mean()) if len(selected) else None,
        "edge_alignment": "score[t] prices t-1 to t; first live-span score is unused",
    }


def _variable_path(probabilities, anchor, fps, costs):
    unary = -np.log(probabilities) / fps
    n = len(unary)
    anchor_code = (anchor == "A").astype(np.int8)
    previous = unary[0].copy()
    previous -= previous.min()
    disagreements = np.array([int(anchor_code[0] != state) for state in (0, 1)])
    back = np.zeros((n, 2), dtype=np.int8)
    ties = 0

    def choose(options, errors, preferred):
        nonlocal ties
        if options[0] != options[1]:
            return int(options[1] < options[0])
        ties += 1
        if errors[0] != errors[1]:
            return int(errors[1] < errors[0])
        return int(preferred)

    for frame in range(1, n):
        current = np.empty(2)
        current_errors = np.empty(2, dtype=np.int64)
        for state in (0, 1):
            options = previous + np.array([costs[frame] if state != prior else 0 for prior in (0, 1)])
            prior = choose(options, disagreements, anchor_code[frame - 1])
            back[frame, state] = prior
            current[state] = options[prior] + unary[frame, state]
            current_errors[state] = disagreements[prior] + int(state != anchor_code[frame])
        current -= current.min()
        previous, disagreements = current, current_errors
    state = choose(previous, disagreements, anchor_code[-1])
    path = np.empty(n, dtype=np.int8)
    for frame in range(n - 1, -1, -1):
        path[frame] = state
        state = int(back[frame, state])
    return np.array(["H", "A"])[path], ties


def decode_adaptive(anchor, owner_probabilities, boundary_scores, blocks, fps):
    """Minimize owner log-cost/second plus variable cost on each H/A switch."""
    fps = _positive_fps(fps)
    anchor = _labels(anchor)
    n = len(anchor)
    probabilities = np.asarray(owner_probabilities, dtype=float)
    if (probabilities.shape != (n, 2) or not np.isfinite(probabilities).all()
            or ((probabilities <= 0) | (probabilities > 1)).any()
            or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("owner_probabilities must be positive normalized N-by-2 H/A scores")
    costs = transition_costs(boundary_scores)
    if costs.shape != (n,):
        raise ValueError("boundary_scores must have the same length as anchor")
    periods = _periods(blocks, n)
    output = anchor.astype("U1", copy=True)
    processed = np.zeros(n, dtype=bool)
    for start, end in periods:
        processed[start:end] = True
    spans, ties = 0, 0
    for start, end in _live_spans(anchor, periods):
        output[start:end], local_ties = _variable_path(
            probabilities[start:end], anchor[start:end], fps, costs[start:end])
        spans += 1
        ties += local_ties
    audit = {
        "live_spans_decoded": spans, "tie_break_decisions": ties,
        "changed_live_frames": int((output != anchor).sum()),
        "home_to_away_frames": int(((anchor == "H") & (output == "A")).sum()),
        "away_to_home_frames": int(((anchor == "A") & (output == "H")).sum()),
        "dead_mask_unchanged": bool(np.array_equal(output == "D", anchor == "D")),
        "unknown_mask_unchanged": bool(np.array_equal(output == "X", anchor == "X")),
        "outside_periods_unchanged": bool(np.array_equal(output[~processed], anchor[~processed])),
        "probabilities_calibrated": False,
        "cost_audit": cost_audit(anchor, boundary_scores, periods),
    }
    if not all(audit[key] for key in ("dead_mask_unchanged", "unknown_mask_unchanged", "outside_periods_unchanged")):
        raise AssertionError("adaptive ownership decoding changed protected support")
    return output, audit
