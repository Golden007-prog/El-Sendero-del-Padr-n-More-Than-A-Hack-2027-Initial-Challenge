"""Player-motion-only signed timing targets and fixed directional regressors.

Training callers must supply only the original four external training matches.
The fixed anchor grid is -3..3 seconds in 0.2-second steps; predictions are
offsets in seconds, with +1 denoting L->D and -1 denoting D->L. Absolute frame
indices and match identities are used for target construction/metadata only.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .motion import PLAYER_SUMMARY_NAMES, motion_design
from .status_model import status_targets


def _blocks(blocks, n):
    result = []
    previous = 0
    for block in blocks:
        if len(block) != 2:
            raise ValueError("Blocks must be ordered half-open (start, end) pairs")
        start, end = block
        if (not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)) or
                start < previous or end <= start or end > n):
            raise ValueError("Blocks must be ordered, disjoint and within the feature rows")
        result.append((int(start), int(end)))
        previous = end
    return result


def _runs(mask, start=0):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(a + start), int(b + start)) for a, b in edges.reshape(-1, 2)]


def motion_support(features):
    """Current motion gate: both teams visible and known median player speed.

    This is the positive v2 player-status quality gate, without ball information.
    Historical/context observations cannot make a missing current frame movable.
    """
    counts = np.asarray(features.player_count, dtype=float)
    if features.player_summary is None:
        speed = np.asarray(features.player_speed_mps, dtype=float)
    else:
        summary = np.asarray(features.player_summary, dtype=float)
        if summary.shape != (len(features), len(PLAYER_SUMMARY_NAMES)):
            raise ValueError("Player summary shape differs from the motion schema")
        speed = summary[:, PLAYER_SUMMARY_NAMES.index("speed_q50_mps")]
    if counts.shape != (len(features), 2) or speed.shape != (len(features),):
        raise ValueError("Player motion arrays have inconsistent shapes")
    return np.isfinite(counts).all(axis=1) & (counts > 0).all(axis=1) & np.isfinite(speed) & (speed >= 0)


def timing_anchors(status, blocks, fps, support=None):
    """Return frames, assigned boundary frames, directions and per-head weights.

    Each snapped anchor appears once per direction and known-status block. A
    nearest same-direction boundary owns it, with the earlier boundary winning a
    tie. Each retained boundary receives equal total weight within its match/head.
    Unknown status separates target blocks and never creates a boundary.
    """
    status = np.asarray(status)
    if status.ndim != 1 or not np.isin(status, [-1, 0, 1]).all():
        raise ValueError("Status must be a vector with -1 unknown, 0 live, 1 dead")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    periods = _blocks(blocks, len(status))
    support = np.ones(len(status), dtype=bool) if support is None else np.asarray(support, dtype=bool)
    if support.shape != status.shape:
        raise ValueError("Motion support must have the same shape as status")
    grid = np.rint(np.arange(-15, 16, dtype=float) * fps / 5).astype(np.int64)
    xs, bs, ds = [], [], []
    for period_start, period_end in periods:
        for start, end in _runs(status[period_start:period_end] >= 0, period_start):
            changes = np.flatnonzero(np.diff(status[start:end])) + start + 1
            directions = status[changes].astype(int) - status[changes - 1].astype(int)
            for direction in (1, -1):
                boundaries = changes[directions == direction]
                if not len(boundaries):
                    continue
                anchors = np.unique((boundaries[:, None] + grid).ravel())
                anchors = anchors[(anchors >= start) & (anchors < end)]
                anchors = anchors[support[anchors]]
                if not len(anchors):
                    continue
                right = np.searchsorted(boundaries, anchors)
                before = boundaries[np.maximum(right - 1, 0)]
                after = boundaries[np.minimum(right, len(boundaries) - 1)]
                assigned = np.where(np.abs(anchors - before) <= np.abs(after - anchors), before, after)
                xs.append(anchors)
                bs.append(assigned)
                ds.append(np.full(len(anchors), direction, dtype=np.int8))
    if not xs:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int8), np.empty(0))
    frames, boundaries, directions = np.concatenate(xs), np.concatenate(bs), np.concatenate(ds)
    weights = np.empty(len(frames), dtype=float)
    for direction in (1, -1):
        selected = directions == direction
        if selected.any():
            _, inverse, counts = np.unique(boundaries[selected], return_inverse=True, return_counts=True)
            weights[selected] = 1 / (len(counts) * counts[inverse])
    return frames, boundaries, directions, weights


def _directions(values):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("Directions must be a vector of +1/-1 or LD/DL")
    if values.dtype.kind in "USO":
        values = np.where(values == "LD", 1, np.where(values == "DL", -1, 0))
    if not np.isin(values, [-1, 1]).all():
        raise ValueError("Directions must be +1/-1 or LD/DL")
    return values.astype(np.int8)


@dataclass
class TimingModel:
    onset: object
    offset: object
    feature_names: list[str]
    train_ids: list[str]
    training_offset_medians: list[float] = field(default_factory=list)
    anchor_counts: list[int] = field(default_factory=list)
    boundary_counts: list[int] = field(default_factory=list)

    def predict_offsets(self, design_rows, directions):
        """Predict signed seconds from N motion rows and +1/-1 (or LD/DL)."""
        design = np.asarray(design_rows, dtype=np.float32)
        direction = _directions(directions)
        if (design.ndim != 2 or design.shape != (len(direction), len(self.feature_names)) or
                not np.isfinite(design).all()):
            raise ValueError("Timing design differs from the trained finite motion schema")
        output = np.empty(len(design), dtype=float)
        with threadpool_limits(limits=4):
            for sign, model in ((1, self.onset), (-1, self.offset)):
                selected = direction == sign
                if selected.any():
                    output[selected] = model.predict(design[selected])
        return output


def _weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, cumulative[-1] / 2)]])


def fit_timing(training_matches):
    """Fit two fixed absolute-error heads on (id, features, labels, blocks, status).

Each match contributes equal total weight to each direction it supplies; its
boundaries have equal weight before that match normalization. No development
or challenge observations belong in this iterable. Early stopping and random
frame validation are deliberately disabled.
    """
    xs, targets, directions, sample_weights, boundary_keys, ids = [], [], [], [], [], []
    names = None
    for match in training_matches:
        match_id, features, labels, blocks = match[:4]
        if match_id in ids:
            raise ValueError("Need distinct training match identities")
        ids.append(match_id)
        status = status_targets(labels, match[4] if len(match) == 5 else None)
        if status.shape != (len(features),):
            raise ValueError("Training labels must match the motion frame count")
        periods = _blocks(blocks, len(features))
        design, schema = motion_design(features, periods)
        if names is not None and names != schema:
            raise ValueError("Training motion feature schemas differ")
        names = schema
        anchors, boundaries, direction, weights = timing_anchors(
            status, periods, features.fps, motion_support(features))
        if not len(anchors):
            continue
        xs.append(design[anchors])
        targets.append((boundaries - anchors) / float(features.fps))
        directions.append(direction)
        sample_weights.append(weights)
        boundary_keys.append([(match_id, int(b), int(d)) for b, d in zip(boundaries, direction)])
    if not xs:
        raise ValueError("No usable external training boundaries")
    x, y = np.concatenate(xs), np.concatenate(targets)
    direction, weights = np.concatenate(directions), np.concatenate(sample_weights)
    keys = [key for group in boundary_keys for key in group]
    common = dict(loss="absolute_error", learning_rate=.07, max_iter=160, max_leaf_nodes=15,
                  min_samples_leaf=60, l2_regularization=2., early_stopping=False, random_state=2027)
    heads, medians, anchor_counts, boundary_counts = [], [], [], []
    with threadpool_limits(limits=4):
        for sign in (1, -1):
            selected = direction == sign
            if not selected.any():
                raise ValueError("Both timing directions require external training boundaries")
            head_weights = weights[selected]
            head_weights = head_weights * len(head_weights) / head_weights.sum()
            model = HistGradientBoostingRegressor(**common)
            model.fit(x[selected], y[selected], sample_weight=head_weights)
            heads.append(model)
            medians.append(_weighted_median(y[selected], head_weights))
            anchor_counts.append(int(selected.sum()))
            boundary_counts.append(len({key for key, chosen in zip(keys, selected) if chosen}))
    return TimingModel(heads[0], heads[1], names, ids, medians, anchor_counts, boundary_counts)
