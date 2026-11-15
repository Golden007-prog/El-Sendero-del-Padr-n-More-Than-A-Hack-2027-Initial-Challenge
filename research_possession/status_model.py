"""Player-only live/dead and boundary models fitted on external training matches."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .motion import motion_design


@dataclass
class StatusModel:
    status: object
    boundary: object
    feature_names: list[str]
    train_ids: list[str]
    training_dead_prior: float

    def predict(self, features, blocks):
        design, names = motion_design(features, blocks)
        if names != self.feature_names:
            raise ValueError("Motion feature schema differs from the trained model")
        with threadpool_limits(limits=4):
            dead = self.status.predict_proba(design)[:, list(self.status.classes_).index(1)]
            # Class-balanced boundary training makes this a detection strength,
            # not a calibrated probability of a transition.
            boundary = self.boundary.predict_proba(design)[:, list(self.boundary.classes_).index(1)]
        return dead, boundary


def boundary_targets(labels, blocks, fps, radius_s=0.4, status=None):
    """External supervision only: live/dead changes, never half separators."""
    labels = np.asarray(labels)
    status = status_targets(labels, status)
    target = np.zeros(len(labels), dtype=np.int8)
    radius = max(1, round(radius_s * fps))
    for start, end in blocks:
        valid = status[start:end] >= 0
        dead = status[start:end] == 1
        changes = np.flatnonzero((dead[1:] != dead[:-1]) & valid[1:] & valid[:-1]) + start + 1
        for frame in changes:
            target[max(start, frame - radius):min(end, frame + radius + 1)] = 1
    return target


def status_targets(labels, status=None):
    """Preserve known binary status even when the owner is unannotated."""
    labels = np.asarray(labels)
    if status is None:
        return np.where(labels == "D", 1, np.where(np.isin(labels, ["H", "A"]), 0, -1)).astype(np.int8)
    status = np.asarray(status)
    if status.shape != labels.shape or not np.isin(status, [-1, 0, 1]).all():
        raise ValueError("Status targets must have shape N with -1 unknown, 0 live, 1 dead")
    return status.astype(np.int8)


def fit_status(training_matches):
    """Each item is (id, Features, labels, blocks[, separate_binary_status]).

    Fixed architecture and seed. No random frame-level validation split or
    internal early stopping; model selection is an explicit OTHER-match stage.
    """
    xs, ys, bs, ws, ids = [], [], [], [], []
    feature_names = None
    for match in training_matches:
        match_id, features, labels, blocks = match[:4]
        status_y = status_targets(labels, match[4] if len(match) == 5 else None)
        design, names = motion_design(features, blocks)
        if feature_names is not None and feature_names != names:
            raise ValueError("Training feature schemas differ")
        feature_names = names
        boundaries = boundary_targets(labels, blocks, features.fps, status=status_y)
        # Fixed 5 Hz sampling, independent of class/labels and match duration.
        stride = max(1, round(features.fps / 5))
        indices = np.concatenate([np.arange(a, b, stride) for a, b in blocks])
        indices = indices[status_y[indices] >= 0]
        if not len(indices):
            raise ValueError("Training match has no usable external targets")
        xs.append(design[indices].astype(np.float32))
        ys.append(status_y[indices])
        bs.append(boundaries[indices])
        ws.append(np.full(len(indices), 1 / len(indices)))
        ids.append(match_id)
    if not xs or len(ids) != len(set(ids)):
        raise ValueError("Need distinct training matches")
    x, y, boundary_y = np.concatenate(xs), np.concatenate(ys), np.concatenate(bs)
    weights = np.concatenate(ws)
    weights *= len(weights) / weights.sum()
    if len(np.unique(y)) != 2 or len(np.unique(boundary_y)) != 2:
        raise ValueError("Both classes required in each prediction head")
    common = dict(learning_rate=0.07, max_iter=160, max_leaf_nodes=15,
                  min_samples_leaf=60, l2_regularization=2.0,
                  early_stopping=False, random_state=2027)
    status = HistGradientBoostingClassifier(**common)
    boundary = HistGradientBoostingClassifier(**common, class_weight="balanced")
    with threadpool_limits(limits=4):
        status.fit(x, y, sample_weight=weights)
        boundary.fit(x, boundary_y, sample_weight=weights)
    return StatusModel(status, boundary, feature_names, ids, float(np.average(y, weights=weights)))
