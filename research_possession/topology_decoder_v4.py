"""Frozen-v2 topology with separate false-island removal and timing refinement.

The identity configuration returns the baseline labels without decoding again.
Original X frames are permanent separators. Quality scores are decision scores,
not calibrated probabilities, and removing short islands does not guarantee
retention of real short stoppages; the audit makes that exposure explicit.
"""

from dataclasses import dataclass

import numpy as np

from .boundary_metrics_v3 import label_status
from .decoder import STATES, _validate
from .precision_decoder import _ownership, _project_boundaries, _retime
from .timing_model import _blocks, _runs, motion_support


@dataclass(frozen=True)
class TopologyConfig:
    timing_scale: float = 0.
    false_threshold: float | None = None

    def __post_init__(self):
        for name in ("timing_scale", "false_threshold"):
            value = getattr(self, name)
            if name == "false_threshold" and value is None:
                continue
            try:
                valid = (np.isscalar(value) and not isinstance(value, (bool, np.bool_))
                         and np.isfinite(value) and 0 <= value <= 1)
            except TypeError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be a finite number between zero and one")


class _ScaledTiming:
    def __init__(self, model, scale):
        self.model, self.scale = model, scale
        if hasattr(model, "feature_names"):
            self.feature_names = model.feature_names

    def predict_offsets(self, design_rows, directions):
        raw = np.asarray(self.model.predict_offsets(design_rows, directions), dtype=float)
        return np.where(np.isfinite(raw), np.clip(raw, -1.5, 1.5) * self.scale, raw)


def _status_counts(status, periods):
    counts = dict(boundaries=0, live_to_dead=0, dead_to_live=0,
                  dead_runs=0, complete_dead_runs=0)
    for period_start, period_end in periods:
        for start, end in _runs(status[period_start:period_end] >= 0, period_start):
            changes = np.diff(status[start:end])
            counts["live_to_dead"] += int(np.sum(changes == 1))
            counts["dead_to_live"] += int(np.sum(changes == -1))
            counts["boundaries"] += int(np.count_nonzero(changes))
            for a, b in _runs(status[start:end] == 1, start):
                counts["dead_runs"] += 1
                counts["complete_dead_runs"] += int(a > start and b < end)
    return counts


def _refine_timing(status, features, periods, known_blocks, evidence, model, scale):
    # Reuse v3's feature schema, directional prediction and unsupported anchors.
    # Its joint projection has a full-scale cap, so project again with the
    # tighter per-boundary bounds required by a fractional timing scale.
    _, audit = _retime(status, features, periods, known_blocks, evidence,
                       _ScaledTiming(model, scale))
    radius = int(np.floor(1.5 * features.fps * scale + 1e-9))
    result = status.copy()
    cursor = 0
    for start, end in known_blocks:
        boundaries = np.flatnonzero(np.diff(status[start:end])) + start + 1
        rows = audit[cursor:cursor + len(boundaries)]
        cursor += len(boundaries)
        if not rows:
            continue
        proposals = np.array([row["proposed_frame"] for row in rows])
        movable = np.array([row["player_support"] and row["predicted_offset_s"] is not None
                            for row in rows])
        lower = np.where(movable, np.maximum(boundaries - radius, start + 1), boundaries)
        upper = np.where(movable, np.minimum(boundaries + radius, end - 1), boundaries)
        final = _project_boundaries(proposals, lower, upper)
        position, state = start, int(status[start])
        for index, (row, actual) in enumerate(zip(rows, final)):
            result[position:actual] = state
            position, state = int(actual), 1 - state
            row["final_frame"] = int(actual)
            row["final_time_s"] = float(actual / features.fps)
            row["timing_scale"] = float(scale)
            row["maximum_move_frames"] = radius
            row["maximum_move_s"] = float(1.5 * scale)
            reasons = [reason for reason in row["reasons"]
                       if reason not in ("order_projection", "frame_move_limit")]
            nearest = int(np.clip(np.ceil(proposals[index] - .5), lower[index], upper[index]))
            if actual != nearest:
                reasons.append("order_projection")
            if abs(proposals[index] - row["original_frame"]) > radius + 1e-9:
                reasons.append("frame_move_limit")
            row["reasons"] = reasons
        result[position:end] = state
    return result, audit


def decode_topology_periods(features, blocks, baseline_labels, baseline_evidence,
                            timing_model=None, quality_model=None, config=TopologyConfig()):
    """Return labels, explicit -1/0/1 status, and local evidence with audit.

Only original complete D proposals can be removed, once, before timing. Their
new live ownership is inferred by the unchanged v3 ownership routine, which
leaves holes unknown when control observations are unavailable. Timing retains
all remaining boundaries in order and cannot cross original X or period edges.
    """
    if not isinstance(config, TopologyConfig):
        raise TypeError("config must be a TopologyConfig instance")
    _validate(features)
    n, fps = len(features), float(features.fps)
    periods = _blocks(blocks, n)
    baseline = np.asarray(baseline_labels)
    if baseline.shape != (n,) or not np.isin(baseline, STATES).all():
        raise ValueError("Baseline labels must be a H/A/D/X vector of length N")
    baseline = baseline.copy()
    evidence = dict(baseline_evidence)
    required = ("fused_dead_score", "live_dead_switch_cost", "status_quality", "unsupported",
                "control_evidence", "team_switch_cost", "support_freshness", "dead_evidence")
    for key in required:
        if key not in evidence or np.asarray(evidence[key]).shape != (n,):
            raise ValueError(f"Frozen v2 evidence requires a length-N {key} vector")
        evidence[key] = np.asarray(evidence[key])
        if not np.isfinite(evidence[key]).all():
            raise ValueError("Frozen v2 evidence must be finite")
    if config.timing_scale > 0 and timing_model is None:
        raise ValueError("A timing model is required for a positive timing scale")
    if config.false_threshold is not None and quality_model is None:
        raise ValueError("An interval quality model is required when removal is enabled")

    from .interval_quality_v4 import interval_candidates, interval_design

    status = label_status(baseline)
    original_status = status.copy()
    original_unknown = baseline == "X"
    current_motion = motion_support(features)
    current_quality = evidence["status_quality"] > 0
    current_supported = ~evidence["unsupported"].astype(bool)
    proposals = interval_candidates(status, periods, fps)
    candidates = []
    for proposal in proposals:
        record = dict(proposal)
        start, end = int(record["start"]), int(record["end"])
        record["start"], record["end"] = start, end
        window = slice(start - 1, end + 1)
        reasons = []
        if not current_motion[window].all():
            reasons.append("missing_current_motion")
        if not current_quality[window].all():
            reasons.append("missing_status_quality")
        if not current_supported[window].all():
            reasons.append("unsupported_interval")
        record.update(false_score=None, removed=False, reasons=reasons)
        candidates.append(record)

    removed = []
    if config.false_threshold is not None and candidates:
        scoreable = [record for record in candidates if not record["reasons"]]
        if scoreable:
            records, design, names = interval_design(features, original_status, evidence,
                                                     periods, candidates=scoreable)
            if [(r["start"], r["end"]) for r in records] != [
                    (r["start"], r["end"]) for r in scoreable]:
                raise ValueError("Interval design must preserve the supplied proposal order")
            if hasattr(quality_model, "feature_names") and names != quality_model.feature_names:
                raise ValueError("Interval quality feature schema differs from the trained model")
            scores = np.asarray(quality_model.predict_false(design), dtype=float)
            if (scores.shape != (len(scoreable),) or not np.isfinite(scores).all()
                    or ((scores < 0) | (scores > 1)).any()):
                raise ValueError("Quality model must return one finite [0,1] score per interval")
            for record, score in zip(scoreable, scores):
                record["false_score"] = float(score)
                if score >= config.false_threshold:
                    status[record["start"]:record["end"]] = 0
                    record["removed"] = True
                    removed.append(dict(record))
                else:
                    record["reasons"].append("below_false_threshold")
    else:
        for record in candidates:
            record["reasons"].append("removal_disabled")

    after_removal = status.copy()
    known_blocks = []
    safe = (status >= 0) & current_supported & current_motion & current_quality
    for start, end in periods:
        known_blocks.extend(_runs(safe[start:end], start))
    retiming = []
    if config.timing_scale > 0:
        status, retiming = _refine_timing(status, features, periods, known_blocks,
                                         evidence, timing_model, config.timing_scale)
    if np.array_equal(status, original_status):
        labels = baseline.copy()
    else:
        labels = _ownership(status, baseline, evidence, known_blocks, fps)
    labels[original_unknown] = "X"
    status[original_unknown] = -1

    counts = {"candidate_intervals": len(candidates), "removed_intervals": len(removed),
              "scored_intervals": sum(record["false_score"] is not None for record in candidates),
              "moved_boundaries": sum(row["final_frame"] != row["original_frame"] for row in retiming),
              "timing_boundaries": len(retiming),
              "removed_short_dead_islands_le_2s": sum((r["end"] - r["start"]) / fps <= 2
                                                       for r in removed),
              "short_dead_safeguard_count": 0}
    for reason in ("missing_current_motion", "missing_status_quality", "unsupported_interval",
                   "below_false_threshold"):
        counts["rejected_" + reason] = sum(reason in record["reasons"] for record in candidates)
    topology_audit = {"before": _status_counts(original_status, periods),
                      "after_removal": _status_counts(after_removal, periods),
                      "after_retime": _status_counts(status, periods),
                      "original_x_preserved": bool(np.all(labels[original_unknown] == "X")
                                                    and np.all(status[original_unknown] == -1)),
                      "short_dead_safeguard_count": 0,
                      "removed_short_dead_islands_le_2s": counts["removed_short_dead_islands_le_2s"]}
    evidence.update(baseline_labels=baseline, baseline_status=original_status,
                    status_before_removal=original_status.copy(), status_after_removal=after_removal,
                    status_before_refine=after_removal.copy(), supported_blocks=known_blocks,
                    removal_candidates=candidates, removed_islands=removed, retiming_audit=retiming,
                    topology_audit=topology_audit, counts=counts)
    evidence["audit"] = {**topology_audit, "counts": counts, "removal_candidates": candidates,
                         "removed_islands": removed, "retiming_audit": retiming,
                         "false_scores_are_calibrated_probabilities": False,
                         "short_dead_retention_guaranteed": False}
    return labels, status, evidence
