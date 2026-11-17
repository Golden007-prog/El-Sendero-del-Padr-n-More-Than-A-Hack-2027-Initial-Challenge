"""Observation-only, globally decoded turnover-preserving sequence refinement.

No model fitting or reference-label interface. See DESIGN.md and PROTOCOL-v1.md.
"""
from collections import deque

import numpy as np


WINDOW_SECONDS = 0.20
MIN_RUN_SECONDS = 0.40
SWITCH_BASE = 0.9
BOUNDARY_DISCOUNT = 0.8


def _fps(value):
    try:
        valid = (np.isscalar(value) and not isinstance(value, (bool, np.bool_))
                 and np.isfinite(value) and value > 0)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("fps must be finite and positive")
    return float(value)


def _inputs(baseline, probabilities, boundary_scores, fps, live_only=False):
    fps = _fps(fps)
    anchor = np.asarray(baseline)
    allowed = list("HA" if live_only else "HADX")
    if anchor.ndim != 1 or not np.isin(anchor, allowed).all():
        raise ValueError("baseline must be a one-dimensional permitted-label vector")
    n = len(anchor)
    try:
        probs = np.asarray(probabilities, dtype=np.float64)
        scores = np.asarray(boundary_scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("scores must be numeric arrays") from exc
    if (probs.shape != (n, 2) or not np.isfinite(probs).all()
            or ((probs <= 0) | (probs > 1)).any()
            or not np.allclose(probs.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("owner probabilities must be positive normalized N-by-2 H/A scores")
    if (scores.shape != (n,) or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("boundary scores must be finite length-N values in [0,1]")
    return anchor, probs, scores, fps


def _blocks(blocks, n):
    try:
        source = iter(blocks)
    except TypeError as exc:
        raise ValueError("blocks must be ordered half-open pairs") from exc
    result, previous = [], 0
    for block in source:
        try:
            pair = isinstance(block, (tuple, list, np.ndarray)) and len(block) == 2
        except TypeError:
            pair = False
        if not pair:
            raise ValueError("blocks must be ordered half-open pairs")
        start, end = block
        if any(not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_))
               for x in (start, end)):
            raise ValueError("block endpoints must be integers")
        if not previous <= start < end <= n:
            raise ValueError("blocks must be disjoint, ordered and in bounds")
        result.append((int(start), int(end)))
        previous = int(end)
    return result


def _layout(anchor, fps):
    n = len(anchor)
    edges = np.flatnonzero(anchor[1:] != anchor[:-1]) + 1
    lengths = np.diff(np.r_[0, edges, n]) if n else np.empty(0, dtype=np.int64)
    radius = round(WINDOW_SECONDS * fps)
    minimum = max(1, round(MIN_RUN_SECONDS * fps))
    windows = np.column_stack((np.maximum(1, edges - radius),
                               np.minimum(n - 1, edges + radius)))
    if len(edges) > 1:
        midpoints = (edges[:-1] + edges[1:]) // 2
        windows[:-1, 1] = np.minimum(windows[:-1, 1], midpoints)
        windows[1:, 0] = np.maximum(windows[1:, 0], midpoints + 1)
    return edges, windows, np.minimum(lengths, minimum), minimum, radius


def _constraint_audit(anchor, result, fps):
    """Verify the decoded path independently of its predecessor-node structure."""
    n = len(anchor)
    old, windows, principal_min, minimum, radius = _layout(anchor, fps)
    edges = np.flatnonzero(result[1:] != result[:-1]) + 1
    segment, phase, last, added, moved, maximum_shift = 0, 0, 0, 0, 0, 0
    minimum_slack = n
    valid = bool(n == 0 or (result[0] == anchor[0] and result[-1] == anchor[-1]))
    for value in edges:
        edge = int(value)
        required = minimum if phase else int(principal_min[segment])
        minimum_slack = min(minimum_slack, edge - last - required)
        if segment < len(old) and edge >= windows[segment, 0]:
            valid &= bool(edge <= windows[segment, 1] and phase == 0)
            shift = abs(edge - int(old[segment]))
            moved += int(shift != 0)
            maximum_shift = max(maximum_shift, shift)
            segment += 1
        else:
            phase = 1 - phase
            added += 1
        last = edge
    if n:
        required = minimum if phase else int(principal_min[segment])
        minimum_slack = min(minimum_slack, n - last - required)
    valid &= bool(segment == len(old) and phase == 0 and minimum_slack >= 0)
    # The parity/direction of every mandatory edge follows from the initial team,
    # the alternating principal sequence, and complete added return pairs.
    return {
        "topology_and_durations_valid": bool(valid),
        "baseline_turnovers": int(len(old)), "output_turnovers": int(len(edges)),
        "added_excursion_pairs": int(added // 2),
        "moved_mandatory_edges": int(moved),
        "maximum_mandatory_shift_frames": int(maximum_shift),
        "minimum_duration_slack_frames": int(minimum_slack),
        "window_radius_frames": int(radius), "minimum_excursion_frames": int(minimum),
    }


class _Nodes:
    """At most two run-entry nodes per frame; arrays keep large spans bounded."""
    def __init__(self, n, prefix_cost, prefix_error):
        size = max(1, 2 * n)
        self.cost = np.empty(size, dtype=np.float64)
        self.error = np.empty(size, dtype=np.int64)
        self.start = np.empty(size, dtype=np.int64)
        self.previous = np.empty(size, dtype=np.int64)
        self.label = np.empty(size, dtype=np.int8)
        self.count = 0
        self.prefix_cost, self.prefix_error = prefix_cost, prefix_error
        self.primary_ties = 0
        self.secondary_ties = 0

    def add(self, start, label, previous, cost_before, error_before):
        node = self.count
        self.count += 1
        if node >= len(self.start):
            raise AssertionError("run-entry allocation bound exceeded")
        self.cost[node] = cost_before - self.prefix_cost[start, label]
        self.error[node] = error_before - self.prefix_error[start, label]
        self.start[node], self.previous[node], self.label[node] = start, previous, label
        return node

    def better(self, candidate, incumbent):
        if incumbent < 0:
            return True
        a, b = self.cost[candidate], self.cost[incumbent]
        if a != b:
            return bool(a < b)
        self.primary_ties += 1
        a, b = self.error[candidate], self.error[incumbent]
        if a != b:
            return bool(a < b)
        self.secondary_ties += 1
        # Within a pool segment/phase are identical. Earlier start means longer
        # current duration; node creation order breaks any remaining exact tie.
        return (int(self.start[candidate]), candidate) < (int(self.start[incumbent]), incumbent)


class _Pool:
    """Minimum-duration run entries, sharing one label and principal segment."""
    def __init__(self, label, minimum, nodes):
        self.label, self.minimum, self.nodes = int(label), int(minimum), nodes
        self.pending = deque()
        self.best = -1

    def add(self, frame, previous, cost_before, error_before):
        node = self.nodes.add(frame, self.label, previous, cost_before, error_before)
        self.pending.append(node)

    def advance(self, frame):
        while self.pending and self.nodes.start[self.pending[0]] + self.minimum - 1 <= frame:
            candidate = self.pending.popleft()
            if self.nodes.better(candidate, self.best):
                self.best = candidate

    def completed(self, end_exclusive):
        if self.best < 0:
            return None
        node = self.best
        return (node,
                float(self.nodes.cost[node] + self.nodes.prefix_cost[end_exclusive, self.label]),
                int(self.nodes.error[node] + self.nodes.prefix_error[end_exclusive, self.label]))


def _span(anchor, probs, boundary, fps):
    n = len(anchor)
    if not n:
        audit = _constraint_audit(anchor, anchor, fps)
        audit.update({"objective": 0.0, "baseline_objective": 0.0,
                      "run_entry_nodes": 0, "primary_tie_comparisons": 0,
                      "secondary_tie_comparisons": 0, "changed_live_frames": 0})
        return anchor.astype("U1", copy=True), audit
    _, windows, principal_min, minimum, _ = _layout(anchor, fps)
    codes = (anchor == "A").astype(np.int8)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        unary = -np.log(probs) / fps
    if not np.isfinite(unary).all():
        raise ValueError("fps/probabilities produce nonfinite emission costs")
    # Removing a framewise constant preserves the objective and exact neutral
    # emissions; all arithmetic remains equivariant under H/A column exchange.
    unary -= unary.min(axis=1)[:, None]
    prefix_cost = np.vstack((np.zeros(2), np.cumsum(unary, axis=0)))
    if not np.isfinite(prefix_cost).all():
        raise ValueError("cumulative emission costs overflow")
    prefix_error = np.vstack((np.zeros(2, dtype=np.int64),
                              np.cumsum(codes[:, None] != np.arange(2), axis=0)))
    costs = SWITCH_BASE * (1 - BOUNDARY_DISCOUNT * boundary)
    nodes = _Nodes(n, prefix_cost, prefix_error)
    principal = _Pool(codes[0], principal_min[0], nodes)
    excursion = _Pool(1 - codes[0], minimum, nodes)
    principal.add(0, -1, 0.0, 0)
    principal.advance(0)
    segment, inside, following = 0, False, None
    for frame in range(1, n):
        if segment < len(windows) and frame == windows[segment, 0]:
            inside = True
            following = _Pool(1 - principal.label, principal_min[segment + 1], nodes)
            # An open excursion cannot close inside a mandatory window and
            # cannot take its mandatory edge. Such histories are infeasible.
            excursion = None
        if inside:
            old = principal.completed(frame)
            if old is not None:
                previous, cost, error = old
                following.add(frame, previous, cost + costs[frame], error)
            principal.advance(frame)
            following.advance(frame)
            if frame == windows[segment, 1]:
                principal = following
                segment += 1
                excursion = _Pool(1 - principal.label, minimum, nodes)
                following, inside = None, False
        else:
            # Snapshot both predecessor pools before inserting either entry;
            # two transitions at the same edge are never possible.
            old_principal = principal.completed(frame)
            old_excursion = excursion.completed(frame)
            if old_principal is not None:
                previous, cost, error = old_principal
                excursion.add(frame, previous, cost + costs[frame], error)
            if old_excursion is not None:
                previous, cost, error = old_excursion
                principal.add(frame, previous, cost + costs[frame], error)
            principal.advance(frame)
            excursion.advance(frame)
    final = principal.completed(n)
    if inside or segment != len(windows) or final is None:
        raise AssertionError("baseline-feasible constrained path became unreachable")
    result = np.empty(n, dtype="U1")
    node, end = final[0], n
    while node >= 0:
        start = int(nodes.start[node])
        result[start:end] = "A" if nodes.label[node] else "H"
        end, node = start, int(nodes.previous[node])
    if end != 0:
        raise AssertionError("traceback did not cover the complete span")
    audit = _constraint_audit(anchor, result, fps)
    if not audit["topology_and_durations_valid"]:
        raise AssertionError("traceback violates frozen path constraints")
    result_codes = (result == "A").astype(np.int8)
    edges = np.flatnonzero(result[1:] != result[:-1]) + 1
    baseline_edges = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    # Report the original objective, rather than its constant-shifted DP form.
    with np.errstate(over="ignore"):
        objective = float((-np.log(probs[np.arange(n), result_codes]) / fps).sum() + costs[edges].sum())
        baseline_objective = float((-np.log(probs[np.arange(n), codes]) / fps).sum() + costs[baseline_edges].sum())
    if not np.isfinite([objective, baseline_objective]).all():
        raise ValueError("reported emission objective overflows")
    audit.update({
        "objective": objective, "baseline_objective": baseline_objective,
        "run_entry_nodes": int(nodes.count),
        "primary_tie_comparisons": int(nodes.primary_ties),
        "secondary_tie_comparisons": int(nodes.secondary_ties),
        "changed_live_frames": int((result != anchor).sum()),
    })
    return result, audit


def refine_span(baseline, owner_probabilities, boundary_scores, fps):
    """Standalone H/A span decoder, including an empty-span case, for review."""
    anchor, probs, boundary, fps = _inputs(baseline, owner_probabilities, boundary_scores, fps, True)
    return _span(anchor, probs, boundary, fps)


def refine(baseline, owner_probabilities, boundary_scores, blocks, fps):
    """Return labels and aggregate audit; accepts predictors and baseline only.

    Probability columns are H,A. Blocks are ordered half-open period intervals.
    Boundary score at index t prices the incoming edge t-1 -> t. Span-first
    scores are unused; all protected support is copied exactly.
    """
    anchor, probs, boundary, fps = _inputs(baseline, owner_probabilities, boundary_scores, fps)
    n = len(anchor)
    periods = _blocks(blocks, n)
    output = anchor.astype("U1", copy=True)
    processed = np.zeros(n, dtype=bool)
    totals = {key: 0 for key in (
        "baseline_turnovers", "output_turnovers", "added_excursion_pairs",
        "moved_mandatory_edges", "run_entry_nodes", "primary_tie_comparisons",
        "secondary_tie_comparisons", "changed_live_frames")}
    objective, baseline_objective, max_shift, spans, min_slack = 0.0, 0.0, 0, 0, n
    for start, end in periods:
        processed[start:end] = True
        live = np.isin(anchor[start:end], ["H", "A"])
        endpoints = np.flatnonzero(np.diff(np.r_[False, live, False]))
        for left, right in endpoints.reshape(-1, 2):
            a, b = start + int(left), start + int(right)
            output[a:b], local = _span(anchor[a:b], probs[a:b], boundary[a:b], fps)
            for key in totals:
                totals[key] += local[key]
            objective += local["objective"]
            baseline_objective += local["baseline_objective"]
            max_shift = max(max_shift, local["maximum_mandatory_shift_frames"])
            min_slack = min(min_slack, local["minimum_duration_slack_frames"])
            spans += 1
    if not np.isfinite([objective, baseline_objective]).all():
        raise ValueError("combined span objective overflows")
    audit = {
        **totals, "live_spans_decoded": spans,
        "objective": float(objective), "baseline_objective": float(baseline_objective),
        "maximum_mandatory_shift_frames": int(max_shift),
        "minimum_duration_slack_frames": int(min_slack if spans else 0),
        "window_radius_frames": int(round(WINDOW_SECONDS * fps)),
        "minimum_excursion_frames": int(max(1, round(MIN_RUN_SECONDS * fps))),
        "topology_and_durations_valid": True,
        "dead_mask_unchanged": bool(np.array_equal(output == "D", anchor == "D")),
        "unknown_mask_unchanged": bool(np.array_equal(output == "X", anchor == "X")),
        "outside_periods_unchanged": bool(np.array_equal(output[~processed], anchor[~processed])),
        "probabilities_calibrated": False,
        "window_seconds": WINDOW_SECONDS, "minimum_run_seconds": MIN_RUN_SECONDS,
        "switch_base": SWITCH_BASE, "boundary_discount": BOUNDARY_DISCOUNT,
    }
    if not all(audit[key] for key in ("dead_mask_unchanged", "unknown_mask_unchanged", "outside_periods_unchanged")):
        raise AssertionError("refinement changed protected support")
    return output, audit
