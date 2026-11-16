"""Minimum-control rule from the organizers' written possession definition.

CHALLENGE.md: possession changes only when the opposing team gains controlled
possession. A simple touch, a deflection or an aerial duel won without
subsequent control does not count; in contested or loose phases possession
stays with the last team that had it; and a proxy for control is that a player
keeps the ball for a few seconds. An opponent run shorter than
MIN_CONTROL_SECONDS that is immediately preceded and followed by the same
team, with no dead ball between them, is therefore not a change of possession.

The threshold is a fixed reading of "a few seconds". It is not fitted to any
match.
"""
import numpy as np

MIN_CONTROL_SECONDS = 3.0
_TEAMS = ("H", "A")


def _scorer_view(segment):
    """Abstentions forward-filled, back-filled at the head, as the scorer does."""
    index = np.where(segment != "X", np.arange(len(segment)), -1)
    np.maximum.accumulate(index, out=index)
    view = np.where(index >= 0, segment[np.clip(index, 0, None)], "X")
    known = np.flatnonzero(view != "X")
    if len(known) and known[0] > 0:
        view[:known[0]] = view[known[0]]
    return view


def _runs(view):
    edges = np.flatnonzero(view[1:] != view[:-1]) + 1
    return np.r_[0, edges], np.r_[edges, len(view)]


def absorb_uncontrolled_excursions(labels, blocks, fps, min_seconds=MIN_CONTROL_SECONDS):
    """Return (labels, audit). Only H/A frames of qualifying runs change."""
    labels = np.array(labels, dtype="<U1")
    limit = int(round(float(min_seconds) * float(fps)))
    absorbed, changed = 0, 0
    for first, stop in blocks:
        first, stop = int(first), int(stop)
        segment = labels[first:stop]
        view = _scorer_view(segment)
        while True:
            starts, ends = _runs(view)
            best = None
            for i in range(1, len(starts) - 1):
                label, before, after = view[starts[i]], view[starts[i - 1]], view[starts[i + 1]]
                length = ends[i] - starts[i]
                if (label in _TEAMS and before in _TEAMS and before == after and before != label
                        and length < limit and (best is None or length < best[0])):
                    best = (length, int(starts[i]), int(ends[i]), before)
            if best is None:
                break
            _, start, end, owner = best
            view[start:end] = owner
            rewrite = segment[start:end] != "X"
            changed += int(rewrite.sum())
            segment[start:end][rewrite] = owner
            absorbed += 1
        labels[first:stop] = segment
    return labels, {"min_control_seconds": float(min_seconds), "min_control_frames": limit,
                    "excursions_absorbed": absorbed, "frames_relabelled": changed}
