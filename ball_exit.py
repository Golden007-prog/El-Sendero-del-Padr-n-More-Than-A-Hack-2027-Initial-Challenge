"""Dead-ball onset at the moment the ball leaves the pitch.

The ball is out of play once it has crossed a boundary line. The status models decide stoppages from
player movement, which reacts late. Where the tracking shows the ball leaving the pitch shortly before a
predicted stoppage, the stoppage starts there instead.

Nothing is fitted. Only live frames (H or A) directly before a predicted stoppage can become D.
"""
import numpy as np

EXIT_LOOKBACK_SECONDS = 3.0     # how far before a predicted stoppage the exit may lie
EXIT_LINE_MARGIN_M = 2.0        # "last seen at the line": distance to the nearest boundary line
EXIT_DELAY_FRAMES = 2           # the stoppage starts this many frames after the last in-pitch sighting
EXIT_CONFIRM_SECONDS = 0.5      # an outside sighting this soon after our onset still confirms the exit
_TEAMS = ("H", "A")


def pitch_margin(ball_xy_m, pitch):
    """Distance in metres to the nearest boundary line: negative outside the pitch, NaN where untracked."""
    ball = np.asarray(ball_xy_m, dtype=float)
    length, width = float(pitch[0]), float(pitch[1])
    return np.minimum.reduce([ball[:, 0], length - ball[:, 0], ball[:, 1], width - ball[:, 1]])


def advance_dead_onsets(labels, ball_xy_m, pitch, blocks, fps):
    """Return (labels, audit). Stoppage onsets move earlier where the ball is shown leaving the pitch."""
    labels = np.array(labels, dtype="<U1")
    out = labels.copy()
    margin = pitch_margin(ball_xy_m, pitch)
    if margin.shape != labels.shape:
        raise ValueError("Ball track and labels must be frame aligned")
    with np.errstate(invalid="ignore"):
        inside, outside = margin >= 0, margin < 0
    lookback = int(round(EXIT_LOOKBACK_SECONDS * float(fps)))
    confirm = int(round(EXIT_CONFIRM_SECONDS * float(fps)))
    moved, frames, by_sighting, by_line = 0, 0, 0, 0
    for first, stop in blocks:
        first, stop = int(first), int(stop)
        seg = labels[first:stop]
        onsets = np.flatnonzero((seg[1:] == "D") & np.isin(seg[:-1], _TEAMS)) + 1 + first
        for onset in onsets.tolist():
            low = max(first, onset - lookback)
            k = onset - 1
            while k >= low and labels[k] in _TEAMS:
                k -= 1
            low = k + 1
            seen_inside = np.flatnonzero(inside[low:onset])
            if not len(seen_inside):
                continue
            last_inside = low + int(seen_inside[-1])
            start = last_inside + EXIT_DELAY_FRAMES
            if start >= onset:
                continue
            sighted = bool(outside[last_inside + 1:min(stop, onset + confirm)].any())
            at_line = bool(margin[last_inside] <= EXIT_LINE_MARGIN_M)
            if not (sighted or at_line):
                continue
            out[start:onset] = "D"
            moved += 1
            frames += onset - start
            by_sighting += int(sighted)
            by_line += int(at_line and not sighted)
    return out, {"lookback_seconds": EXIT_LOOKBACK_SECONDS, "line_margin_m": EXIT_LINE_MARGIN_M,
                 "delay_frames": EXIT_DELAY_FRAMES, "onsets_moved": moved, "frames_relabelled": frames,
                 "confirmed_by_outside_sighting": by_sighting, "confirmed_by_last_sighting_at_line": by_line}
