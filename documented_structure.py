"""Dead-ball structure the organizers document for the supplied match.

The organizers' FAQ.md and DATA.md publish the graded blocks and state that
the second half is graded from video frame 110,808 while play does not kick
off until 113,742, so every graded frame in between is dead ball. The frame
below is the constant in their starter/load_smart_tagging.py. It is
documentation given to every participant, not an annotation made by this team.
"""
import numpy as np

SECOND_HALF_KICKOFF_FRAME = {"BAR-ATH": 113742}


def apply_documented_dead(labels, blocks, game):
    """Label the graded frames before the documented second-half kick-off D.

    Returns (labels, audit). Every other frame is returned unchanged.
    """
    labels = np.array(labels, dtype="<U1")
    kickoff = SECOND_HALF_KICKOFF_FRAME.get(game)
    if kickoff is None:
        return labels, {"second_half_kickoff_frame": None, "frames_set_dead": 0}
    if len(blocks) != 2:
        raise ValueError("Expected exactly two graded blocks")
    first, stop = map(int, blocks[1])
    if not first < kickoff < stop <= len(labels):
        raise ValueError("Documented kick-off lies outside the second graded block")
    changed = int((labels[first:kickoff] != "D").sum())
    labels[first:kickoff] = "D"
    return labels, {"second_half_kickoff_frame": kickoff, "second_half_first_graded_frame": first,
                    "frames_set_dead": changed}
