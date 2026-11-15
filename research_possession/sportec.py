"""Load an OTHER-match Sportec/DFL holdout with target fields kept separate.

The provider's BallStatus and BallPossession are evaluation targets. Geometry,
sample membership, frame rate and half boundaries never depend on those fields.
The two streaming passes avoid materialising the large XML tree.
"""

from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .tracking import Features
from .motion import summarize_player_frames


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _locate(folder, short_name, pattern):
    path = folder / short_name
    if path.is_file():
        return path
    paths = list(folder.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one Sportec {short_name} file")
    return paths[0]


def _metadata(path):
    teams = {}
    pitch = None
    match_id = None
    for element in ET.parse(path).iter():
        name = _tag(element)
        if name == "Team" and element.get("Role") in ("home", "guest", "away"):
            side = 0 if element.get("Role") == "home" else 1
            team_id = element.get("TeamId")
            if not team_id or team_id in teams or side in teams.values():
                raise ValueError("Sportec metadata must declare one home and one away team")
            teams[team_id] = side
        elif name == "Environment":
            pitch = (float(element.get("PitchX", "nan")),
                     float(element.get("PitchY", "nan")))
        elif name == "General":
            match_id = element.get("MatchId")
    if len(teams) != 2 or set(teams.values()) != {0, 1}:
        raise ValueError("Sportec metadata must declare one home and one away team")
    if pitch is None or not np.isfinite(pitch).all() or min(pitch) <= 0:
        raise ValueError("Sportec metadata needs positive finite PitchX and PitchY")
    return teams, pitch, match_id


def _frames(path):
    context = None
    set_id = -1
    for event, element in ET.iterparse(path, events=("start", "end")):
        name = _tag(element)
        if event == "start" and name == "FrameSet":
            context = dict(element.attrib)
            set_id += 1
        elif event == "end" and name == "Frame" and context is not None:
            yield set_id, context, dict(element.attrib)
            element.clear()
        elif event == "end" and name == "FrameSet":
            context = None
            element.clear()


def _time(value):
    if not value:
        raise ValueError("Sportec frame timestamp is missing")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp


def _geometry(attributes):
    """The entire predictor-value whitelist; deliberately no target fields."""
    values = []
    for name in ("X", "Y", "S"):
        raw = attributes.get(name)
        value = float(raw) if raw not in (None, "") else np.nan
        values.append(value if np.isfinite(value) else np.nan)
    return values


def _label(attributes):
    status = attributes.get("BallStatus")
    if status == "0":
        return "D"
    if status == "1":
        return {"1": "H", "2": "A"}.get(attributes.get("BallPossession"), "X")
    return "X"


def _status_target(attributes):
    """Separate dead/live supervision, independent of possession ownership."""
    return {"0": 1, "1": 0}.get(attributes.get("BallStatus"), -1)


def load_sportec(folder, include_status=False):
    """Return ``(Features, labels: ndarray[U1], blocks: list[(start, end)])``.

    With ``include_status=True``, append a fourth result: an ``int8[N]`` status
    target where 1 means dead, 0 means live, and -1 means unknown. This target
    uses only raw BallStatus, so live frames with unknown possession ownership
    still have status 0 even though their four-state possession label is X.
    Unobserved or invalid-status rows remain -1. Status never enters Features.

    ``folder`` contains ``tracking.xml`` and ``matchinformation.xml`` (original
    DFL filenames are also accepted). Each block is a half-open slice for one
    GameSection, ordered by its ISO timestamps. Native frame numbers are kept on
    their uniform lattice within a half, including missing and dead-ball rows;
    the halftime numbering gap is not expanded into artificial match samples.

    X/Y are centred metres and become positive-pitch coordinates by adding half
    the metadata dimensions. Player S is km/h and becomes m/s. The speed feature
    is the median of finite, nonnegative speeds of geometrically visible players.
    """
    folder = Path(folder)
    meta_path = _locate(folder, "matchinformation.xml", "DFL_02_01_matchinformation_*.xml")
    raw_path = _locate(folder, "tracking.xml", "DFL_04_03_positions_raw_observed_*.xml")
    teams, pitch, match_id = _metadata(meta_path)
    periods, timing, players = {}, {}, set()
    ball_rows, target_rows = {}, {}

    # Pass one determines the clock from all recognised objects and stores only
    # ball geometry. Targets go into a separate map and do not control the clock.
    for set_id, context, attributes in _frames(raw_path):
        team = context.get("TeamId")
        if team not in teams and team != "BALL":
            continue
        if match_id and context.get("MatchId") != match_id:
            raise ValueError("Sportec tracking and metadata match IDs disagree")
        section = context.get("GameSection")
        if not section:
            raise ValueError("Sportec FrameSet needs a GameSection")
        frame = int(attributes["N"])
        if frame < 0:
            raise ValueError("Sportec frame IDs must be nonnegative")
        period = periods.setdefault(section, [frame, frame, None])
        period[0], period[1] = min(period[0], frame), max(period[1], frame)
        if set_id not in timing:
            stamp = _time(attributes.get("T"))
            timing[set_id] = [frame, stamp, frame, attributes.get("T")]
            period[2] = stamp if period[2] is None else min(period[2], stamp)
        else:
            timing[set_id][2:] = [frame, attributes.get("T")]
        if team == "BALL":
            key = section, frame
            if key in ball_rows:
                raise ValueError("Duplicate Sportec ball frame")
            ball_rows[key] = _geometry(attributes)[:2]
            target_rows[key] = (_label(attributes), _status_target(attributes))
        else:
            player = context.get("PersonId")
            if not player:
                raise ValueError("Sportec player FrameSet needs a PersonId")
            players.add((team, player))
    if not periods:
        raise ValueError("Sportec tracking contains no recognised match frames")

    rates = []
    for first_n, first_t, last_n, last_t in timing.values():
        if last_n > first_n:
            elapsed = (_time(last_t) - first_t).total_seconds()
            if elapsed <= 0:
                raise ValueError("Sportec timestamps must advance with frame IDs")
            rates.append((last_n - first_n) / elapsed)
    if not rates:
        raise ValueError("At least two timestamped Sportec frames are needed to infer fps")
    fps = float(np.median(rates))
    if not np.allclose(rates, fps, rtol=1e-5, atol=1e-6):
        raise ValueError("Sportec FrameSets disagree on the frame rate")

    blocks, offsets, total = [], {}, 0
    for section in sorted(periods, key=lambda s: periods[s][2]):
        first, last, _ = periods[section]
        offsets[section] = total - first
        blocks.append((total, total + last - first + 1))
        total = blocks[-1][1]
    ball = np.full((total, 2), np.nan)
    nearest = np.full((total, 2), np.inf)
    counts = np.zeros((total, 2), dtype=np.int16)
    labels = np.full(total, "X", dtype="U1")
    status = np.full(total, -1, dtype=np.int8)
    origin = np.asarray(pitch) / 2
    for (section, frame), xy in ball_rows.items():
        row = offsets[section] + frame
        if np.isfinite(xy).all():
            ball[row] = np.asarray(xy) + origin
    for (section, frame), (label, state) in target_rows.items():
        row = offsets[section] + frame
        labels[row] = label
        status[row] = state

    slots = {key: index for index, key in enumerate(sorted(players))}
    speeds = np.full((total, len(slots)), np.nan)
    player_xy = np.full((total, len(slots), 2), np.nan)
    seen = np.zeros((total, len(slots)), dtype=bool)
    # Pass two reads only player geometry and speed into predictor features.
    for _, context, attributes in _frames(raw_path):
        team = context.get("TeamId")
        if team not in teams:
            continue
        row = offsets[context["GameSection"]] + int(attributes["N"])
        slot = slots[(team, context["PersonId"])]
        if seen[row, slot]:
            raise ValueError("Duplicate Sportec player frame")
        seen[row, slot] = True
        x, y, speed = _geometry(attributes)
        if np.isfinite([x, y]).all():
            player_xy[row, slot] = np.array([x, y]) + origin
            side = teams[team]
            counts[row, side] += 1
            if np.isfinite(speed) and speed >= 0:
                speeds[row, slot] = speed / 3.6
            if np.isfinite(ball[row]).all():
                distance = np.linalg.norm(np.array([x, y]) + origin - ball[row])
                nearest[row, side] = min(nearest[row, side], distance)
    median_speed = np.full(total, np.nan)
    valid_speed = np.isfinite(speeds).any(axis=1)
    if valid_speed.any():
        median_speed[valid_speed] = np.nanmedian(speeds[valid_speed], axis=1)
    slot_teams = np.array([teams[team] for team, _ in sorted(players)], dtype=np.int8)
    summary = summarize_player_frames(player_xy, speeds, slot_teams, pitch)
    features = Features(ball, nearest, median_speed, counts, fps, pitch, summary)
    if include_status:
        return features, labels, blocks, status
    return features, labels, blocks
