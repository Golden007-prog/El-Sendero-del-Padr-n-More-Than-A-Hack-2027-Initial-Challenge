"""Fit the ownership and boundary heads bundled with TRACE on seven external matches.

Not used at inference. It is here so the bundled heads can be inspected and refitted.
The two files in models/ are the output of this script.

Training matches (none is distributed in this repository):

  Sportec Open DFL   J03WOH, J03WMX, J03WN1, J03WOY. Targets are the provider's
                     BallStatus / BallPossession fields mapped to H/A/D/X.
  Metrica Sports     Sample Games 1 to 3 from github.com/metrica-sports/sample-data
                     (commit e706dd506b360d69d9d123d5b8026e7294b13996). Targets are
                     derived from Metrica's event data by the fixed rule in
                     `event_labels`: possession changes only at a RECOVERY or
                     SET PIECE of the other team, and the ball is dead from the end
                     of the event before each SET PIECE to that set piece.

Status, timing and interval-quality models are not refitted here.

    python -I -B training/fit_heads_seven_matches.py --sportec DIR --metrica DIR --out DIR

  --sportec  one <match>.npz per Sportec match with the predictor fields ball_xy_m,
             nearest_m, player_speed_mps, player_count, player_summary, fps, pitch,
             plus `labels` (H/A/D/X per frame) and `blocks` (period frame ranges)
  --metrica  folder holding the sample tracking and event files under their
             original names
  --out      new folder for owner-model.joblib, full330-boundary.joblib and a summary
"""
import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptive_turnovers import (BoundaryModel, TARGET_RADIUS_SECONDS, VARIANTS, _balanced_weight, _labels,  # noqa: E402
                                _sample_indices, boundary_targets)
from candidate_pipeline import HEAD_TRAIN_IDS, control_evidence  # noqa: E402
from owner_model import FEATURE_NAMES, MODEL_SETTINGS, THREAD_LIMIT, OwnershipModel, feature_design, swap_design  # noqa: E402
from research_possession.motion import summarize_player_frames  # noqa: E402
from research_possession.tracking import Features  # noqa: E402

FPS, PITCH = 25.0, (105.0, 68.0)
SPORTEC = HEAD_TRAIN_IDS[:4]
SAMPLE_IDS = HEAD_TRAIN_IDS[4:]


def load_sportec(folder, match):
    with np.load(Path(folder) / f"{match}.npz", allow_pickle=False) as d:
        features = Features(d["ball_xy_m"], d["nearest_m"], d["player_speed_mps"], d["player_count"], float(d["fps"]),
                            tuple(float(v) for v in d["pitch"]), d["player_summary"])
        return features, d["labels"], [tuple(map(int, row)) for row in d["blocks"]]


def load_events(folder, game):
    """Event table with team mapped to H/A and times on the 25 Hz tracking clock."""
    folder = Path(folder)
    if game < 3:
        c = pd.read_csv(folder / f"Sample_Game_{game}_RawEventsData.csv")
        d = pd.DataFrame({"team": c["Team"].map({"Home": "H", "Away": "A"}), "type": c["Type"], "period": c["Period"],
                          "sf": np.rint(c["Start Time [s]"] * FPS).astype(int), "ef": np.rint(c["End Time [s]"] * FPS).astype(int)})
        return d
    j = json.loads((folder / "Sample_Game_3_events.json").read_text(encoding="utf-8"))["data"]
    d = pd.DataFrame({"team": [{"Team A": "H", "Team B": "A"}[e["team"]["name"]] for e in j],
                      "type": [e["type"]["name"] for e in j], "period": [e["period"] for e in j],
                      "sf": np.rint(np.array([e["start"]["time"] for e in j]) * FPS).astype(int),
                      "ef": np.rint(np.array([e["end"]["time"] for e in j]) * FPS).astype(int)})
    return d[d["type"] != "CARRY"].reset_index(drop=True)      # CARRY exists only in game 3


def event_labels(events, n_rows):
    """Per-row H/A/D/X labels. Row r is tracking frame r + 1, at (r + 1) / 25 seconds."""
    size = max(n_rows + 2, int(events["ef"].max()) + 2)
    labels = np.full(size, "X", dtype="<U1")
    blocks = []
    for period in sorted(events["period"].unique()):
        idx = np.flatnonzero((events["period"] == period).to_numpy())
        pieces = [i for i in idx if events["type"].iat[i] == "SET PIECE"]
        if not pieces or pieces[0] != idx[0]:
            raise ValueError("a period must begin with a set piece")
        p_start, p_end = int(events["sf"].iat[idx[0]]), int(events["ef"].iloc[idx].max())
        blocks.append((p_start, p_end))
        labels[p_start:p_end] = "D"
        for k, a in enumerate(pieces):
            b = pieces[k + 1] if k + 1 < len(pieces) else idx[-1] + 1
            lo = int(events["sf"].iat[a])
            hi = max(int(events["ef"].iat[b - 1]) if k + 1 < len(pieces) else p_end, lo)
            owner = None
            for i in range(a, b):
                kind, team = events["type"].iat[i], events["team"].iat[i]
                if owner is None or (kind in ("RECOVERY", "SET PIECE") and team != owner):
                    owner = team
                    labels[min(int(events["sf"].iat[i]), hi):hi] = team
    return labels[1:n_rows + 1].copy(), [(max(0, s - 1), min(n_rows, e - 1)) for s, e in blocks]


def load_tracking(folder, game):
    """Return normalised player positions (frames, players, 2), ball (frames, 2) and team per player."""
    folder = Path(folder)
    if game < 3:
        sides, ball = [], None
        for name in ("Home", "Away"):
            df = pd.read_csv(folder / f"Sample_Game_{game}_RawTrackingData_{name}_Team.csv", skiprows=2)
            values = df.iloc[:, 3:].to_numpy(dtype=float)
            sides.append(values[:, :-2].reshape(len(df), -1, 2))
            ball = values[:, -2:]
        teams = np.r_[np.zeros(sides[0].shape[1], np.int8), np.ones(sides[1].shape[1], np.int8)]
        return np.concatenate(sides, axis=1), ball, teams
    root = ET.parse(folder / "Sample_Game_3_metadata.xml").getroot()
    tag = lambda el: el.tag.split("}")[-1]
    team_of = {el.get("id"): el.get("teamId") for el in root.iter() if tag(el) == "Player"}
    channel = {el.get("id"): el.get("playerId") for el in root.iter() if tag(el) == "PlayerChannel"}
    team_ids = [el.get("id") for el in root.iter() if tag(el) == "Team"]
    players = sorted(team_of, key=lambda p: (team_ids.index(team_of[p]), p))
    index = {p: i for i, p in enumerate(players)}
    specs = []
    for el in root.iter():
        if tag(el) == "DataFormatSpecification":
            outer = [c for c in el if tag(c) == "SplitRegister" and c.get("separator") == ";"][0]
            groups = [index[channel[[r.get("playerChannelId") for r in g if tag(r) == "PlayerChannelRef"][0]]] for g in outer]
            specs.append((int(el.get("startFrame")), int(el.get("endFrame")), groups))
    lines = (folder / "Sample_Game_3_tracking.txt").read_text(encoding="utf-8").splitlines()
    xy = np.full((len(lines), len(players), 2), np.nan)
    ball = np.full((len(lines), 2), np.nan)
    for row, line in enumerate(lines):
        frame, body, sphere = line.split(":")
        groups = next(g for s, e, g in specs if s <= int(frame) <= e)
        xy[row, groups] = np.array([p.split(",") for p in body.split(";")], dtype=float)
        ball[row] = np.array(sphere.split(",")[:2], dtype=float)
    return xy, ball, np.array([team_ids.index(team_of[p]) for p in players], dtype=np.int8)


def build_features(xy, ball, teams, blocks):
    """Same predictor fields the supplied-match reader produces; speeds come from positions."""
    xy_m, ball_m = xy * np.array(PITCH), ball * np.array(PITCH)
    n = len(xy_m)
    speed = np.full(xy_m.shape[:2], np.nan)
    for start, end in blocks:                       # central difference over +/- 2 frames, inside a period
        if end - start > 4:
            speed[start + 2:end - 2] = np.linalg.norm(xy_m[start + 4:end] - xy_m[start:end - 4], axis=2) / (4 / FPS)
    speed = np.where(speed > 12.0, 12.0, speed)
    visible = np.isfinite(xy_m).all(axis=2)
    counts = np.stack([(visible & (teams == s)).sum(axis=1) for s in (0, 1)], axis=1).astype(np.int16)
    with np.errstate(all="ignore"):
        median_speed = np.nanmedian(np.where(visible & np.isfinite(speed), speed, np.nan), axis=1)
    have_ball = np.isfinite(ball_m).all(axis=1)
    distance = np.linalg.norm(xy_m - ball_m[:, None, :], axis=2)
    nearest = np.full((n, 2), np.inf)
    for s in (0, 1):
        nearest[:, s] = np.where(have_ball, np.where(visible & (teams == s)[None, :], distance, np.inf).min(axis=1), np.inf)
    summary = np.concatenate([summarize_player_frames(xy_m[i:i + 2048], speed[i:i + 2048], teams, PITCH) for i in range(0, n, 2048)])
    return Features(np.where(have_ball[:, None], ball_m, np.nan), nearest, median_speed, counts, FPS, PITCH, summary)


def training_rows(features, labels, blocks):
    """Sampled 5 Hz rows for both heads, with team-swap augmentation and per-match class balance."""
    design, names = feature_design(features, control_evidence(features, blocks), blocks)
    if names != FEATURE_NAMES:
        raise ValueError("feature schema differs")
    labels = _labels(labels, len(design))
    sampled, _, periods = _sample_indices(features, blocks, len(design))
    rows = {}
    for head in ("owner", "boundary"):
        if head == "owner":
            idx = sampled[np.isin(labels[sampled], ["H", "A"])]
            target = (labels[idx] == "H").astype(np.int8)
            mirrored = 1 - target
        else:
            marks = boundary_targets(labels, periods, features.fps)
            idx = sampled[marks[sampled] >= 0]
            target = marks[idx]
            mirrored = target.copy()
        weight, _ = _balanced_weight(target)
        observed = design[idx]
        rows[head] = (np.concatenate((observed, swap_design(observed))), np.concatenate((target, mirrored)),
                      np.concatenate((weight / 2, weight / 2)), int(len(target)))
    return rows


def fit(rows, ids, head):
    x = np.concatenate([rows[i][head][0] for i in ids])
    y = np.concatenate([rows[i][head][1] for i in ids])
    w = np.concatenate([rows[i][head][2] for i in ids])
    scale = len(w) / w.sum()
    classifier = HistGradientBoostingClassifier(**MODEL_SETTINGS)
    with threadpool_limits(limits=THREAD_LIMIT):
        classifier.fit(x, y, sample_weight=w * scale)
    counts = {i: {"sampled_original_rows": rows[i][head][3], "augmented_rows": 2 * rows[i][head][3]} for i in ids}
    settings = {**MODEL_SETTINGS, "thread_limit": THREAD_LIMIT, "sample_target_hz": 5, "global_weight_scale": float(scale),
                "total_augmented_rows": int(len(y)), "feature_count": len(FEATURE_NAMES), "probabilities_calibrated": False,
                "augmentation": "exact team swap; half weight per copy"}
    if head == "owner":
        return OwnershipModel(classifier, FEATURE_NAMES, list(ids), counts, settings)
    settings.update({"variant": VARIANTS[0], "target_radius_seconds": TARGET_RADIUS_SECONDS,
                     "edge_alignment": "score[t] prices the incoming edge t-1 to t", "dropped_feature_names": []})
    return BoundaryModel(classifier, FEATURE_NAMES, list(ids), counts, settings, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sportec", required=True)
    parser.add_argument("--metrica", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = {}
    for match in SPORTEC:
        rows[match] = training_rows(*load_sportec(args.sportec, match))
    for game, name in enumerate(SAMPLE_IDS, start=1):
        xy, ball, teams = load_tracking(args.metrica, game)
        labels, blocks = event_labels(load_events(args.metrica, game), len(xy))
        rows[name] = training_rows(build_features(xy, ball, teams, blocks), labels, blocks)
    summary = {"train_ids": list(HEAD_TRAIN_IDS), "settings": MODEL_SETTINGS, "heads": {}}
    for head, filename in (("owner", "owner-model.joblib"), ("boundary", "full330-boundary.joblib")):
        model = fit(rows, HEAD_TRAIN_IDS, head)
        joblib.dump(model, out / filename)
        summary["heads"][head] = {"file": filename, "sha256": hashlib.sha256((out / filename).read_bytes()).hexdigest(),
                                  "training_counts": model.training_counts, "total_augmented_rows": model.settings["total_augmented_rows"]}
    (out / "training-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({head: v["sha256"] for head, v in summary["heads"].items()}))


if __name__ == "__main__":
    main()
