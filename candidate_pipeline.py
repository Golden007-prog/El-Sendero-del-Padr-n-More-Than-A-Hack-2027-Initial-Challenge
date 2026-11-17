"""Self-contained frozen v4 -> owner(.9) -> constrained research pipeline."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np

from adaptive_turnovers import BoundaryModel, predict_boundary
from constrained_turnovers import refine
from frozen_helpers import baseline
from owner_model import EVIDENCE_KEYS, OwnershipModel, TRAIN_IDS, _periods, _positive_fps, decode_owner, predict_owner
from research_possession.decoder import decode
from research_possession.topology_decoder_v4 import TopologyConfig, decode_topology_periods
from research_possession.tracking import Features

ROOT = Path(__file__).resolve().parent
FEATURE_KEYS = frozenset(("ball_xy_m", "nearest_m", "player_speed_mps", "player_count", "fps", "pitch",
                          "player_summary", "has_player_summary", "blocks"))
PREDICTION_KEYS = frozenset(("anchor_v4", "baseline_c0", "constrained"))
# The ownership and boundary heads are fitted on the four Sportec matches plus the three
# public Metrica sample matches (training/fit_heads_seven_matches.py).
HEAD_TRAIN_IDS = TRAIN_IDS + ("METRICA_SAMPLE_1", "METRICA_SAMPLE_2", "METRICA_SAMPLE_3")
# The supplied automatic tracking carries a ball track that is smoothed over about half a second; the
# external training matches do not. Two pairs of heads are bundled, one fitted on tracks as recorded and
# one fitted with smoothed tracks added, and the pipeline chooses by measuring the track it is given.
SMOOTHNESS_WINDOW_SECONDS = 0.2
SMOOTHED_BELOW_MPS = 12.0
SMOOTHED_HEAD_WIDTHS = [0, 6]


def ball_velocity_change_p99(ball_xy_m, blocks, fps):
    """99th percentile of the change in ball velocity between two consecutive windows of 0.2 s.

    Taken over frames inside the periods where the ball is tracked at all three window ends. A track as
    recorded gives 16 to 21 m/s on the external matches; one smoothed over half a second gives 5 to 10.
    """
    k = max(1, int(round(SMOOTHNESS_WINDOW_SECONDS * fps)))
    xy = np.asarray(ball_xy_m, dtype=float)
    values = []
    for start, end in blocks:
        segment = xy[start:end]
        if len(segment) <= 2 * k:
            continue
        p0, p1, p2 = segment[:-2 * k], segment[k:-k], segment[2 * k:]
        ok = np.isfinite(p0).all(axis=1) & np.isfinite(p1).all(axis=1) & np.isfinite(p2).all(axis=1)
        values.append(np.linalg.norm((p2[ok] - p1[ok]) - (p1[ok] - p0[ok]), axis=1) * fps / k)
    values = np.concatenate(values) if values else np.zeros(0)
    return float(np.percentile(values, 99)) if len(values) else float("nan")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(values):
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def validate_package():
    manifest = json.loads((ROOT / "artifacts.json").read_text(encoding="utf-8"))
    for name, record in manifest["artifacts"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != record["sha256"]:
            raise ValueError("Packaged source/model identity mismatch: " + name)
    expected = {}
    for filename in ("requirements-lock.txt", "requirements-research-lock.txt"):
        for line in (ROOT / filename).read_text(encoding="utf-8-sig").splitlines():
            if "==" in line and not line.startswith("#"):
                name, version = line.strip().split("==")
                expected[name] = version
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected or platform.python_version() != "3.11.9":
        raise ValueError("Use the exact locks with Python 3.11.9")
    return manifest, {"python": platform.python_version(), **actual}


def load_models():
    models = {name: joblib.load(ROOT / "models" / filename) for name, filename in (
        ("status", "status-model.joblib"), ("timing", "timing-model.joblib"),
        ("quality", "quality-model.joblib"), ("owner", "owner-model.joblib"),
        ("boundary", "full330-boundary.joblib"), ("owner_smoothed", "owner-model-smoothed.joblib"),
        ("boundary_smoothed", "full330-boundary-smoothed.joblib"))}
    for owner, boundary, widths in (("owner", "boundary", None), ("owner_smoothed", "boundary_smoothed", SMOOTHED_HEAD_WIDTHS)):
        if not isinstance(models[owner], OwnershipModel) or not isinstance(models[boundary], BoundaryModel):
            raise ValueError("Unexpected packaged model types")
        for name in (owner, boundary):
            if tuple(models[name].train_ids) != HEAD_TRAIN_IDS or len(models[name].feature_names) != 330:
                raise ValueError("Packaged supervised-head provenance differs")
            if models[name].settings.get("ball_smoothing_frames") != widths:
                raise ValueError("Packaged heads were fitted on a different kind of ball track")
        if models[boundary].use_memory is not True:
            raise ValueError("The candidate requires the full330 boundary head")
    return models


def load_features(path):
    """Only explicit predictor members are accepted; target/status fields fail."""
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != FEATURE_KEYS:
            raise ValueError("Feature archive must contain exactly the documented predictor-only schema")
        fps = _positive_fps(float(data["fps"]))
        ball, nearest = data["ball_xy_m"], data["nearest_m"]
        speed, counts, summary = data["player_speed_mps"], data["player_count"], data["player_summary"]
        pitch = np.asarray(data["pitch"], dtype=float)
        n = len(ball)
        if (ball.shape != (n, 2) or nearest.shape != (n, 2) or speed.shape != (n,)
                or counts.shape != (n, 2) or summary.shape != (n, 18)
                or np.asarray(data["has_player_summary"]).shape != () or not bool(data["has_player_summary"])
                or pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any()):
            raise ValueError("Predictor geometry does not match the frozen schema")
        blocks = _periods(data["blocks"], n)
        features = Features(ball, nearest, speed, counts, fps, tuple(pitch.tolist()), summary)
    return features, blocks


def control_evidence(features, blocks):
    """Exact observation-only evidence construction used by the frozen study."""
    evidence = {}
    for start, end in blocks:
        _, local = decode(features.sliced(start, end))
        for key in EVIDENCE_KEYS:
            if key not in evidence:
                evidence[key] = np.zeros(len(features), dtype=local[key].dtype)
            evidence[key][start:end] = local[key]
    if not blocks:
        evidence = {key: np.zeros(len(features)) for key in EVIDENCE_KEYS}
    return evidence


def infer_v4(features, blocks, models):
    dead, boundaries = models["status"].predict(features, blocks)
    original, evidence = baseline(features, blocks, dead, boundaries)
    predicted, _, _ = decode_topology_periods(
        features, blocks, original, evidence, timing_model=models["timing"],
        quality_model=models["quality"], config=TopologyConfig(false_threshold=.9, timing_scale=.5))
    return predicted


def infer(features, blocks, models, frozen_v4=None):
    """Normal inference recomputes v4; optional anchor is for explicit replay only."""
    anchor = infer_v4(features, blocks, models) if frozen_v4 is None else np.asarray(frozen_v4).copy()
    if anchor.shape != (len(features),) or not np.isin(anchor, list("HADX")).all():
        raise ValueError("v4 anchor shape/alphabet mismatch")
    evidence = control_evidence(features, blocks)
    statistic = ball_velocity_change_p99(features.ball_xy_m, blocks, features.fps)
    smoothed = bool(np.isfinite(statistic) and statistic < SMOOTHED_BELOW_MPS)
    owner, boundary = ("owner_smoothed", "boundary_smoothed") if smoothed else ("owner", "boundary")
    probabilities = predict_owner(models[owner], features, evidence, blocks)
    c0, owner_audit = decode_owner(anchor, probabilities, blocks, features.fps, .9)
    scores = predict_boundary(models[boundary], features, evidence, blocks)
    constrained, audit = refine(c0, probabilities, scores, blocks, features.fps)
    for output in (c0, constrained):
        if not np.array_equal(output == "D", anchor == "D") or not np.array_equal(output == "X", anchor == "X"):
            raise AssertionError("Candidate changed frozen v4 status")
    return {"anchor_v4": anchor, "baseline_c0": c0, "constrained": constrained}, {
        "v4_recomputed": frozen_v4 is None, "owner_switch_cost": .9,
        "ball_track": {"velocity_change_p99_mps": statistic, "smoothed_below_mps": SMOOTHED_BELOW_MPS, "smoothed": smoothed,
                       "heads_used": "fitted with smoothed tracks" if smoothed else "fitted on tracks as recorded"},
        "owner_audit": owner_audit, "constrained_audit": audit,
        "training_or_selection_run": False, "promotion_to_submission": False}


def write_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
