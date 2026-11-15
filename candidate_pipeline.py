"""Self-contained frozen v4 -> owner(.6) -> constrained research pipeline."""
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
        ("boundary", "full330-boundary.joblib"))}
    if not isinstance(models["owner"], OwnershipModel) or not isinstance(models["boundary"], BoundaryModel):
        raise ValueError("Unexpected packaged model types")
    for name in ("owner", "boundary"):
        if tuple(models[name].train_ids) != TRAIN_IDS or len(models[name].feature_names) != 330:
            raise ValueError("Packaged supervised-head provenance differs")
    if models["boundary"].use_memory is not True:
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
    probabilities = predict_owner(models["owner"], features, evidence, blocks)
    c0, owner_audit = decode_owner(anchor, probabilities, blocks, features.fps, .6)
    scores = predict_boundary(models["boundary"], features, evidence, blocks)
    constrained, audit = refine(c0, probabilities, scores, blocks, features.fps)
    for output in (c0, constrained):
        if not np.array_equal(output == "D", anchor == "D") or not np.array_equal(output == "X", anchor == "X"):
            raise AssertionError("Candidate changed frozen v4 status")
    return {"anchor_v4": anchor, "baseline_c0": c0, "constrained": constrained}, {
        "v4_recomputed": frozen_v4 is None, "owner_switch_cost": .6,
        "owner_audit": owner_audit, "constrained_audit": audit,
        "training_or_selection_run": False, "promotion_to_submission": False}


def write_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
