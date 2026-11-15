"""Transparent, untuned possession evidence and a four-state sequence decoder.

States are H (home control), A (away control), D (conservative dead-ball
evidence), and X (unknown). All constants are engineering priors, not estimates
from the challenge match. Emissions are rates integrated over frame duration;
transition costs penalize a change once, independently of the frame rate.
"""

from dataclasses import dataclass, fields

import numpy as np

from .tracking import Features


STATES = np.array(["H", "A", "D", "X"], dtype="U1")


@dataclass(frozen=True)
class Config:
    """Exposed engineering priors; no match-specific calibration is performed."""

    control_radius_m: float = 8.0
    advantage_scale_m: float = 3.0
    minimum_control_support: float = 0.15
    memory_decay_s: float = 2.0
    max_memory_s: float = 5.0
    ball_jump_mps: float = 45.0
    outside_margin_m: float = 0.8
    outside_confirm_s: float = 0.8
    outside_ramp_s: float = 0.4
    control_weight: float = 3.0
    dead_weight: float = 4.0
    unknown_weight: float = 2.0
    control_switch_cost: float = 1.2
    live_dead_switch_cost: float = 0.6
    unknown_switch_cost: float = 0.7
    tie_tolerance: float = 1e-9

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not np.isscalar(value) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{field.name} must be a finite positive number")
            try:
                valid = np.isfinite(value) and 0 < value < 1e6
            except TypeError:
                valid = False
            if not valid:
                raise ValueError(f"{field.name} must be finite, positive, and below 1e6")
        if self.minimum_control_support > 1:
            raise ValueError("minimum_control_support must be at most one")
        if self.memory_decay_s > self.max_memory_s:
            raise ValueError("memory_decay_s must not exceed max_memory_s")
        if self.tie_tolerance > 1e-3:
            raise ValueError("tie_tolerance must be at most 1e-3")


def _validate(features):
    ball = np.asarray(features.ball_xy_m, dtype=float)
    nearest = np.asarray(features.nearest_m, dtype=float)
    counts = np.asarray(features.player_count, dtype=float)
    speeds = np.asarray(features.player_speed_mps, dtype=float)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError("ball_xy_m must have shape (N, 2)")
    n = len(ball)
    if nearest.shape != (n, 2) or counts.shape != (n, 2) or speeds.shape != (n,):
        raise ValueError("Feature arrays must share the same frame count and declared shapes")
    fps = float(features.fps)
    pitch = np.asarray(features.pitch, dtype=float)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("pitch must contain two finite positive dimensions")
    if np.isinf(ball).any() or np.isinf(speeds).any():
        raise ValueError("Ball positions and player speeds may be NaN, but not infinite")
    if (nearest < 0).any() or (speeds < 0).any():
        raise ValueError("Distances and player speeds cannot be negative")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("player_count must contain finite nonnegative integers")
    return ball, nearest, counts, fps, pitch


def _transition(config, unknown_previous, unknown_current):
    """Waive uncertainty-boundary costs so unsupported prefixes cannot inherit H/A."""
    switch = config.control_switch_cost
    dead = config.live_dead_switch_cost
    unknown = config.unknown_switch_cost
    costs = np.array([[0, switch, dead, unknown],
                      [switch, 0, dead, unknown],
                      [dead, dead, 0, unknown],
                      [unknown, unknown, unknown, 0]], dtype=float)
    if unknown_previous:
        costs[3, :] = 0
    if unknown_current:
        costs[:, 3] = 0
    return costs


def _viterbi(emission, unknown, dt, config):
    """Minimum path costs with abstention wherever optimal paths disagree.

    Forward/backward min-cost recurrences identify every state compatible with
    an optimal path. A numerical tie becomes X rather than selecting the first
    team in an array. This abstention overlay preserves home/away symmetry;
    tied output frames need not themselves form one minimum-cost path.
    """
    n = len(emission)
    if not n:
        return np.empty(0, dtype="U1"), np.empty(0, dtype=bool)
    forward = np.empty((n, 4), dtype=float)
    backward = np.zeros((n, 4), dtype=float)
    forward[0] = emission[0] * dt
    forward[0] -= forward[0].min()
    for i in range(1, n):
        transition = _transition(config, unknown[i - 1], unknown[i])
        forward[i] = np.min(forward[i - 1, :, None] + transition, axis=0) + emission[i] * dt
        forward[i] -= forward[i].min()
    for i in range(n - 2, -1, -1):
        transition = _transition(config, unknown[i], unknown[i + 1])
        backward[i] = np.min(transition + emission[i + 1] * dt + backward[i + 1], axis=1)
        backward[i] -= backward[i].min()
    marginal = forward + backward
    marginal -= marginal.min(axis=1, keepdims=True)
    tied = np.count_nonzero(marginal <= config.tie_tolerance, axis=1) > 1
    labels = STATES[np.argmin(marginal, axis=1)]
    labels[tied] = "X"
    return labels, tied


def decode(features: Features, config: Config = Config()):
    """Decode one independent, ordered tracking block, returning labels/evidence.

    Call separately for each official playing period. Evidence uses only the
    current and earlier observations; the sequence objective uses the whole
    supplied block. Positive control evidence favors H, negative favors A.
    Both teams must be observed with finite distances to refresh team support.
    Support decays exponentially and is discarded after ``max_memory_s``;
    it is never refreshed by the decoder's own inferred states.

    D requires a sustained, credible ball observation outside the pitch plus
    a safety margin. Missing ball, distant players, or a stationary ball do
    not supply dead-ball evidence. This deliberately misses many stoppages.
    """
    if not isinstance(config, Config):
        raise TypeError("config must be a Config instance")
    ball, nearest, counts, fps, pitch = _validate(features)
    n = len(ball)
    dt = 1.0 / fps
    observed = np.isfinite(ball).all(axis=1)
    quality = observed.astype(float)
    ball_speed = np.zeros(n, dtype=float)
    speed_observed = np.zeros(n, dtype=bool)
    if n > 1:
        adjacent = observed[1:] & observed[:-1]
        with np.errstate(over="ignore", invalid="ignore"):
            displacement = ball[1:] - ball[:-1]
            speed = np.hypot(displacement[:, 0], displacement[:, 1]) * fps
        speed_observed[1:] = adjacent & np.isfinite(speed)
        ball_speed[1:] = np.where(speed_observed[1:], speed, 0)
        rejected = adjacent & (~np.isfinite(speed) | (speed > config.ball_jump_mps))
        quality[1:][rejected] = 0

    both_visible = (counts > 0).all(axis=1)
    comparable = both_visible & np.isfinite(nearest).all(axis=1) & observed
    advantage = np.zeros(n, dtype=float)
    proximity = np.zeros(n, dtype=float)
    # Clipping before subtraction also makes very large finite coordinates safe.
    distance_limit = config.control_radius_m + 20 * config.advantage_scale_m
    clipped = np.clip(nearest[comparable], 0, distance_limit)
    advantage[comparable] = np.tanh((clipped[:, 1] - clipped[:, 0]) / config.advantage_scale_m)
    proximity[comparable] = np.clip(1 - np.min(clipped, axis=1) / config.control_radius_m, 0, 1)

    outside = np.zeros(n, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        distance = np.maximum.reduce((-ball[:, 0], ball[:, 0] - pitch[0],
                                      -ball[:, 1], ball[:, 1] - pitch[1], np.zeros(n)))
    outside[observed] = distance[observed]
    clearly_outside = (outside > config.outside_margin_m) & (quality > 0)
    direct_control = advantage * proximity * quality
    direct_control[outside > config.outside_margin_m] = 0
    direct_support = np.abs(direct_control)

    outside_duration = np.zeros(n, dtype=float)
    dead_evidence = np.zeros(n, dtype=float)
    control_evidence = np.zeros(n, dtype=float)
    freshness = np.zeros(n, dtype=float)
    # The finite cap means "older than the support horizon", including no history.
    age = np.full(n, config.max_memory_s + dt, dtype=float)
    last_supported = None
    last_control = 0.0
    outside_run = 0.0
    for i in range(n):
        outside_run = outside_run + dt if clearly_outside[i] else 0.0
        outside_duration[i] = outside_run
        dead_evidence[i] = np.clip((outside_run - config.outside_confirm_s) / config.outside_ramp_s, 0, 1)
        if dead_evidence[i] > 0:
            last_supported = None
            last_control = 0.0
        elif direct_support[i] >= config.minimum_control_support:
            last_supported = i
            last_control = direct_control[i]
        if last_supported is not None:
            elapsed = (i - last_supported) * dt
            age[i] = min(elapsed, config.max_memory_s + dt)
            if elapsed <= config.max_memory_s:
                freshness[i] = np.exp(-elapsed / config.memory_decay_s)
                control_evidence[i] = last_control * freshness[i]

    unknown_evidence = ((freshness == 0) & (dead_evidence == 0)).astype(float)
    emission = np.column_stack((
        -config.control_weight * control_evidence + config.dead_weight * dead_evidence,
        config.control_weight * control_evidence + config.dead_weight * dead_evidence,
        config.dead_weight * (1 - 2 * dead_evidence),
        -config.unknown_weight * unknown_evidence,
    ))
    labels, tied = _viterbi(emission, unknown_evidence > 0, dt, config)
    evidence = {
        "ball_observed": observed,
        "ball_quality": quality,
        "ball_speed_mps": ball_speed,
        "ball_speed_observed": speed_observed,
        "both_teams_visible": both_visible,
        "proximity_confidence": proximity,
        "control_advantage": advantage,
        "direct_support": direct_support,
        "control_evidence": control_evidence,
        "support_age_s": age,
        "support_freshness": freshness,
        "outside_distance_m": outside,
        "outside_duration_s": outside_duration,
        "dead_evidence": dead_evidence,
        "unknown_evidence": unknown_evidence,
        "emission_cost": emission,
        "ambiguous_optimum": tied,
    }
    return labels, evidence
