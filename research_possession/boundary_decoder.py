"""Factorized status/control inference with evidence-adaptive switching costs.

This is a project-specific combination of existing ideas, not a claimed new
class of probabilistic model. Model outputs and min-cost gaps are uncalibrated.
"""

from dataclasses import dataclass

import numpy as np

from .decoder import Config, STATES, decode


@dataclass(frozen=True)
class BoundaryConfig:
    status_weight: float = 2.0
    live_dead_switch_cost: float = 0.8
    control_switch_cost: float = 1.2
    boundary_discount: float = 0.65
    missing_control_penalty: float = 0.5
    control_gain: float = 3.0
    unknown_cost: float = 3.0
    adaptive: bool = True

    def __post_init__(self):
        for name in ("status_weight", "live_dead_switch_cost", "control_switch_cost",
                     "missing_control_penalty", "control_gain", "unknown_cost"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.boundary_discount) or not 0 <= self.boundary_discount < 1:
            raise ValueError("boundary_discount must be in [0,1)")


def _path(emission, ld_cost, team_cost, fps, unsupported=None):
    n = len(emission)
    if n == 0:
        return np.empty(0, dtype="U1"), np.empty((0, 4))
    fwd, bwd = np.zeros((n, 4)), np.zeros((n, 4))
    if unsupported is None:
        unsupported = np.zeros(n, dtype=bool)
    # Two supported X edges must never undercut any direct state switch,
    # including when adaptive costs differ between adjacent frames.
    x_switch = 0.5 * max(float(np.max(ld_cost)), float(np.max(team_cost)))

    def matrix(i):
        ld, team = ld_cost[i], team_cost[i]
        matrix = np.array([[0, team, ld, x_switch], [team, 0, ld, x_switch],
                           [ld, ld, 0, x_switch], [x_switch, x_switch, x_switch, 0]])
        if unsupported[i - 1]:
            matrix[3, :] = 0
        if unsupported[i]:
            matrix[:, 3] = 0
        return matrix

    fwd[0] = emission[0] / fps
    fwd[0] -= fwd[0].min()
    for i in range(1, n):
        fwd[i] = np.min(fwd[i - 1, :, None] + matrix(i), axis=0) + emission[i] / fps
        fwd[i] -= fwd[i].min()
    for i in range(n - 2, -1, -1):
        bwd[i] = np.min(matrix(i + 1) + emission[i + 1] / fps + bwd[i + 1], axis=1)
        bwd[i] -= bwd[i].min()
    marginal = fwd + bwd
    marginal -= marginal.min(axis=1, keepdims=True)
    labels = STATES[np.argmin(marginal, axis=1)]
    labels[(marginal < 1e-9).sum(axis=1) > 1] = "X"
    return labels, marginal


def decode_boundary(features, dead_score, boundary_strength, config=BoundaryConfig()):
    """One period only. No external targets enter this inference function."""
    n = len(features)
    dead_score = np.asarray(dead_score, dtype=float)
    boundary_strength = np.asarray(boundary_strength, dtype=float)
    for name, values in (("dead_score", dead_score), ("boundary_strength", boundary_strength)):
        if values.shape != (n,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{name} must be a finite [0,1] vector of length N")
    _, evidence = decode(features, Config())
    control = evidence["control_evidence"]
    # Boundary geometry can strengthen D, but missing ball never does so.
    status_quality = np.clip(np.min(features.player_count, axis=1) / 4, 0, 1)
    supported_status = 0.5 + status_quality * (dead_score - 0.5)
    p_dead = np.clip(np.maximum(supported_status, 0.95 * evidence["dead_evidence"]), 0.01, 0.99)
    p_home_live = np.clip(1 / (1 + np.exp(-config.control_gain * control)), 0.01, 0.99)
    emission = np.column_stack((
        -config.status_weight * np.log1p(-p_dead) - np.log(p_home_live) - np.log(2),
        -config.status_weight * np.log1p(-p_dead) - np.log1p(-p_home_live) - np.log(2),
        -config.status_weight * np.log(p_dead),
        np.full(n, config.unknown_cost),
    ))
    # X is available when neither side has control support, without converting
    # missing observations into D. Strong independently learned D can still win.
    unknown = evidence["support_freshness"] == 0
    emission[unknown, 3] = 1.2
    # Conditional team information is centred at its neutral 50/50 value above;
    # otherwise no team information would give D twice the mass of either team.
    unsupported = unknown & (status_quality == 0) & (evidence["dead_evidence"] == 0)
    emission[unsupported, 3] = 0
    ld_cost = np.full(n, config.live_dead_switch_cost)
    team_cost = np.full(n, config.control_switch_cost)
    if config.adaptive and n:
        radius = max(1, round(0.4 * features.fps))
        index = np.arange(n)
        left, right = np.maximum(0, index - radius), np.minimum(n - 1, index + radius)
        status_change = np.abs(p_dead[right] - p_dead[left])
        # A class-balanced head alone cannot open a cheap transition: it must
        # coincide with an actual change in the independently predicted status.
        gate = np.clip(status_change * boundary_strength, 0, 1)
        ld_cost *= 1 - config.boundary_discount * gate
        control_change = np.clip(np.abs(control[right] - control[left]) / 2, 0, 1)
        direct = evidence["ball_quality"] * evidence["direct_support"]
        team_cost *= 1 - config.boundary_discount * control_change * direct
        team_cost += config.missing_control_penalty * (1 - evidence["ball_quality"])
    labels, marginal = _path(emission, ld_cost, team_cost, features.fps, unsupported)
    evidence.update({"learned_dead_score": dead_score, "boundary_strength": boundary_strength,
                     "fused_dead_score": p_dead, "live_dead_switch_cost": ld_cost,
                     "status_quality": status_quality, "unsupported": unsupported,
                     "team_switch_cost": team_cost, "emission_cost": emission,
                     "min_marginal_cost": marginal})
    return labels, evidence


def decode_periods(features, dead_score, boundary_strength, blocks, config):
    labels = np.full(len(features), "X", dtype="U1")
    for start, end in blocks:
        labels[start:end], _ = decode_boundary(features.sliced(start, end),
                                               dead_score[start:end], boundary_strength[start:end], config)
    return labels
