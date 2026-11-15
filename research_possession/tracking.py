"""Strict, local-only ingestion of public-format automatic tracking."""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .motion import PLAYER_SUMMARY_NAMES, summarize_player_frames


@dataclass
class Features:
    ball_xy_m: np.ndarray
    nearest_m: np.ndarray
    player_speed_mps: np.ndarray
    player_count: np.ndarray
    fps: float
    pitch: tuple[float, float]
    player_summary: np.ndarray | None = None

    def __len__(self):
        return len(self.ball_xy_m)

    def sliced(self, start, end):
        return Features(self.ball_xy_m[start:end], self.nearest_m[start:end],
                        self.player_speed_mps[start:end], self.player_count[start:end],
                        self.fps, self.pitch,
                        None if self.player_summary is None else self.player_summary[start:end])


def read_teams(path):
    root = ET.parse(path).getroot()
    teams = [team.get("id") for team in root.iter("Team")]
    if len(teams) != 2 or len(set(teams)) != 2 or None in teams:
        raise ValueError("Expected exactly two distinct metadata teams")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots or any(team not in teams for team in slots):
        raise ValueError("Each track slot must belong to a declared team")
    return np.array([teams.index(team) for team in slots], dtype=np.int8)


def read_atd(meta_path, raw_path, n_frames, fps, pitch):
    """Raw ATD is one-based tactical-video time; no corrected-GT offsets apply.

    Slots are tracklets: only within-frame team geometry is used. No temporal
    differentiation of individual slots is performed. DATA.md specifies m/s.
    Error messages deliberately omit restricted input contents.
    """
    teams = read_teams(meta_path)
    ball = np.full((n_frames, 2), np.nan)
    nearest = np.full((n_frames, 2), np.inf)
    speed = np.full(n_frames, np.nan)
    counts = np.zeros((n_frames, 2), dtype=np.int16)
    summary = np.full((n_frames, len(PLAYER_SUMMARY_NAMES)), np.nan)
    chunk_size = 2048
    player_chunk = np.full((min(chunk_size, n_frames), len(teams), 3), np.nan)
    scale = np.array(pitch, dtype=float)
    observed = 0
    with Path(raw_path).open(encoding="utf-8-sig") as handle:
        for row, line in enumerate(handle):
            if row >= n_frames:
                raise ValueError("ATD has more frames than declared")
            fields = line.strip().split(":")
            if len(fields) != 3:
                raise ValueError(f"Malformed ATD structure at row {row + 1}")
            try:
                frame = int(fields[0])
                groups = fields[1].split(";")
                if groups and groups[-1] == "":
                    groups.pop()
                if frame != row + 1 or len(groups) != len(teams):
                    raise ValueError
                players = np.array([[float(x) for x in group.split(",")]
                                    for group in groups])
                sphere = np.array([float(x) for x in fields[2].split(",")])
                if players.shape != (len(teams), 3) or sphere.shape != (3,):
                    raise ValueError
            except (ValueError, IndexError):
                raise ValueError(f"Invalid ATD index, slot count or numeric format at row {row + 1}") from None
            if np.isinf(players).any() or np.isinf(sphere).any():
                raise ValueError(f"Infinite ATD value at row {row + 1}")
            player_chunk[row % chunk_size] = players
            visible = np.isfinite(players[:, :2]).all(axis=1)
            for side in (0, 1):
                counts[row, side] = np.count_nonzero(visible & (teams == side))
            speeds = players[visible, 2]
            speeds = speeds[np.isfinite(speeds) & (speeds >= 0)]
            if len(speeds):
                speed[row] = np.median(speeds)
            if np.isfinite(sphere[:2]).all():
                ball[row] = sphere[:2] * scale
                distances = np.linalg.norm(players[:, :2] * scale - ball[row], axis=1)
                for side in (0, 1):
                    selected = visible & (teams == side)
                    if selected.any():
                        nearest[row, side] = distances[selected].min()
            observed += 1
            if observed % chunk_size == 0 or observed == n_frames:
                chunk_count = (row % chunk_size) + 1
                chunk = player_chunk[:chunk_count]
                summary[observed - chunk_count:observed] = summarize_player_frames(
                    chunk[:, :, :2] * scale, chunk[:, :, 2], teams, pitch)
    if observed != n_frames:
        raise ValueError(f"ATD frame count mismatch: expected {n_frames}, got {observed}")
    return Features(ball, nearest, speed, counts, float(fps), tuple(pitch), summary)
