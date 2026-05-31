from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable

import numpy as np

from entrance_counter.config import CountingConfig, Line, Point


@dataclass
class TrackState:
    points: Deque[Point] = field(default_factory=lambda: deque(maxlen=32))
    sides: Deque[float] = field(default_factory=lambda: deque(maxlen=32))
    first_frame: int | None = None
    first_point: Point | None = None
    first_side: float | None = None
    last_count_frame: int = -10_000
    last_direction: str | None = None


def bottom_center_xy(xyxy: Iterable[float]) -> Point:
    x1, _y1, x2, y2 = map(float, xyxy)
    return ((x1 + x2) / 2.0, y2)


def signed_line_side(point: Point, line: Line) -> float:
    (x1, y1), (x2, y2) = line
    x, y = point
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def normalized_line_side(point: Point, line: Line) -> float:
    (x1, y1), (x2, y2) = line
    length = math.hypot(x2 - x1, y2 - y1)
    return signed_line_side(point, line) / max(length, 1.0)


def projection_t(point: Point, line: Line) -> float:
    (x1, y1), (x2, y2) = line
    px, py = point
    vx, vy = x2 - x1, y2 - y1
    denom = vx * vx + vy * vy
    if denom <= 0:
        return 0.0
    return ((px - x1) * vx + (py - y1) * vy) / denom


def orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def movement_direction(prev_point: Point, curr_point: Point, line: Line, config: CountingConfig) -> str:
    (x1, y1), (x2, y2) = line
    line_dx, line_dy = x2 - x1, y2 - y1
    normal = np.array([-line_dy, line_dx], dtype=float)
    motion = np.array([curr_point[0] - prev_point[0], curr_point[1] - prev_point[1]], dtype=float)
    positive = float(motion @ normal) >= 0
    if config.invert_directions:
        positive = not positive
    return config.positive_direction_label if positive else config.negative_direction_label


def eligible_crossing(
    prev_point: Point,
    curr_point: Point,
    prev_side: float,
    curr_side: float,
    line: Line,
    config: CountingConfig,
) -> bool:
    if abs(prev_side) < config.min_side_abs and abs(curr_side) < config.min_side_abs:
        return False

    if prev_side == 0 or curr_side == 0:
        changed_side = True
    else:
        changed_side = (prev_side > 0) != (curr_side > 0)
    if not changed_side:
        return False

    displacement = math.hypot(curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
    if displacement < config.min_displacement_px:
        return False

    if segments_intersect(prev_point, curr_point, line[0], line[1]):
        return True

    t0 = projection_t(prev_point, line)
    t1 = projection_t(curr_point, line)
    return (max(t0, t1) >= -0.05) and (min(t0, t1) <= 1.05)


def maybe_late_init_crossing(
    state: TrackState,
    curr_point: Point,
    curr_side: float,
    frame_idx: int,
    line: Line,
    config: CountingConfig,
) -> str | None:
    if state.first_frame is None or state.first_point is None or state.first_side is None:
        return None

    age = frame_idx - state.first_frame
    if age > config.late_init_max_age_frames:
        return None
    if abs(state.first_side) > config.late_init_buffer_px:
        return None
    if abs(curr_side) <= config.late_init_buffer_px:
        return None

    displacement = math.hypot(curr_point[0] - state.first_point[0], curr_point[1] - state.first_point[1])
    if displacement < max(config.min_displacement_px * 2.0, 42.0):
        return None

    t = projection_t(state.first_point, line)
    if not (-0.08 <= t <= 1.08):
        return None

    return movement_direction(state.first_point, curr_point, line, config)


def count_event_allowed(state: TrackState, frame_idx: int, config: CountingConfig) -> bool:
    return frame_idx - state.last_count_frame >= config.count_cooldown_frames
