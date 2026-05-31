from collections import deque

import numpy as np

from entrance_counter.config import CountingConfig
from entrance_counter.geometry import (
    TrackState,
    bottom_center_xy,
    count_event_allowed,
    eligible_crossing,
    maybe_late_init_crossing,
    movement_direction,
    normalized_line_side,
    projection_t,
    reset_track_state,
    segments_intersect,
    transform_line,
    transform_point,
)


def test_bottom_center_uses_box_floor_midpoint() -> None:
    assert bottom_center_xy((10, 20, 30, 80)) == (20.0, 80.0)


def test_projection_t_maps_point_to_line_extent() -> None:
    line = ((10, 50), (90, 50))

    assert projection_t((10, 50), line) == 0.0
    assert projection_t((50, 50), line) == 0.5
    assert projection_t((90, 50), line) == 1.0


def test_segments_intersect_detects_crossing_segments() -> None:
    assert segments_intersect((50, 20), (50, 80), (10, 50), (90, 50))
    assert not segments_intersect((50, 20), (60, 30), (10, 50), (90, 50))


def test_eligible_crossing_requires_side_change_and_real_motion() -> None:
    config = CountingConfig(line=((10, 50), (90, 50)))
    prev_point = (50.0, 30.0)
    curr_point = (50.0, 80.0)
    prev_side = normalized_line_side(prev_point, config.line)
    curr_side = normalized_line_side(curr_point, config.line)

    assert eligible_crossing(prev_point, curr_point, prev_side, curr_side, config.line, config)


def test_eligible_crossing_rejects_small_jitter_around_line() -> None:
    config = CountingConfig(line=((10, 50), (90, 50)), min_side_abs=8.0)
    prev_point = (50.0, 49.0)
    curr_point = (50.0, 51.0)
    prev_side = normalized_line_side(prev_point, config.line)
    curr_side = normalized_line_side(curr_point, config.line)

    assert not eligible_crossing(prev_point, curr_point, prev_side, curr_side, config.line, config)


def test_movement_direction_uses_line_normal_and_inversion_flag() -> None:
    normal_config = CountingConfig(line=((10, 50), (90, 50)))
    inverted_config = CountingConfig(line=((10, 50), (90, 50)), invert_directions=True)

    assert movement_direction((50.0, 30.0), (50.0, 80.0), normal_config.line, normal_config) == "out"
    assert movement_direction((50.0, 30.0), (50.0, 80.0), inverted_config.line, inverted_config) == "in"


def test_late_init_crossing_counts_track_initialized_inside_gate_zone() -> None:
    config = CountingConfig(line=((10, 50), (90, 50)))
    state = TrackState(
        points=deque(maxlen=32),
        sides=deque(maxlen=32),
        first_frame=10,
        first_point=(50.0, 50.0),
        first_side=0.0,
    )

    direction = maybe_late_init_crossing(
        state=state,
        curr_point=(50.0, 100.0),
        curr_side=50.0,
        frame_idx=16,
        line=config.line,
        config=config,
    )

    assert direction == "out"


def test_count_event_allowed_enforces_cooldown() -> None:
    config = CountingConfig(count_cooldown_frames=18)
    state = TrackState(last_count_frame=100)

    assert not count_event_allowed(state, frame_idx=110, config=config)
    assert count_event_allowed(state, frame_idx=118, config=config)


def test_transform_point_applies_affine_matrix() -> None:
    matrix = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -5.0]], dtype=float)

    assert transform_point((10.0, 20.0), matrix) == (22.0, 15.0)


def test_transform_line_rounds_affine_transformed_endpoints() -> None:
    matrix = np.array([[1.0, 0.0, 12.4], [0.0, 1.0, -5.2]], dtype=float)

    assert transform_line(((10, 20), (30, 40)), matrix) == ((22, 15), (42, 35))


def test_transform_point_returns_original_when_matrix_missing() -> None:
    assert transform_point((10.0, 20.0), None) == (10.0, 20.0)


def test_reset_track_state_clears_motion_but_preserves_count_cooldown() -> None:
    state = TrackState(last_count_frame=100, last_direction="out")
    state.points.append((1.0, 2.0))
    state.sides.append(3.0)
    state.first_frame = 10
    state.first_point = (1.0, 2.0)
    state.first_side = 3.0
    state.last_seen_frame = 20

    reset_track_state(state)

    assert list(state.points) == []
    assert list(state.sides) == []
    assert state.first_frame is None
    assert state.first_point is None
    assert state.first_side is None
    assert state.last_seen_frame is None
    assert state.last_count_frame == 100
    assert state.last_direction == "out"
