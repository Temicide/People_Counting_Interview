# Door ROI and Gate State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragile single-line side-change counting with a doorway corridor polygon and gate state machine that only counts people who clearly pass through the physical door opening.

**Architecture:** Add a corridor polygon to `CountingConfig` that defines the doorway area where counting is eligible. Replace the `eligible_crossing` side-change logic with a gate state machine that classizes footpoints into zones (positive/negative/threshold/ineligible) and only fires count events after multiple consecutive stable frames confirm a zone transition. Add EMA footpoint smoothing per track ID to handle overlapping people, person-masked ORB stabilization to prevent moving bodies from corrupting the transform, and per-frame diagnostics for debugging.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, pandas, Ultralytics YOLO with ByteTrack or BoT-SORT, pytest.

---

## File Structure

- Modify `src/entrance_counter/config.py`: Add `Polygon` type alias, corridor polygon, gate state machine params, footpoint smoother params to `CountingConfig`. Add `parse_corridor_polygon`.
- Modify `src/entrance_counter/geometry.py`: Add `point_in_polygon`, `classify_gate_zone`, `GateTracker`, `update_gate_tracker`, `FootpointSmoother`. Later remove `eligible_crossing`, `maybe_late_init_crossing`, `count_event_allowed`.
- Modify `src/entrance_counter/stabilization.py`: Add person mask support to `FrameStabilizer.estimate` and `_detect`.
- Modify `src/entrance_counter/pipeline.py`: Integrate gate state machine, footpoint smoother, person mask forwarding, diagnostics columns, optional per-frame trace CSV.
- Modify `src/entrance_counter/rendering.py`: Draw corridor polygon overlay, gate zone indicators.
- Modify `src/entrance_counter/cli.py`: Add `--corridor`, `--gate-stable-frames`, `--gate-threshold-px`, `--smooth-alpha`, `--max-jump-px`, `--write-frame-trace` flags.
- Modify `tests/test_config.py`: Validate new config fields, corridor parser, updated defaults.
- Modify `tests/test_geometry.py`: Add zone classification, gate state machine, footpoint smoother tests. Remove old crossing tests in cleanup.
- Modify `tests/test_stabilization.py`: Add person mask test.
- Modify `tests/test_pipeline.py`: Update scenarios for gate state machine behavior.
- Modify `tests/test_cli.py`: Add new CLI flag tests.
- Modify `tests/test_video_rendering.py`: Add corridor rendering test.

---

## Task 1: Corridor Polygon and Gate Configuration

**Files:**
- Modify: `src/entrance_counter/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Add these imports to `tests/test_config.py`:

```python
from entrance_counter.config import Polygon, parse_corridor_polygon
```

Add these tests to `tests/test_config.py`:

```python
def test_counting_config_has_corridor_and_gate_defaults() -> None:
    config = CountingConfig()

    assert config.corridor == ()
    assert config.gate_threshold_px == 8.0
    assert config.gate_stable_frames == 3
    assert config.smooth_alpha == 0.4
    assert config.max_jump_px == 80.0


def test_counting_config_accepts_corridor_polygon() -> None:
    config = CountingConfig(corridor=((10, 20), (30, 20), (30, 40), (10, 40)))

    assert len(config.corridor) == 4
    assert config.corridor[0] == (10, 20)


def test_counting_config_rejects_corridor_with_fewer_than_three_vertices() -> None:
    with pytest.raises(ValueError, match="at least 3 vertices"):
        CountingConfig(corridor=((10, 20), (30, 20)))


def test_counting_config_rejects_invalid_gate_stable_frames() -> None:
    with pytest.raises(ValueError, match="gate_stable_frames must be at least 1"):
        CountingConfig(gate_stable_frames=0)


def test_counting_config_rejects_invalid_smooth_alpha() -> None:
    with pytest.raises(ValueError, match="smooth_alpha must be between"):
        CountingConfig(smooth_alpha=0.0)


def test_counting_config_rejects_negative_max_jump() -> None:
    with pytest.raises(ValueError, match="max_jump_px must be non-negative"):
        CountingConfig(max_jump_px=-1.0)


def test_parse_corridor_polygon_accepts_flat_integer_string() -> None:
    result = parse_corridor_polygon("180,380,520,380,520,650,180,650")

    assert result == ((180, 380), (520, 380), (520, 650), (180, 650))


def test_parse_corridor_polygon_accepts_tuple_string() -> None:
    result = parse_corridor_polygon("(10, 20), (30, 20), (30, 40)")

    assert result == ((10, 20), (30, 20), (30, 40))


def test_parse_corridor_polygon_returns_empty_for_blank_input() -> None:
    assert parse_corridor_polygon("") == ()


def test_parse_corridor_polygon_rejects_odd_coordinate_count() -> None:
    with pytest.raises(ValueError, match="even number"):
        parse_corridor_polygon("10,20,30")


def test_parse_corridor_polygon_rejects_fewer_than_six_coordinates() -> None:
    with pytest.raises(ValueError, match="at least 3 vertices"):
        parse_corridor_polygon("10,20,30,40")
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: FAIL because `Polygon`, `parse_corridor_polygon`, and the new `CountingConfig` fields do not exist.

- [ ] **Step 3: Implement config changes**

In `src/entrance_counter/config.py`, add the `Polygon` type alias after the existing `Line` alias:

```python
Polygon: TypeAlias = tuple[tuple[int, int], ...]
```

Add new fields to `CountingConfig` (insert after `invert_directions`):

```python
@dataclass(frozen=True)
class CountingConfig:
    line: Line = ((330, 785), (870, 812))
    corridor: Polygon = ()
    positive_direction_label: str = "out"
    negative_direction_label: str = "in"
    invert_directions: bool = False
    gate_threshold_px: float = 8.0
    gate_stable_frames: int = 3
    smooth_alpha: float = 0.4
    max_jump_px: float = 80.0
    min_side_abs: float = 8.0
    min_track_age_frames: int = 3
    count_cooldown_frames: int = 18
    min_displacement_px: float = 18.0
    late_init_buffer_px: float = 45.0
    late_init_max_age_frames: int = 18
    max_track_gap_frames: int = 45
    draw_track_history: bool = True

    def __post_init__(self) -> None:
        if self.positive_direction_label == self.negative_direction_label:
            raise ValueError("positive_direction_label and negative_direction_label must differ")
        if self.gate_threshold_px < 0:
            raise ValueError("gate_threshold_px must be non-negative")
        if self.gate_stable_frames < 1:
            raise ValueError("gate_stable_frames must be at least 1")
        if not 0.0 < self.smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha must be between 0 (exclusive) and 1 (inclusive)")
        if self.max_jump_px < 0:
            raise ValueError("max_jump_px must be non-negative")
        if len(self.corridor) > 0 and len(self.corridor) < 3:
            raise ValueError("corridor polygon requires at least 3 vertices")
        if self.min_track_age_frames < 0:
            raise ValueError("min_track_age_frames must be non-negative")
        if self.count_cooldown_frames < 0:
            raise ValueError("count_cooldown_frames must be non-negative")
        if self.min_displacement_px < 0:
            raise ValueError("min_displacement_px must be non-negative")
        if self.max_track_gap_frames < 0:
            raise ValueError("max_track_gap_frames must be non-negative")
```

Add `parse_corridor_polygon` after `parse_counting_line`:

```python
def parse_corridor_polygon(raw: str) -> Polygon:
    cleaned = (
        raw.replace("(", " ")
        .replace(")", " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace(";", ",")
    )
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not parts:
        return ()
    if len(parts) < 6 or len(parts) % 2 != 0:
        raise ValueError(
            "corridor polygon requires at least 3 vertices (even number of integer coordinates)"
        )
    try:
        coords = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("corridor polygon coordinates must be integers") from exc
    return tuple((coords[i], coords[i + 1]) for i in range(0, len(coords), 2))
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: PASS (all config tests including new ones).

- [ ] **Step 5: Commit**

```bash
git add src/entrance_counter/config.py tests/test_config.py
git commit -m "feat: add corridor polygon and gate config fields"
```

---

## Task 2: Zone Classification, Gate State Machine, and Footpoint Smoother

**Files:**
- Modify: `src/entrance_counter/geometry.py`
- Test: `tests/test_geometry.py`

- [ ] **Step 1: Write failing geometry tests**

Add these imports to `tests/test_geometry.py`:

```python
from entrance_counter.config import Polygon
from entrance_counter.geometry import (
    FootpointSmoother,
    GateTracker,
    classify_gate_zone,
    point_in_polygon,
    update_gate_tracker,
)
```

Add these tests to `tests/test_geometry.py`:

```python
def test_point_in_polygon_detects_interior_point() -> None:
    polygon: Polygon = ((0, 0), (100, 0), (100, 100), (0, 100))

    assert point_in_polygon((50.0, 50.0), polygon) is True


def test_point_in_polygon_rejects_exterior_point() -> None:
    polygon: Polygon = ((0, 0), (100, 0), (100, 100), (0, 100))

    assert point_in_polygon((150.0, 50.0), polygon) is False


def test_point_in_polygon_handles_concave_shape() -> None:
    polygon: Polygon = ((0, 0), (100, 0), (100, 50), (50, 50), (50, 100), (0, 100))

    assert point_in_polygon((25.0, 75.0), polygon) is True
    assert point_in_polygon((75.0, 75.0), polygon) is False


def test_classify_gate_zone_returns_threshold_near_line() -> None:
    line = ((10, 50), (90, 50))

    assert classify_gate_zone((50.0, 52.0), line, (), gate_threshold_px=8.0) == "threshold"


def test_classify_gate_zone_returns_positive_or_negative_side() -> None:
    line = ((10, 50), (90, 50))

    assert classify_gate_zone((50.0, 20.0), line, (), gate_threshold_px=8.0) == "negative"
    assert classify_gate_zone((50.0, 80.0), line, (), gate_threshold_px=8.0) == "positive"


def test_classify_gate_zone_returns_ineligible_outside_corridor() -> None:
    line = ((10, 50), (90, 50))
    corridor: Polygon = ((20, 20), (80, 20), (80, 80), (20, 80))

    assert classify_gate_zone((5.0, 50.0), line, corridor, gate_threshold_px=8.0) == "ineligible"


def test_classify_gate_zone_returns_side_when_inside_corridor() -> None:
    line = ((10, 50), (90, 50))
    corridor: Polygon = ((0, 0), (100, 0), (100, 100), (0, 100))

    assert classify_gate_zone((50.0, 80.0), line, corridor, gate_threshold_px=8.0) == "positive"


def test_gate_tracker_default_state_is_unknown() -> None:
    gate = GateTracker()

    assert gate.confirmed_zone == "unknown"
    assert gate.candidate_zone == "unknown"
    assert gate.candidate_count == 0


def test_update_gate_tracker_confirms_zone_after_stable_frames() -> None:
    config = CountingConfig(line=((10, 50), (90, 50)), gate_stable_frames=3)
    gate = GateTracker()
    point = (50.0, 80.0)

    assert update_gate_tracker(gate, point, config, frame_idx=0, last_count_frame=-10000) is None
    assert gate.candidate_zone == "positive"
    assert gate.candidate_count == 1

    assert update_gate_tracker(gate, point, config, frame_idx=1, last_count_frame=-10000) is None
    assert gate.candidate_count == 2

    assert update_gate_tracker(gate, point, config, frame_idx=2, last_count_frame=-10000) is None
    assert gate.confirmed_zone == "positive"


def test_update_gate_tracker_fires_direction_on_zone_transition() -> None:
    config = CountingConfig(line=((10, 50), (90, 50)), gate_stable_frames=2)
    gate = GateTracker()
    outside_point = (50.0, 20.0)
    inside_point = (50.0, 80.0)

    update_gate_tracker(gate, outside_point, config, frame_idx=0, last_count_frame=-10000)
    update_gate_tracker(gate, outside_point, config, frame_idx=1, last_count_frame=-10000)
    assert gate.confirmed_zone == "negative"

    assert update_gate_tracker(gate, inside_point, config, frame_idx=2, last_count_frame=-10000) is None
    direction = update_gate_tracker(gate, inside_point, config, frame_idx=3, last_count_frame=-10000)
    assert direction == "out"


def test_update_gate_tracker_resets_candidate_on_threshold() -> None:
    config = CountingConfig(line=((10, 50), (90, 50)), gate_stable_frames=2)
    gate = GateTracker()
    outside_point = (50.0, 20.0)
    threshold_point = (50.0, 50.0)

    update_gate_tracker(gate, outside_point, config, frame_idx=0, last_count_frame=-10000)
    update_gate_tracker(gate, threshold_point, config, frame_idx=1, last_count_frame=-10000)

    assert gate.candidate_zone == "unknown"
    assert gate.candidate_count == 0


def test_update_gate_tracker_resets_candidate_on_ineligible() -> None:
    config = CountingConfig(
        line=((10, 50), (90, 50)),
        corridor=((20, 20), (80, 20), (80, 80), (20, 80)),
        gate_stable_frames=2,
    )
    gate = GateTracker()
    outside_corridor = (5.0, 50.0)

    update_gate_tracker(gate, outside_corridor, config, frame_idx=0, last_count_frame=-10000)

    assert gate.candidate_zone == "unknown"
    assert gate.candidate_count == 0


def test_update_gate_tracker_enforces_cooldown() -> None:
    config = CountingConfig(
        line=((10, 50), (90, 50)),
        gate_stable_frames=1,
        count_cooldown_frames=10,
    )
    gate = GateTracker()
    outside_point = (50.0, 20.0)
    inside_point = (50.0, 80.0)

    update_gate_tracker(gate, outside_point, config, frame_idx=0, last_count_frame=-10000)
    direction = update_gate_tracker(gate, inside_point, config, frame_idx=1, last_count_frame=-10000)
    assert direction == "out"

    gate2 = GateTracker()
    update_gate_tracker(gate2, outside_point, config, frame_idx=0, last_count_frame=-10000)
    direction2 = update_gate_tracker(gate2, inside_point, config, frame_idx=1, last_count_frame=0)
    assert direction2 is None


def test_update_gate_tracker_inverts_direction_with_config_flag() -> None:
    config = CountingConfig(
        line=((10, 50), (90, 50)),
        gate_stable_frames=1,
        invert_directions=True,
    )
    gate = GateTracker()
    outside_point = (50.0, 20.0)
    inside_point = (50.0, 80.0)

    update_gate_tracker(gate, outside_point, config, frame_idx=0, last_count_frame=-10000)
    direction = update_gate_tracker(gate, inside_point, config, frame_idx=1, last_count_frame=-10000)
    assert direction == "in"


def test_footpoint_smoother_returns_raw_on_first_observation() -> None:
    smoother = FootpointSmoother(alpha=0.4, max_jump_px=80.0)

    result = smoother.update(track_id=1, raw_point=(100.0, 200.0))

    assert result == (100.0, 200.0)


def test_footpoint_smoother_applies_ema_smoothing() -> None:
    smoother = FootpointSmoother(alpha=0.5, max_jump_px=80.0)
    smoother.update(track_id=1, raw_point=(100.0, 200.0))

    result = smoother.update(track_id=1, raw_point=(110.0, 210.0))

    assert abs(result[0] - 105.0) < 0.01
    assert abs(result[1] - 205.0) < 0.01


def test_footpoint_smoother_rejects_large_jump() -> None:
    smoother = FootpointSmoother(alpha=0.5, max_jump_px=20.0)
    smoother.update(track_id=1, raw_point=(100.0, 200.0))

    result = smoother.update(track_id=1, raw_point=(200.0, 300.0))

    assert result == (100.0, 200.0)


def test_footpoint_smoother_tracks_independent_ids() -> None:
    smoother = FootpointSmoother(alpha=0.5, max_jump_px=80.0)
    smoother.update(track_id=1, raw_point=(100.0, 200.0))
    smoother.update(track_id=2, raw_point=(300.0, 400.0))

    r1 = smoother.update(track_id=1, raw_point=(110.0, 210.0))
    r2 = smoother.update(track_id=2, raw_point=(310.0, 410.0))

    assert abs(r1[0] - 105.0) < 0.01
    assert abs(r2[0] - 305.0) < 0.01


def test_footpoint_smoother_remove_clears_track() -> None:
    smoother = FootpointSmoother(alpha=0.5, max_jump_px=80.0)
    smoother.update(track_id=1, raw_point=(100.0, 200.0))
    smoother.remove(track_id=1)

    result = smoother.update(track_id=1, raw_point=(500.0, 600.0))

    assert result == (500.0, 600.0)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_geometry.py -q
```

Expected: FAIL because `point_in_polygon`, `classify_gate_zone`, `GateTracker`, `update_gate_tracker`, and `FootpointSmoother` do not exist.

- [ ] **Step 3: Implement zone classification, gate state machine, and smoother**

Add `Polygon` to the import in `src/entrance_counter/geometry.py`:

```python
from entrance_counter.config import CountingConfig, Line, Point, Polygon
```

Add `GateTracker` **before** the existing `TrackState` dataclass (so `TrackState` can reference it via `default_factory`):

```python
@dataclass
class GateTracker:
    confirmed_zone: str = "unknown"
    candidate_zone: str = "unknown"
    candidate_count: int = 0
```

Add these functions and classes **after** the existing functions at the end of `src/entrance_counter/geometry.py`:

```python
def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(yj - yi, 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def classify_gate_zone(
    point: Point, line: Line, corridor: Polygon, gate_threshold_px: float
) -> str:
    if corridor and not point_in_polygon(point, corridor):
        return "ineligible"
    side = normalized_line_side(point, line)
    if abs(side) < gate_threshold_px:
        return "threshold"
    return "positive" if side > 0 else "negative"


def update_gate_tracker(
    gate: GateTracker,
    point: Point,
    config: CountingConfig,
    frame_idx: int,
    last_count_frame: int,
) -> str | None:
    zone = classify_gate_zone(point, config.line, config.corridor, config.gate_threshold_px)

    if zone in ("ineligible", "threshold"):
        gate.candidate_zone = "unknown"
        gate.candidate_count = 0
        return None

    if zone == gate.candidate_zone:
        gate.candidate_count += 1
    else:
        gate.candidate_zone = zone
        gate.candidate_count = 1

    if gate.candidate_count >= config.gate_stable_frames:
        direction = None
        if gate.confirmed_zone != "unknown" and zone != gate.confirmed_zone:
            if frame_idx - last_count_frame >= config.count_cooldown_frames:
                moved_positive = zone == "positive"
                if config.invert_directions:
                    moved_positive = not moved_positive
                direction = (
                    config.positive_direction_label
                    if moved_positive
                    else config.negative_direction_label
                )
        gate.confirmed_zone = zone
        gate.candidate_zone = "unknown"
        gate.candidate_count = 0
        return direction

    return None


@dataclass
class FootpointSmoother:
    alpha: float = 0.4
    max_jump_px: float = 80.0

    def __post_init__(self) -> None:
        self._last: dict[int, Point] = {}

    def update(self, track_id: int, raw_point: Point) -> Point:
        if track_id not in self._last:
            self._last[track_id] = raw_point
            return raw_point

        last = self._last[track_id]
        jump = math.hypot(raw_point[0] - last[0], raw_point[1] - last[1])
        if jump > self.max_jump_px:
            return last

        smoothed = (
            self.alpha * raw_point[0] + (1.0 - self.alpha) * last[0],
            self.alpha * raw_point[1] + (1.0 - self.alpha) * last[1],
        )
        self._last[track_id] = smoothed
        return smoothed

    def remove(self, track_id: int) -> None:
        self._last.pop(track_id, None)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_geometry.py -q
```

Expected: PASS (all geometry tests including new ones).

- [ ] **Step 5: Commit**

```bash
git add src/entrance_counter/geometry.py tests/test_geometry.py
git commit -m "feat: add zone classification, gate state machine, and footpoint smoother"
```

---

## Task 3: Pipeline Integration with Gate State Machine

**Files:**
- Modify: `src/entrance_counter/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

Replace the existing `test_people_counter_counts_crossing_and_writes_csvs` test in `tests/test_pipeline.py` with this version that provides enough frames for the gate state machine:

```python
def test_people_counter_counts_crossing_and_writes_csvs(tmp_path: Path) -> None:
    video_path = tmp_path / "entrance.mp4"
    _write_video(video_path, frame_count=10)
    output_dir = tmp_path / "output"
    fake_model = FakeModel(
        frames=[
            [(_box_with_bottom_center_y(20.0), 7, 0.90)],
            [(_box_with_bottom_center_y(20.0), 7, 0.90)],
            [(_box_with_bottom_center_y(20.0), 7, 0.90)],
            [(_box_with_bottom_center_y(20.0), 7, 0.90)],
            [(_box_with_bottom_center_y(80.0), 7, 0.91)],
            [(_box_with_bottom_center_y(80.0), 7, 0.92)],
            [(_box_with_bottom_center_y(80.0), 7, 0.93)],
            [(_box_with_bottom_center_y(80.0), 7, 0.94)],
            [(_box_with_bottom_center_y(80.0), 7, 0.95)],
            [(_box_with_bottom_center_y(80.0), 7, 0.96)],
        ]
    )
    config = RunConfig(
        paths=PathConfig(project_root=tmp_path, video_path=video_path, output_dir=output_dir),
        counting=CountingConfig(line=((10, 50), (90, 50)), gate_stable_frames=3),
        model=ModelConfig(model_name="fake.pt"),
        processing=ProcessingConfig(max_frames=10),
    )

    result = PeopleCounter(config=config, model=fake_model, device="cpu").run(
        write_annotated_video=False,
        progress_every=0,
    )

    assert result.events_path.exists()
    assert result.summary_path.exists()
    assert fake_model.calls == 10

    events = pd.read_csv(result.events_path)
    summary = pd.read_csv(result.summary_path).iloc[0]

    assert len(events) == 1
    assert events.iloc[0]["track_id"] == 7
    assert events.iloc[0]["direction"] == "out"
    assert events.iloc[0]["counted_by"] == "gate_transition"
    assert int(summary["frames_processed"]) == 10
    assert int(summary["in_count"]) == 0
    assert int(summary["out_count"]) == 1
    assert int(summary["total_count"]) == 1
```

Add this new test for corridor filtering:

```python
def test_people_counter_ignores_detections_outside_corridor(tmp_path: Path) -> None:
    video_path = tmp_path / "entrance.mp4"
    _write_video(video_path, frame_count=6)
    output_dir = tmp_path / "output"
    fake_model = FakeModel(
        frames=[
            [((5.0, 0.0, 15.0, 20.0), 7, 0.90)],
            [((5.0, 0.0, 15.0, 20.0), 7, 0.90)],
            [((5.0, 0.0, 15.0, 20.0), 7, 0.90)],
            [((5.0, 60.0, 15.0, 80.0), 7, 0.91)],
            [((5.0, 60.0, 15.0, 80.0), 7, 0.92)],
            [((5.0, 60.0, 15.0, 80.0), 7, 0.93)],
        ]
    )
    config = RunConfig(
        paths=PathConfig(project_root=tmp_path, video_path=video_path, output_dir=output_dir),
        counting=CountingConfig(
            line=((10, 50), (90, 50)),
            corridor=((20, 0), (80, 0), (80, 100), (20, 100)),
            gate_stable_frames=2,
        ),
        model=ModelConfig(model_name="fake.pt"),
        processing=ProcessingConfig(max_frames=6),
    )

    result = PeopleCounter(config=config, model=fake_model, device="cpu").run(
        write_annotated_video=False,
        progress_every=0,
    )

    assert len(result.events) == 0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_pipeline.py -q
```

Expected: FAIL because pipeline still uses `eligible_crossing` and does not recognize `counted_by="gate_transition"`.

- [ ] **Step 3: Implement pipeline integration**

Update imports in `src/entrance_counter/pipeline.py`. Replace the geometry import block:

```python
from entrance_counter.geometry import (
    FootpointSmoother,
    GateTracker,
    TrackState,
    bottom_center_xy,
    classify_gate_zone,
    normalized_line_side,
    projection_t,
    point_in_polygon,
    reset_track_state,
    transform_line,
    transform_point,
    update_gate_tracker,
)
```

Add `GateTracker` to `TrackState` in `geometry.py`. Add this field at the end of the `TrackState` dataclass:

```python
    gate: GateTracker = field(default_factory=GateTracker)
```

Update `reset_track_state` in `geometry.py` to also reset the gate:

```python
def reset_track_state(state: TrackState) -> None:
    state.points.clear()
    state.sides.clear()
    state.first_frame = None
    state.first_point = None
    state.first_side = None
    state.last_seen_frame = None
    state.gate = GateTracker()
```

Update `EVENT_COLUMNS` in `pipeline.py` to add diagnostic columns:

```python
EVENT_COLUMNS = [
    "event_index",
    "frame",
    "time_seconds",
    "track_id",
    "direction",
    "counted_by",
    "bottom_center_x",
    "bottom_center_y",
    "reference_bottom_center_x",
    "reference_bottom_center_y",
    "side_distance",
    "projection_t",
    "in_corridor",
    "gate_zone",
    "gate_confirmed_zone",
    "confidence",
    "stabilization_valid",
    "in_count",
    "out_count",
    "total_count",
]
```

Add `TRACE_COLUMNS` after `EVENT_COLUMNS`:

```python
TRACE_COLUMNS = [
    "frame",
    "track_id",
    "raw_x",
    "raw_y",
    "stabilized_x",
    "stabilized_y",
    "side_distance",
    "projection_t",
    "in_corridor",
    "gate_zone",
    "gate_confirmed_zone",
    "gate_candidate_count",
    "counted",
    "direction",
]
```

Add `write_frame_trace: bool = False` parameter to `PeopleCounter.run`:

```python
def run(
    self,
    write_annotated_video: bool = True,
    write_frame_trace: bool = False,
    progress_every: int = 100,
) -> CountingResult:
```

After the `events` list initialization in `run`, add the trace list and smoother:

```python
        trace_rows: list[dict[str, object]] = []
        smoother = FootpointSmoother(
            alpha=self.config.counting.smooth_alpha,
            max_jump_px=self.config.counting.max_jump_px,
        )
```

Replace the per-detection loop body (the `for xyxy, track_id, conf in zip(...)` block) with:

```python
                person_boxes = list(zip(boxes_xyxy, track_ids, confs))

                for xyxy, track_id, conf in person_boxes:
                    tid = int(track_id)
                    state = states[tid]
                    raw_point = bottom_center_xy(xyxy)
                    smoothed_point = smoother.update(tid, raw_point)
                    image_point = smoothed_point
                    point = transform_point(image_point, current_to_reference)
                    side = normalized_line_side(point, self.config.counting.line)
                    proj_t = projection_t(point, self.config.counting.line)
                    corridor = self.config.counting.corridor
                    in_corridor = (
                        not corridor or point_in_polygon(point, corridor)
                    )
                    raw_zone = classify_gate_zone(
                        point,
                        self.config.counting.line,
                        corridor,
                        self.config.counting.gate_threshold_px,
                    )

                    if (
                        state.last_seen_frame is not None
                        and frame_idx - state.last_seen_frame
                        > self.config.counting.max_track_gap_frames
                    ):
                        reset_track_state(state)
                        smoother.remove(tid)

                    if state.first_frame is None:
                        state.first_frame = frame_idx
                        state.first_point = point
                        state.first_side = side

                    counted_direction = update_gate_tracker(
                        state.gate,
                        point,
                        self.config.counting,
                        frame_idx,
                        state.last_count_frame,
                    )
                    counted_by = None
                    if counted_direction is not None:
                        counted_by = "gate_transition"
                        state.last_count_frame = frame_idx
                        state.last_direction = counted_direction

                    if counted_direction is not None:
                        counts[counted_direction] = counts.get(counted_direction, 0) + 1
                        events.append(
                            {
                                "event_index": len(events) + 1,
                                "frame": frame_idx,
                                "time_seconds": frame_idx / max(fps, 1e-6),
                                "track_id": tid,
                                "direction": counted_direction,
                                "counted_by": counted_by,
                                "bottom_center_x": image_point[0],
                                "bottom_center_y": image_point[1],
                                "reference_bottom_center_x": point[0],
                                "reference_bottom_center_y": point[1],
                                "side_distance": side,
                                "projection_t": proj_t,
                                "in_corridor": in_corridor,
                                "gate_zone": raw_zone,
                                "gate_confirmed_zone": state.gate.confirmed_zone,
                                "confidence": float(conf),
                                "stabilization_valid": stabilization_valid,
                                "in_count": counts.get(
                                    self.config.counting.negative_direction_label, 0
                                ),
                                "out_count": counts.get(
                                    self.config.counting.positive_direction_label, 0
                                ),
                                "total_count": (
                                    counts.get(
                                        self.config.counting.negative_direction_label, 0
                                    )
                                    + counts.get(
                                        self.config.counting.positive_direction_label, 0
                                    )
                                ),
                            }
                        )

                    if write_frame_trace:
                        trace_rows.append(
                            {
                                "frame": frame_idx,
                                "track_id": tid,
                                "raw_x": raw_point[0],
                                "raw_y": raw_point[1],
                                "stabilized_x": point[0],
                                "stabilized_y": point[1],
                                "side_distance": side,
                                "projection_t": proj_t,
                                "in_corridor": in_corridor,
                                "gate_zone": raw_zone,
                                "gate_confirmed_zone": state.gate.confirmed_zone,
                                "gate_candidate_count": state.gate.candidate_count,
                                "counted": counted_direction is not None,
                                "direction": counted_direction or "",
                            }
                        )

                    state.points.append(point)
                    state.sides.append(side)
                    state.last_seen_frame = frame_idx
```

After the events CSV write, add the trace CSV write:

```python
        if write_frame_trace:
            trace_path = self.config.paths.output_dir / "entrance_frame_trace.csv"
            trace_df = pd.DataFrame(trace_rows, columns=TRACE_COLUMNS)
            trace_df.to_csv(trace_path, index=False)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/entrance_counter/geometry.py src/entrance_counter/pipeline.py tests/test_pipeline.py
git commit -m "feat: integrate gate state machine and footpoint smoother into pipeline"
```

---

## Task 4: Cleanup Deprecated Fields and Functions

**Files:**
- Modify: `src/entrance_counter/config.py`
- Modify: `src/entrance_counter/geometry.py`
- Modify: `src/entrance_counter/pipeline.py`
- Test: `tests/test_config.py`
- Test: `tests/test_geometry.py`

- [ ] **Step 1: Update config defaults test**

In `tests/test_config.py`, update `test_run_config_defaults_match_current_baseline_parameters` to reflect the new field names and remove references to deprecated fields:

```python
def test_run_config_defaults_match_current_baseline_parameters() -> None:
    config = RunConfig()

    assert config.counting.line == ((330, 785), (870, 812))
    assert config.counting.corridor == ()
    assert config.counting.positive_direction_label == "out"
    assert config.counting.negative_direction_label == "in"
    assert config.counting.gate_threshold_px == 8.0
    assert config.counting.gate_stable_frames == 3
    assert config.counting.smooth_alpha == 0.4
    assert config.counting.max_jump_px == 80.0
    assert config.counting.count_cooldown_frames == 18
    assert config.counting.max_track_gap_frames == 45
    assert config.model.model_name == "yolov8n.pt"
    assert config.model.confidence == 0.22
    assert config.model.image_size == 960
    assert config.processing.process_every_n_frames == 1
    assert config.processing.max_frames is None
```

- [ ] **Step 2: Update geometry tests**

In `tests/test_geometry.py`, remove these test functions that test deprecated functions:
- `test_eligible_crossing_requires_side_change_and_real_motion`
- `test_eligible_crossing_rejects_small_jitter_around_line`
- `test_late_init_crossing_counts_track_initialized_inside_gate_zone`
- `test_count_event_allowed_enforces_cooldown`

Also remove the imports of deprecated functions:
- `count_event_allowed`
- `eligible_crossing`
- `maybe_late_init_crossing`

- [ ] **Step 3: Remove deprecated config fields**

In `src/entrance_counter/config.py`, remove these fields from `CountingConfig`:
- `min_side_abs`
- `min_track_age_frames`
- `min_displacement_px`
- `late_init_buffer_px`
- `late_init_max_age_frames`

Remove their corresponding validations from `__post_init__`:
- `min_track_age_frames < 0` check
- `min_displacement_px < 0` check

Keep the `count_cooldown_frames` and `max_track_gap_frames` validations.

- [ ] **Step 4: Remove deprecated geometry functions**

In `src/entrance_counter/geometry.py`, remove these functions:
- `eligible_crossing`
- `maybe_late_init_crossing`
- `count_event_allowed`

Remove unused imports if any (check `math.hypot` is still used by `FootpointSmoother`).

Remove `first_point` and `first_side` from `TrackState`:

```python
@dataclass
class TrackState:
    points: Deque[Point] = field(default_factory=lambda: deque(maxlen=32))
    sides: Deque[float] = field(default_factory=lambda: deque(maxlen=32))
    first_frame: int | None = None
    last_count_frame: int = -10_000
    last_direction: str | None = None
    last_seen_frame: int | None = None
    gate: GateTracker = field(default_factory=GateTracker)
```

Update `reset_track_state` to match:

```python
def reset_track_state(state: TrackState) -> None:
    state.points.clear()
    state.sides.clear()
    state.first_frame = None
    state.last_seen_frame = None
    state.gate = GateTracker()
```

Update `test_reset_track_state_clears_motion_but_preserves_count_cooldown` in `tests/test_geometry.py`:

```python
def test_reset_track_state_clears_motion_but_preserves_count_cooldown() -> None:
    state = TrackState(last_count_frame=100, last_direction="out")
    state.points.append((1.0, 2.0))
    state.sides.append(3.0)
    state.first_frame = 10
    state.last_seen_frame = 20

    reset_track_state(state)

    assert list(state.points) == []
    assert list(state.sides) == []
    assert state.first_frame is None
    assert state.last_seen_frame is None
    assert state.last_count_frame == 100
    assert state.last_direction == "out"
    assert state.gate.confirmed_zone == "unknown"
```

- [ ] **Step 5: Update pipeline to remove deprecated references**

In `src/entrance_counter/pipeline.py`, remove the lines that set `state.first_point` and `state.first_side`:

```python
                    if state.first_frame is None:
                        state.first_frame = frame_idx
```

Remove the import of `maybe_late_init_crossing` if still present (it should have been removed in Task 3).

- [ ] **Step 6: Run all tests**

Run:

```bash
python -m pytest -q
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/entrance_counter/config.py src/entrance_counter/geometry.py src/entrance_counter/pipeline.py tests/test_config.py tests/test_geometry.py
git commit -m "refactor: remove deprecated crossing logic and config fields"
```

---

## Task 5: Person-Masked Stabilization

**Files:**
- Modify: `src/entrance_counter/stabilization.py`
- Modify: `src/entrance_counter/pipeline.py`
- Test: `tests/test_stabilization.py`

- [ ] **Step 1: Write failing stabilization mask test**

Add this test to `tests/test_stabilization.py`:

```python
def test_stabilizer_ignores_features_inside_person_mask() -> None:
    reference = _feature_frame()
    current = _feature_frame(shift_x=10, shift_y=6)
    stabilizer = FrameStabilizer(
        StabilizationConfig(
            enabled=True,
            max_features=500,
            min_matches=4,
            min_inlier_ratio=0.25,
            max_reprojection_error=5.0,
            smoothing_alpha=1.0,
        ),
        reference,
    )

    person_masks = [(30, 30, 200, 160)]
    transform_masked = stabilizer.estimate(current, person_masks=person_masks)
    transform_unmasked = stabilizer.estimate(current)

    assert transform_masked.valid is True
    assert transform_masked.match_count <= transform_unmasked.match_count
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python -m pytest tests/test_stabilization.py::test_stabilizer_ignores_features_inside_person_mask -v
```

Expected: FAIL because `estimate` does not accept `person_masks`.

- [ ] **Step 3: Implement person mask support**

In `src/entrance_counter/stabilization.py`, update `estimate` to accept an optional `person_masks` parameter:

```python
    def estimate(
        self,
        frame: np.ndarray,
        person_masks: list[tuple[int, int, int, int]] | None = None,
    ) -> FrameTransform:
        if not self.config.enabled:
            return self._identity(valid=False)
        if self._reference_descriptors is None or len(self._reference_keypoints) < self.config.min_matches:
            return self._identity(valid=False)

        current_keypoints, current_descriptors = self._detect(frame, person_masks)
        if current_descriptors is None or len(current_keypoints) < self.config.min_matches:
            return self._fallback()

        matches = sorted(
            self._matcher.match(current_descriptors, self._reference_descriptors),
            key=lambda match: match.distance,
        )
        if len(matches) < self.config.min_matches:
            return self._fallback(match_count=len(matches))

        current_points = np.float32([current_keypoints[match.queryIdx].pt for match in matches])
        reference_points = np.float32([self._reference_keypoints[match.trainIdx].pt for match in matches])
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            current_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.max_reprojection_error,
        )
        if matrix is None or inlier_mask is None:
            return self._fallback(match_count=len(matches))

        inlier_count = int(inlier_mask.sum())
        inlier_ratio = inlier_count / max(len(matches), 1)
        if inlier_count < self.config.min_matches or inlier_ratio < self.config.min_inlier_ratio:
            return self._fallback(match_count=len(matches), inlier_count=inlier_count, inlier_ratio=inlier_ratio)

        matrix = matrix.astype(float)
        if self._previous_current_to_reference is not None and self.config.smoothing_alpha < 1.0:
            alpha = self.config.smoothing_alpha
            matrix = alpha * matrix + (1.0 - alpha) * self._previous_current_to_reference
        self._previous_current_to_reference = matrix

        return FrameTransform(
            current_to_reference=matrix,
            reference_to_current=invert_affine(matrix),
            valid=True,
            match_count=len(matches),
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
        )
```

Update `_detect` to accept and apply a mask:

```python
    def _detect(
        self,
        frame: np.ndarray,
        person_masks: list[tuple[int, int, int, int]] | None = None,
    ) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._build_detection_mask(gray.shape, person_masks)
        keypoints, descriptors = self._orb.detectAndCompute(gray, mask)
        return tuple(keypoints or ()), descriptors

    def _build_detection_mask(
        self,
        shape: tuple[int, ...],
        person_masks: list[tuple[int, int, int, int]] | None,
    ) -> np.ndarray | None:
        if not person_masks:
            return None
        mask = np.ones(shape[:2], dtype=np.uint8) * 255
        h, w = shape[:2]
        for x1, y1, x2, y2 in person_masks:
            mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 0
        return mask
```

- [ ] **Step 4: Forward person boxes from pipeline to stabilizer**

In `src/entrance_counter/pipeline.py`, after the detection loop but before the stabilizer call, build person masks from the current frame's boxes. Move the stabilizer `estimate` call to after detection and pass masks:

Replace the stabilization block (before the detection loop) with a deferred approach. First, compute the transform without masks (for the first frame or when no detections exist). Then, after detections, re-estimate with masks if available.

Actually, the simpler approach: compute the transform before detections (as currently), but on subsequent frames, pass the previous frame's person boxes as masks. This avoids a chicken-and-egg problem.

Add a `prev_person_masks` variable before the main loop:

```python
        prev_person_masks: list[tuple[int, int, int, int]] | None = None
```

Update the stabilization block to use `prev_person_masks`:

```python
                elif stabilizer is not None:
                    transform = stabilizer.estimate(frame, person_masks=prev_person_masks)
                    current_to_reference = transform.current_to_reference
                    reference_to_current = transform.reference_to_current
                    stabilization_valid = transform.valid
```

After the detection loop, build `prev_person_masks` from current boxes:

```python
                prev_person_masks = [
                    (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    for xyxy in boxes_xyxy
                ]
```

- [ ] **Step 5: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_stabilization.py tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/entrance_counter/stabilization.py src/entrance_counter/pipeline.py tests/test_stabilization.py
git commit -m "feat: add person-masked stabilization feature detection"
```

---

## Task 6: Rendering Door Corridor and Gate State

**Files:**
- Modify: `src/entrance_counter/rendering.py`
- Test: `tests/test_video_rendering.py`

- [ ] **Step 1: Write failing rendering test**

Add this import to `tests/test_video_rendering.py`:

```python
from entrance_counter.config import Polygon
```

Add these tests:

```python
def test_draw_corridor_overlay_renders_polygon() -> None:
    from entrance_counter.rendering import draw_corridor

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    corridor: Polygon = ((20, 20), (140, 20), (140, 100), (20, 100))

    result = draw_corridor(frame, corridor)

    assert result.shape == frame.shape
    assert int(result.sum()) > 0


def test_draw_corridor_overlay_returns_frame_unchanged_for_empty_corridor() -> None:
    from entrance_counter.rendering import draw_corridor

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    result = draw_corridor(frame, ())

    np.testing.assert_array_equal(result, frame)


def test_draw_annotations_renders_corridor_when_configured() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    config = CountingConfig(
        line=((20, 60), (120, 60)),
        corridor=((10, 10), (150, 10), (150, 110), (10, 110)),
    )

    annotated = draw_annotations(
        frame=frame,
        boxes_xyxy=np.empty((0, 4), dtype=float),
        track_ids=np.empty((0,), dtype=int),
        states={},
        counts={"in": 0, "out": 0},
        frame_idx=0,
        fps=10.0,
        config=config,
    )

    assert annotated.shape == frame.shape
    assert int(annotated.sum()) > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_video_rendering.py -q
```

Expected: FAIL because `draw_corridor` does not exist.

- [ ] **Step 3: Implement corridor rendering**

Add `Polygon` to the import in `src/entrance_counter/rendering.py`:

```python
from entrance_counter.config import CountingConfig, Line, Polygon
```

Add `draw_corridor` function:

```python
def draw_corridor(
    frame: np.ndarray,
    corridor: Polygon,
    color: tuple[int, int, int] = (0, 200, 0),
    alpha: float = 0.12,
    border_thickness: int = 2,
) -> np.ndarray:
    if not corridor:
        return frame
    overlay = frame.copy()
    pts = np.array(corridor, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    result = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
    cv2.polylines(result, [pts], isClosed=True, color=color, thickness=border_thickness, lineType=cv2.LINE_AA)
    return result
```

Update `draw_annotations` to render the corridor before the counting line:

```python
def draw_annotations(
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,
    track_ids: np.ndarray,
    states: Mapping[int, TrackState],
    counts: Mapping[str, int],
    frame_idx: int,
    fps: float,
    config: CountingConfig,
    line: Line | None = None,
) -> np.ndarray:
    display_line = line if line is not None else config.line
    annotated = draw_corridor(frame, config.corridor)
    annotated = draw_counting_line(annotated, display_line, config, thickness=4)
    # ... rest of function unchanged
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_video_rendering.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/entrance_counter/rendering.py tests/test_video_rendering.py
git commit -m "feat: render door corridor polygon and gate zone overlay"
```

---

## Task 7: CLI Wiring for Corridor and Gate Parameters

**Files:**
- Modify: `src/entrance_counter/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Add these assertions to `tests/test_cli.py::test_cli_defaults_match_run_config`:

```python
    assert config.counting.corridor == ()
    assert config.counting.gate_stable_frames == 3
    assert config.counting.gate_threshold_px == 8.0
    assert config.counting.smooth_alpha == 0.4
    assert config.counting.max_jump_px == 80.0
```

Add these arguments to `tests/test_cli.py::test_cli_maps_overrides_to_run_config`:

```python
            "--corridor",
            "180,380,520,380,520,650,180,650",
            "--gate-stable-frames",
            "5",
            "--gate-threshold-px",
            "12.0",
            "--smooth-alpha",
            "0.6",
            "--max-jump-px",
            "100.0",
```

Add these assertions to the same test:

```python
    assert config.counting.corridor == ((180, 380), (520, 380), (520, 650), (180, 650))
    assert config.counting.gate_stable_frames == 5
    assert config.counting.gate_threshold_px == 12.0
    assert config.counting.smooth_alpha == 0.6
    assert config.counting.max_jump_px == 100.0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: FAIL because CLI flags do not exist.

- [ ] **Step 3: Implement CLI flags**

In `src/entrance_counter/cli.py`, add `parse_corridor_polygon` to the import:

```python
from entrance_counter.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO_PATH,
    CountingConfig,
    ModelConfig,
    PathConfig,
    ProcessingConfig,
    RunConfig,
    StabilizationConfig,
    parse_corridor_polygon,
    parse_counting_line,
)
```

Add parser arguments in `build_parser` (after `--line`):

```python
    parser.add_argument(
        "--corridor",
        default="",
        help="Door corridor polygon as flat x1,y1,x2,y2,... coordinates. Empty means no corridor filter.",
    )
    parser.add_argument(
        "--gate-stable-frames",
        type=int,
        default=3,
        help="Consecutive frames required in a zone to confirm a gate transition.",
    )
    parser.add_argument(
        "--gate-threshold-px",
        type=float,
        default=8.0,
        help="Perpendicular pixel distance from the line that defines the threshold zone.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.4,
        help="EMA alpha for footpoint smoothing (0=full history, 1=raw observation).",
    )
    parser.add_argument(
        "--max-jump-px",
        type=float,
        default=80.0,
        help="Maximum per-frame footpoint jump in pixels before rejection.",
    )
    parser.add_argument(
        "--write-frame-trace",
        action="store_true",
        help="Write per-frame diagnostic trace CSV alongside events.",
    )
```

Update `config_from_args` to pass new fields to `CountingConfig`:

```python
    corridor_raw = parse_corridor_polygon(args.corridor)
    counting = CountingConfig(
        line=parse_counting_line(args.line),
        corridor=corridor_raw,
        invert_directions=args.invert_directions,
        gate_stable_frames=args.gate_stable_frames,
        gate_threshold_px=args.gate_threshold_px,
        smooth_alpha=args.smooth_alpha,
        max_jump_px=args.max_jump_px,
    )
```

Update `main` to pass `write_frame_trace`:

```python
    result = counter.run(
        write_annotated_video=not args.skip_annotated_video,
        write_frame_trace=args.write_frame_trace,
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/entrance_counter/cli.py tests/test_cli.py
git commit -m "feat: wire corridor polygon, gate params, and frame trace CLI flags"
```

---

## Task 8: Regression Fixtures, Recalibrated Defaults, and Validation

**Files:**
- Modify: `src/entrance_counter/config.py`
- Test: `tests/test_geometry.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write regression test for overlapping people at door**

Add this test to `tests/test_pipeline.py`:

```python
def test_overlapping_people_at_door_count_once_each(tmp_path: Path) -> None:
    video_path = tmp_path / "entrance.mp4"
    _write_video(video_path, frame_count=12)
    output_dir = tmp_path / "output"
    fake_model = FakeModel(
        frames=[
            [
                (_box_with_bottom_center_y(20.0), 1, 0.90),
                (_box_with_bottom_center_y(22.0), 2, 0.88),
            ],
            [
                (_box_with_bottom_center_y(20.0), 1, 0.90),
                (_box_with_bottom_center_y(22.0), 2, 0.88),
            ],
            [
                (_box_with_bottom_center_y(20.0), 1, 0.90),
                (_box_with_bottom_center_y(22.0), 2, 0.88),
            ],
            [
                (_box_with_bottom_center_y(20.0), 1, 0.90),
                (_box_with_bottom_center_y(22.0), 2, 0.88),
            ],
            [
                (_box_with_bottom_center_y(80.0), 1, 0.91),
                (_box_with_bottom_center_y(82.0), 2, 0.89),
            ],
            [
                (_box_with_bottom_center_y(80.0), 1, 0.92),
                (_box_with_bottom_center_y(82.0), 2, 0.90),
            ],
            [
                (_box_with_bottom_center_y(80.0), 1, 0.93),
                (_box_with_bottom_center_y(82.0), 2, 0.91),
            ],
            [
                (_box_with_bottom_center_y(80.0), 1, 0.94),
                (_box_with_bottom_center_y(82.0), 2, 0.92),
            ],
            [
                (_box_with_bottom_center_y(80.0), 1, 0.95),
                (_box_with_bottom_center_y(82.0), 2, 0.93),
            ],
            [
                (_box_with_bottom_center_y(80.0), 1, 0.96),
                (_box_with_bottom_center_y(82.0), 2, 0.94),
            ],
            [
                (_box_with_bottom_center_y(80.0), 1, 0.97),
                (_box_with_bottom_center_y(82.0), 2, 0.95),
            ],
            [
                (_box_with_bottom_center_y(80.0), 1, 0.98),
                (_box_with_bottom_center_y(82.0), 2, 0.96),
            ],
        ]
    )
    config = RunConfig(
        paths=PathConfig(project_root=tmp_path, video_path=video_path, output_dir=output_dir),
        counting=CountingConfig(line=((10, 50), (90, 50)), gate_stable_frames=3),
        model=ModelConfig(model_name="fake.pt"),
        processing=ProcessingConfig(max_frames=12),
    )

    result = PeopleCounter(config=config, model=fake_model, device="cpu").run(
        write_annotated_video=False,
        progress_every=0,
    )

    assert len(result.events) == 2
    assert set(result.events["track_id"].tolist()) == {1, 2}
    assert all(result.events["direction"] == "out")
```

- [ ] **Step 2: Write regression test for lateral walker crossing foreground**

Add this test to `tests/test_pipeline.py`:

```python
def test_lateral_walker_along_line_does_not_count(tmp_path: Path) -> None:
    video_path = tmp_path / "entrance.mp4"
    _write_video(video_path, frame_count=8)
    output_dir = tmp_path / "output"
    fake_model = FakeModel(
        frames=[
            [((20.0, 40.0, 40.0, 55.0), 3, 0.85)],
            [((30.0, 40.0, 50.0, 55.0), 3, 0.85)],
            [((40.0, 40.0, 60.0, 55.0), 3, 0.85)],
            [((50.0, 40.0, 70.0, 55.0), 3, 0.85)],
            [((60.0, 40.0, 80.0, 55.0), 3, 0.85)],
            [((70.0, 40.0, 90.0, 55.0), 3, 0.85)],
            [((80.0, 40.0, 100.0, 55.0), 3, 0.85)],
            [((90.0, 40.0, 110.0, 55.0), 3, 0.85)],
        ]
    )
    config = RunConfig(
        paths=PathConfig(project_root=tmp_path, video_path=video_path, output_dir=output_dir),
        counting=CountingConfig(line=((10, 50), (90, 50)), gate_stable_frames=2),
        model=ModelConfig(model_name="fake.pt"),
        processing=ProcessingConfig(max_frames=8),
    )

    result = PeopleCounter(config=config, model=fake_model, device="cpu").run(
        write_annotated_video=False,
        progress_every=0,
    )

    assert len(result.events) == 0
```

- [ ] **Step 3: Write regression test for person initialized inside doorway**

Add this test to `tests/test_pipeline.py`:

```python
def test_person_appearing_inside_door_does_not_count(tmp_path: Path) -> None:
    video_path = tmp_path / "entrance.mp4"
    _write_video(video_path, frame_count=5)
    output_dir = tmp_path / "output"
    fake_model = FakeModel(
        frames=[
            [(_box_with_bottom_center_y(80.0), 9, 0.90)],
            [(_box_with_bottom_center_y(80.0), 9, 0.90)],
            [(_box_with_bottom_center_y(80.0), 9, 0.90)],
            [(_box_with_bottom_center_y(80.0), 9, 0.90)],
            [(_box_with_bottom_center_y(80.0), 9, 0.90)],
        ]
    )
    config = RunConfig(
        paths=PathConfig(project_root=tmp_path, video_path=video_path, output_dir=output_dir),
        counting=CountingConfig(line=((10, 50), (90, 50)), gate_stable_frames=2),
        model=ModelConfig(model_name="fake.pt"),
        processing=ProcessingConfig(max_frames=5),
    )

    result = PeopleCounter(config=config, model=fake_model, device="cpu").run(
        write_annotated_video=False,
        progress_every=0,
    )

    assert len(result.events) == 0
```

- [ ] **Step 4: Write regression test for footpoint jump rejection**

Add this test to `tests/test_pipeline.py`:

```python
def test_footpoint_jump_rejected_by_smoother(tmp_path: Path) -> None:
    video_path = tmp_path / "entrance.mp4"
    _write_video(video_path, frame_count=6)
    output_dir = tmp_path / "output"
    fake_model = FakeModel(
        frames=[
            [(_box_with_bottom_center_y(20.0), 5, 0.90)],
            [(_box_with_bottom_center_y(20.0), 5, 0.90)],
            [(_box_with_bottom_center_y(20.0), 5, 0.90)],
            [(_box_with_bottom_center_y(200.0), 5, 0.91)],
            [(_box_with_bottom_center_y(200.0), 5, 0.92)],
            [(_box_with_bottom_center_y(200.0), 5, 0.93)],
        ]
    )
    config = RunConfig(
        paths=PathConfig(project_root=tmp_path, video_path=video_path, output_dir=output_dir),
        counting=CountingConfig(
            line=((10, 50), (90, 50)),
            gate_stable_frames=2,
            max_jump_px=30.0,
        ),
        model=ModelConfig(model_name="fake.pt"),
        processing=ProcessingConfig(max_frames=6),
    )

    result = PeopleCounter(config=config, model=fake_model, device="cpu").run(
        write_annotated_video=False,
        progress_every=0,
    )

    assert len(result.events) == 0
```

- [ ] **Step 5: Run regression tests**

Run:

```bash
python -m pytest tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 6: Recalibrate default line and corridor**

Run the ROI preview to inspect the actual doorway:

```bash
entrance-counter --video resources/data/entrance.mov --output-dir output --no-preview --max-frames 1 --skip-annotated-video
```

Open `output/entrance_roi_preview.jpg` and identify the door threshold. Update the default line and corridor in `src/entrance_counter/config.py` to match the actual doorway. Example (coordinates must be calibrated from the preview):

```python
    line: Line = ((220, 520), (480, 530))
    corridor: Polygon = ((180, 380), (520, 380), (520, 650), (180, 650))
```

Update `test_run_config_defaults_match_current_baseline_parameters` in `tests/test_config.py` to match the new defaults.

Update the `--line` default in `src/entrance_counter/cli.py` to match:

```python
    parser.add_argument("--line", default="220,520,480,530", help="Counting line as x1,y1,x2,y2.")
```

Update `test_cli_defaults_match_run_config` in `tests/test_cli.py` to match.

- [ ] **Step 7: Run full test suite**

Run:

```bash
python -m pytest -q
```

Expected: All tests pass (40+ tests).

- [ ] **Step 8: Run smoke CLI with gate and corridor**

Run:

```bash
entrance-counter --video resources/data/entrance.mov --output-dir output --max-frames 300 --skip-annotated-video --no-preview --gate-stable-frames 3 --write-frame-trace
```

Expected: Command completes, writes `output/entrance_people_count_events.csv`, `output/entrance_people_count_summary.csv`, and `output/entrance_frame_trace.csv`.

- [ ] **Step 9: Commit**

```bash
git add src/entrance_counter/config.py src/entrance_counter/cli.py tests/test_config.py tests/test_cli.py tests/test_pipeline.py
git commit -m "feat: add regression fixtures and recalibrate door defaults"
```

- [ ] **Step 10: Final verification**

Run:

```bash
python -m pytest -q
git status --short
git log --oneline -10
```

Expected: All tests pass, clean working tree, commit history shows: config, geometry, pipeline, cleanup, stabilization, rendering, CLI, regression.
