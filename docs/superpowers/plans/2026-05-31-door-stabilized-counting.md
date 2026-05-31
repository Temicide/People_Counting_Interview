# Door Stabilized Counting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the entrance counter robust to a handheld camera by evaluating crossing events in a doorway-stabilized coordinate system instead of raw moving-frame pixels.

**Architecture:** Add optional background stabilization that estimates a current-frame-to-reference transform from doorway/background features, transforms person footpoints into the reference coordinate system, and transforms the configured reference counting line forward for annotation. Keep the original fixed-pixel behavior as the default-compatible fallback when stabilization is disabled or transform quality is insufficient.

**Tech Stack:** Python 3.10+, OpenCV ORB/BFMatcher/RANSAC, NumPy, pandas, Ultralytics YOLO/ByteTrack or BoT-SORT, pytest.

---

## File Structure

- Modify `src/entrance_counter/config.py`: add `StabilizationConfig` and attach it to `RunConfig`.
- Modify `src/entrance_counter/cli.py`: expose stabilization flags and tracker override.
- Modify `src/entrance_counter/geometry.py`: add transform helpers and track-gap bookkeeping.
- Create `src/entrance_counter/stabilization.py`: estimate and carry frame-to-reference transforms.
- Modify `src/entrance_counter/rendering.py`: draw a per-frame counting line while preserving the existing API.
- Modify `src/entrance_counter/pipeline.py`: integrate stabilization into counting, event logging, rendering, and summary metadata.
- Modify `tests/test_config.py`: validate stabilization defaults and CLI config mapping.
- Modify `tests/test_geometry.py`: validate point/line transform helpers and track reset helpers.
- Create `tests/test_stabilization.py`: validate identity fallback, translation estimation, quality gating, and forward-line projection.
- Modify `tests/test_pipeline.py`: validate that stabilized counting ignores pure camera translation and counts true reference-frame motion.
- Modify `tests/test_cli.py`: validate new command-line flags.
- Modify `README.md`: document handheld-camera behavior and usage.

## Tasks

### Task 1: Configuration And CLI Surface

**Files:**
- Modify: `src/entrance_counter/config.py`
- Modify: `src/entrance_counter/cli.py`
- Test: `tests/test_config.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing config and CLI tests**

Add this import in `tests/test_config.py`:

```python
from entrance_counter.config import StabilizationConfig
```

Add these tests to `tests/test_config.py`:

```python
def test_stabilization_config_defaults_disable_stabilization() -> None:
    config = StabilizationConfig()

    assert config.enabled is False
    assert config.reference_second == 25.0
    assert config.max_features == 1200
    assert config.min_matches == 18
    assert config.min_inlier_ratio == 0.35
    assert config.max_reprojection_error == 4.0
    assert config.smoothing_alpha == 0.25


def test_run_config_contains_stabilization_config() -> None:
    config = RunConfig()

    assert config.stabilization == StabilizationConfig()
```

Add these assertions to `tests/test_cli.py::test_cli_defaults_match_run_config`:

```python
assert config.model.tracker == "bytetrack.yaml"
assert config.stabilization.enabled is False
assert config.stabilization.reference_second == 25.0
```

Add these arguments to `tests/test_cli.py::test_cli_maps_overrides_to_run_config`:

```python
"--tracker",
"botsort.yaml",
"--stabilize",
"--stabilization-reference-second",
"12.5",
"--stabilization-max-features",
"800",
"--stabilization-min-matches",
"12",
"--stabilization-min-inlier-ratio",
"0.45",
"--stabilization-max-reprojection-error",
"3.5",
"--stabilization-smoothing-alpha",
"0.4",
```

Add these assertions to the same CLI override test:

```python
assert config.model.tracker == "botsort.yaml"
assert config.stabilization.enabled is True
assert config.stabilization.reference_second == 12.5
assert config.stabilization.max_features == 800
assert config.stabilization.min_matches == 12
assert config.stabilization.min_inlier_ratio == 0.45
assert config.stabilization.max_reprojection_error == 3.5
assert config.stabilization.smoothing_alpha == 0.4
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_config.py tests/test_cli.py -q
```

Expected: FAIL because `StabilizationConfig` and CLI flags do not exist yet.

- [ ] **Step 3: Implement config and CLI flags**

In `src/entrance_counter/config.py`, add:

```python
@dataclass(frozen=True)
class StabilizationConfig:
    enabled: bool = False
    reference_second: float = 25.0
    max_features: int = 1200
    min_matches: int = 18
    min_inlier_ratio: float = 0.35
    max_reprojection_error: float = 4.0
    smoothing_alpha: float = 0.25

    def __post_init__(self) -> None:
        if self.reference_second < 0:
            raise ValueError("reference_second must be non-negative")
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if self.min_matches < 1:
            raise ValueError("min_matches must be positive")
        if not 0.0 <= self.min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be between 0 and 1")
        if self.max_reprojection_error <= 0:
            raise ValueError("max_reprojection_error must be positive")
        if not 0.0 <= self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be between 0 and 1")
```

Change `RunConfig` to:

```python
@dataclass(frozen=True)
class RunConfig:
    paths: PathConfig = PathConfig()
    counting: CountingConfig = CountingConfig()
    model: ModelConfig = ModelConfig()
    processing: ProcessingConfig = ProcessingConfig()
    stabilization: StabilizationConfig = StabilizationConfig()
```

In `src/entrance_counter/cli.py`, import `StabilizationConfig`, add parser flags:

```python
parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config, for example bytetrack.yaml or botsort.yaml.")
parser.add_argument("--stabilize", action="store_true", help="Compensate handheld camera motion before line-crossing geometry.")
parser.add_argument("--stabilization-reference-second", type=float, default=25.0, help="Timestamp used as the doorway reference frame.")
parser.add_argument("--stabilization-max-features", type=int, default=1200, help="Maximum ORB features for background alignment.")
parser.add_argument("--stabilization-min-matches", type=int, default=18, help="Minimum feature matches required for a stabilization transform.")
parser.add_argument("--stabilization-min-inlier-ratio", type=float, default=0.35, help="Minimum RANSAC inlier ratio required for stabilization.")
parser.add_argument("--stabilization-max-reprojection-error", type=float, default=4.0, help="RANSAC reprojection threshold in pixels.")
parser.add_argument("--stabilization-smoothing-alpha", type=float, default=0.25, help="EMA alpha for smoothing stabilization transforms.")
```

Set `tracker=args.tracker` in `ModelConfig` and build:

```python
stabilization = StabilizationConfig(
    enabled=args.stabilize,
    reference_second=args.stabilization_reference_second,
    max_features=args.stabilization_max_features,
    min_matches=args.stabilization_min_matches,
    min_inlier_ratio=args.stabilization_min_inlier_ratio,
    max_reprojection_error=args.stabilization_max_reprojection_error,
    smoothing_alpha=args.stabilization_smoothing_alpha,
)
return RunConfig(paths=paths, counting=counting, model=model, processing=processing, stabilization=stabilization)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_config.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/entrance_counter/config.py src/entrance_counter/cli.py tests/test_config.py tests/test_cli.py
git commit -m "Add stabilization configuration"
```

### Task 2: Geometry Transform Helpers

**Files:**
- Modify: `src/entrance_counter/geometry.py`
- Test: `tests/test_geometry.py`

- [ ] **Step 1: Write failing geometry tests**

Add imports in `tests/test_geometry.py`:

```python
import numpy as np
from entrance_counter.geometry import reset_track_state, transform_line, transform_point
```

Add tests:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_geometry.py -q
```

Expected: FAIL because transform helpers and `last_seen_frame` do not exist.

- [ ] **Step 3: Implement transform helpers and track reset**

In `TrackState`, add:

```python
last_seen_frame: int | None = None
```

Add functions:

```python
def transform_point(point: Point, matrix: np.ndarray | None) -> Point:
    if matrix is None:
        return point
    x, y = point
    if matrix.shape == (2, 3):
        transformed = matrix @ np.array([x, y, 1.0], dtype=float)
        return (float(transformed[0]), float(transformed[1]))
    if matrix.shape == (3, 3):
        transformed = matrix @ np.array([x, y, 1.0], dtype=float)
        scale = max(float(transformed[2]), 1e-9)
        return (float(transformed[0] / scale), float(transformed[1] / scale))
    raise ValueError("matrix must have shape (2, 3) or (3, 3)")


def transform_line(line: Line, matrix: np.ndarray | None) -> Line:
    p1 = transform_point((float(line[0][0]), float(line[0][1])), matrix)
    p2 = transform_point((float(line[1][0]), float(line[1][1])), matrix)
    return ((int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))))


def reset_track_state(state: TrackState) -> None:
    state.points.clear()
    state.sides.clear()
    state.first_frame = None
    state.first_point = None
    state.first_side = None
    state.last_seen_frame = None
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_geometry.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/entrance_counter/geometry.py tests/test_geometry.py
git commit -m "Add stabilized geometry helpers"
```

### Task 3: Camera Stabilizer

**Files:**
- Create: `src/entrance_counter/stabilization.py`
- Test: `tests/test_stabilization.py`

- [ ] **Step 1: Write failing stabilizer tests**

Create `tests/test_stabilization.py`:

```python
import cv2
import numpy as np

from entrance_counter.config import StabilizationConfig
from entrance_counter.geometry import transform_line, transform_point
from entrance_counter.stabilization import FrameTransform, FrameStabilizer, identity_affine


def _feature_frame(shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    for x, y in [(40, 40), (180, 45), (60, 130), (190, 135), (120, 90), (85, 80), (150, 110)]:
        cv2.circle(frame, (x + shift_x, y + shift_y), 8, (255, 255, 255), -1)
        cv2.rectangle(frame, (x + shift_x - 4, y + shift_y + 12), (x + shift_x + 12, y + shift_y + 22), (180, 180, 180), -1)
    return frame


def test_identity_affine_keeps_points_unchanged() -> None:
    matrix = identity_affine()

    assert transform_point((10.0, 20.0), matrix) == (10.0, 20.0)


def test_stabilizer_returns_identity_when_disabled() -> None:
    stabilizer = FrameStabilizer(StabilizationConfig(enabled=False), _feature_frame())

    transform = stabilizer.estimate(_feature_frame(shift_x=8, shift_y=4))

    assert transform.valid is False
    np.testing.assert_allclose(transform.current_to_reference, identity_affine())
    np.testing.assert_allclose(transform.reference_to_current, identity_affine())


def test_stabilizer_estimates_translation_to_reference() -> None:
    reference = _feature_frame()
    current = _feature_frame(shift_x=10, shift_y=6)
    stabilizer = FrameStabilizer(
        StabilizationConfig(
            enabled=True,
            max_features=500,
            min_matches=8,
            min_inlier_ratio=0.25,
            max_reprojection_error=5.0,
            smoothing_alpha=1.0,
        ),
        reference,
    )

    transform = stabilizer.estimate(current)

    assert transform.valid is True
    aligned = transform_point((110.0, 86.0), transform.current_to_reference)
    assert abs(aligned[0] - 100.0) < 3.0
    assert abs(aligned[1] - 80.0) < 3.0


def test_frame_transform_projects_reference_line_to_current() -> None:
    transform = FrameTransform(
        current_to_reference=np.array([[1.0, 0.0, -10.0], [0.0, 1.0, -6.0]], dtype=float),
        reference_to_current=np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 6.0]], dtype=float),
        valid=True,
        match_count=20,
        inlier_count=18,
        inlier_ratio=0.9,
    )

    assert transform_line(((20, 30), (80, 30)), transform.reference_to_current) == ((30, 36), (90, 36))
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_stabilization.py -q
```

Expected: FAIL because `entrance_counter.stabilization` does not exist.

- [ ] **Step 3: Implement stabilizer**

Create `src/entrance_counter/stabilization.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from entrance_counter.config import StabilizationConfig


def identity_affine() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)


def _to_homogeneous(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (3, 3):
        return matrix.astype(float)
    if matrix.shape == (2, 3):
        return np.vstack([matrix.astype(float), np.array([0.0, 0.0, 1.0], dtype=float)])
    raise ValueError("matrix must have shape (2, 3) or (3, 3)")


def _from_homogeneous(matrix: np.ndarray) -> np.ndarray:
    return matrix[:2, :].astype(float)


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    return _from_homogeneous(np.linalg.inv(_to_homogeneous(matrix)))


@dataclass(frozen=True)
class FrameTransform:
    current_to_reference: np.ndarray
    reference_to_current: np.ndarray
    valid: bool
    match_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0


class FrameStabilizer:
    def __init__(self, config: StabilizationConfig, reference_frame: np.ndarray) -> None:
        self.config = config
        self.reference_frame = reference_frame
        self._previous_current_to_reference: np.ndarray | None = None
        self._orb = cv2.ORB_create(nfeatures=config.max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._reference_keypoints, self._reference_descriptors = self._detect(reference_frame)

    def estimate(self, frame: np.ndarray) -> FrameTransform:
        if not self.config.enabled:
            return self._identity(valid=False)
        if self._reference_descriptors is None or len(self._reference_keypoints) < self.config.min_matches:
            return self._identity(valid=False)

        current_keypoints, current_descriptors = self._detect(frame)
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

    def _detect(self, frame: np.ndarray) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        return tuple(keypoints or ()), descriptors

    def _identity(self, valid: bool) -> FrameTransform:
        matrix = identity_affine()
        return FrameTransform(current_to_reference=matrix, reference_to_current=matrix.copy(), valid=valid)

    def _fallback(self, match_count: int = 0, inlier_count: int = 0, inlier_ratio: float = 0.0) -> FrameTransform:
        if self._previous_current_to_reference is None:
            matrix = identity_affine()
            valid = False
        else:
            matrix = self._previous_current_to_reference
            valid = True
        return FrameTransform(
            current_to_reference=matrix,
            reference_to_current=invert_affine(matrix),
            valid=valid,
            match_count=match_count,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_stabilization.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/entrance_counter/stabilization.py tests/test_stabilization.py
git commit -m "Add frame stabilizer"
```

### Task 4: Pipeline Integration And Track-Gap Guard

**Files:**
- Modify: `src/entrance_counter/pipeline.py`
- Modify: `src/entrance_counter/rendering.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_video_rendering.py`

- [ ] **Step 1: Write failing pipeline tests**

In `tests/test_pipeline.py`, add:

```python
def test_people_counter_uses_stabilized_reference_points_for_crossing(tmp_path: Path) -> None:
    video_path = tmp_path / "entrance.mp4"
    _write_video(video_path, frame_count=5)
    output_dir = tmp_path / "output"
    fake_model = FakeModel(
        frames=[
            [(_box_with_bottom_center_y(30.0), 7, 0.90)],
            [(_box_with_bottom_center_y(35.0), 7, 0.91)],
            [(_box_with_bottom_center_y(40.0), 7, 0.92)],
            [(_box_with_bottom_center_y(90.0), 7, 0.93)],
            [(_box_with_bottom_center_y(100.0), 7, 0.94)],
        ]
    )
    config = RunConfig(
        paths=PathConfig(project_root=tmp_path, video_path=video_path, output_dir=output_dir),
        counting=CountingConfig(line=((10, 50), (90, 50))),
        model=ModelConfig(model_name="fake.pt"),
        processing=ProcessingConfig(max_frames=5),
        stabilization=StabilizationConfig(enabled=True, reference_second=0.0),
    )

    result = PeopleCounter(
        config=config,
        model=fake_model,
        device="cpu",
        transform_provider=lambda frame, frame_idx: np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -10.0]], dtype=float),
    ).run(write_annotated_video=False, progress_every=0)

    assert len(result.events) == 1
    assert result.events.iloc[0]["bottom_center_y"] == 90.0
    assert result.events.iloc[0]["reference_bottom_center_y"] == 80.0
    assert result.summary.iloc[0]["stabilization_enabled"] is True


def test_people_counter_resets_stale_track_before_crossing(tmp_path: Path) -> None:
    video_path = tmp_path / "entrance.mp4"
    _write_video(video_path, frame_count=8)
    output_dir = tmp_path / "output"
    fake_model = FakeModel(
        frames=[
            [(_box_with_bottom_center_y(20.0), 7, 0.90)],
            [],
            [],
            [],
            [],
            [],
            [],
            [(_box_with_bottom_center_y(90.0), 7, 0.94)],
        ]
    )
    config = RunConfig(
        paths=PathConfig(project_root=tmp_path, video_path=video_path, output_dir=output_dir),
        counting=CountingConfig(line=((10, 50), (90, 50)), max_track_gap_frames=3),
        model=ModelConfig(model_name="fake.pt"),
        processing=ProcessingConfig(max_frames=8),
    )

    result = PeopleCounter(config=config, model=fake_model, device="cpu").run(write_annotated_video=False, progress_every=0)

    assert len(result.events) == 0
```

In `tests/test_video_rendering.py`, add:

```python
def test_draw_annotations_accepts_frame_specific_line() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    config = CountingConfig(line=((20, 60), (120, 60)))

    annotated = draw_annotations(
        frame=frame,
        boxes_xyxy=np.empty((0, 4), dtype=float),
        track_ids=np.empty((0,), dtype=int),
        states={},
        counts={"in": 1, "out": 2},
        frame_idx=30,
        fps=10.0,
        config=config,
        line=((30, 70), (130, 70)),
    )

    assert annotated.shape == frame.shape
    assert int(annotated[70].sum()) > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_pipeline.py tests/test_video_rendering.py -q
```

Expected: FAIL because pipeline does not accept `transform_provider`, event columns do not include reference points, `max_track_gap_frames` is missing, and rendering has no frame-specific line parameter.

- [ ] **Step 3: Implement pipeline integration**

Add `max_track_gap_frames: int = 45` to `CountingConfig` and validate it is non-negative.

Extend `EVENT_COLUMNS` in `pipeline.py` with:

```python
"reference_bottom_center_x",
"reference_bottom_center_y",
"stabilization_valid",
```

Modify `PeopleCounter.__init__` to accept:

```python
transform_provider: Callable[[Any, int], np.ndarray | None] | None = None
```

Use a `FrameStabilizer` when `config.stabilization.enabled` and no provider is injected. On each processed frame:

```python
current_to_reference = None
reference_to_current = None
stabilization_valid = False
if transform_provider is not None:
    current_to_reference = transform_provider(frame, frame_idx)
    reference_to_current = invert_affine(current_to_reference) if current_to_reference is not None else None
    stabilization_valid = current_to_reference is not None
elif stabilizer is not None:
    transform = stabilizer.estimate(frame)
    current_to_reference = transform.current_to_reference
    reference_to_current = transform.reference_to_current
    stabilization_valid = transform.valid
```

For each detection, compute:

```python
image_point = bottom_center_xy(xyxy)
point = transform_point(image_point, current_to_reference)
side = normalized_line_side(point, self.config.counting.line)
```

Before evaluating crossing, reset stale track state:

```python
if state.last_seen_frame is not None and frame_idx - state.last_seen_frame > self.config.counting.max_track_gap_frames:
    reset_track_state(state)
```

After appending points and sides:

```python
state.last_seen_frame = frame_idx
```

Record both image and reference points in events.

For rendering:

```python
display_line = transform_line(self.config.counting.line, reference_to_current)
annotated = draw_annotations(..., line=display_line)
```

Add summary fields:

```python
"stabilization_enabled": self.config.stabilization.enabled,
"stabilization_reference_second": self.config.stabilization.reference_second,
```

In `rendering.py`, add optional `line: Line | None = None` to `draw_annotations` and use `line or config.line` for drawing.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_pipeline.py tests/test_video_rendering.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/entrance_counter/config.py src/entrance_counter/pipeline.py src/entrance_counter/rendering.py tests/test_pipeline.py tests/test_video_rendering.py
git commit -m "Integrate stabilized counting pipeline"
```

### Task 5: Documentation And Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-05-31-door-stabilized-counting.md`

- [ ] **Step 1: Update README usage**

Add a section:

```markdown
## Handheld Camera Stabilization

The default line-crossing mode assumes the camera is static. For handheld videos where the doorway shifts in the frame, enable stabilization so crossing geometry is evaluated in a doorway reference frame:

```bash
entrance-counter --video resources/data/entrance.mov --output-dir output --stabilize --stabilization-reference-second 25
```

The configured `--line` is interpreted in the reference frame selected by `--stabilization-reference-second`. The annotated video projects that reference line back into each current frame. If the doorway alignment is weak or the line drifts, choose a reference second where the doorway is clear and increase feature quality by avoiding frames dominated by moving people.

For handheld footage with identity switches, try BoT-SORT:

```bash
entrance-counter --video resources/data/entrance.mov --output-dir output --stabilize --tracker botsort.yaml
```
```

- [ ] **Step 2: Run full automated verification**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run smoke CLI without model download**

Run:

```bash
python -m entrance_counter --video resources/data/entrance.mov --output-dir output --max-frames 3 --skip-annotated-video --no-preview --stabilize
```

Expected: command completes and writes CSV outputs. If model weights are missing and network is unavailable, record that manual smoke verification is blocked by model download.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/superpowers/plans/2026-05-31-door-stabilized-counting.md
git commit -m "Document handheld camera stabilization"
```

- [ ] **Step 5: Final status**

Run:

```bash
git status --short
git log --oneline -5
```

Expected: clean working tree, recent commits show stabilization config, geometry helpers, frame stabilizer, pipeline integration, and docs.
