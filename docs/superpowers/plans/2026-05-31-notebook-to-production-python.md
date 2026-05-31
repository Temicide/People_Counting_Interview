# Notebook To Production Python Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the notebook-only entrance people counting workflow with a tested, importable Python package and command-line entrypoint.

**Architecture:** Move the current `src/src.ipynb` logic into focused modules under `src/entrance_counter/`: configuration, geometry/counting decisions, video I/O, rendering, model loading, pipeline orchestration, and CLI. Keep YOLO/ByteTrack behavior equivalent to the notebook, but make pure logic unit-testable without loading the model or processing the full video.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, pandas, PyTorch, Ultralytics YOLO/ByteTrack, setuptools, pytest.

---

## File Structure

- Create `pyproject.toml`: package metadata, dependencies, console script, pytest config.
- Modify `requirements.txt`: install the local package through `pip install -r requirements.txt`.
- Create `src/entrance_counter/__init__.py`: package exports and version.
- Create `src/entrance_counter/config.py`: dataclasses for paths, counting parameters, model parameters, processing parameters, line parsing, display-path formatting.
- Create `src/entrance_counter/geometry.py`: pure geometry helpers and per-track state.
- Create `src/entrance_counter/video_io.py`: video metadata, sampled frame loading, output directory creation, video writer creation.
- Create `src/entrance_counter/rendering.py`: OpenCV drawing helpers for the counting line, tracks, counts, and overlay panel.
- Create `src/entrance_counter/modeling.py`: device selection, YOLO model loading, Ultralytics result extraction.
- Create `src/entrance_counter/pipeline.py`: `PeopleCounter` orchestration class that writes ROI preview, event CSV, summary CSV, and optional annotated video.
- Create `src/entrance_counter/cli.py`: `entrance-counter` command implementation.
- Create `src/entrance_counter/__main__.py`: `python -m entrance_counter` entrypoint.
- Create `tests/test_config.py`: config and counting-line parsing tests.
- Create `tests/test_geometry.py`: deterministic line-crossing and stabilization tests.
- Create `tests/test_video_rendering.py`: OpenCV video/frame and drawing tests.
- Create `tests/test_modeling.py`: Ultralytics adapter tests using fake tensor objects.
- Create `tests/test_pipeline.py`: fake-model integration test against a tiny generated video.
- Create `tests/test_cli.py`: CLI parser-to-config tests.
- Modify `README.md`: document production Python workflow and CLI.
- Modify `AGENTS.md`: update repository guidance so production Python code is the source of truth.
- Delete `src/src.ipynb`: remove the notebook as the primary implementation after parity is verified.

## Tasks

### Task 1: Package Metadata And Configuration

**Files:**
- Create: `pyproject.toml`
- Modify: `requirements.txt`
- Create: `src/entrance_counter/__init__.py`
- Create: `src/entrance_counter/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing config tests**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from entrance_counter.config import (
    CountingConfig,
    PathConfig,
    RunConfig,
    display_path,
    parse_counting_line,
)


def test_parse_counting_line_accepts_plain_four_integer_string() -> None:
    assert parse_counting_line("330,785,870,812") == ((330, 785), (870, 812))


def test_parse_counting_line_accepts_notebook_tuple_string() -> None:
    assert parse_counting_line("((330, 785), (870, 812))") == ((330, 785), (870, 812))


def test_parse_counting_line_rejects_wrong_coordinate_count() -> None:
    with pytest.raises(ValueError, match="four integer coordinates"):
        parse_counting_line("330,785,870")


def test_path_config_derives_output_paths(tmp_path: Path) -> None:
    paths = PathConfig(
        project_root=tmp_path,
        video_path=tmp_path / "resources" / "data" / "entrance.mov",
        output_dir=tmp_path / "output",
    )

    assert paths.model_dir == tmp_path / "output" / "models"
    assert paths.roi_preview_path == tmp_path / "output" / "entrance_roi_preview.jpg"
    assert paths.annotated_video_path == tmp_path / "output" / "entrance_people_count_annotated.mp4"
    assert paths.events_csv_path == tmp_path / "output" / "entrance_people_count_events.csv"
    assert paths.summary_csv_path == tmp_path / "output" / "entrance_people_count_summary.csv"


def test_run_config_defaults_match_current_notebook_parameters() -> None:
    config = RunConfig()

    assert config.counting.line == ((330, 785), (870, 812))
    assert config.counting.positive_direction_label == "out"
    assert config.counting.negative_direction_label == "in"
    assert config.counting.min_side_abs == 8.0
    assert config.counting.min_track_age_frames == 3
    assert config.counting.count_cooldown_frames == 18
    assert config.model.model_name == "yolov8n.pt"
    assert config.model.confidence == 0.22
    assert config.model.image_size == 960
    assert config.processing.process_every_n_frames == 1
    assert config.processing.max_frames is None


def test_display_path_prefers_project_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "project"
    path = root / "output" / "summary.csv"

    assert display_path(path, root) == "output/summary.csv"


def test_counting_config_rejects_duplicate_direction_labels() -> None:
    with pytest.raises(ValueError, match="must differ"):
        CountingConfig(positive_direction_label="in", negative_direction_label="in")
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'entrance_counter'`.

- [ ] **Step 3: Add package metadata, dependency install path, and config module**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "entrance-counter"
version = "0.1.0"
description = "Production Python pipeline for entrance people counting with YOLO and ByteTrack."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "opencv-python>=4.9,<5",
    "numpy>=1.26,<3",
    "pandas>=2.2,<3",
    "torch>=2.2",
    "ultralytics>=8.2",
    "lap>=0.5.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "ruff>=0.5",
]

[project.scripts]
entrance-counter = "entrance_counter.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Replace `requirements.txt` with:

```text
-e .
```

Create `src/entrance_counter/__init__.py`:

```python
"""Entrance people counting package."""

__version__ = "0.1.0"
```

Create `src/entrance_counter/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

Point: TypeAlias = tuple[float, float]
Line: TypeAlias = tuple[tuple[int, int], tuple[int, int]]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "resources" / "data" / "entrance.mov"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_MODEL_NAME = "yolov8n.pt"


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = PROJECT_ROOT
    video_path: Path = DEFAULT_VIDEO_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR

    @property
    def model_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def roi_preview_path(self) -> Path:
        return self.output_dir / "entrance_roi_preview.jpg"

    @property
    def annotated_video_path(self) -> Path:
        return self.output_dir / "entrance_people_count_annotated.mp4"

    @property
    def events_csv_path(self) -> Path:
        return self.output_dir / "entrance_people_count_events.csv"

    @property
    def summary_csv_path(self) -> Path:
        return self.output_dir / "entrance_people_count_summary.csv"


@dataclass(frozen=True)
class CountingConfig:
    line: Line = ((330, 785), (870, 812))
    positive_direction_label: str = "out"
    negative_direction_label: str = "in"
    invert_directions: bool = False
    min_side_abs: float = 8.0
    min_track_age_frames: int = 3
    count_cooldown_frames: int = 18
    min_displacement_px: float = 18.0
    late_init_buffer_px: float = 45.0
    late_init_max_age_frames: int = 18
    draw_track_history: bool = True

    def __post_init__(self) -> None:
        if self.positive_direction_label == self.negative_direction_label:
            raise ValueError("positive_direction_label and negative_direction_label must differ")
        if self.min_track_age_frames < 0:
            raise ValueError("min_track_age_frames must be non-negative")
        if self.count_cooldown_frames < 0:
            raise ValueError("count_cooldown_frames must be non-negative")
        if self.min_displacement_px < 0:
            raise ValueError("min_displacement_px must be non-negative")


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = DEFAULT_MODEL_NAME
    person_class_id: int = 0
    confidence: float = 0.22
    iou_threshold: float = 0.50
    image_size: int = 960
    tracker: str = "bytetrack.yaml"


@dataclass(frozen=True)
class ProcessingConfig:
    process_every_n_frames: int = 1
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if self.process_every_n_frames < 1:
            raise ValueError("process_every_n_frames must be at least 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be positive when provided")


@dataclass(frozen=True)
class RunConfig:
    paths: PathConfig = PathConfig()
    counting: CountingConfig = CountingConfig()
    model: ModelConfig = ModelConfig()
    processing: ProcessingConfig = ProcessingConfig()


def parse_counting_line(raw: str) -> Line:
    cleaned = (
        raw.replace("(", " ")
        .replace(")", " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace(";", ",")
    )
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("counting line must contain four integer coordinates: x1,y1,x2,y2")

    try:
        x1, y1, x2, y2 = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("counting line coordinates must be integers") from exc

    return ((x1, y1), (x2, y2))


def display_path(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)
```

- [ ] **Step 4: Run the config tests to verify they pass**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: PASS with `7 passed`.

- [ ] **Step 5: Commit package metadata and config**

Run:

```bash
git add pyproject.toml requirements.txt src/entrance_counter/__init__.py src/entrance_counter/config.py tests/test_config.py
git commit -m "Add production package configuration"
```

Expected: commit succeeds.

### Task 2: Geometry And Counting Decisions

**Files:**
- Create: `src/entrance_counter/geometry.py`
- Test: `tests/test_geometry.py`

- [ ] **Step 1: Write the failing geometry tests**

Create `tests/test_geometry.py`:

```python
from collections import deque

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
    segments_intersect,
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
```

- [ ] **Step 2: Run the geometry tests to verify they fail**

Run:

```bash
python -m pytest tests/test_geometry.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'entrance_counter.geometry'`.

- [ ] **Step 3: Add the geometry module**

Create `src/entrance_counter/geometry.py`:

```python
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
```

- [ ] **Step 4: Run config and geometry tests to verify they pass**

Run:

```bash
python -m pytest tests/test_config.py tests/test_geometry.py -q
```

Expected: PASS with `15 passed`.

- [ ] **Step 5: Commit geometry logic**

Run:

```bash
git add src/entrance_counter/geometry.py tests/test_geometry.py
git commit -m "Extract line crossing geometry"
```

Expected: commit succeeds.

### Task 3: Video I/O And Rendering Helpers

**Files:**
- Create: `src/entrance_counter/video_io.py`
- Create: `src/entrance_counter/rendering.py`
- Test: `tests/test_video_rendering.py`

- [ ] **Step 1: Write the failing video and rendering tests**

Create `tests/test_video_rendering.py`:

```python
from pathlib import Path

import cv2
import numpy as np
import pytest

from entrance_counter.config import CountingConfig, PathConfig
from entrance_counter.rendering import draw_annotations, draw_counting_line
from entrance_counter.video_io import create_video_writer, ensure_output_dirs, frame_at, read_video_metadata


def _write_tiny_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened()
    for idx in range(frame_count):
        frame = np.full((48, 64, 3), idx * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_ensure_output_dirs_creates_output_and_model_dirs(tmp_path: Path) -> None:
    paths = PathConfig(project_root=tmp_path, video_path=tmp_path / "entrance.mov", output_dir=tmp_path / "output")

    ensure_output_dirs(paths)

    assert paths.output_dir.is_dir()
    assert paths.model_dir.is_dir()


def test_read_video_metadata_and_frame_at(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.mp4"
    _write_tiny_video(video_path)

    metadata = read_video_metadata(video_path)
    frame = frame_at(video_path, second=0.1)

    assert metadata.width == 64
    assert metadata.height == 48
    assert metadata.frame_count == 3
    assert metadata.fps > 0
    assert frame.shape == (48, 64, 3)


def test_read_video_metadata_raises_for_missing_video(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Could not open video"):
        read_video_metadata(tmp_path / "missing.mp4")


def test_create_video_writer_creates_parent_directory(tmp_path: Path) -> None:
    writer_path = tmp_path / "nested" / "out.mp4"

    writer = create_video_writer(writer_path, fps=10.0, size=(64, 48))
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    assert writer_path.exists()


def test_draw_counting_line_marks_frame_pixels() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    config = CountingConfig(line=((20, 60), (120, 60)))

    annotated = draw_counting_line(frame, config.line, config)

    assert annotated.shape == frame.shape
    assert int(annotated.sum()) > 0


def test_draw_annotations_handles_empty_tracks() -> None:
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
    )

    assert annotated.shape == frame.shape
    assert int(annotated.sum()) > 0
```

- [ ] **Step 2: Run the video and rendering tests to verify they fail**

Run:

```bash
python -m pytest tests/test_video_rendering.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `entrance_counter.rendering` or `entrance_counter.video_io`.

- [ ] **Step 3: Add video I/O and rendering modules**

Create `src/entrance_counter/video_io.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from entrance_counter.config import PathConfig


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float | None


def ensure_output_dirs(paths: PathConfig) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)


def read_video_metadata(video_path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else None
    cap.release()

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
    )


def frame_at(video_path: Path, second: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame at {second:.2f}s from {video_path}")
    return frame


def create_video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer
```

Create `src/entrance_counter/rendering.py`:

```python
from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from entrance_counter.config import CountingConfig, Line
from entrance_counter.geometry import TrackState, bottom_center_xy


def draw_counting_line(
    frame: np.ndarray,
    line: Line,
    config: CountingConfig,
    thickness: int = 5,
) -> np.ndarray:
    annotated = frame.copy()
    p1, p2 = line
    cv2.line(annotated, p1, p2, (0, 220, 255), thickness, cv2.LINE_AA)
    cv2.circle(annotated, p1, 10, (0, 220, 255), -1, cv2.LINE_AA)
    cv2.circle(annotated, p2, 10, (0, 220, 255), -1, cv2.LINE_AA)

    midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    normal = np.array([-dy, dx], dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1.0)
    arrow_end = (int(midpoint[0] + normal[0] * 90), int(midpoint[1] + normal[1] * 90))

    cv2.arrowedLine(annotated, midpoint, arrow_end, (0, 120, 255), 4, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(
        annotated,
        f"+ = {config.positive_direction_label}",
        (arrow_end[0] + 12, arrow_end[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 120, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "Entrance counting line",
        (p1[0], max(45, p1[1] - 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 220, 255),
        3,
        cv2.LINE_AA,
    )
    return annotated


def draw_annotations(
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,
    track_ids: np.ndarray,
    states: Mapping[int, TrackState],
    counts: Mapping[str, int],
    frame_idx: int,
    fps: float,
    config: CountingConfig,
) -> np.ndarray:
    annotated = draw_counting_line(frame, config.line, config, thickness=4)

    for xyxy, track_id in zip(boxes_xyxy, track_ids):
        x1, y1, x2, y2 = map(int, xyxy)
        tid = int(track_id)
        point = bottom_center_xy(xyxy)
        color = (80, 220, 80)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.circle(annotated, (int(point[0]), int(point[1])), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"ID {tid}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        if config.draw_track_history and tid in states:
            pts = list(states[tid].points)
            for a, b in zip(pts[-16:-1], pts[-15:]):
                cv2.line(
                    annotated,
                    (int(a[0]), int(a[1])),
                    (int(b[0]), int(b[1])),
                    (80, 180, 255),
                    2,
                    cv2.LINE_AA,
                )

    panel_x, panel_y = 24, 28
    panel_w, panel_h = 390, 150
    overlay = annotated.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)

    total = counts.get(config.positive_direction_label, 0) + counts.get(config.negative_direction_label, 0)
    seconds = frame_idx / max(fps, 1e-6)
    lines = [
        f"People Count: {total}",
        (
            f"{config.negative_direction_label}: {counts.get(config.negative_direction_label, 0)}   "
            f"{config.positive_direction_label}: {counts.get(config.positive_direction_label, 0)}"
        ),
        f"Frame: {frame_idx}   Time: {seconds:0.1f}s",
    ]

    for idx, text in enumerate(lines):
        cv2.putText(
            annotated,
            text,
            (panel_x + 18, panel_y + 42 + idx * 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

    return annotated
```

- [ ] **Step 4: Run tests through rendering to verify they pass**

Run:

```bash
python -m pytest tests/test_config.py tests/test_geometry.py tests/test_video_rendering.py -q
```

Expected: PASS with `21 passed`.

- [ ] **Step 5: Commit video and rendering helpers**

Run:

```bash
git add src/entrance_counter/video_io.py src/entrance_counter/rendering.py tests/test_video_rendering.py
git commit -m "Add video IO and rendering helpers"
```

Expected: commit succeeds.

### Task 4: Model Loading And Ultralytics Result Extraction

**Files:**
- Create: `src/entrance_counter/modeling.py`
- Test: `tests/test_modeling.py`

- [ ] **Step 1: Write the failing model adapter tests**

Create `tests/test_modeling.py`:

```python
import numpy as np

from entrance_counter.modeling import extract_track_outputs


class FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = FakeTensor(np.array([[1.0, 2.0, 11.0, 22.0]], dtype=float))
        self.id = FakeTensor(np.array([42], dtype=float))
        self.conf = FakeTensor(np.array([0.87], dtype=float))


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None) -> None:
        self.boxes = boxes


def test_extract_track_outputs_returns_empty_arrays_when_no_boxes() -> None:
    boxes, ids, confs = extract_track_outputs(FakeResult(boxes=None))

    assert boxes.shape == (0, 4)
    assert ids.shape == (0,)
    assert confs.shape == (0,)


def test_extract_track_outputs_returns_empty_arrays_when_no_track_ids() -> None:
    fake_boxes = FakeBoxes()
    fake_boxes.id = None

    boxes, ids, confs = extract_track_outputs(FakeResult(boxes=fake_boxes))

    assert boxes.shape == (0, 4)
    assert ids.shape == (0,)
    assert confs.shape == (0,)


def test_extract_track_outputs_converts_fake_ultralytics_tensors() -> None:
    boxes, ids, confs = extract_track_outputs(FakeResult(boxes=FakeBoxes()))

    np.testing.assert_allclose(boxes, np.array([[1.0, 2.0, 11.0, 22.0]], dtype=float))
    np.testing.assert_array_equal(ids, np.array([42], dtype=int))
    np.testing.assert_allclose(confs, np.array([0.87], dtype=float))
```

- [ ] **Step 2: Run the model adapter tests to verify they fail**

Run:

```bash
python -m pytest tests/test_modeling.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'entrance_counter.modeling'`.

- [ ] **Step 3: Add the modeling module**

Create `src/entrance_counter/modeling.py`:

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def choose_device() -> str | int:
    import torch

    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_yolo_model(model_name: str, model_dir: Path) -> Any:
    from ultralytics import YOLO

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_name
    if model_path.exists():
        return YOLO(str(model_path))

    old_cwd = Path.cwd()
    try:
        os.chdir(model_dir)
        return YOLO(model_name)
    finally:
        os.chdir(old_cwd)


def extract_track_outputs(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if result.boxes is None or result.boxes.id is None:
        return np.empty((0, 4), dtype=float), np.empty((0,), dtype=int), np.empty((0,), dtype=float)

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    ids = result.boxes.id.detach().cpu().numpy().astype(int)
    if result.boxes.conf is None:
        confs = np.ones(len(ids), dtype=float)
    else:
        confs = result.boxes.conf.detach().cpu().numpy()

    return boxes, ids, confs
```

- [ ] **Step 4: Run tests through modeling to verify they pass**

Run:

```bash
python -m pytest tests/test_config.py tests/test_geometry.py tests/test_video_rendering.py tests/test_modeling.py -q
```

Expected: PASS with `24 passed`.

- [ ] **Step 5: Commit model adapter**

Run:

```bash
git add src/entrance_counter/modeling.py tests/test_modeling.py
git commit -m "Add model loading adapter"
```

Expected: commit succeeds.

### Task 5: Production Counting Pipeline

**Files:**
- Create: `src/entrance_counter/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing pipeline integration test with a fake tracker**

Create `tests/test_pipeline.py`:

```python
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from entrance_counter.config import CountingConfig, ModelConfig, PathConfig, ProcessingConfig, RunConfig
from entrance_counter.pipeline import PeopleCounter


class FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class FakeBoxes:
    def __init__(self, rows: list[tuple[tuple[float, float, float, float], int, float]]) -> None:
        self.xyxy = FakeTensor(np.array([row[0] for row in rows], dtype=float).reshape((-1, 4)))
        self.id = FakeTensor(np.array([row[1] for row in rows], dtype=float))
        self.conf = FakeTensor(np.array([row[2] for row in rows], dtype=float))


class FakeResult:
    def __init__(self, rows: list[tuple[tuple[float, float, float, float], int, float]]) -> None:
        self.boxes = FakeBoxes(rows) if rows else None


class FakeModel:
    def __init__(self, frames: list[list[tuple[tuple[float, float, float, float], int, float]]]) -> None:
        self.frames = frames
        self.calls = 0

    def track(self, *args: object, **kwargs: object) -> list[FakeResult]:
        rows = self.frames[self.calls]
        self.calls += 1
        return [FakeResult(rows)]


def _write_video(path: Path, frame_count: int = 5) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (120, 100))
    assert writer.isOpened()
    for idx in range(frame_count):
        frame = np.full((100, 120, 3), idx * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _box_with_bottom_center_y(y: float) -> tuple[float, float, float, float]:
    return (40.0, y - 20.0, 60.0, y)


def test_people_counter_counts_crossing_and_writes_csvs(tmp_path: Path) -> None:
    video_path = tmp_path / "entrance.mp4"
    _write_video(video_path)
    output_dir = tmp_path / "output"
    fake_model = FakeModel(
        frames=[
            [(_box_with_bottom_center_y(20.0), 7, 0.90)],
            [(_box_with_bottom_center_y(25.0), 7, 0.91)],
            [(_box_with_bottom_center_y(30.0), 7, 0.92)],
            [(_box_with_bottom_center_y(80.0), 7, 0.93)],
            [(_box_with_bottom_center_y(90.0), 7, 0.94)],
        ]
    )
    config = RunConfig(
        paths=PathConfig(project_root=tmp_path, video_path=video_path, output_dir=output_dir),
        counting=CountingConfig(line=((10, 50), (90, 50))),
        model=ModelConfig(model_name="fake.pt"),
        processing=ProcessingConfig(max_frames=5),
    )

    result = PeopleCounter(config=config, model=fake_model, device="cpu").run(
        write_annotated_video=False,
        progress_every=0,
    )

    assert result.events_path == output_dir / "entrance_people_count_events.csv"
    assert result.summary_path == output_dir / "entrance_people_count_summary.csv"
    assert result.events_path.exists()
    assert result.summary_path.exists()
    assert fake_model.calls == 5

    events = pd.read_csv(result.events_path)
    summary = pd.read_csv(result.summary_path).iloc[0]

    assert len(events) == 1
    assert events.iloc[0]["track_id"] == 7
    assert events.iloc[0]["direction"] == "out"
    assert events.iloc[0]["counted_by"] == "line_crossing"
    assert int(summary["frames_processed"]) == 5
    assert int(summary["in_count"]) == 0
    assert int(summary["out_count"]) == 1
    assert int(summary["total_count"]) == 1
```

- [ ] **Step 2: Run the pipeline test to verify it fails**

Run:

```bash
python -m pytest tests/test_pipeline.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'entrance_counter.pipeline'`.

- [ ] **Step 3: Add the production pipeline**

Create `src/entrance_counter/pipeline.py`:

```python
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import cv2
import pandas as pd

from entrance_counter.config import RunConfig, display_path
from entrance_counter.geometry import (
    TrackState,
    bottom_center_xy,
    count_event_allowed,
    eligible_crossing,
    maybe_late_init_crossing,
    movement_direction,
    normalized_line_side,
)
from entrance_counter.modeling import choose_device, extract_track_outputs, load_yolo_model
from entrance_counter.rendering import draw_annotations, draw_counting_line
from entrance_counter.video_io import create_video_writer, ensure_output_dirs, frame_at, read_video_metadata

EVENT_COLUMNS = [
    "event_index",
    "frame",
    "time_seconds",
    "track_id",
    "direction",
    "counted_by",
    "bottom_center_x",
    "bottom_center_y",
    "confidence",
    "in_count",
    "out_count",
    "total_count",
]


@dataclass(frozen=True)
class CountingResult:
    events: pd.DataFrame
    summary: pd.DataFrame
    events_path: object
    summary_path: object
    annotated_video_path: object


class PeopleCounter:
    def __init__(self, config: RunConfig, model: Any | None = None, device: str | int | None = None) -> None:
        self.config = config
        self.model = model
        self.device = device

    def _model(self) -> Any:
        if self.model is None:
            self.model = load_yolo_model(self.config.model.model_name, self.config.paths.model_dir)
        return self.model

    def _device(self) -> str | int:
        if self.device is None:
            self.device = choose_device()
        return self.device

    def save_roi_preview(self, second: float = 25.0) -> object:
        ensure_output_dirs(self.config.paths)
        metadata = read_video_metadata(self.config.paths.video_path)
        last_second = max((metadata.frame_count - 1) / max(metadata.fps, 1e-6), 0.0)
        sample_second = min(max(second, 0.0), last_second)
        preview_frame = frame_at(self.config.paths.video_path, sample_second)
        preview = draw_counting_line(preview_frame, self.config.counting.line, self.config.counting)
        ok = cv2.imwrite(str(self.config.paths.roi_preview_path), preview)
        if not ok:
            raise RuntimeError(f"Could not write ROI preview: {self.config.paths.roi_preview_path}")
        return self.config.paths.roi_preview_path

    def run(self, write_annotated_video: bool = True, progress_every: int = 100) -> CountingResult:
        ensure_output_dirs(self.config.paths)

        cap = cv2.VideoCapture(str(self.config.paths.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.config.paths.video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_fps = fps / max(self.config.processing.process_every_n_frames, 1)

        writer = None
        if write_annotated_video:
            writer = create_video_writer(self.config.paths.annotated_video_path, output_fps, (width, height))

        states: dict[int, TrackState] = defaultdict(TrackState)
        counts = {
            self.config.counting.negative_direction_label: 0,
            self.config.counting.positive_direction_label: 0,
        }
        events: list[dict[str, object]] = []
        processed = 0
        frame_idx = -1
        start = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_idx += 1
                if self.config.processing.max_frames is not None and frame_idx >= self.config.processing.max_frames:
                    break
                if frame_idx % self.config.processing.process_every_n_frames != 0:
                    continue

                results = self._model().track(
                    frame,
                    persist=True,
                    classes=[self.config.model.person_class_id],
                    conf=self.config.model.confidence,
                    iou=self.config.model.iou_threshold,
                    imgsz=self.config.model.image_size,
                    tracker=self.config.model.tracker,
                    device=self._device(),
                    verbose=False,
                )
                result = results[0]
                boxes_xyxy, track_ids, confs = extract_track_outputs(result)

                for xyxy, track_id, conf in zip(boxes_xyxy, track_ids, confs):
                    tid = int(track_id)
                    state = states[tid]
                    point = bottom_center_xy(xyxy)
                    side = normalized_line_side(point, self.config.counting.line)

                    if state.first_frame is None:
                        state.first_frame = frame_idx
                        state.first_point = point
                        state.first_side = side

                    counted_direction = None
                    counted_by = None

                    if len(state.points) >= 1:
                        prev_point = state.points[-1]
                        prev_side = state.sides[-1]
                        track_age = frame_idx - (state.first_frame if state.first_frame is not None else frame_idx)
                        if (
                            track_age >= self.config.counting.min_track_age_frames
                            and eligible_crossing(
                                prev_point,
                                point,
                                prev_side,
                                side,
                                self.config.counting.line,
                                self.config.counting,
                            )
                        ):
                            direction = movement_direction(
                                prev_point,
                                point,
                                self.config.counting.line,
                                self.config.counting,
                            )
                            if count_event_allowed(state, frame_idx, self.config.counting):
                                counted_direction = direction
                                counted_by = "line_crossing"

                    if counted_direction is None:
                        direction = maybe_late_init_crossing(
                            state,
                            point,
                            side,
                            frame_idx,
                            self.config.counting.line,
                            self.config.counting,
                        )
                        if direction is not None and count_event_allowed(state, frame_idx, self.config.counting):
                            counted_direction = direction
                            counted_by = "late_init_buffer"

                    if counted_direction is not None:
                        counts[counted_direction] = counts.get(counted_direction, 0) + 1
                        state.last_count_frame = frame_idx
                        state.last_direction = counted_direction
                        events.append(
                            {
                                "event_index": len(events) + 1,
                                "frame": frame_idx,
                                "time_seconds": frame_idx / max(fps, 1e-6),
                                "track_id": tid,
                                "direction": counted_direction,
                                "counted_by": counted_by,
                                "bottom_center_x": point[0],
                                "bottom_center_y": point[1],
                                "confidence": float(conf),
                                "in_count": counts.get(self.config.counting.negative_direction_label, 0),
                                "out_count": counts.get(self.config.counting.positive_direction_label, 0),
                                "total_count": (
                                    counts.get(self.config.counting.negative_direction_label, 0)
                                    + counts.get(self.config.counting.positive_direction_label, 0)
                                ),
                            }
                        )

                    state.points.append(point)
                    state.sides.append(side)

                if writer is not None:
                    annotated = draw_annotations(
                        frame,
                        boxes_xyxy,
                        track_ids,
                        states,
                        counts,
                        frame_idx,
                        fps,
                        self.config.counting,
                    )
                    writer.write(annotated)

                processed += 1
                if progress_every > 0 and processed % progress_every == 0:
                    elapsed = max(time.time() - start, 1e-6)
                    total = (
                        counts.get(self.config.counting.negative_direction_label, 0)
                        + counts.get(self.config.counting.positive_direction_label, 0)
                    )
                    print(f"Processed {processed} frames at {processed / elapsed:0.2f} FPS | total={total}")
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        events_df = pd.DataFrame(events, columns=EVENT_COLUMNS)
        summary_df = pd.DataFrame(
            [
                {
                    "video": display_path(self.config.paths.video_path, self.config.paths.project_root),
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frames_in_video": frame_count,
                    "frames_processed": processed,
                    "duration_seconds": frame_count / max(fps, 1e-6),
                    "counting_line": str(self.config.counting.line),
                    "in_count": counts.get(self.config.counting.negative_direction_label, 0),
                    "out_count": counts.get(self.config.counting.positive_direction_label, 0),
                    "total_count": (
                        counts.get(self.config.counting.negative_direction_label, 0)
                        + counts.get(self.config.counting.positive_direction_label, 0)
                    ),
                    "model": self.config.model.model_name,
                    "confidence": self.config.model.confidence,
                    "image_size": self.config.model.image_size,
                    "process_every_n_frames": self.config.processing.process_every_n_frames,
                    "annotated_video": display_path(
                        self.config.paths.annotated_video_path,
                        self.config.paths.project_root,
                    ),
                    "events_csv": display_path(self.config.paths.events_csv_path, self.config.paths.project_root),
                }
            ]
        )

        events_df.to_csv(self.config.paths.events_csv_path, index=False)
        summary_df.to_csv(self.config.paths.summary_csv_path, index=False)

        elapsed = max(time.time() - start, 1e-6)
        print(f"Done. Processed {processed} frames in {elapsed:0.1f}s ({processed / elapsed:0.2f} FPS).")
        print(f"Saved events: {self.config.paths.events_csv_path}")
        print(f"Saved summary: {self.config.paths.summary_csv_path}")
        if writer is not None:
            print(f"Saved annotated video: {self.config.paths.annotated_video_path}")

        return CountingResult(
            events=events_df,
            summary=summary_df,
            events_path=self.config.paths.events_csv_path,
            summary_path=self.config.paths.summary_csv_path,
            annotated_video_path=self.config.paths.annotated_video_path,
        )
```

- [ ] **Step 4: Run tests through pipeline to verify they pass**

Run:

```bash
python -m pytest tests/test_config.py tests/test_geometry.py tests/test_video_rendering.py tests/test_modeling.py tests/test_pipeline.py -q
```

Expected: PASS with `25 passed`.

- [ ] **Step 5: Commit production pipeline**

Run:

```bash
git add src/entrance_counter/pipeline.py tests/test_pipeline.py
git commit -m "Add production people counting pipeline"
```

Expected: commit succeeds.

### Task 6: Command-Line Entrypoint

**Files:**
- Create: `src/entrance_counter/cli.py`
- Create: `src/entrance_counter/__main__.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

Create `tests/test_cli.py`:

```python
from pathlib import Path

from entrance_counter.cli import build_parser, config_from_args


def test_cli_defaults_match_run_config() -> None:
    args = build_parser().parse_args([])
    config = config_from_args(args)

    assert config.paths.video_path.name == "entrance.mov"
    assert config.paths.output_dir.name == "output"
    assert config.counting.line == ((330, 785), (870, 812))
    assert config.model.model_name == "yolov8n.pt"
    assert config.processing.process_every_n_frames == 1
    assert config.processing.max_frames is None


def test_cli_maps_overrides_to_run_config(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mov"
    output_dir = tmp_path / "artifacts"
    args = build_parser().parse_args(
        [
            "--video",
            str(video_path),
            "--output-dir",
            str(output_dir),
            "--line",
            "10,20,30,40",
            "--model-name",
            "yolov8s.pt",
            "--confidence",
            "0.35",
            "--iou",
            "0.60",
            "--image-size",
            "640",
            "--process-every-n-frames",
            "2",
            "--max-frames",
            "100",
            "--invert-directions",
        ]
    )

    config = config_from_args(args)

    assert config.paths.video_path == video_path
    assert config.paths.output_dir == output_dir
    assert config.counting.line == ((10, 20), (30, 40))
    assert config.counting.invert_directions is True
    assert config.model.model_name == "yolov8s.pt"
    assert config.model.confidence == 0.35
    assert config.model.iou_threshold == 0.60
    assert config.model.image_size == 640
    assert config.processing.process_every_n_frames == 2
    assert config.processing.max_frames == 100
```

- [ ] **Step 2: Run the CLI tests to verify they fail**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'entrance_counter.cli'`.

- [ ] **Step 3: Add CLI and module entrypoint**

Create `src/entrance_counter/cli.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from entrance_counter.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO_PATH,
    CountingConfig,
    ModelConfig,
    PathConfig,
    ProcessingConfig,
    RunConfig,
    parse_counting_line,
)
from entrance_counter.pipeline import PeopleCounter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count people crossing an entrance line in a video.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH, help="Input video path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated outputs.")
    parser.add_argument("--line", default="330,785,870,812", help="Counting line as x1,y1,x2,y2.")
    parser.add_argument("--model-name", default="yolov8n.pt", help="YOLO model name or local weight filename.")
    parser.add_argument("--confidence", type=float, default=0.22, help="YOLO detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.50, help="YOLO IoU threshold.")
    parser.add_argument("--image-size", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument("--process-every-n-frames", type=int, default=1, help="Frame sampling interval.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for short runs.")
    parser.add_argument("--invert-directions", action="store_true", help="Swap in/out direction labels.")
    parser.add_argument("--skip-annotated-video", action="store_true", help="Write CSV outputs without annotated MP4.")
    parser.add_argument("--preview-second", type=float, default=25.0, help="Timestamp used for ROI preview image.")
    parser.add_argument("--no-preview", action="store_true", help="Skip ROI preview image generation.")
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    project_root = Path.cwd()
    paths = PathConfig(
        project_root=project_root,
        video_path=args.video,
        output_dir=args.output_dir,
    )
    counting = CountingConfig(
        line=parse_counting_line(args.line),
        invert_directions=args.invert_directions,
    )
    model = ModelConfig(
        model_name=args.model_name,
        confidence=args.confidence,
        iou_threshold=args.iou,
        image_size=args.image_size,
    )
    processing = ProcessingConfig(
        process_every_n_frames=args.process_every_n_frames,
        max_frames=args.max_frames,
    )
    return RunConfig(paths=paths, counting=counting, model=model, processing=processing)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    counter = PeopleCounter(config)

    if not args.no_preview:
        preview_path = counter.save_roi_preview(second=args.preview_second)
        print(f"Saved ROI preview: {preview_path}")

    result = counter.run(write_annotated_video=not args.skip_annotated_video)
    row = result.summary.iloc[0]
    print(
        "Final count: "
        f"in={int(row['in_count'])} "
        f"out={int(row['out_count'])} "
        f"total={int(row['total_count'])}"
    )
    return 0
```

Create `src/entrance_counter/__main__.py`:

```python
from __future__ import annotations

from entrance_counter.cli import main

raise SystemExit(main())
```

- [ ] **Step 4: Run all unit and fake-integration tests**

Run:

```bash
python -m pytest -q
```

Expected: PASS with `27 passed`.

- [ ] **Step 5: Run the CLI help command**

Run:

```bash
python -m entrance_counter --help
```

Expected: PASS and output starts with `usage:`.

- [ ] **Step 6: Commit CLI entrypoint**

Run:

```bash
git add src/entrance_counter/cli.py src/entrance_counter/__main__.py tests/test_cli.py
git commit -m "Add people counting CLI"
```

Expected: commit succeeds.

### Task 7: Documentation Migration And Notebook Removal

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Delete: `src/src.ipynb`

- [ ] **Step 1: Replace README with production Python instructions**

Replace `README.md` with:

```markdown
# Entrance People Counting

This project counts people crossing an entrance in `resources/data/entrance.mov`.
The implementation is a production Python package with a command-line entrypoint.

The pipeline uses:

- YOLOv8 person detection.
- ByteTrack tracking through Ultralytics.
- A doorway threshold line for directional crossing events.
- The bottom-center of each person box as the tracked floor-contact point.
- Jitter suppression, late track initialization handling, and per-track cooldown.

## Current Baseline

The latest validated run against `resources/data/entrance.mov` produced:

| Metric | Value |
| --- | ---: |
| Video duration | 85.535 seconds |
| Frames processed | 2,548 |
| In count | 7 |
| Out count | 18 |
| Total crossings | 25 |

The total is a crossing-event count, not a frame-level occupancy count.

## Project Structure

```text
.
├── pyproject.toml
├── requirements.txt
├── src
│   └── entrance_counter
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── geometry.py
│       ├── modeling.py
│       ├── pipeline.py
│       ├── rendering.py
│       └── video_io.py
├── tests
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_geometry.py
│   ├── test_modeling.py
│   ├── test_pipeline.py
│   └── test_video_rendering.py
├── resources
│   ├── data
│   │   └── entrance.mov
│   └── docs
│       └── research-1.md
└── output
    ├── entrance_roi_preview.jpg
    ├── entrance_people_count_annotated.mp4
    ├── entrance_people_count_events.csv
    ├── entrance_people_count_summary.csv
    └── models
        └── yolov8n.pt
```

Keep raw input media under `resources/data/`. Generated files belong under `output/`.

## Setup

Create a local Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The first full run uses `output/models/yolov8n.pt` when present. If the weight file is missing, Ultralytics downloads it.

## Run

Generate the ROI preview, annotated video, event log, and summary CSV:

```bash
entrance-counter --video resources/data/entrance.mov --output-dir output
```

Equivalent module invocation:

```bash
python -m entrance_counter --video resources/data/entrance.mov --output-dir output
```

For a short smoke run:

```bash
entrance-counter --video resources/data/entrance.mov --output-dir output --max-frames 300 --skip-annotated-video --no-preview
```

## Test

Run the automated tests:

```bash
python -m pytest -q
```

## Outputs

Generated artifacts are written to `output/`:

- `entrance_roi_preview.jpg`: sampled frame showing the configured entrance counting line.
- `entrance_people_count_annotated.mp4`: processed video with boxes, tracks, crossing line, and live counts.
- `entrance_people_count_events.csv`: one row per counted crossing event.
- `entrance_people_count_summary.csv`: run metadata and final directional totals.
- `models/yolov8n.pt`: local YOLO model weights, when available.

## Tuning

The main parameters are exposed through CLI flags:

- `--line x1,y1,x2,y2`: doorway threshold coordinates.
- `--confidence`: YOLO detection confidence threshold.
- `--image-size`: inference image size.
- `--process-every-n-frames`: frame sampling rate.
- `--max-frames`: optional cap for quick runs.
- `--invert-directions`: flips the `in` and `out` direction convention.

If the ROI preview shows the threshold in the wrong place, adjust `--line` first.
If visual review shows that direction labels are reversed, use `--invert-directions`.

## Reference Material

`resources/docs/research-1.md` contains background notes on pedestrian counting methods, including tracking-by-detection, line-crossing geometry, video individual counting, and dense-crowd alternatives.
```

- [ ] **Step 2: Replace AGENTS.md with production Python repository guidance**

Replace `AGENTS.md` with:

```markdown
# Repository Guidelines

## Project Structure & Module Organization

This is a compact production Python computer-vision project. The source of truth lives in `src/entrance_counter/`. Keep modules focused: configuration in `config.py`, line-crossing geometry in `geometry.py`, model loading in `modeling.py`, video helpers in `video_io.py`, drawing in `rendering.py`, orchestration in `pipeline.py`, and CLI wiring in `cli.py`. Tests belong in `tests/` and should avoid loading YOLO weights unless a full manual verification run requires it. Source notes and research material belong in `resources/docs/`. Raw local inputs, including videos, belong in `resources/data/`. Generated artifacts belong in `output/`; do not treat files there as source inputs unless code or docs explicitly document that dependency.

## Build, Test, and Development Commands

Create and activate a local Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the CLI on the entrance video:

```bash
entrance-counter --video resources/data/entrance.mov --output-dir output
```

Run a short smoke execution:

```bash
entrance-counter --video resources/data/entrance.mov --output-dir output --max-frames 300 --skip-annotated-video --no-preview
```

Run automated tests:

```bash
python -m pytest -q
```

## Coding Style & Naming Conventions

Use standard Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for constants. Keep file paths relative to the repository root in docs and examples, such as `resources/data/entrance.mov`. Prefer dataclasses for structured configuration and pure functions for geometry so behavior stays easy to test. Keep generated filenames descriptive, for example `output/entrance_people_count_summary.csv`.

## Testing Guidelines

Unit tests should cover pure geometry, config parsing, rendering helpers, model-result adapters, and CLI config mapping. Integration tests should use fake model outputs and tiny generated videos so the suite remains fast and deterministic. Validate full reproducibility manually by running the CLI from a clean environment and confirming expected outputs appear under `output/`.

## Commit & Pull Request Guidelines

Use concise imperative commit messages, such as `Add people counting CLI`. Pull requests should include a short summary, verification commands, notes about changed inputs or generated outputs, and screenshots or sample frames when visual processing changes.

## Security & Configuration Tips

Do not commit secrets, private datasets, or large derived files unless intentionally required. Keep raw media in `resources/data/` and generated results in `output/` so source material and artifacts remain easy to distinguish. Model weights and large generated videos should remain ignored unless there is an explicit reason to version them.
```

- [ ] **Step 3: Remove the notebook implementation**

Run:

```bash
git rm src/src.ipynb
```

Expected: `src/src.ipynb` is staged for deletion.

- [ ] **Step 4: Verify docs no longer point to the notebook workflow**

Run:

```bash
rg "ipynb|notebook|jupyter|nbconvert" README.md AGENTS.md src tests
```

Expected: no matches.

- [ ] **Step 5: Run all tests after docs migration**

Run:

```bash
python -m pytest -q
```

Expected: PASS with `27 passed`.

- [ ] **Step 6: Commit docs migration and notebook removal**

Run:

```bash
git add README.md AGENTS.md
git commit -m "Document production Python workflow"
```

Expected: commit succeeds and includes the staged notebook deletion.

### Task 8: Full Reproducibility And Baseline Parity

**Files:**
- Verify: `output/entrance_people_count_summary.csv`
- Verify: `output/entrance_people_count_events.csv`
- Verify: `output/entrance_people_count_annotated.mp4`
- Verify: `output/entrance_roi_preview.jpg`

- [ ] **Step 1: Install the package in editable development mode**

Run:

```bash
python -m pip install -e ".[dev]"
```

Expected: package installs successfully and exposes the `entrance-counter` command.

- [ ] **Step 2: Run the full production CLI**

Run:

```bash
entrance-counter --video resources/data/entrance.mov --output-dir output
```

Expected: command completes and prints `Final count: in=7 out=18 total=25`.

- [ ] **Step 3: Verify generated artifacts exist**

Run:

```bash
python -c "from pathlib import Path; paths = ['output/entrance_roi_preview.jpg', 'output/entrance_people_count_annotated.mp4', 'output/entrance_people_count_events.csv', 'output/entrance_people_count_summary.csv']; missing = [p for p in paths if not Path(p).exists()]; assert not missing, missing"
```

Expected: no output and exit code 0.

- [ ] **Step 4: Verify baseline counts and frame count**

Run:

```bash
python -c "import pandas as pd; row = pd.read_csv('output/entrance_people_count_summary.csv').iloc[0]; assert int(row['in_count']) == 7; assert int(row['out_count']) == 18; assert int(row['total_count']) == 25; assert int(row['frames_processed']) == 2548"
```

Expected: no output and exit code 0.

- [ ] **Step 5: Run the full automated test suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS with `27 passed`.

- [ ] **Step 6: Check Git status**

Run:

```bash
git status --short
```

Expected: only ignored generated files under `output/` are absent from status; tracked source, tests, docs, and config changes are committed.

## Self Review

- Spec coverage: The migration from notebook-only workflow to production Python code is covered by Tasks 1 through 7, and Task 8 verifies output parity with the current baseline.
- Red-flag scan: The plan uses concrete file paths, exact code, exact commands, and exact expected results.
- Type consistency: `RunConfig`, `PathConfig`, `CountingConfig`, `ModelConfig`, `ProcessingConfig`, `Line`, `Point`, `TrackState`, `PeopleCounter`, and `CountingResult` are used consistently across tasks.
- Test coverage: Pure logic, config parsing, rendering, model result extraction, fake-model pipeline execution, CLI mapping, and full manual parity are all covered.
