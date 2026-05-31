from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from entrance_counter.config import CountingConfig, ModelConfig, PathConfig, ProcessingConfig, RunConfig, StabilizationConfig
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
    assert bool(result.summary.iloc[0]["stabilization_enabled"]) is True


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

    result = PeopleCounter(config=config, model=fake_model, device="cpu").run(
        write_annotated_video=False,
        progress_every=0,
    )

    assert len(result.events) == 0
