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
