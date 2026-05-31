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


def test_parse_counting_line_accepts_tuple_string() -> None:
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


def test_run_config_defaults_match_current_baseline_parameters() -> None:
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
