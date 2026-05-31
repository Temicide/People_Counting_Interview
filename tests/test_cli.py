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
