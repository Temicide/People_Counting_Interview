from pathlib import Path

from entrance_counter.cli import build_parser, config_from_args


def test_cli_defaults_match_run_config() -> None:
    args = build_parser().parse_args([])
    config = config_from_args(args)

    assert config.paths.video_path.name == "entrance.mov"
    assert config.paths.output_dir.name == "output"
    assert config.counting.line == ((330, 785), (870, 812))
    assert config.model.model_name == "yolov8n.pt"
    assert config.model.tracker == "bytetrack.yaml"
    assert config.processing.process_every_n_frames == 1
    assert config.processing.max_frames is None
    assert config.stabilization.enabled is False
    assert config.stabilization.reference_second == 25.0


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
        ]
    )

    config = config_from_args(args)

    assert config.paths.video_path == video_path
    assert config.paths.output_dir == output_dir
    assert config.counting.line == ((10, 20), (30, 40))
    assert config.counting.invert_directions is True
    assert config.model.model_name == "yolov8s.pt"
    assert config.model.tracker == "botsort.yaml"
    assert config.model.confidence == 0.35
    assert config.model.iou_threshold == 0.60
    assert config.model.image_size == 640
    assert config.processing.process_every_n_frames == 2
    assert config.processing.max_frames == 100
    assert config.stabilization.enabled is True
    assert config.stabilization.reference_second == 12.5
    assert config.stabilization.max_features == 800
    assert config.stabilization.min_matches == 12
    assert config.stabilization.min_inlier_ratio == 0.45
    assert config.stabilization.max_reprojection_error == 3.5
    assert config.stabilization.smoothing_alpha == 0.4
