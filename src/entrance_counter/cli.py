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
    StabilizationConfig,
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
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config, for example bytetrack.yaml or botsort.yaml.",
    )
    parser.add_argument("--process-every-n-frames", type=int, default=1, help="Frame sampling interval.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for short runs.")
    parser.add_argument("--invert-directions", action="store_true", help="Swap in/out direction labels.")
    parser.add_argument(
        "--stabilize",
        action="store_true",
        help="Compensate handheld camera motion before line-crossing geometry.",
    )
    parser.add_argument(
        "--stabilization-reference-second",
        type=float,
        default=25.0,
        help="Timestamp used as the doorway reference frame.",
    )
    parser.add_argument(
        "--stabilization-max-features",
        type=int,
        default=1200,
        help="Maximum ORB features for background alignment.",
    )
    parser.add_argument(
        "--stabilization-min-matches",
        type=int,
        default=18,
        help="Minimum feature matches required for a stabilization transform.",
    )
    parser.add_argument(
        "--stabilization-min-inlier-ratio",
        type=float,
        default=0.35,
        help="Minimum RANSAC inlier ratio required for stabilization.",
    )
    parser.add_argument(
        "--stabilization-max-reprojection-error",
        type=float,
        default=4.0,
        help="RANSAC reprojection threshold in pixels.",
    )
    parser.add_argument(
        "--stabilization-smoothing-alpha",
        type=float,
        default=0.25,
        help="EMA alpha for smoothing stabilization transforms.",
    )
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
        tracker=args.tracker,
    )
    processing = ProcessingConfig(
        process_every_n_frames=args.process_every_n_frames,
        max_frames=args.max_frames,
    )
    stabilization = StabilizationConfig(
        enabled=args.stabilize,
        reference_second=args.stabilization_reference_second,
        max_features=args.stabilization_max_features,
        min_matches=args.stabilization_min_matches,
        min_inlier_ratio=args.stabilization_min_inlier_ratio,
        max_reprojection_error=args.stabilization_max_reprojection_error,
        smoothing_alpha=args.stabilization_smoothing_alpha,
    )
    return RunConfig(
        paths=paths,
        counting=counting,
        model=model,
        processing=processing,
        stabilization=stabilization,
    )


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
