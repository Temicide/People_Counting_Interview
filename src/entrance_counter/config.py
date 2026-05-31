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
