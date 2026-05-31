from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
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
    events_path: Path
    summary_path: Path
    annotated_video_path: Path


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

    def save_roi_preview(self, second: float = 25.0) -> Path:
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
