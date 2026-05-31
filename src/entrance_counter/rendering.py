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
