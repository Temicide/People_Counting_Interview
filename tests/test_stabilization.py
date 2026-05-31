import cv2
import numpy as np

from entrance_counter.config import StabilizationConfig
from entrance_counter.geometry import transform_line, transform_point
from entrance_counter.stabilization import FrameTransform, FrameStabilizer, identity_affine


def _feature_frame(shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    for x, y in [(40, 40), (180, 45), (60, 130), (190, 135), (120, 90), (85, 80), (150, 110)]:
        cv2.circle(frame, (x + shift_x, y + shift_y), 8, (255, 255, 255), -1)
        cv2.rectangle(
            frame,
            (x + shift_x - 4, y + shift_y + 12),
            (x + shift_x + 12, y + shift_y + 22),
            (180, 180, 180),
            -1,
        )
    return frame


def test_identity_affine_keeps_points_unchanged() -> None:
    matrix = identity_affine()

    assert transform_point((10.0, 20.0), matrix) == (10.0, 20.0)


def test_stabilizer_returns_identity_when_disabled() -> None:
    stabilizer = FrameStabilizer(StabilizationConfig(enabled=False), _feature_frame())

    transform = stabilizer.estimate(_feature_frame(shift_x=8, shift_y=4))

    assert transform.valid is False
    np.testing.assert_allclose(transform.current_to_reference, identity_affine())
    np.testing.assert_allclose(transform.reference_to_current, identity_affine())


def test_stabilizer_estimates_translation_to_reference() -> None:
    reference = _feature_frame()
    current = _feature_frame(shift_x=10, shift_y=6)
    stabilizer = FrameStabilizer(
        StabilizationConfig(
            enabled=True,
            max_features=500,
            min_matches=8,
            min_inlier_ratio=0.25,
            max_reprojection_error=5.0,
            smoothing_alpha=1.0,
        ),
        reference,
    )

    transform = stabilizer.estimate(current)

    assert transform.valid is True
    aligned = transform_point((110.0, 86.0), transform.current_to_reference)
    assert abs(aligned[0] - 100.0) < 3.0
    assert abs(aligned[1] - 80.0) < 3.0


def test_frame_transform_projects_reference_line_to_current() -> None:
    transform = FrameTransform(
        current_to_reference=np.array([[1.0, 0.0, -10.0], [0.0, 1.0, -6.0]], dtype=float),
        reference_to_current=np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 6.0]], dtype=float),
        valid=True,
        match_count=20,
        inlier_count=18,
        inlier_ratio=0.9,
    )

    assert transform_line(((20, 30), (80, 30)), transform.reference_to_current) == ((30, 36), (90, 36))
