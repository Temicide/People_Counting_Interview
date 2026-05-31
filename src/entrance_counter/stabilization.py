from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from entrance_counter.config import StabilizationConfig


def identity_affine() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)


def _to_homogeneous(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (3, 3):
        return matrix.astype(float)
    if matrix.shape == (2, 3):
        return np.vstack([matrix.astype(float), np.array([0.0, 0.0, 1.0], dtype=float)])
    raise ValueError("matrix must have shape (2, 3) or (3, 3)")


def _from_homogeneous(matrix: np.ndarray) -> np.ndarray:
    return matrix[:2, :].astype(float)


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    return _from_homogeneous(np.linalg.inv(_to_homogeneous(matrix)))


@dataclass(frozen=True)
class FrameTransform:
    current_to_reference: np.ndarray
    reference_to_current: np.ndarray
    valid: bool
    match_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0


class FrameStabilizer:
    def __init__(self, config: StabilizationConfig, reference_frame: np.ndarray) -> None:
        self.config = config
        self.reference_frame = reference_frame
        self._previous_current_to_reference: np.ndarray | None = None
        self._orb = cv2.ORB_create(nfeatures=config.max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._reference_keypoints, self._reference_descriptors = self._detect(reference_frame)

    def estimate(self, frame: np.ndarray) -> FrameTransform:
        if not self.config.enabled:
            return self._identity(valid=False)
        if self._reference_descriptors is None or len(self._reference_keypoints) < self.config.min_matches:
            return self._identity(valid=False)

        current_keypoints, current_descriptors = self._detect(frame)
        if current_descriptors is None or len(current_keypoints) < self.config.min_matches:
            return self._fallback()

        matches = sorted(
            self._matcher.match(current_descriptors, self._reference_descriptors),
            key=lambda match: match.distance,
        )
        if len(matches) < self.config.min_matches:
            return self._fallback(match_count=len(matches))

        current_points = np.float32([current_keypoints[match.queryIdx].pt for match in matches])
        reference_points = np.float32([self._reference_keypoints[match.trainIdx].pt for match in matches])
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            current_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.max_reprojection_error,
        )
        if matrix is None or inlier_mask is None:
            return self._fallback(match_count=len(matches))

        inlier_count = int(inlier_mask.sum())
        inlier_ratio = inlier_count / max(len(matches), 1)
        if inlier_count < self.config.min_matches or inlier_ratio < self.config.min_inlier_ratio:
            return self._fallback(match_count=len(matches), inlier_count=inlier_count, inlier_ratio=inlier_ratio)

        matrix = matrix.astype(float)
        if self._previous_current_to_reference is not None and self.config.smoothing_alpha < 1.0:
            alpha = self.config.smoothing_alpha
            matrix = alpha * matrix + (1.0 - alpha) * self._previous_current_to_reference
        self._previous_current_to_reference = matrix

        return FrameTransform(
            current_to_reference=matrix,
            reference_to_current=invert_affine(matrix),
            valid=True,
            match_count=len(matches),
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
        )

    def _detect(self, frame: np.ndarray) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        return tuple(keypoints or ()), descriptors

    def _identity(self, valid: bool) -> FrameTransform:
        matrix = identity_affine()
        return FrameTransform(current_to_reference=matrix, reference_to_current=matrix.copy(), valid=valid)

    def _fallback(
        self,
        match_count: int = 0,
        inlier_count: int = 0,
        inlier_ratio: float = 0.0,
    ) -> FrameTransform:
        if self._previous_current_to_reference is None:
            matrix = identity_affine()
            valid = False
        else:
            matrix = self._previous_current_to_reference
            valid = True
        return FrameTransform(
            current_to_reference=matrix,
            reference_to_current=invert_affine(matrix),
            valid=valid,
            match_count=match_count,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
        )
