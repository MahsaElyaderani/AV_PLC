"""MediaPipe reference-face alignment for the AV_PLC pixel-frame path.

AV-HuBERT visual features are intentionally NOT produced here.  Their existing
preprocessing remains unchanged in dataset/features/save_features.py.
"""
from __future__ import annotations

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh_connections import (
    FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE, FACEMESH_LIPS,
)

LEFT_EYE_IDS = np.array(sorted({i for edge in FACEMESH_LEFT_EYE for i in edge}), dtype=np.int32)
RIGHT_EYE_IDS = np.array(sorted({i for edge in FACEMESH_RIGHT_EYE for i in edge}), dtype=np.int32)
ALIGN_IDS = np.array(sorted(set(LEFT_EYE_IDS.tolist() + RIGHT_EYE_IDS.tolist())), dtype=np.int32)
LIP_IDS = np.array(sorted({i for edge in FACEMESH_LIPS for i in edge}), dtype=np.int32)


def interpolate_missing_landmarks(landmarks: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(landmarks, dtype=np.float32).copy()
    valid = np.asarray(valid, dtype=bool)
    good = np.flatnonzero(valid)
    if good.size < 2:
        raise ValueError("At least two valid face detections are required for alignment")
    x = np.arange(out.shape[0])
    for p in range(out.shape[1]):
        for c in range(2):
            out[:, p, c] = np.interp(x, good, out[good, p, c])
    return out


def smooth_landmarks(landmarks: np.ndarray, window: int = 5) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return landmarks.astype(np.float32, copy=True)
    radius = window // 2
    padded = np.pad(landmarks, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    out = np.empty_like(landmarks, dtype=np.float32)
    for t in range(landmarks.shape[0]):
        out[t] = padded[t:t + window].mean(axis=0)
    return out


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    return np.concatenate([points.astype(np.float32), ones], axis=1) @ matrix.T


def _fixed_crop(gray: np.ndarray, center_xy, size: int = 96) -> np.ndarray:
    cx, cy = map(float, center_xy)
    half = size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2, y2 = x1 + size, y1 + size
    out = np.zeros((size, size), dtype=np.uint8)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
    if sx2 <= sx1 or sy2 <= sy1:
        return out
    dx1, dy1 = sx1 - x1, sy1 - y1
    out[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = gray[sy1:sy2, sx1:sx2]
    return out


class ReferenceFaceAligner:
    def __init__(self, reference_landmarks: np.ndarray, canvas_size: int = 256,
                 crop_size: int = 96, smooth_window: int = 5):
        ref = np.asarray(reference_landmarks, dtype=np.float32)
        if ref.ndim != 2 or ref.shape[1] != 2 or ref.shape[0] <= int(ALIGN_IDS.max()):
            raise ValueError(f"reference_landmarks must be [N,2], got {ref.shape}")
        self.reference = ref
        self.canvas_size = int(canvas_size)
        self.crop_size = int(crop_size)
        self.smooth_window = int(smooth_window)

    @classmethod
    def from_npy(cls, path: str, **kwargs):
        return cls(np.load(path), **kwargs)

    def align_sequence(self, rgb_frames: np.ndarray, full_landmarks: np.ndarray,
                       detected_valid: np.ndarray) -> np.ndarray:
        frames = np.asarray(rgb_frames)
        lm = interpolate_missing_landmarks(full_landmarks, detected_valid)
        smooth = smooth_landmarks(lm, self.smooth_window)
        out = np.zeros((frames.shape[0], self.crop_size, self.crop_size, 1), dtype=np.uint8)

        dst = self.reference[ALIGN_IDS]
        for t in range(frames.shape[0]):
            matrix, _ = cv2.estimateAffinePartial2D(
                smooth[t, ALIGN_IDS], dst, method=cv2.LMEDS
            )
            if matrix is None or not np.isfinite(matrix).all():
                continue
            warped = cv2.warpAffine(
                frames[t], matrix, (self.canvas_size, self.canvas_size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            lips = _transform_points(lm[t, LIP_IDS], matrix)
            mouth_center = lips.mean(axis=0)
            gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
            out[t, :, :, 0] = _fixed_crop(gray, mouth_center, self.crop_size)
        return out
