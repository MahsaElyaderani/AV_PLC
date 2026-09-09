"""Build a fixed MediaPipe reference-face template from training videos.

Each detected face is first normalized by a two-eye similarity transform to a
256x256 canvas, then full-face landmarks are averaged.  The resulting .npy is
used only for AV_PLC pixel-frame alignment.
"""
from __future__ import annotations

import argparse
import glob
import os
import cv2
import mediapipe as mp
import numpy as np

from AV_PLC.video_preprocessing import LEFT_EYE_IDS, RIGHT_EYE_IDS


def _eye_center(lm, ids):
    return lm[ids].mean(axis=0)


def build_reference(video_patterns, output, max_faces=2000, frame_stride=10, canvas=256):
    paths = []
    for pattern in video_patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("No videos matched the supplied patterns")

    target_x_left = np.array([0.35 * canvas, 0.35 * canvas], dtype=np.float32)
    target_x_right = np.array([0.65 * canvas, 0.35 * canvas], dtype=np.float32)

    samples = []
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    try:
        for path in paths:
            cap = cv2.VideoCapture(path)
            i = 0
            while len(samples) < max_faces:
                ok, bgr = cap.read()
                if not ok:
                    break
                if i % frame_stride:
                    i += 1
                    continue
                i += 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                if not result.multi_face_landmarks:
                    continue
                h, w = rgb.shape[:2]
                lm = np.array([[p.x * w, p.y * h] for p in result.multi_face_landmarks[0].landmark], dtype=np.float32)
                src = np.stack([_eye_center(lm, LEFT_EYE_IDS), _eye_center(lm, RIGHT_EYE_IDS)])
                # FACEMESH_LEFT_EYE/RIGHT_EYE are anatomical labels, not
                # image-left/image-right.  Preserve the observed horizontal
                # ordering so the normalization can never introduce a flip or
                # a 180-degree rotation merely because of landmark naming.
                if src[0, 0] <= src[1, 0]:
                    dst = np.stack([target_x_left, target_x_right])
                else:
                    dst = np.stack([target_x_right, target_x_left])
                matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
                if matrix is None or not np.isfinite(matrix).all():
                    continue
                ones = np.ones((lm.shape[0], 1), dtype=np.float32)
                samples.append(np.concatenate([lm, ones], 1) @ matrix.T)
                if len(samples) >= max_faces:
                    break
            cap.release()
            if len(samples) >= max_faces:
                break
    finally:
        face_mesh.close()

    if len(samples) < 10:
        raise RuntimeError(f"Only {len(samples)} valid faces were found; reference is unreliable")
    reference = np.mean(np.stack(samples), axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.save(output, reference)
    print(f"Saved {reference.shape} reference face from {len(samples)} detections to {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="+", required=True, help="Glob pattern(s), e.g. '/data/grid/train/**/*.mpg'")
    p.add_argument("--output", required=True)
    p.add_argument("--max-faces", type=int, default=2000)
    p.add_argument("--frame-stride", type=int, default=10)
    args = p.parse_args()
    build_reference(args.videos, args.output, args.max_faces, args.frame_stride)


if __name__ == "__main__":
    main()
