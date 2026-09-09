import os
import cv2
import torch
import mediapipe
from tqdm import tqdm
from pathlib import Path


def crop_faces(video_path):
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_detector = mediapipe.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    output_dir = Path(video_path).stem  # = `.split('.mp4')[0]`
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = face_detector.process(rgb_frame).detections

        if detections:
            for det in detections:
                bboxC = det.location_data.relative_bounding_box
                x, y = max(0, int(bboxC.xmin * w)), max(0, int(bboxC.ymin * h))
                x2, y2 = min(w, x + int(bboxC.width * w)), min(h, y + int(bboxC.height * h))
                face = frame[y:y2, x:x2]

                if face.size > 0:
                    cv2.imwrite(f"{output_dir}/frame_{frame_id}.png", face)

        frame_id += 1
    cap.release()


def visualize_motion(motion_features, landmark_video, frames, video_path):

    # output_path = video_path.split('.mp4')[0]
    output_path = video_path.split('.mpg')[0]
    os.makedirs(output_path, exist_ok=True)

    for i in range(len(frames)):
        frame = frames[i].copy()
        for j, (dx, dy) in enumerate(motion_features[i]):
            x_prev, y_prev = landmark_video[i][j]
            x_new, y_new = x_prev + dx, y_prev + dy
            cv2.arrowedLine(frame, pt1=(int(x_prev), int(y_prev)), pt2=(int(x_new), int(y_new)),
                            color=(0, 255, 0), thickness=3, tipLength=0.3)
        frame_filename = os.path.join(output_path, f"frame_{i}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(frame_filename)

