import cv2
import torch
import numpy as np
import utils as avhubert_utils
from av_hubert.fairseq.fairseq.checkpoint_utils import load_model_ensemble_and_task

def extract_roi_landmarks(video, face_mesh, video_duration=TARGET_DURATION_S * FPS):

    INDICES = [
        # Outer lips
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267,
        269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80,
        81, 82, 13, 312, 311, 310, 415,
        # Inner lips
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402,
        318, 324, 308,
        # Jaw line
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267,
        269, 270, 409,
    ]

    # Remove duplicates while preserving order
    INDICES = list(dict.fromkeys(INDICES))
    LIP_PTS = len(INDICES)
    CFRAME_SIZE = (96, 96) #(64, 64)

    landmark_video = []
    cropped_frames = []

    for image in video.numpy():
        h, w, _ = image.shape
        process_landmarks = face_mesh.process(image)
        landmark_frame = np.zeros((LANDMARK_PTS, 2), dtype=np.float16)

        if process_landmarks.multi_face_landmarks:
            face = process_landmarks.multi_face_landmarks[0]
            landmark_frame = np.array([(lm.x * w, lm.y * h) for lm in face.landmark],
                                      dtype=np.float16)

            lip_landmarks = landmark_frame[INDICES]

            # Calculate bounding box with padding
            min_x, min_y = np.min(lip_landmarks, axis=0)
            max_x, max_y = np.max(lip_landmarks, axis=0)

            # Add padding (50% of width/height)
            pad_x = (max_x - min_x) * 0.5
            pad_y = (max_y - min_y) * 0.5

            # Ensure bounding box is within image boundaries
            x1 = max(0, int(min_x - pad_x))
            y1 = max(0, int(min_y - pad_y))
            x2 = min(w, int(max_x + pad_x))
            y2 = min(h, int(max_y + pad_y))

            cropped_image = image[y1:y2, x1:x2]
            cropped_image = cv2.resize(cropped_image, CFRAME_SIZE)
            #pil_img = Image.fromarray(cropped_image)
            #resized_img = pil_img.resize(CFRAME_SIZE)  # CFRAME_SIZE should be (width, height)
            #cropped_image = np.array(resized_img)
            cropped_frames.append(cropped_image)
        else:
            cropped_frames.append(np.zeros((CFRAME_SIZE[0], CFRAME_SIZE[1], 3)))

        landmark_video.append(landmark_frame[INDICES])

    if not landmark_video:
        return None, None, None

    cropped_frames = np.array(cropped_frames, dtype=np.float32)
    landmark_video = np.array(landmark_video, dtype=np.float16)

    video_duration = round(video_duration)
    len_frames = len(cropped_frames)
    if len_frames < video_duration:
        pad_size = video_duration - len_frames
        padding_vecs = np.zeros((pad_size, LIP_PTS, 2), dtype=np.float16)
        landmark_video = np.concatenate((landmark_video, padding_vecs), axis=0)

        ch, cw, cc = cropped_frames.shape[1:]
        padding_cropped = np.zeros((pad_size, ch, cw, cc), dtype=np.float16)
        cropped_frames = np.concatenate((cropped_frames, padding_cropped), axis=0)
    else:
        landmark_video = landmark_video[:video_duration]
        cropped_frames = cropped_frames[:video_duration]

    #motion_features = np.diff(landmark_video, axis=0)

    return cropped_frames, landmark_video#, motion_features


def extract_visual_feature(roi_frames):

    try:
        gray_frames = []
        for roi_frame in roi_frames:  # each frame shape: (96, 96, 3)
            #roi_frame = roi_frame.astype(np.float32)
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_RGB2GRAY)
            #gray = np.dot(roi_frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            gray_frames.append(gray)

        gray_frames = np.stack(gray_frames)  # shape: (102, 96, 96)
        processed_frames = transform(gray_frames)
        frames_tensor = torch.FloatTensor(processed_frames).unsqueeze(0).unsqueeze(0).cuda()

        with torch.no_grad():
            feature, _= model.extract_finetune(
                source={'video': frames_tensor, 'audio': None},
                padding_mask=None,
                output_layer=None
            )
            feature = feature.squeeze(dim=0)

        feature_np = feature.cpu().numpy()
        del frames_tensor, feature

        return feature_np
    finally:
        # Ensure memory is cleared even if an exception occurs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
