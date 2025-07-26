import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.io import read_video
import torchaudio.transforms as T
import mediapipe as mp
import numpy as np
import cv2
import random
import librosa
from pathlib import Path
import torchvision
from PIL import Image
from typing import Tuple, List
from resemblyzer import VoiceEncoder, preprocess_wav
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS

from masking import GilbertElliottModel

"""
This Dataset reads the videos directly in every call and crop video frames. 
It slows down the training, that's why I have a script (save_features.py) to 
save video and audio features on the disk in .h5 file and read those during training.

        Args:
            video_paths: List of paths to video files
            sample_rate: Target audio sampling rate
            n_mels: Number of mel spectrogram bins
            crop_size: Output size for cropped lip region
            max_audio_length: Duration in seconds to truncate/pad audio
        """
class AV_Dataset(Dataset):
    def __init__(self, video_paths: List[str], loss_rate: str = 'rand',
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 crop_size: int = 96,
                 chunk_len_sec: float = 2.0):

        self.video_paths = video_paths
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.crop_size = crop_size
        self.chunk_len_sec = chunk_len_sec
        self.loss_rate = loss_rate

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=640,
            win_length=640,
            hop_length=160,
            center=False,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=self.n_mels,
            mel_scale="slaney",
        )
        self.voice_encoder = VoiceEncoder()

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1
        )
        self.video_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([112, 112]),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.421, std=0.165),
        ])

        self.lip_indices = sorted(set(i for connection in FACEMESH_LIPS for i in connection))

        self.landmark_dim = 478  # total landmarks in MediaPipe
        self.frame_size = (crop_size, crop_size)

    def __len__(self):
        return len(self.video_paths)

    def _read_video(self, path: str) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        video, audio, info = read_video(path)
        video_fps = info['video_fps']
        audio_fps = info['audio_fps']

        audio = audio.numpy().astype(np.float32)
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio

        if audio.ndim > 1 and audio.shape[0] > 1:
            audio = np.mean(audio, axis=0, keepdims=True)

        if audio_fps != self.sample_rate:
            audio = librosa.resample(audio[0], orig_sr=audio_fps, target_sr=self.sample_rate)

        audio_len = int(self.chunk_len_sec * self.sample_rate)
        video_len = int(self.chunk_len_sec * video_fps)

        #max_audio_start = max(0, len(audio) - audio_len)
        #max_video_start = max(0, video.shape[0] - video_len)

        #start_audio = random.randint(0, max_audio_start) if max_audio_start > 0 else 0
        #start_video = random.randint(0, max_video_start) if max_video_start > 0 else 0

        audio = audio[: audio_len]
        video = video[: video_len]

        return video, torch.from_numpy(audio), info

    def _video_roi(self, video: torch.Tensor, fps: float) -> Tuple[np.ndarray, np.ndarray]:
        video_np = video.numpy()
        h, w, _ = video_np[0].shape
        landmark_seq = []
        cropped_seq = []

        for frame in video_np:
            result = self.face_mesh.process(frame)
            landmarks = np.zeros((self.landmark_dim, 2), dtype=np.float32)

            if result.multi_face_landmarks:
                face = result.multi_face_landmarks[0]
                landmarks = np.array([[lm.x * w, lm.y * h] for lm in face.landmark], dtype=np.float32)
                lip_pts = landmarks[self.lip_indices]

                x1, y1 = np.min(lip_pts, axis=0)
                x2, y2 = np.max(lip_pts, axis=0)

                pad_x = (x2 - x1) * 0.5
                pad_y = (y2 - y1) * 0.5

                x1 = int(max(0, x1 - pad_x))
                y1 = int(max(0, y1 - pad_y))
                x2 = int(min(w, x2 + pad_x))
                y2 = int(min(h, y2 + pad_y))

                cropped = frame[y1:y2, x1:x2]
                cropped = cv2.resize(cropped, self.frame_size)
            else:
                cropped = np.zeros((*self.frame_size, 3), dtype=np.float32)

            landmark_seq.append(landmarks[self.lip_indices])
            cropped_seq.append(cropped)

        landmark_seq = np.stack(landmark_seq).astype(np.float32)
        cropped_seq = np.stack(cropped_seq).astype(np.float32)

        return cropped_seq, landmark_seq

    def _audio_mel(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        #audio = (audio - audio.mean()) / (audio.std() + 1e-5)
        padding = (640 - 160) // 2
        audio = torch.nn.functional.pad(audio, (padding, padding), "constant")
        mel_spec = self.mel_transform(audio)
        logmel = torch.log(torch.clamp(mel_spec, min=1e-5))
        #mel_spec = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0)

        return audio, logmel

    def _uniform_mask(self, size, loss_bounds=(0.3, 0.7)):

        model = GilbertElliottModel(loss_rate=np.random.uniform(*loss_bounds))
        return model.simulate(*size)

    def _random_mask(self, size, loss_rate):

        model = GilbertElliottModel(loss_rate=loss_rate)
        return model.simulate(*size)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        video, audio, info = self._read_video(path)

        cropped_frames, landmarks = self._video_roi(video, info['video_fps'])
        audio, mel_spec = self._audio_mel(audio)
        mel_spec = torch.nn.functional.layer_norm(mel_spec, mel_spec.shape)

        # cropped_frames = torch.from_numpy(cropped_frames).float() / 255.0  # [T, C, H, W]
        processed_frames = []

        for frame in cropped_frames:
            # Convert from NumPy to PIL for torchvision transforms (Resize, Grayscale, etc.)
            pil_frame = Image.fromarray(frame.astype(np.uint8))  # Use fromarray safely
            transformed = self.video_transform(pil_frame)
            processed_frames.append(transformed)

        # Stack into a Tensor: shape (T, C, H, W)
        lip_frames = torch.stack(processed_frames)

        if 'train' in path or 'val' in path:
            mask = torch.from_numpy(self._uniform_mask(mel_spec.shape))
        else:
            mask = torch.from_numpy(self._random_mask(mel_spec.shape, self.loss_rate))

        spkr_embed = self.voice_encoder.embed_utterance(preprocess_wav(Path(path)))
        np.set_printoptions(precision=3, suppress=True)

        return lip_frames, spkr_embed, mask * mel_spec, mel_spec, mask  # [T, C, crop_size, crop_size], [n_mels, time]
