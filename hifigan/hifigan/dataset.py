from pathlib import Path
import math
import random
import glob
import os
import numpy as np
import torch
import torch.nn.functional as F

import torchaudio
from torch.utils.data import Dataset
from torchvision.io import read_video


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.melspctrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        center=False,
        power=1.0,
        norm="slaney",
        onesided=True,
        n_mels=80,
        mel_scale="slaney",
    )

        self.amp2db_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)
    def forward(self, wav):
        padding = (512 - 160) // 2
        wav = torch.nn.functional.pad(wav, (padding, padding), "reflect")
        mel = self.melspctrogram(wav)
        mel = mel.clamp(min=1e-5)  # avoid log(0)
        logmel = self.amp2db_transform(mel)
        return logmel


class MelDataset(Dataset):
    def __init__(
        self,
        root: Path,
        segment_length: int,
        sample_rate: int,
        hop_length: int,
        train: bool = True,
        finetune: bool = False,
    ):
        self.wavs_dir = root #/ "wavs"
        self.mels_dir = root / "mels"
        self.data_dir = self.wavs_dir if not finetune else self.mels_dir

        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.train = train
        self.finetune = finetune

        #suffix = ".wav" if not finetune else ".npy"
        #pattern = f"train/**/*{suffix}" if train else f"dev/**/*{suffix}"

        #self.metadata = [
        #    path.relative_to(self.data_dir).with_suffix("")
        #    for path in self.data_dir.rglob(pattern)
        #]

        suffix = ".mp4" if not finetune else ".npy"
        pattern = f"*/*/*{suffix}"
        video_list = glob.glob(os.path.join(self.data_dir, pattern))
        random.shuffle(video_list)
        if len(video_list) > 30000:
            video_list = video_list[:30000]
        split_point = int(len(video_list) * 0.9)
        if train:
            self.metadata = video_list[:split_point]
        else:
            self.metadata = video_list[split_point:]

        self.logmel = LogMelSpectrogram()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]
        #wav_path = self.wavs_dir / path
        wav_path = self.metadata[index]

        # info = torchaudio.info(wav_path.with_suffix(".wav"))
        # if info.sample_rate != self.sample_rate:
        #     raise ValueError(
        #         f"Sample rate {info.sample_rate} doesn't match target of {self.sample_rate}"
        #     )
        _, _wav, _meta = read_video(wav_path)
        if _wav.dim() == 2 and _wav.size(0) > 1:  # [C, T]
            _wav = _wav.mean(dim=0, keepdim=True)  # [1, T]

        sr = _meta['audio_fps']  # sr = info.sample_rate
        if sr != self.sample_rate:
            raise ValueError(
                f"Sample rate {sr} doesn't match target of {self.sample_rate}"
            )

        if self.finetune:
            mel_path = self.mels_dir / path
            src_logmel = torch.from_numpy(np.load(mel_path.with_suffix(".npy")))
            src_logmel = src_logmel.unsqueeze(0)

            mel_frames_per_segment = math.ceil(self.segment_length / self.hop_length)
            mel_diff = src_logmel.size(-1) - mel_frames_per_segment if self.train else 0
            mel_offset = random.randint(0, max(mel_diff, 0))

            frame_offset = self.hop_length * mel_offset
        else:
            #frame_diff = info.num_frames - self.segment_length
            frame_diff = _wav.size(-1) - self.segment_length
            frame_offset = random.randint(0, max(frame_diff, 0))

        # wav, _ = torchaudio.load(
        #     filepath=wav_path.with_suffix(".wav"),
        #     frame_offset=frame_offset if self.train else 0,
        #     num_frames=self.segment_length if self.train else -1,
        # )
        wav = _wav[:, frame_offset : frame_offset + self.segment_length] if self.train else _wav
        if wav.size(-1) < self.segment_length:
            wav = F.pad(wav, (0, self.segment_length - wav.size(-1)))

        if not self.finetune and self.train:
            gain = random.random() * (0.99 - 0.4) + 0.4
            flip = -1 if random.random() > 0.5 else 1
            wav = flip * gain * wav / max(wav.abs().max(), 1e-5)

        tgt_logmel = self.logmel(wav.unsqueeze(0)).squeeze(0)

        if self.finetune:
            if self.train:
                src_logmel = src_logmel[
                    :, :, mel_offset : mel_offset + mel_frames_per_segment
                ]

            if src_logmel.size(-1) < mel_frames_per_segment:
                src_logmel = F.pad(
                    src_logmel,
                    (0, mel_frames_per_segment - src_logmel.size(-1)),
                    "constant",
                    src_logmel.min(),
                )
        else:
            src_logmel = tgt_logmel.clone()

        return wav, src_logmel, tgt_logmel

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    dataset = MelDataset(
        root=Path("/home/ai/Projects/Mahsa/datasets/vox2_short/vox2_dev_mp4"),
        segment_length=8192,
        sample_rate=16000,
        hop_length=160,
        train=True,
    )
    print(f"Number of training utterances: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    for i, batch in enumerate(dataloader):
        wavs, src_mels, tgt_mels = batch
        #print(wavs.shape, src_mels.shape, tgt_mels.shape)
        #plt.imshow(tgt_mels[0][0].numpy(), aspect='auto', origin='lower')
        #plt.show()
        print(src_mels.min(), src_mels.max())
        if i == 10:
            break
