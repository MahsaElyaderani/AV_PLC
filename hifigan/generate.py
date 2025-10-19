from pathlib import Path
import numpy as np
import argparse
import torch
import torchaudio
from tqdm import tqdm
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from AV_PLC.hifigan.hifigan.generator import HifiganGenerator


def generate(args):
    print("Loading checkpoint")
    model_name = f"hifigan_hubert_{args.model}" if args.model != "base" else "hifigan"
    hifigan = torch.hub.load("bshall/hifigan:main", model_name).cuda()

    print(f"Generating audio from {args.in_dir}")
    for path in tqdm(list(args.in_dir.rglob("*.npy"))):
        mel = torch.from_numpy(np.load(path))
        mel = mel.unsqueeze(0).cuda()

        wav, sr = hifigan.generate(mel)
        wav = wav.squeeze(0).cpu()

        out_path = args.out_dir / path.relative_to(args.in_dir)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        torchaudio.save(out_path.with_suffix(".wav"), wav, sr)


class Vocoder:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = HifiganGenerator().to(self.device)

        print(f"Loading hifi_gan model from {checkpoint_path}")
        checkpoint_dict = torch.load(checkpoint_path, map_location={"cuda:0": f"cuda:{0}"})
        if 'generator' in checkpoint_dict:
            consume_prefix_in_state_dict_if_present(checkpoint_dict["generator"]["model"], "module.")
            self.generator.load_state_dict(checkpoint_dict["generator"]["model"])
        else:
            consume_prefix_in_state_dict_if_present(checkpoint_dict, "module.")
            self.generator.load_state_dict(checkpoint_dict)

        self.generator.eval()
        self.generator.remove_weight_norm()
        print("hifi_gan vocoder loaded successfully")


    @torch.no_grad()
    def infer(self, mel):
        # mel: torch.Tensor [80, T] or [1, 80, T] (already in the CORRECT units used during training)
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel)
        mel = mel.to(self.device, dtype=torch.float32)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if mel.shape[1] != 80 and mel.shape[2] == 80:
            mel = mel.transpose(1, 2)
        y = self.generator(mel)
        return y.squeeze(0).squeeze(0).detach().cpu()

if __name__ == "__main__":
    import h5py
    import matplotlib.pyplot as plt

    vocoder_path = '/home/ai/Projects/Mahsa/sources/AV_PLC/hifigan/checkpoints/model-best.pt'
    vocoder = Vocoder(vocoder_path)

    vox2_path = Path('/home/ai/Projects/Mahsa/datasets/vox2_short/vox2_short_test_features_chunk0.h5')
    #open h5 file and read mel
    f = h5py.File(vox2_path, 'r')
    keys = list(f.keys())
    mel = f[f'{keys[100]}/spec']
    mel = torch.from_numpy(mel[:])
    print(mel.shape, mel.min(), mel.max())
    wav = vocoder.infer(mel.to('cuda'))
    print(wav.shape, wav.max(), wav.min())
    # plot wav in non-interactive mode
    #save wav to test.wav
    torchaudio.save('test.wav', wav.unsqueeze(0), 16000)
    plt.plot(wav.numpy())
    plt.savefig('test.png')
    # parser = argparse.ArgumentParser(
    #     description="Generate audio for a directory of mel-spectrogams using HiFi-GAN."
    # )
    # parser.add_argument(
    #     "model",
    #     help="available models (HuBERT-Soft, HuBERT-Discrete, or Base).",
    #     choices=["soft", "discrete", "base"],
    # )
    # parser.add_argument(
    #     "in_dir",
    #     metavar="in-dir",
    #     help="path to input directory containing the mel-spectrograms.",
    #     type=Path,
    # )
    # parser.add_argument(
    #     "out_dir",
    #     metavar="out-dir",
    #     help="path to output directory.",
    #     type=Path,
    # )
    # args = parser.parse_args()
    #
    # generate(args)
