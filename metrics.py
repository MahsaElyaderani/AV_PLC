import os
import json
import glob
import torch
import librosa
import warnings
import numpy as np
from pesq import pesq
from pystoi import stoi
from jiwer import wer
from speechmos import plcmos
from editdistance import eval as edit_eval
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from hifigan.hifigan.generator import HifiganGenerator
from audio_processing import torch_mel2audio
#from audio_processing import inv_spectrogram, inv_melspectrogram


class Vocoder:

    def __init__(self, checkpoint_path, device=None):#, config_path):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.generator = HifiganGenerator()
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

        # self.hifigan = torch.hub.load("https://github.com/bshall/hifigan/tree/main",
        #                                "hifigan_hubert_soft").cuda()

    def convert_torch(self, mel_spectrogram):

        self.generator.to('cuda')
        audio = self.generator(mel_spectrogram)
        return audio

    def convert(self, mel_spectrogram):

        self.generator.to('cuda')
        mel = torch.from_numpy(mel_spectrogram)
        mel = mel.unsqueeze(0).cuda()
        audio = self.generator(mel)

        # mel = torch.from_numpy(mel_spectrogram).unsqueeze(0).cuda()
        # audio, sr = self.hifigan.generate(mel)
        audio = audio.squeeze().cpu().detach().numpy()
        return audio

def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)


def calculate_psnr(original, reconstructed, max_val=None):
    mse = calculate_mse(original, reconstructed)
    if mse == 0:
        return float('inf')

    if max_val is None:
        max_val = np.max(np.abs(original))

    return 20 * np.log10(max_val / np.sqrt(mse))

def torch_mel_to_audio(spectrogram, phase):
    return torch_mel2audio(spectrogram)


def mel_to_audio_hifigan(mel_spectrogram, hifigan_vocoder):
    return hifigan_vocoder.convert(mel_spectrogram)


def calculate_pesq(original_audio, reconstructed_audio, sr=16000, mode='wb'):

    if sr not in [8000, 16000]:
        warnings.warn(f"PESQ requires sample rate of 8000 or 16000 Hz, got {sr}. Resampling...")
        target_sr = 16000
        original_audio = librosa.resample(original_audio, orig_sr=sr, target_sr=target_sr)
        reconstructed_audio = librosa.resample(reconstructed_audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    min_len = min(len(original_audio), len(reconstructed_audio))
    original_audio = original_audio[:min_len]
    reconstructed_audio = reconstructed_audio[:min_len]

    try:
        score = pesq(sr, original_audio, reconstructed_audio, mode)
        return score
    except Exception as e:
        warnings.warn(f"PESQ calculation failed: {str(e)}")
        return None


def calculate_stoi(original_audio, reconstructed_audio, sr=16000, extended=False):

    min_len = min(len(original_audio), len(reconstructed_audio))
    original_audio = original_audio[:min_len]
    reconstructed_audio = reconstructed_audio[:min_len]

    try:
        score = stoi(original_audio, reconstructed_audio, sr, extended=extended)
        return score
    except Exception as e:
        warnings.warn(f"STOI calculation failed: {str(e)}")
        return None


def calculate_metrics(original_spec, reconstructed_spec, phase, hifigan_vocoder, sample_rate=16000):

    metrics = {'mse': 0.0, 'psnr': 0.0, 'pesq': 0.0, 'stoi': 0.0}

    try:
        if hifigan_vocoder:
            original_audio = mel_to_audio_hifigan(original_spec, hifigan_vocoder)
            reconstructed_audio = mel_to_audio_hifigan(reconstructed_spec, hifigan_vocoder)

            # original_audio = original_audio.cpu().numpy()
            # reconstructed_audio = reconstructed_audio.cpu().numpy()
        else:
            original_audio = torch_mel_to_audio(original_spec.cpu(), phase)
            reconstructed_audio = torch_mel_to_audio(reconstructed_spec.cpu(), phase)

        original_spec_np = original_spec.cpu().numpy()
        reconstructed_spec_np = reconstructed_spec.cpu().numpy()
        original_audio_np = original_audio.cpu().numpy()
        reconstructed_audio_np = reconstructed_audio.cpu().numpy()

        metrics['mse'] = calculate_mse(original_spec_np, reconstructed_spec_np)
        metrics['psnr'] = calculate_psnr(original_spec_np, reconstructed_spec_np)

        pesq_score = calculate_pesq(original_audio_np, reconstructed_audio_np, sr=sample_rate)
        if pesq_score is not None:
            metrics['pesq'] = pesq_score

        stoi_score = calculate_stoi(original_audio_np, reconstructed_audio_np, sr=sample_rate)
        if stoi_score is not None:
            metrics['stoi'] = stoi_score

        #clean_p = plcmos.run(original_audio, sr=16000)
        #lossy_p = plcmos.run(os.path.join(lossy_dir, f), sr=16000)
        #inpainted_p = plcmos.run(reconstructed_audio, sr=16000)

        # clean_plcmos_values.append(clean_p['plcmos'])
        # lossy_plcmos_values.append(lossy_p['plcmos'])
        # inpainted_plcmos_values.append(inpainted_p['plcmos'])

    except Exception as e:
        warnings.warn(f"Failed to calculate audio metrics for sample: {str(e)}")

    return metrics

def calculate_batch_metrics(original_batch, reconstructed_batch,
                            phase, all_text, all_pred_text,
                            hifigan_vocoder, tokenizer, max_samples, sample_rate=16000):

    batch_metrics = {'mse': [], 'psnr': [],
                     'pesq': [], 'stoi': [], #'plcmos':[],
                     'cer': [], 'wer': []}
    #num_samples = min(max_samples, len(original_batch))

    for i in range(max_samples):
        try:
            if original_batch is not None and reconstructed_batch is not None:

                if hifigan_vocoder:
                    original_audio = mel_to_audio_hifigan(original_batch[i], hifigan_vocoder)
                    reconstructed_audio = mel_to_audio_hifigan(reconstructed_batch[i], hifigan_vocoder)
                else:
                    phase_i = phase[i] if phase is not None else None
                    original_audio = torch_mel_to_audio(original_batch[i].cpu(), phase_i)
                    reconstructed_audio = torch_mel_to_audio(reconstructed_batch[i].cpu(), phase_i)

                original_batch_np = original_batch[i].cpu().numpy()
                reconstructed_batch_np = reconstructed_batch[i].cpu().numpy()
                original_audio_np = original_audio.cpu().numpy()
                reconstructed_audio_np = reconstructed_audio.cpu().numpy()

                batch_metrics['mse'].append(calculate_mse(original_batch_np,
                                                          reconstructed_batch_np))
                batch_metrics['psnr'].append(calculate_psnr(original_batch_np,
                                                            reconstructed_batch_np))

                pesq_score = calculate_pesq(original_audio_np,
                                            reconstructed_audio_np,
                                            sr=sample_rate)
                if pesq_score is not None:
                    batch_metrics['pesq'].append(pesq_score)

                stoi_score = calculate_stoi(original_audio_np,
                                            reconstructed_audio_np,
                                            sr=sample_rate)
                if stoi_score is not None:
                    batch_metrics['stoi'].append(stoi_score)

                #clean_p = plcmos.run(original_audio, sr=16000)
                # lossy_p = plcmos.run(os.path.join(lossy_dir, f), sr=16000)
                #inpainted_p = plcmos.run(reconstructed_audio, sr=16000)

                #clean_plcmos_values.append(clean_p['plcmos'])
                #lossy_plcmos_values.append(lossy_p['plcmos'])
                #batch_metrics['plcmos'].append(inpainted_p['plcmos'])

            if tokenizer is not None:
                all_dec_text = tokenizer._decode_greedy(all_text[i])
                all_dec_pred_text = tokenizer._decode_greedy(all_pred_text[i])

                wer_score = wer(all_dec_text, all_dec_pred_text)
                batch_metrics['wer'].append(wer_score)

                cer_score = edit_eval(all_dec_text, all_dec_pred_text)
                batch_metrics['cer'].append(cer_score)

        except Exception as e:
            warnings.warn(f"Failed to calculate audio metrics for sample {i}: {str(e)}")

    result_metrics = {}
    for metric_name, values in batch_metrics.items():
        if values:
            result_metrics[metric_name] = np.mean(values)

    return result_metrics
