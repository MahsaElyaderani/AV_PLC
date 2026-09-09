import os
import json
import glob
import torch
import torchaudio
import librosa
import soundfile as sf
import warnings
import numpy as np
from pesq import pesq
from pystoi import stoi
import jiwer
from speechmos import plcmos
from editdistance import eval as edit_eval
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

try:
    from hifigan.hifigan.generator import HifiganGenerator
except ImportError:
    HifiganGenerator = None
from shared.audio_processing import torch_mel2audio, librosa_mel2audio, load_audio_ffmpeg
from concurrent.futures import ProcessPoolExecutor, as_completed
import re, unicodedata
import whisper
from shared.text_processing import TextTokenizer
from evaluations.runtime_config import DATA_ROOT

text_decoder = TextTokenizer()

_ASR_MODEL = None
def get_asr_model(model_size="tiny.en", device="cuda"):
    global _ASR_MODEL
    if _ASR_MODEL is None:
        _ASR_MODEL = whisper.load_model(model_size, device=device)
    return _ASR_MODEL

def asr_transcribe_np(audio_np_16k: np.ndarray, language="en") -> str:
    model = get_asr_model()
    # expects 16 kHz mono float32
    out = model.transcribe(audio_np_16k, language=language, task="transcribe", fp16=True)
    return out["text"]

def batch_asr_transcribe(audio_list, model=None, language="en", batch_size=32):
    if model is None:
        model = get_asr_model()
    device = next(model.parameters()).device
    texts = []

    for start in range(0, len(audio_list), batch_size):
        chunk = audio_list[start:start + batch_size]

        mels = []
        for audio_np in chunk:
            audio_t = whisper.pad_or_trim(torch.from_numpy(audio_np).float())
            mel = whisper.log_mel_spectrogram(audio_t).to(device)
            mels.append(mel)
        mel_batch = torch.stack(mels, dim=0)

        # decode does encoding internally — no need for explicit encoder call
        options = whisper.DecodingOptions(language=language, fp16=(device.type == 'cuda'),
                                          without_timestamps=True)
        with torch.no_grad():
            results = whisper.decode(model, mel_batch, options)
        texts.extend([r.text for r in results])

    return texts

def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)      # keep apostrophes; drop other punct
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _levenshtein_distance(seq_a, seq_b):
    """Simple Levenshtein (O(len(a)*len(b))) for fallback WER/CER."""
    m, n = len(seq_a), len(seq_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + 1,           # deletion
                dp[j-1] + 1,         # insertion
                prev + (seq_a[i-1] != seq_b[j-1])  # substitution
            )
            prev = cur
    return dp[-1]

def compute_wer_cer(ref_text: str, hyp_text: str):
    """WER/CER with jiwer if available; fallback to Levenshtein."""
    try:
        transf = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.Strip()])
        wer_val = jiwer.wer(ref_text, hyp_text, truth_transform=transf, hypothesis_transform=transf)
        cer_val = jiwer.cer(ref_text, hyp_text, truth_transform=transf, hypothesis_transform=transf)
        return float(wer_val), float(cer_val)
    except Exception:
        ref_norm, hyp_norm = normalize_text(ref_text), normalize_text(hyp_text)
        ref_words, hyp_words = ref_norm.split(), hyp_norm.split()
        if len(ref_words) == 0:
            wer_val = 0.0 if len(hyp_words) == 0 else 1.0
            cer_val = 0.0 if len(ref_norm) == 0 else 1.0
        else:
            wer_val = _levenshtein_distance(ref_words, hyp_words) / len(ref_words)
            cer_val = _levenshtein_distance(ref_norm, hyp_norm) / len(ref_norm)
        return float(wer_val), float(cer_val)


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

def torch_mel_to_audio(spectrogram, mel_mean=-56.775, mel_std=19.707):
    return librosa_mel2audio(
        spectrogram, mel_mean=mel_mean, mel_std=mel_std
    )

def mel_to_audio_hifigan(mel_spectrogram, hifigan_vocoder,
                          mel_mean=-56.775, mel_std=19.707):
    mel_spectrogram = mel_spectrogram * mel_std + mel_mean
    return hifigan_vocoder.infer(mel_spectrogram)

def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def calculate_psnr(original, reconstructed, max_val=None):
    mse = calculate_mse(original, reconstructed)
    if mse == 0:
        return float('nan')

    if max_val is None:
        max_val = np.max(np.abs(original))

    return 10 * np.log10((max_val ** 2) / mse)

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

def calculate_estoi(original_audio, reconstructed_audio, sr=16000, extended=True):

    min_len = min(len(original_audio), len(reconstructed_audio))
    original_audio = original_audio[:min_len]
    reconstructed_audio = reconstructed_audio[:min_len]

    try:
        score = stoi(original_audio, reconstructed_audio, sr, extended=extended)
        return score
    except Exception as e:
        warnings.warn(f"ESTOI calculation failed: {str(e)}")
        return None


def calculate_metrics(original_spec, reconstructed_spec,
                      path, mask, text, hifigan_vocoder, sample_rate=16000,
                      mel_mean=-56.775, mel_std=19.707, masked_input=False,
                      reconstructed_audio=None):

    get_asr_model()
    metrics = {'mse': [], 'psnr': [],
               'pesq': [], 'stoi': [], 'estoi': [],
                'plcmos': [],
                'cer': [], 'wer': []}

    rel_path = path.split("datasets", 1)[-1].lstrip(os.sep)
    audio_path = os.path.join(str(DATA_ROOT), rel_path)
    try:
        original_audio_np = load_audio_ffmpeg(audio_path, sr=sample_rate, fixlen_sec=3)
        if masked_input:
            if mask is None:
                raise ValueError("mask is required when masked_input=True")
            time_keep = mask[0].numpy().astype(np.float32)
            sample_mask = np.repeat(time_keep, 160)
            sample_mask = sample_mask[:len(original_audio_np)]
            if len(sample_mask) < len(original_audio_np):
                sample_mask = np.pad(sample_mask, (0, len(original_audio_np) - len(sample_mask)), constant_values=1.0)
            reconstructed_audio_np = original_audio_np * sample_mask
        else:
            if reconstructed_audio is None:
                if hifigan_vocoder:
                    reconstructed_audio = mel_to_audio_hifigan(reconstructed_spec, hifigan_vocoder, mel_mean, mel_std)
                else:
                    reconstructed_audio = torch_mel_to_audio(reconstructed_spec.cpu(), mel_mean, mel_std)
            reconstructed_audio_np = np.asarray(reconstructed_audio.detach().cpu()
                                                if torch.is_tensor(reconstructed_audio)
                                                else reconstructed_audio, dtype=np.float32,).squeeze()
            if mask is not None:
                time_keep = mask[0].numpy().astype(np.float32)
                sample_mask = np.repeat(time_keep, 160)[:len(original_audio_np)]
                n = min(len(original_audio_np), len(reconstructed_audio_np), len(sample_mask))
                reconstructed_audio_np = (original_audio_np[:n] * sample_mask[:n] +
                                          reconstructed_audio_np[:n] * (1.0 - sample_mask[:n]))

        original_spec_np = original_spec.cpu().numpy()
        reconstructed_spec_np = reconstructed_spec.cpu().numpy()

        metrics['mse'] = calculate_mse(original_spec_np, reconstructed_spec_np)
        metrics['psnr'] = calculate_psnr(original_spec_np, reconstructed_spec_np)

        pesq_score = calculate_pesq(original_audio_np, reconstructed_audio_np, sr=sample_rate)
        if pesq_score is not None:
            metrics['pesq'] = pesq_score

        stoi_score = calculate_stoi(original_audio_np, reconstructed_audio_np, sr=sample_rate)
        if stoi_score is not None:
            metrics['stoi'] = stoi_score

        estoi_score = calculate_estoi(original_audio_np, reconstructed_audio_np, sr=sample_rate)
        if estoi_score is not None:
            metrics['estoi'] = estoi_score

        ref_text = text_decoder.decode(text) #asr_transcribe_np(original_audio_np, language="en")
        hyp_text = asr_transcribe_np(reconstructed_audio_np, language="en")

        wer_score, cer_score = compute_wer_cer(ref_text, hyp_text)
        metrics['wer'] = wer_score
        metrics['cer'] = cer_score

        if plcmos is not None:
            metrics['plcmos'] = plcmos.run(reconstructed_audio_np/ (np.max(np.abs(reconstructed_audio_np)) + 1e-8),
                                           sr=sample_rate, return_df=False)['plcmos']

    except Exception as e:
        warnings.warn(f"Failed to calculate audio metrics for sample: {str(e)}")

    return metrics

def calculate_batch_metrics(original_batch, reconstructed_batch,
                            path, mask, max_samples,
                            texts, pred_texts=None,
                            phones=None, pred_phones=None,
                            hifigan_vocoder=None, tokenizer=None,
                            sample_rate=16000, asr_lang="en",
                            mel_mean=-56.775, mel_std=19.707, masked_input=False,
                            reconstructed_audio_batch=None,
                            original_audio_batch=None,
                            sample_mask_batch=None,
                            audio_lengths=None,
                            frame_valid_batch=None,
                            normalize_wer_text=False):

    get_asr_model()

    batch_metrics = {'mse': [], 'psnr': [],
                     'pesq': [], 'stoi': [], 'estoi': [],
                     'plcmos': [],
                     'cer': [], 'wer': [],
                     'wer_vsr': [], 'cer_vsr': []}

    for i in range(max_samples):
        try:
            if original_batch is not None and reconstructed_batch is not None:

                if original_audio_batch is not None:
                    original_audio_np = np.asarray(
                        original_audio_batch[i].detach().cpu() if torch.is_tensor(original_audio_batch[i])
                        else original_audio_batch[i], dtype=np.float32
                    ).squeeze()
                else:
                    rel_path = path[i].split("datasets", 1)[-1].lstrip(os.sep)
                    audio_path = os.path.join(str(DATA_ROOT), rel_path)
                    original_audio_np = load_audio_ffmpeg(audio_path, sr=sample_rate, fixlen_sec=3)

                # AV/audio PLC uses a mask and inserts only the reconstructed gap.
                # Video-only synthesis passes mask=None and must be evaluated over
                # the complete synthesized waveform.
                
                sample_mask = None
                # A sample-domain mask is meaningful only for PLC conditions.
                # Video-only synthesis deliberately passes mask=None and must
                # be scored over the complete synthesized waveform.
                if sample_mask_batch is not None and (mask is not None or masked_input):
                    sample_mask = np.asarray(
                        sample_mask_batch[i].detach().cpu() if torch.is_tensor(sample_mask_batch[i])
                        else sample_mask_batch[i], dtype=np.float32
                    ).squeeze()[:len(original_audio_np)]
                elif mask is not None:
                    time_keep = mask[i, 0].cpu().numpy().astype(np.float32)
                    sample_mask = np.repeat(time_keep, 160)[:len(original_audio_np)]
                    if len(sample_mask) < len(original_audio_np):
                        sample_mask = np.pad(
                            sample_mask,
                            (0, len(original_audio_np) - len(sample_mask)),
                            constant_values=1.0,
                        )

                if masked_input:
                    if sample_mask is None:
                        raise ValueError("masked_input=True requires a non-None mask")
                    # Exact masked-input waveform baseline: no Griffin-Lim or vocoder.
                    reconstructed_audio_np = original_audio_np * sample_mask
                else:
                    if reconstructed_audio_batch is not None:
                        reconstructed_audio = reconstructed_audio_batch[i]
                    elif hifigan_vocoder:
                        reconstructed_audio = mel_to_audio_hifigan(reconstructed_batch[i],
                                                                   hifigan_vocoder,
                                                                   mel_mean, mel_std)
                    else:
                        reconstructed_audio = torch_mel_to_audio(reconstructed_batch[i].cpu(),
                                                                 mel_mean, mel_std)
                    reconstructed_audio = np.asarray(
                        reconstructed_audio.detach().cpu()
                        if torch.is_tensor(reconstructed_audio)
                        else reconstructed_audio,
                        dtype=np.float32,
                    ).squeeze()

                    if sample_mask is None:
                        # Video-only: compare the complete synthesized waveform.
                        n = min(len(original_audio_np), len(reconstructed_audio))
                        original_audio_np = original_audio_np[:n]
                        reconstructed_audio_np = reconstructed_audio[:n]
                    else:
                        # AV/audio PLC: retain original audio outside the gap.
                        n = min(
                            len(original_audio_np),
                            len(reconstructed_audio),
                            len(sample_mask),
                        )
                        original_audio_np = original_audio_np[:n]
                        reconstructed_audio_np = (
                            original_audio_np * sample_mask[:n]
                            + reconstructed_audio[:n] * (1.0 - sample_mask[:n])
                        )

                if audio_lengths is not None:
                    valid_n = int(audio_lengths[i])
                    valid_n = max(0, min(valid_n, len(original_audio_np), len(reconstructed_audio_np)))
                    original_audio_np = original_audio_np[:valid_n]
                    reconstructed_audio_np = reconstructed_audio_np[:valid_n]

                original_batch_np = original_batch[i].cpu().numpy()
                reconstructed_batch_np = reconstructed_batch[i].cpu().numpy()
                if frame_valid_batch is not None:
                    fv = np.asarray(
                        frame_valid_batch[i].detach().cpu() if torch.is_tensor(frame_valid_batch[i])
                        else frame_valid_batch[i], dtype=bool
                    ).reshape(-1)
                    n_t = min(original_batch_np.shape[-1], reconstructed_batch_np.shape[-1], fv.size)
                    fv = fv[:n_t]
                    original_batch_np = original_batch_np[..., :n_t][..., fv]
                    reconstructed_batch_np = reconstructed_batch_np[..., :n_t][..., fv]

                # --------- MSE, PSNR, PESQ, STOI metrics ----------
                batch_metrics['mse'].append(calculate_mse(original_batch_np, reconstructed_batch_np))
                batch_metrics['psnr'].append(calculate_psnr(original_batch_np, reconstructed_batch_np))

                pesq_score = calculate_pesq(original_audio_np, reconstructed_audio_np, sr=sample_rate)
                if pesq_score is not None:
                    batch_metrics['pesq'].append(pesq_score)

                stoi_score = calculate_stoi(original_audio_np, reconstructed_audio_np, sr=sample_rate)
                if stoi_score is not None:
                    batch_metrics['stoi'].append(stoi_score)

                estoi_score = calculate_estoi(original_audio_np, reconstructed_audio_np, sr=sample_rate)
                if estoi_score is not None:
                    batch_metrics['estoi'].append(estoi_score)

                # --------- ASR-based WER/CER  ----------

                #ref_text = asr_transcribe_np(original_audio_np, language=asr_lang)
                ref_text = text_decoder.decode(texts[i])
                hyp_text = asr_transcribe_np(reconstructed_audio_np, language=asr_lang)
                if normalize_wer_text:
                    ref_text = text_decoder.normalize_text(ref_text)
                    hyp_text = text_decoder.normalize_text(hyp_text)

                wer_score, cer_score = compute_wer_cer(ref_text, hyp_text)
                batch_metrics['wer'].append(wer_score)
                batch_metrics['cer'].append(cer_score)

                # --------- PLCMOS  ----------
                if plcmos is not None:
                    plc = plcmos.run(reconstructed_audio_np/ (np.max(np.abs(reconstructed_audio_np)) + 1e-8),
                                     sr=sample_rate, return_df=False)
                    batch_metrics['plcmos'].append(plc['plcmos'])

            #  model-tokenized WER/CER:
            if tokenizer is not None and pred_phones is not None and phones is not None:

                ref_tok = tokenizer.decode(phones[i])
                hyp_tok = tokenizer.decode_greedy(pred_phones[i])
                wer_tok, cer_tok = compute_wer_cer(ref_tok, hyp_tok)
                batch_metrics['wer_vsr'].append(wer_tok)
                batch_metrics['cer_vsr'].append(cer_tok)

            if tokenizer is not None and pred_texts is not None and texts is not None:

                ref_tok = tokenizer.decode(texts[i])
                hyp_tok = tokenizer.decode_greedy(pred_texts[i])
                wer_tok, cer_tok = compute_wer_cer(ref_tok, hyp_tok)
                batch_metrics['wer_vsr'].append(wer_tok)
                batch_metrics['cer_vsr'].append(cer_tok)

        except Exception as e:
            warnings.warn(f"Failed to calculate metrics for sample {i}: {str(e)}")

    # reduce to means
    result_metrics = {k: float(np.nanmean(v)) for k, v in batch_metrics.items() if v}
    return result_metrics
