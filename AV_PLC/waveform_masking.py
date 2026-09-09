"""Waveform-domain packet loss masking used only by AV_PLC."""
from __future__ import annotations

import hashlib
import numpy as np
from shared.masking import generate_ge_trace_bursty, generate_single_gap_trace


def stable_seed(base_seed: int, condition: str, sample_id: str) -> int:
    token = f"{int(base_seed)}:{condition}:{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def packet_count(audio_len: int, packet_samples: int = 160) -> int:
    audio_len = max(0, int(audio_len))
    return (audio_len + packet_samples - 1) // packet_samples


def trace_to_sample_mask(trace, total_samples: int, audio_len: int,
                         packet_samples: int = 160) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)
    mask = np.ones(int(total_samples), dtype=np.float32)
    valid = min(int(audio_len), int(total_samples))
    if valid > 0:
        expanded = np.repeat(trace, packet_samples)
        mask[:valid] = expanded[:valid]
    return mask


def generate_ge_sample_mask(total_samples: int, audio_len: int, loss_rate: float,
                            packet_samples: int = 160) -> tuple[np.ndarray, np.ndarray]:
    n = packet_count(audio_len, packet_samples)
    trace = generate_ge_trace_bursty(n, loss_rate)
    return trace, trace_to_sample_mask(trace, total_samples, audio_len, packet_samples)


def generate_single_gap_sample_mask(total_samples: int, audio_len: int, gap_ms: float,
                                    sample_id: str, seed: int = 42,
                                    packet_samples: int = 160,
                                    sample_rate: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    n = packet_count(audio_len, packet_samples)
    hop_ms = 1000.0 * packet_samples / sample_rate
    trace = generate_single_gap_trace(n, gap_ms, sample_id, seed=seed, hop_ms=hop_ms)
    return trace, trace_to_sample_mask(trace, total_samples, audio_len, packet_samples)
