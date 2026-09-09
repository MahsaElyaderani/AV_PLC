import numpy as np

class GilbertElliottModel:
    def __init__(self, loss_rate=None, p_range=(0.2, 0.8), q_range=(0.2, 0.8), initial_state=None):
        if loss_rate is not None:

            #q_vals = np.linspace(q_range[0], q_range[1], 1000)
            q_vals = np.exp(np.linspace(np.log(q_range[0]), np.log(q_range[1]), 1000))
            p_vals = (loss_rate * q_vals) / (1 - loss_rate)
            mask = (p_vals >= p_range[0]) & (p_vals <= p_range[1])
            valid_p = p_vals[mask]
            valid_q = q_vals[mask]

            if len(valid_p) == 0:
                raise ValueError("No valid (p, q) found for given loss rate and ranges.")

            idx = np.random.randint(len(valid_p))
            self.p, self.q = valid_p[idx], valid_q[idx]

        else:
            self.p = np.random.uniform(*p_range)
            self.q = np.random.uniform(*q_range)
            loss_rate = self.p / (self.p + self.q)

        if initial_state is None:
            self.state = np.random.choice([0, 1], p=[1 - loss_rate, loss_rate])
        else:
            self.state = initial_state

    def step(self):
        if self.state == 0:
            self.state = np.random.choice([0, 1], p=[1 - self.p, self.p])
        else:
            self.state = np.random.choice([0, 1], p=[self.q, 1 - self.q])
        return self.state

    def simulate(self, spec_dim, spec_len, burn_in=10):
       trace = [self.step() for _ in range(spec_len + burn_in)]
       empirical_r = sum(trace) / len(trace)
       trace = np.array([1 - x for x in trace[burn_in:]])
       mask = np.repeat(trace[:, None], spec_dim, axis=1)
       return np.transpose(mask) #, empirical_r

def generate_ge_mask_bursty(spec_shape, loss_rate):

    model = GilbertElliottModel(
        loss_rate=loss_rate,
        p_range=(0.001, 0.80),
        q_range=(0.025, 0.06),
    )
    return model.simulate(*spec_shape).astype(np.float32)

def generate_ge_mask(spec_shape, loss_rate):
    """
    Generate a Gilbert-Elliott mask for the full spectrogram shape (F, T).
    Uses widened p/q ranges to support loss_rate ∈ [~0.05, ~0.95].
    Returns: np.float32 array of shape (F, T), with 1 = kept, 0 = lost.
    """
    model = GilbertElliottModel(
        loss_rate=loss_rate,
        p_range=(0.01, 0.99),
        q_range=(0.01, 0.99),
    )
    return model.simulate(*spec_shape).astype(np.float32)

def generate_single_gap_mask(spec_shape, gap_ms, sample_id, seed=42, hop_ms=10.0):
    """Return a deterministic single contiguous gap mask (1=kept, 0=lost).

    The stable sample_id makes the gap location identical across projects and
    DataLoader workers for the same dataset sample and gap duration.
    """
    import hashlib

    if len(spec_shape) != 2:
        raise ValueError(f"spec_shape must be (frequency, time), got {spec_shape}")
    freq_bins, time_frames = int(spec_shape[0]), int(spec_shape[1])
    if gap_ms <= 0:
        raise ValueError(f"gap_ms must be positive, got {gap_ms}")

    gap_frames = max(1, int(round(float(gap_ms) / float(hop_ms))))
    gap_frames = min(gap_frames, time_frames)

    token = f"{int(seed)}:{float(gap_ms):g}:{sample_id}".encode("utf-8")
    item_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    rng = np.random.default_rng(item_seed)
    start = int(rng.integers(0, time_frames - gap_frames + 1))

    mask = np.ones((freq_bins, time_frames), dtype=np.float32)
    mask[:, start:start + gap_frames] = 0.0
    return mask

# ---------------------------------------------------------------------------
# Additive valid-length helpers.  Existing mask APIs above are intentionally
# unchanged because AV_LSTM/AV_S2S/AV_Transformer reproduce spectrogram-domain
# methods.  These helpers expose the common 1-D packet trace so every project
# can use the same valid 10-ms packet locations.
# ---------------------------------------------------------------------------

def generate_ge_trace_bursty(time_steps, loss_rate):
    """Return a 1-D Gilbert-Elliott keep trace of length ``time_steps``."""
    time_steps = int(time_steps)
    if time_steps <= 0:
        return np.ones((0,), dtype=np.float32)
    model = GilbertElliottModel(
        loss_rate=loss_rate,
        p_range=(0.001, 0.80),
        q_range=(0.025, 0.06),
    )
    burn_in = 10
    trace = [model.step() for _ in range(time_steps + burn_in)]
    return np.asarray([1 - x for x in trace[burn_in:]], dtype=np.float32)


def generate_single_gap_trace(time_steps, gap_ms, sample_id, seed=42, hop_ms=10.0):
    """Deterministic 1-D contiguous keep trace (1=kept, 0=lost)."""
    import hashlib
    time_steps = int(time_steps)
    if time_steps <= 0:
        return np.ones((0,), dtype=np.float32)
    if gap_ms <= 0:
        raise ValueError(f"gap_ms must be positive, got {gap_ms}")
    gap_frames = max(1, int(round(float(gap_ms) / float(hop_ms))))
    gap_frames = min(gap_frames, time_steps)
    token = f"{int(seed)}:{float(gap_ms):g}:{sample_id}".encode("utf-8")
    item_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    rng = np.random.default_rng(item_seed)
    start = int(rng.integers(0, time_steps - gap_frames + 1))
    trace = np.ones(time_steps, dtype=np.float32)
    trace[start:start + gap_frames] = 0.0
    return trace


def packet_count_from_audio_len(audio_len, packet_samples=160):
    audio_len = max(0, int(audio_len))
    return (audio_len + int(packet_samples) - 1) // int(packet_samples)


def trace_to_spec_mask(trace, spec_shape):
    """Fill valid trace into a full [F,T] mask; padded tail is always kept."""
    if len(spec_shape) != 2:
        raise ValueError(f"spec_shape must be (F,T), got {spec_shape}")
    f, t = map(int, spec_shape)
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)
    mask = np.ones((f, t), dtype=np.float32)
    n = min(t, trace.size)
    if n:
        mask[:, :n] = trace[:n][None, :]
    return mask
