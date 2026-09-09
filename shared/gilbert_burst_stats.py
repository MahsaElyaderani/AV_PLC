import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.runtime_config import SEED
from av_dataloader import AVDataloader


LOSS_RATES = ["10", "20", "30", "40", "50"]
CLIP_SEC = 3.0


def get_burst_lengths(mask_1d: np.ndarray) -> list[int]:
    """
    mask_1d:
        1 = available
        0 = missing

    Returns lengths of all contiguous missing regions in frames.
    """
    missing = mask_1d == 0

    padded = np.pad(missing.astype(np.int8), (1, 1))
    changes = np.diff(padded)

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    return (ends - starts).tolist()


def report_loss_rate(dataset_name: str, loss_rate: str) -> None:
    loader_builder = AVDataloader(
        dataset_name=dataset_name,
        mode="a",
        batch_size=16,
        num_workers=4,
        video_aug=False,
        dropout_modality=False,
    )

    loader = loader_builder.test_dataloader(
        mask_range=loss_rate,
        mask_type="gilbert",
        seed=SEED,
    )

    total_missing_ms = []
    mean_burst_ms = []
    max_burst_ms = []
    burst_counts = []

    for batch in loader:
        # Current audio-only batch:
        # masked_spec, mel_spec, audio_length, text, mask, video_path
        _, _, _, _, masks, _ = batch

        masks = masks.detach().cpu().numpy()

        for mask in masks:
            # The same time mask is repeated over mel bins.
            time_mask = mask[0]

            num_frames = time_mask.shape[0]
            frame_ms = CLIP_SEC * 1000.0 / num_frames

            burst_lengths = get_burst_lengths(time_mask)

            total_missing_ms.append(
                float(np.sum(time_mask == 0) * frame_ms)
            )

            burst_counts.append(len(burst_lengths))

            if burst_lengths:
                durations = np.asarray(burst_lengths) * frame_ms
                mean_burst_ms.append(float(np.mean(durations)))
                max_burst_ms.append(float(np.max(durations)))
            else:
                mean_burst_ms.append(0.0)
                max_burst_ms.append(0.0)

    print(f"\nGilbert–Elliott loss rate: {loss_rate}%")
    print(f"Samples: {len(total_missing_ms)}")
    print(
        f"Total missing duration: "
        f"{np.mean(total_missing_ms):.1f} ± "
        f"{np.std(total_missing_ms):.1f} ms"
    )
    print(
        f"Number of bursts: "
        f"{np.mean(burst_counts):.2f} ± "
        f"{np.std(burst_counts):.2f}"
    )
    print(
        f"Mean burst duration: "
        f"{np.mean(mean_burst_ms):.1f} ± "
        f"{np.std(mean_burst_ms):.1f} ms"
    )
    print(
        f"Maximum burst duration: "
        f"{np.mean(max_burst_ms):.1f} ± "
        f"{np.std(max_burst_ms):.1f} ms"
    )
    print(
        f"90th-percentile maximum burst: "
        f"{np.percentile(max_burst_ms, 90):.1f} ms"
    )


def main():
    dataset_name = "grid"

    for loss_rate in LOSS_RATES:
        report_loss_rate(dataset_name, loss_rate)


if __name__ == "__main__":
    main()

