#!/usr/bin/env python3
"""Standalone Monte Carlo statistics for generate_ge_mask_bursty."""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from shared.masking import generate_ge_mask_bursty

RATES = (10, 20, 30, 40, 50)
METRICS = (
    "realized_loss_rate_pct",
    "total_loss_ms",
    "longest_gap_ms",
    "number_of_gaps",
    "mean_gap_ms",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-masks", type=int, default=100_000)
    p.add_argument("--rates", type=int, nargs="+", default=list(RATES))
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--hop-ms", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--chunks-per-rate", type=int, default=200)
    p.add_argument("--output-dir", type=Path, default=Path("ge_monte_carlo_results"))
    p.add_argument("--save-raw", action="store_true")
    return p.parse_args()


def gap_lengths(loss_trace: np.ndarray) -> np.ndarray:
    """Lengths of contiguous True runs in frames."""
    padded = np.pad(loss_trace.astype(np.int8), (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return ends - starts


def measure_mask(mask: np.ndarray, hop_ms: float) -> tuple[float, ...]:
    # generate_ge_mask_bursty returns 1=kept and 0=lost.
    loss_trace = np.asarray(mask[0]) < 0.5
    lost_frames = int(loss_trace.sum())
    runs = gap_lengths(loss_trace)

    realized_loss_rate_pct = 100.0 * lost_frames / loss_trace.size
    total_loss_ms = lost_frames * hop_ms

    if runs.size:
        longest_gap_ms = float(runs.max() * hop_ms)
        number_of_gaps = float(runs.size)
        mean_gap_ms = float(runs.mean() * hop_ms)
    else:
        longest_gap_ms = 0.0
        number_of_gaps = 0.0
        mean_gap_ms = 0.0

    return (
        realized_loss_rate_pct,
        total_loss_ms,
        longest_gap_ms,
        number_of_gaps,
        mean_gap_ms,
    )


def simulate_chunk(task: tuple[int, int, int, int, float]) -> tuple[int, np.ndarray]:
    """Generate and measure one deterministic chunk of masks."""
    rate_pct, count, chunk_seed, frames, hop_ms = task
    np.random.seed(chunk_seed)

    out = np.empty((count, len(METRICS)), dtype=np.float64)
    for i in range(count):
        mask = generate_ge_mask_bursty(
            spec_shape=(1, frames),
            loss_rate=rate_pct / 100.0,
        )
        out[i] = measure_mask(mask, hop_ms)

    return rate_pct, out


def split_count(total: int, chunks: int) -> list[int]:
    chunks = min(max(1, chunks), total)
    base, remainder = divmod(total, chunks)
    return [base + int(i < remainder) for i in range(chunks)]


def build_tasks(args: argparse.Namespace) -> list[tuple[int, int, int, int, float]]:
    tasks = []
    for rate in args.rates:
        for chunk_idx, count in enumerate(split_count(args.num_masks, args.chunks_per_rate)):
            # This seed depends only on master seed, rate, and fixed chunk index.
            # Therefore, changing --workers does not change the generated masks.
            ss = np.random.SeedSequence([args.seed, rate, chunk_idx])
            chunk_seed = int(ss.generate_state(1, dtype=np.uint32)[0])
            tasks.append((rate, count, chunk_seed, args.frames, args.hop_ms))
    return tasks


def summarize(values: np.ndarray, rate: int) -> list[dict[str, float]]:
    rows = []
    for col, name in enumerate(METRICS):
        x = values[:, col]
        rows.append(
            {
                "target_loss_rate_pct": rate,
                "metric": name,
                "mean": float(np.mean(x)),
                "std": float(np.std(x, ddof=1)),
                "median": float(np.median(x)),
                "p2_5": float(np.percentile(x, 2.5)),
                "p97_5": float(np.percentile(x, 97.5)),
                "num_masks": int(x.size),
            }
        )
    return rows


def main() -> None:
    args = parse_args()

    if args.num_masks <= 0 or args.frames <= 0 or args.hop_ms <= 0:
        raise ValueError("num-masks, frames, and hop-ms must be positive.")
    if args.workers <= 0 or args.chunks_per_rate <= 0:
        raise ValueError("workers and chunks-per-rate must be positive.")
    if any(rate <= 0 or rate >= 100 for rate in args.rates):
        raise ValueError("All rates must be between 0 and 100.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)
    collected: dict[int, list[np.ndarray]] = {rate: [] for rate in args.rates}

    print(
        f"Generating {args.num_masks:,} masks per rate for {len(args.rates)} rates "
        f"with {args.workers} CPU workers..."
    )

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (rate, block) in enumerate(pool.map(simulate_chunk, tasks), start=1):
            collected[rate].append(block)
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                print(f"Completed {done}/{len(tasks)} chunks")

    summary_rows = []
    raw_tables = []

    for rate in args.rates:
        values = np.concatenate(collected[rate], axis=0)
        if len(values) != args.num_masks:
            raise RuntimeError(f"Expected {args.num_masks} masks for {rate}%, got {len(values)}")

        summary_rows.extend(summarize(values, rate))

        if args.save_raw:
            raw = pd.DataFrame(values, columns=METRICS)
            raw.insert(0, "target_loss_rate_pct", rate)
            raw_tables.append(raw)

    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "gilbert_elliott_summary.csv"
    summary.to_csv(summary_path, index=False)

    if args.save_raw:
        raw_path = args.output_dir / "gilbert_elliott_raw.csv"
        pd.concat(raw_tables, ignore_index=True).to_csv(raw_path, index=False)
        print(f"Raw results: {raw_path}")

    print(f"Summary results: {summary_path}\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
