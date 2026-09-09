#!/usr/bin/env python3
"""Run AV-PLC ablations and model evaluations sequentially.

Ablations use dataset-native single-gap masking:
- gap sweep: the wide gap range defined in ablation_gap_sweep.py
- loss/modality ablation: 160, 500, and 1000 ms, defined in ablation.py
- video reduction: 160, 500, and 1000 ms, defined in
  ablation_video_reduction.py

The main model comparison always evaluates both Gilbert-Elliott and single-gap
masking through run_all_evaluations.py.

Each command runs in a separate Python process, so CUDA memory is released
between experiments. A completion marker is written only after a successful
command, allowing safe resume with --resume.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MODELS_ROOT = Path(__file__).resolve().parent
AV_PLC_DIR = MODELS_ROOT / "AV_PLC"
LOG_ROOT = AV_PLC_DIR / "logs" / "all_evaluations"

# Main model comparison always uses both mask types.
MODEL_EVALUATION_MASK_TYPES = ("gilbert", "single_gap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("grid", "lrs2", "voxceleb2"),
        default=["grid", "lrs2", "voxceleb2"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("av_lstm", "av_s2s", "av_transformer", "av_plc"),
        default=["av_lstm", "av_s2s", "av_transformer", "av_plc"],
        help="Models passed to run_all_evaluations.py.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_step(
    name: str,
    cmd: list[str],
    *,
    env: dict[str, str],
    resume: bool,
    continue_on_error: bool,
    dry_run: bool,
) -> bool:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    marker = LOG_ROOT / f"{name}.done"
    log_path = LOG_ROOT / f"{name}.log"

    if resume and marker.is_file():
        print(f"[skip] {name}: completion marker exists")
        return True

    print(f"\n[run] {name}")
    print("      " + " ".join(cmd))
    if dry_run:
        return True

    with log_path.open("a", encoding="utf-8") as log_file:
        started = datetime.now().isoformat(timespec="seconds")
        log_file.write(f"\n\n===== {started} | {' '.join(cmd)} =====\n")
        log_file.flush()
        result = subprocess.run(
            cmd,
            cwd=MODELS_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode == 0:
        marker.write_text(
            datetime.now().isoformat(timespec="seconds"),
            encoding="utf-8",
        )
        print(f"[ok]  {name} (log: {log_path})")
        return True

    print(f"[fail] {name}: exit code {result.returncode} (log: {log_path})")
    if not continue_on_error:
        raise SystemExit(result.returncode)
    return False


def main() -> None:
    args = parse_args()
    python = sys.executable
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    common_ablation_args = [
        "--datasets",
        *args.datasets,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
    ]

    steps: list[tuple[str, list[str]]] = [
        # Wide single-gap sweep. The gap list is defined in the script itself.
        (
            "01_ablation_gap_sweep",
            [
                python,
                str(AV_PLC_DIR / "ablation_gap_sweep.py"),
                *common_ablation_args,
            ],
        ),
        # Representative 160/500/1000 ms video-frame ablations.
        (
            "02_ablation_video_reduction",
            [
                python,
                str(AV_PLC_DIR / "ablation_video_reduction.py"),
                *common_ablation_args,
            ],
        ),
    ]

    # ablation.py accepts one dataset per invocation and internally evaluates
    # only the representative 160, 500, and 1000 ms single gaps.
    for index, dataset in enumerate(args.datasets, start=1):
        steps.append(
            (
                f"03_{index:02d}_loss_modality_ablation_{dataset}",
                [
                    python,
                    str(AV_PLC_DIR / "ablation.py"),
                    "--dataset",
                    dataset,
                    "--batch-size",
                    str(args.batch_size),
                    "--num-workers",
                    str(args.num_workers),
                ],
            )
        )

    # Main model comparison always uses both Gilbert-Elliott and single-gap
    # masking. Ablation scripts do not receive a mask-type argument.
    steps.append(
        (
            "04_run_all_evaluations",
            [
                python,
                str(MODELS_ROOT / "run_all_evaluations.py"),
                "--datasets",
                *args.datasets,
                "--models",
                *args.models,
                "--mask-types",
                *MODEL_EVALUATION_MASK_TYPES,
                "--continue-on-error",
            ],
        )
    )

    failures: list[str] = []
    for name, cmd in steps:
        ok = run_step(
            name,
            cmd,
            env=env,
            resume=args.resume,
            continue_on_error=args.continue_on_error,
            dry_run=args.dry_run,
        )
        if not ok:
            failures.append(name)

    if failures:
        print("\nCompleted with failures:")
        for name in failures:
            print(f"  - {name}")
        raise SystemExit(1)

    print("\nAll requested ablations and model evaluations completed successfully.")


if __name__ == "__main__":
    main()
