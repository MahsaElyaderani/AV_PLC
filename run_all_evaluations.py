#!/usr/bin/env python3
"""Run every project runner sequentially for GE loss rates and/or fixed gaps."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = {
    "av_lstm": ("AV_LSTM", ["--mode", "av", "--phase", "test", "--asr"]),
    "av_s2s": ("AV_S2S", ["--mode", "av", "--phase", "test", "--asr"]),
    "av_transformer": ("AV_Transformer", ["--mode", "av", "--phase", "test"]),
    "av_plc": ("AV_PLC", ["--mode", "av", "--phase", "test", "--asr", "--pmsqe"]),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["grid", "lrs2", "voxceleb2"])
    p.add_argument("--models", nargs="+", choices=JOBS, default=list(JOBS))
    p.add_argument("--mask-types", nargs="+", choices=["gilbert", "single_gap"],
                   default=["gilbert", "single_gap"])
    p.add_argument("--loss-rates", nargs="+", default=["10", "20", "30", "40", "50"])
    p.add_argument("--gap-durations", nargs="+", type=int,
                   default=[20, 40, 80, 160, 320, 500, 750, 1000, 1250, 1500])
    p.add_argument("--save-output", action="store_true",
                   help="Save reconstructed outputs from each model.")
    p.add_argument("--sample-paths", nargs="+", default=None,
                   help="Optional dataset-relative or absolute video paths to save. "
                        "If omitted with --save-output, save all test utterances.")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    failures = []
    for dataset in args.datasets:
        for key in args.models:
            project, fixed_args = JOBS[key]
            cmd = [
                sys.executable, "runner.py", "--dataset", dataset, *fixed_args,
                "--mask_types", *args.mask_types,
                "--plc_loss_rates", *args.loss_rates,
                "--gap_durations", *map(str, args.gap_durations),
            ]
            if args.save_output:
                cmd.append("--save_output")
            if args.sample_paths:
                cmd.extend(["--sample_paths", *args.sample_paths])
            print(f"\n=== {dataset} | {key} ===")
            print(" ".join(cmd))
            if args.dry_run:
                continue
            result = subprocess.run(cmd, cwd=ROOT / project)
            if result.returncode:
                failures.append((dataset, key, result.returncode))
                if not args.continue_on_error:
                    raise SystemExit(result.returncode)
    if failures:
        print("Failures:", *failures, sep="\n  ")
        raise SystemExit(1)
    print("\nAll requested evaluations finished.")


if __name__ == "__main__":
    main()

# python run_all_evaluations.py \
#     --datasets lrs2\
#     --mask-types single_gap \
#     --gap-durations 20, 40, 80, 160, 320, 500, 750, 1000, 1500
#     --models av_lstm av_s2s av_transformer av_plc

# python run_all_evaluations.py \
#     --datasets lrs2\
#     --mask-types gilbert \
#     --loss-rates 10 20 30 40 50
#     --models av_lstm av_s2s av_transformer av_plc
