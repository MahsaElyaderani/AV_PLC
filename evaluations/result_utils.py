"""Shared helpers for combining evaluation results across masking conditions."""

import csv
import math
import os
from collections import defaultdict


def _finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def save_combined_results(model_log_dir, mask_type, records):
    """Save condition-level and mean metrics for one mask type.

    Parameters
    ----------
    model_log_dir : str
        Log directory of the current model, normally ``trainer.run_dir``.
    mask_type : str
        For example ``gilbert``, ``single_gap``, or ``video_only``.
    records : list[dict]
        Each record must contain ``source`` and ``results`` and may contain
        ``loss_rate`` or ``gap_ms``. ``results`` may be the return value from
        either ``Trainer.evaluate`` or ``Trainer.evaluate_ablation``.

    Returns
    -------
    str | None
        Path to the combined CSV, or ``None`` when there are no valid rows.
    """
    rows = []

    for record in records:
        results = record.get("results")

        # evaluate_ablation returns (metrics_dict, csv_path).
        if isinstance(results, tuple):
            results = results[0]

        if not isinstance(results, dict):
            continue

        for output_name, metrics in results.items():
            if output_name == "csv_path" or not isinstance(metrics, dict):
                continue

            row = {
                "source": record.get("source", "evaluate"),
                "mask_type": mask_type,
                "loss_rate": record.get("loss_rate", "") if mask_type == "gilbert" else "",
                "gap_ms": record.get("gap_ms", "") if mask_type == "single_gap" else "",
                "output": output_name,
                "summary": "condition",
            }

            for metric_name, value in metrics.items():
                if _finite_number(value):
                    row[metric_name] = float(value)

            rows.append(row)

    if not rows:
        return None

    fixed_fields = [
        "source", "mask_type", "loss_rate", "gap_ms", "output", "summary"
    ]

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["output"])].append(row)

    mean_rows = []
    for (source, output_name), group in grouped.items():
        mean_row = {
            "source": source,
            "mask_type": mask_type,
            "loss_rate": "",
            "gap_ms": "",
            "output": output_name,
            "summary": (
                "mean_over_loss_rates"
                if mask_type == "gilbert"
                else "mean_over_gap_durations"
                if mask_type == "single_gap"
                else "single_evaluation"
            ),
        }

        metric_names = {
            key for row in group for key in row if key not in fixed_fields
        }
        for metric_name in metric_names:
            values = [
                float(row[metric_name])
                for row in group
                if _finite_number(row.get(metric_name))
            ]
            if values:
                mean_row[metric_name] = sum(values) / len(values)

        mean_rows.append(mean_row)

    all_rows = rows + mean_rows
    metric_fields = sorted({
        key for row in all_rows for key in row if key not in fixed_fields
    })

    os.makedirs(model_log_dir, exist_ok=True)
    csv_path = os.path.join(model_log_dir, f"combined_{mask_type}_results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fixed_fields + metric_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(all_rows)

    return csv_path
