"""Cross-model evaluation CSV writer.

The evaluator is intentionally independent of model architecture. Each model's
existing trainer computes metrics using ``shared.metrics`` and then records two
rows per masking condition:

* ``masked_input``: common baseline against ground truth;
* the model's reconstructed mel-spectrogram against ground truth.

One CSV is maintained per dataset. Re-running a model/loss-rate combination
updates that row instead of creating duplicates. Detail rows are followed by
unweighted means over all evaluated loss rates for each model, as requested.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from .runtime_config import RESULTS_ROOT, normalize_dataset_name

DEFAULT_METRICS = (
    "mse", "mae", "psnr", "ssim",
    "pesq", "stoi", "estoi", "plcmos",
    "wer", "cer", "wer_vsr", "cer_vsr",
)

_IDENTIFIER_FIELDS = (
    "dataset", "row_type", "project", "model", "architecture",
    "loss_rate", "loss_rate_percent", "loss_gap_ms", "sample_count",
)


def _finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _normalize_rate(value: object) -> float:
    rate = float(value)
    if rate < 0:
        raise ValueError("loss_rate must be non-negative")
    return rate / 100.0 if rate > 1.0 else rate


def _gap_key(value: object) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):g}"


class DatasetEvaluation:
    """Incrementally maintain one cross-model CSV for one dataset."""

    def __init__(
        self,
        dataset: str,
        output_dir: str | os.PathLike[str] | None = None,
        metrics: Iterable[str] = DEFAULT_METRICS,
        filename: str | None = None,
        load_existing: bool = True,
    ) -> None:
        self.dataset = normalize_dataset_name(dataset)
        self.metrics = tuple(dict.fromkeys(str(m).lower() for m in metrics))
        self.output_dir = Path(output_dir) if output_dir else RESULTS_ROOT
        self.path = self.output_dir / (
            filename or f"{self.dataset}_all_models_metrics.csv"
        )
        self.rows: list[dict[str, object]] = []
        if load_existing and self.path.is_file():
            self._load_existing()

    @property
    def fieldnames(self) -> list[str]:
        return [*_IDENTIFIER_FIELDS, *self.metrics]

    def _load_existing(self) -> None:
        with self.path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("row_type") != "loss_rate":
                    continue
                clean: dict[str, object] = dict(row)
                clean["loss_rate"] = _normalize_rate(row["loss_rate"])
                clean["loss_rate_percent"] = 100.0 * float(clean["loss_rate"])
                clean["loss_gap_ms"] = _gap_key(row.get("loss_gap_ms"))
                try:
                    clean["sample_count"] = int(float(row.get("sample_count") or 0))
                except ValueError:
                    clean["sample_count"] = 0
                for metric in self.metrics:
                    clean[metric] = _finite_float(row.get(metric))
                self.rows.append(clean)

    def _clean_metrics(self, values: Mapping[str, object]) -> Dict[str, float]:
        lower = {str(k).lower(): v for k, v in values.items()}
        return {metric: _finite_float(lower.get(metric)) for metric in self.metrics}

    @staticmethod
    def _row_key(row: Mapping[str, object]) -> tuple[object, ...]:
        return (
            row.get("dataset"), row.get("project"), row.get("model"),
            float(row.get("loss_rate", 0.0)), _gap_key(row.get("loss_gap_ms")),
        )

    def add_result(
        self,
        model_name: str,
        loss_rate: object,
        metric_values: Mapping[str, object],
        *,
        project: str = "",
        architecture: str = "",
        loss_gap_ms: float | None = None,
        sample_count: int | None = None,
        save: bool = False,
    ) -> None:
        rate = _normalize_rate(loss_rate)
        row: dict[str, object] = {
            "dataset": self.dataset,
            "row_type": "loss_rate",
            "project": project,
            "model": model_name,
            "architecture": architecture,
            "loss_rate": rate,
            "loss_rate_percent": 100.0 * rate,
            "loss_gap_ms": _gap_key(loss_gap_ms),
            "sample_count": int(sample_count or 0),
        }
        row.update(self._clean_metrics(metric_values))

        key = self._row_key(row)
        self.rows = [old for old in self.rows if self._row_key(old) != key]
        self.rows.append(row)
        if save:
            self.save_csv()

    def add_masked_baseline(
        self,
        loss_rate: object,
        metric_values: Mapping[str, object],
        *,
        loss_gap_ms: float | None = None,
        sample_count: int | None = None,
        save: bool = False,
    ) -> None:
        self.add_result(
            model_name="masked_input",
            loss_rate=loss_rate,
            metric_values=metric_values,
            project="shared",
            architecture="baseline",
            loss_gap_ms=loss_gap_ms,
            sample_count=sample_count,
            save=save,
        )

    def add_pair(
        self,
        *,
        project: str,
        model_name: str,
        architecture: str,
        loss_rate: object,
        model_metrics: Mapping[str, object],
        masked_metrics: Mapping[str, object],
        loss_gap_ms: float | None = None,
        sample_count: int | None = None,
        save: bool = True,
    ) -> None:
        """Record baseline and model metrics for exactly the same mask condition."""
        self.add_masked_baseline(
            loss_rate,
            masked_metrics,
            loss_gap_ms=loss_gap_ms,
            sample_count=sample_count,
        )
        self.add_result(
            model_name,
            loss_rate,
            model_metrics,
            project=project,
            architecture=architecture,
            loss_gap_ms=loss_gap_ms,
            sample_count=sample_count,
        )
        if save:
            self.save_csv()

    def _mean_rows(self) -> list[dict[str, object]]:
        grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
        for row in self.rows:
            group = (
                str(row.get("project", "")),
                str(row.get("model", "")),
                str(row.get("architecture", "")),
            )
            grouped.setdefault(group, []).append(row)

        means: list[dict[str, object]] = []
        for (project, model, architecture), rows in sorted(grouped.items()):
            mean_row: dict[str, object] = {
                "dataset": self.dataset,
                "row_type": "mean_all_loss_rates",
                "project": project,
                "model": model,
                "architecture": architecture,
                "loss_rate": "mean",
                "loss_rate_percent": "mean",
                "loss_gap_ms": "",
                "sample_count": sum(int(r.get("sample_count") or 0) for r in rows),
            }
            # Deliberately unweighted: every evaluated loss rate contributes once.
            for metric in self.metrics:
                values = np.asarray([_finite_float(r.get(metric)) for r in rows], dtype=float)
                mean_row[metric] = (
                    float(np.nanmean(values))
                    if values.size and not np.all(np.isnan(values))
                    else float("nan")
                )
            means.append(mean_row)
        return means

    def save_csv(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        detail_rows = sorted(
            self.rows,
            key=lambda row: (
                float(row["loss_rate"]),
                _gap_key(row.get("loss_gap_ms")),
                str(row.get("model", "")),
            ),
        )

        # Atomic replacement avoids leaving a partial CSV after interruption.
        with NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=self.output_dir,
            prefix=f".{self.dataset}_", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(detail_rows)
            writer.writerows(self._mean_rows())
        temp_path.replace(self.path)
        return self.path


def aggregate_sample_metrics(
    sample_metrics: Iterable[Mapping[str, object]],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> Dict[str, float]:
    buckets: MutableMapping[str, list[float]] = {m: [] for m in metrics}
    for sample in sample_metrics:
        lower = {str(k).lower(): v for k, v in sample.items()}
        for metric in metrics:
            buckets[metric].append(_finite_float(lower.get(metric)))
    return {
        metric: (
            float(np.nanmean(values))
            if values and not np.all(np.isnan(np.asarray(values, dtype=float)))
            else float("nan")
        )
        for metric, values in buckets.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or initialize a dataset evaluation CSV")
    parser.add_argument("--dataset", required=True, choices=("grid", "lrs2", "voxceleb2"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    evaluator = DatasetEvaluation(args.dataset, output_dir=args.output_dir)
    path = evaluator.save_csv()
    print(path)


if __name__ == "__main__":
    main()
