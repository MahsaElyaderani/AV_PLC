"""Portable runtime paths for the multi-project AV reconstruction repository.

Default repository layout::

    parent/
    ├── datasets/                 # HDF5 datasets
    └── models/
        ├── evaluations/
        ├── AV_LSTM/logs/          # project-local logs
        ├── AV_LSTM/checkpoints/
        ├── AV_S2S/logs/
        ├── AV_S2S/checkpoints/
        ├── AV_Transformer/logs/
        ├── AV_Transformer/checkpoints/
        └── AV_PLC/logs/ and checkpoints/

All defaults can be overridden with environment variables. Existing model names
are never changed; these helpers only return directories.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Mapping

MODELS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODELS_ROOT.parent

# The datasets directory is next to the models directory, as requested.
DATA_ROOT = Path(os.getenv("AV_DATA_ROOT", REPOSITORY_ROOT / "datasets")).expanduser().resolve()

# Cross-model CSV files are kept outside every model's TensorBoard/log folder.
RESULTS_ROOT = Path(
    os.getenv("AV_RESULTS_ROOT", MODELS_ROOT / "evaluations" / "results")
).expanduser().resolve()

DEVICE = os.getenv("AV_DEVICE", "cuda")
NUM_WORKERS = int(os.getenv("AV_NUM_WORKERS", "4"))
SEED = int(os.getenv("AV_SEED", "42"))

PROJECT_FOLDERS = ("AV_LSTM", "AV_S2S", "AV_Transformer", "AV_PLC")


def set_global_seed(seed: int = SEED, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch once at process startup."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

_DATASET_ALIASES = {
    "grid": "grid",
    "lrs2": "lrs2",
    "voxceleb2": "voxceleb2",
    "vox2": "voxceleb2",
    "vox2_short": "voxceleb2",
}

_DATASET_DIRS: Mapping[str, str] = {
    "grid": "grid",
    "lrs2": "lrs2",
    "voxceleb2": "vox2_short",
}

_DATASET_PATTERNS: Mapping[str, Mapping[str, str]] = {
    "grid": {
        "train": "grid_train_features_chunk*.h5",
        "val": "grid_val_features_chunk*.h5",
        "test": "grid_test_features_chunk*.h5",
    },
    "lrs2": {
        "train": "lrs2_*train_features_chunk*.h5",
        "val": "lrs2_val_features_chunk*.h5",
        "test": "lrs2_test_features_chunk*.h5",
    },
    "voxceleb2": {
        "train": "vox2_short_dev_features_chunk*.h5",
        "val": "vox2_short_val_features_chunk*.h5",
        "test": "vox2_short_test_features_chunk*.h5",
    },
}


def normalize_dataset_name(dataset_name: str) -> str:
    key = str(dataset_name).strip().lower()
    try:
        return _DATASET_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(_DATASET_ALIASES))
        raise ValueError(f"Unknown dataset {dataset_name!r}. Valid names: {valid}") from exc


def dataset_dir(dataset_name: str) -> Path:
    canonical = normalize_dataset_name(dataset_name)
    return DATA_ROOT / _DATASET_DIRS[canonical]


def dataset_patterns(dataset_name: str) -> Dict[str, str]:
    """Return train/val/test HDF5 glob patterns expected by all dataloaders."""
    canonical = normalize_dataset_name(dataset_name)
    base = dataset_dir(canonical)
    return {
        split: str(base / pattern)
        for split, pattern in _DATASET_PATTERNS[canonical].items()
    }


def project_dir(project_folder: str) -> Path:
    if project_folder not in PROJECT_FOLDERS:
        raise ValueError(
            f"Unknown project folder {project_folder!r}; expected one of {PROJECT_FOLDERS}"
        )
    return MODELS_ROOT / project_folder


def project_log_dir(project_folder: str) -> str:
    """Return ``<project>/logs`` and create it when missing."""
    path = project_dir(project_folder) / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def project_checkpoint_dir(project_folder: str) -> str:
    """Return ``<project>/checkpoints`` and create it when missing."""
    path = project_dir(project_folder) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def model_log_dir(project_folder: str, model_name: str) -> Path:
    """Return the existing model-name-specific log directory without renaming it."""
    path = Path(project_log_dir(project_folder)) / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_checkpoint_dir(project_folder: str, model_name: str) -> Path:
    """Return the existing model-name-specific checkpoint directory."""
    path = Path(project_checkpoint_dir(project_folder)) / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_vocoder_path() -> str | None:
    """Resolve an optional vocoder checkpoint.

    ``AV_VOCODER_PATH`` has priority. Otherwise a few repository-local legacy
    locations are checked. Returning ``None`` intentionally keeps Griffin-Lim
    evaluation active when no vocoder is installed.
    """
    configured = os.getenv("AV_VOCODER_PATH")
    if configured:
        return str(Path(configured).expanduser().resolve())

    candidates = (
        MODELS_ROOT / "AV_PLC" / "hifigan" / "checkpoints" / "model-best.pt",
        MODELS_ROOT / "hifigan" / "checkpoints" / "model-best.pt",
        REPOSITORY_ROOT / "hifigan" / "checkpoints" / "model-best.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


VOCODER_PATH = find_vocoder_path()


def ensure_runtime_dirs() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for project in PROJECT_FOLDERS:
        Path(project_log_dir(project)).mkdir(parents=True, exist_ok=True)
        Path(project_checkpoint_dir(project)).mkdir(parents=True, exist_ok=True)


def describe_runtime() -> Dict[str, object]:
    return {
        "models_root": str(MODELS_ROOT),
        "data_root": str(DATA_ROOT),
        "results_root": str(RESULTS_ROOT),
        "vocoder_path": VOCODER_PATH,
        "device": DEVICE,
        "num_workers": NUM_WORKERS,
        "seed": SEED,
        "projects": {
            project: {
                "logs": project_log_dir(project),
                "checkpoints": project_checkpoint_dir(project),
            }
            for project in PROJECT_FOLDERS
        },
    }
