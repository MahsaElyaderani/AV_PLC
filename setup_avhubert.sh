#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# AV-HuBERT setup (macOS / conda) — the combo that worked
# ------------------------------------------------------------
# Usage:
#   bash setup_avhubert.sh [--env avhubert_env] [--python 3.9] [--ckpt /full/path/to/avhubert_base_vox_iter5.pt]
#
# After it finishes:
#   conda activate <env>
#   # If you didn’t pass --ckpt, export it now:
#   # export AVHUBERT_CKPT="/absolute/path/to/your/checkpoint.pt"
#   python your_feature_script.py
# ------------------------------------------------------------

ENV_NAME="avhubert_env"
PY_VER="3.9"
CKPT_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)    ENV_NAME="$2"; shift 2 ;;
    --python) PY_VER="$2";   shift 2 ;;
    --ckpt)   CKPT_PATH="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  case
done

# --- find and enable conda in this shell ---
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install miniforge/miniconda first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- create & activate env ---
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  echo "Conda env '${ENV_NAME}' already exists."
else
  conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
fi
conda activate "${ENV_NAME}"

# --- ensure modern tooling, but keep pip <24.1 (OmegaConf 2.0.x metadata quirk) ---
python -m pip install --upgrade "pip<24.1" setuptools wheel

# --- PyTorch (CPU/MPS wheels from PyPI are fine on macOS) ---
# If you want CUDA (not on Mac), you'd pick the CUDA index instead.
pip install torch torchvision torchaudio

# --- ffmpeg (needed for audio/video utilities) ---
if command -v brew >/dev/null 2>&1; then
  brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
else
  conda install -y -c conda-forge ffmpeg
fi

# --- deps that avoid the classic conflicts with fairseq ---
pip install "omegaconf==2.0.6" "hydra-core==1.0.7" sentencepiece

# --- fairseq (pin 0.12.2 to avoid 0.12.1 sdist packaging bug) ---
if ! pip install --no-cache-dir "fairseq==0.12.2"; then
  echo "Binary not available; falling back to GitHub tag v0.12.2..."
  pip install --no-cache-dir "git+https://github.com/facebookresearch/fairseq@v0.12.2"
fi

# (Occasionally required; uncomment only if fairseq later complains)
# pip install "antlr4-python3-runtime==4.8"

# --- optionally register the checkpoint path in this env ---
if [[ -n "${CKPT_PATH}" ]]; then
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "WARNING: --ckpt path not found: ${CKPT_PATH}"
  fi
  conda env config vars set "AVHUBERT_CKPT=${CKPT_PATH}"
  # refresh env vars
  conda deactivate && conda activate "${ENV_NAME}"
fi

# --- quick sanity check ---
python - <<'PY'
import os, sys
print("python:", sys.version.split()[0])
try:
    import omegaconf, hydra, fairseq
    print("omegaconf:", omegaconf.__version__)
    print("hydra-core:", hydra.__version__)
    print("fairseq:", fairseq.__version__)
except Exception as e:
    print("IMPORT ERROR:", e); raise

ckpt = os.environ.get("AVHUBERT_CKPT")
print("AVHUBERT_CKPT:", ckpt if ckpt else "(not set)")
PY

cat <<'MSG'

AV-HuBERT environment is ready.

HOW TO USE WITH YOUR FEATURE EXTRACTIONS
----------------------------------------
1) Activate the env:
   conda activate '"${ENV_NAME}"'

2) Make sure the checkpoint path is set (either you passed --ckpt to this script,
   or export it now):
   export AVHUBERT_CKPT="/absolute/path/to/avhubert_base_vox_iter5.pt"

3) Run your feature-extraction script that already integrates AV-HuBERT (the one we set up earlier):
   python your_feature_script.py

   - Your script will:
     • stream video with OpenCV (grayscale ROIs)
     • stream audio via ffmpeg
     • compute log-mel specs
     • (if AVHUBERT_CKPT is set) load AV-HuBERT and save visual feats as /video_i/avhubert_vis in HDF5.

4) PyCharm tip:
   In Run/Debug Config → Environment variables, add AVHUBERT_CKPT with the same absolute path.

If you want to make this permanent for the env:
   conda env config vars set AVHUBERT_CKPT="/absolute/path/to/avhubert_base_vox_iter5.pt"
   conda deactivate && conda activate '"${ENV_NAME}"'

