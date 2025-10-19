# Audio-Visual Packet Loss Concealment (AV-PLC)

This repository provides an end-to-end framework for **audio-visual speech inpainting** — reconstructing missing or degraded speech segments using both **audio** and **visual ** cues.  
It supports feature extraction, multimodal model training, and evaluation across multiple datasets (e.g., **GRID**, **LRS2**, **VoxCeleb2**).

---

## Overview

The main entry point is `main.py`, which handles:
- **Feature extraction** for each dataset split  
- **Model initialization** with dataset-specific parameters  
- **Training** with optional perceptual and spectral losses  
- **Evaluation** across multiple packet-loss rates  

---

## Repository Structure

```

.
├── main.py                  # Entry point for training / evaluation
├── save_features.py         # Parallelized video feature extraction
├── audio_encoder.py         # Audio encoder network
├── video_encoder.py         # Video encoder network
├── multimodal_decoder.py    # AV_PLC model definition (audio-visual decoder)
├── trainer.py               # Training, validation, evaluation logic
├── av_dataloader.py         # Audio-visual dataset loader and batching
├── datasets/                # Dataset root directory (Grid, LRS2, VoxCeleb2, etc.)

````

---

## 1. Installation

Create the environment from the YAML file:
```bash
conda env create -f speech_environment.yml
conda activate speech_environment
````

---

## 2. Feature Extraction

Before training, extract audio-visual features for your dataset:

```bash
python main.py --save_features --dataset grid
```

This will:

* Process all videos under `datasets/grid/{train,val,test}/`
* Save `.h5` feature files in `datasets/grid/`

---

## 3. Training the Model

Train the **Audio-Visual PLC** model with default settings:

```bash
python main.py --datasets grid --batch-size 8 --epochs 100
```

### Optional arguments (loss toggles)

| Argument    | Type | Default | Description                                                        |
| ----------- | ---- | ------- | ------------------------------------------------------------------ |
| `--pesq`    | bool | `true`  | Enable PESQ perceptual loss                                        |
| `--stoi`    | bool | `false` | Enable STOI intelligibility loss                                   |
| `--asr`     | bool | `true`  | Enable ASR perceptual loss                                         |
| `--sc-flag` | bool | `false` | Enable spectral-consistency loss                                   |
| `--l2s`     | bool | `true`  | Enable lip-to-speech (AV fusion) mode; if `false`, uses audio-only |

### Other training options

| Argument            | Default | Description                    |
| ------------------- |---------| ------------------------------ |
| `--batch-size`      | 8       | Mini-batch size                |
| `--epochs`          | 100     | Number of epochs               |
| `--learning-rate`   | 1e-4    | Learning rate for optimizer    |
| `--mixed-precision` | true    | Use automatic mixed precision  |
| `--use-bf16`        | true    | Prefer bfloat16 when supported |

**Example:**

```bash
python main.py --datasets grid --batch-size 8 --epochs 100 --pesq true --asr true --plc-loss-rates 20 40 60
```

---

## 4. Checkpoints and Logging

* **Checkpoints:** saved automatically in `checkpoints/`
* **Logs:** stored under `logs/` (training, validation, and evaluation metrics)
* **Model naming:** built dynamically based on enabled loss flags and dataset (e.g. `av_plc_a0.05_v0.1_pesq_0.01_asr_0.1(grid)`)

To resume from the latest checkpoint:

```bash
python main.py --datasets grid
```

The trainer will automatically load the latest model.

---

## 5. Evaluation

After training, the model evaluates automatically on specified **packet-loss rates**:

```bash
python main.py --datasets grid --plc-loss-rates 20 30 40 50 60
```

Results and evaluation metrics (e.g., PESQ/STOI) are logged under the corresponding log directory.

---

<!--## Citation

If you use this repository, please cite:

```text
@misc{av_plc_2025,
  title        = {Audio-Visual Packet Loss Concealment (AV-PLC)},
  author       = {Your Name},
  year         = {2025},
  note         = {GitHub repository},
  howpublished = {\url{https://github.com/<your_username>/<repo_name>}}
}-->
```

```

---


