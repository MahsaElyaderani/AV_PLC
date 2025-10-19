# AV_PLC

AV_PLC is an audio-visual speech packet loss concealment (PLC) framework that trains a neural model to reconstruct speech by leveraging both audio and visual inputs (e.g., lip movements). 

## Project Structure

```

.
├── datasets/                # Grid, LRS2, VoxCeleb2
├── save_features.py         # Parallel video feature extraction
├── audio_encoder.py         # audio encoder model definition
├── video_encoder.py         # video encoder model definition
├── multimodal_decoder.py    # audio_video decoder model definition
├── trainer.py               # Training and evaluation logic
├── av_dataloader.py         # dataset and dataloader for AV inputs
├── main.py                  # Main training/feature extraction script

````

---

### 1. Install Dependencies

```bash
conda env create -f speech_environment.yml
````

### 2. Prepare Dataset

Videos of each dataset are under the `datasets/` directory with the following structure:

```
datasets/
└── grid/
    ├── train/
    ├── val/
    └── test/
```

---

## Feature Extraction

To extract audio-visual features from the dataset:

```bash
python main.py --save_features --dataset grid
```

Features will be saved as `.h5` files under `datasets/grid/`.

---

## Training the Model

To train the AV_PLC model:

```bash
python main.py --dataset grid --batch_size 4 --epochs 200
```

Optional arguments:

* `--pesq`: Enable PESQ loss .
* `--log_dir`: Set custom log directory.
* `--checkpoint_dir`: Set checkpoint save directory.
* `--plc_rates`: Specify PLC rates for inference to evaluate on (default: 20 30 40 50 60 rand).

Example:

```bash
python main.py --dataset grid --pesq --plc_rates 20 40 rand
```

---

## Evaluation

After training, the model is automatically evaluated on the specified PLC loss rates.

Evaluation logs and final test losses are saved in the log directory.

