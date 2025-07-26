# AV_ReVoice

AV_ReVoice is an Audio-Visual speech reconstruction framework that trains a neural model to improve packet loss concealment (PLC) in speech by leveraging both audio and visual inputs (e.g., lip movements). It supports GRID Corpus dataset for now.

## Project Structure

```

.
├── datasets/                # Contains GRID examples
├── checkpoints/             # Trained model checkpoints
├── logs/                    # Training and evaluation logs
├── save_features.py         # Parallel video feature extraction
├── model.py                 # AV\_ReVoice model definition
├── trainer.py               # Training and evaluation logic
├── av_l_dataloader.py       # Custom dataloader for AV inputs
├── main.py                  # Main training/feature extraction script

````

---

### 1. Install Dependencies

```bash
conda env create -f speech_environment.yml
````

### 2. Prepare Dataset

I placed some video example from the GRID dataset under the `datasets/` directory with the following structure:

```
datasets/
└── grid/
    ├── train/
    ├── val/
    └── test/
```

Each folder contains speaker directories with `.mpg` video files.

---

## Feature Extraction

To extract audio-visual features from the dataset:

```bash
python main.py --save_features --dataset grid
```

Features will be saved as `.h5` files under `datasets/grid/`.

---

## Training the Model

To train the AV_ReVoice model:

```bash
python main.py --dataset grid --batch_size 4 --epochs 200
```

Optional arguments:

* `--pesq`: Enable PESQ loss (option is not available, set False for now).
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
