# NeuroNet-AD

NeuroNet-AD is a multi-view deep learning pipeline for Alzheimer's disease classification from MRI data.  
It uses 3 views (axial, sagittal, coronal), feature extraction with MobileNetV3, attention/CBAM modules, and 3D convolutional fusion for final classification.

## Features

- Converts `.nii` MRI volumes into 2D multi-view slice datasets
- Custom PyTorch dataset loader for aligned axial/coronal/sagittal samples
- Multi-branch model with:
  - 3x `MobileNetV3-Large` backbones (one per view)
  - Residual CBAM (channel + spatial attention)
  - Multi-level attention encoder
  - 3D convolutional fusion head
- Train and evaluate scripts with command-line arguments
- Checkpoint saving for best validation F1

## Project Structure

```text
.
|-- NeuroNet-AD Model.ipynb          # Original notebook
|-- NeuroNet-AD Model.py             # Original exported script
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- scripts
|   |-- preprocess.py                # NIfTI -> PNG preprocessing
|   |-- train.py                     # Training entrypoint
|   `-- evaluate.py                  # Evaluation entrypoint
`-- src
    `-- neuronet_ad
        |-- __init__.py
        |-- data_preprocessing.py    # MRI slice extraction
        |-- dataset.py               # AlzheimerDataset + transforms
        |-- model.py                 # Model architecture
        |-- train.py                 # Training logic
        `-- evaluate.py              # Evaluation logic
```

## Data Format

### 1) Raw NIfTI input (before preprocessing)

Expected source format:

```text
OASIS_2/
|-- Converted/
|   |-- sample1.nii
|   `-- ...
|-- Demented/
`-- Nondemented/
```

### 2) Processed multi-view dataset (after preprocessing)

Expected training/evaluation format:

```text
Output/
|-- Converted/
|   |-- axial/
|   |-- coronal/
|   `-- sagittal/
|-- Demented/
`-- Nondemented/
```

For each class, filenames are aligned across the three view folders (same filename for the same sample index).

## Setup

### Option A: `pip` with `requirements.txt`

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Option B: Editable package install (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Usage

### 1) Preprocess NIfTI to slices

```bash
python scripts/preprocess.py --input-dir OASIS_2 --output-dir Output --classes Converted Demented Nondemented
```

### 2) Train model

```bash
python scripts/train.py --data-dir Output --epochs 20 --batch-size 8 --lr 1e-3 --train-split 0.8 --save-dir checkpoints
```

Optional flag:
- `--no-pretrained`: disables pretrained MobileNetV3 weights

### 3) Evaluate model

```bash
python scripts/evaluate.py --data-dir Output --checkpoint checkpoints/best_model.pt --batch-size 8
```

## Model Overview

1. Each view (`axial`, `sagittal`, `coronal`) is passed through a dedicated `MobileNetV3-Large` feature extractor.
2. Feature maps are refined using `ResidualCBAM`:
   - Channel attention (`ChannelAttention`)
   - Spatial attention (`SpatialAttention`)
   - Residual skip connection
3. Refined maps are processed by `MultiLevelAttentionNet` to capture local-to-global context.
4. The three view feature maps are concatenated.
5. A `1x1` convolution adjusts channels.
6. 3D convolutions fuse cross-view information.
7. Adaptive pooling + fully connected layer produce class logits.

## Training Details

- Loss: `CrossEntropyLoss`
- Optimizer: `Adam`
- Metrics:
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1 score (weighted)
- Best model checkpoint is selected by **validation F1**

## Important Notes

- This code preserves your original architecture and logic but organizes it into reusable modules.
- `RegionSimamModule` is still a placeholder block (as in your original code).
- Validation split currently uses random split from the provided dataset directory.
- For production research workflow, consider:
  - patient-level train/val/test split (avoid subject leakage)
  - experiment tracking (e.g., TensorBoard/W&B)
  - augmentation and class balancing
  - deterministic seeds and reproducibility controls

## Upload to GitHub

Run these commands from the project root:

```bash
git init
git add .
git commit -m "Refactor notebook into modular codebase with documentation"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If the GitHub repo already exists with commits, pull first before pushing:


```bash
git pull --rebase origin main
git push -u origin main
```

For any query, please contact at usaeed534@gmail.com
