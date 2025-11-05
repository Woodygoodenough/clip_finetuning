# CLIP Fine-tuning on Fashion Dataset

This repository contains a full pipeline to fine‑tune and evaluate OpenCLIP models on a Fashion‑Gen–derived image–caption dataset using WebDataset shards.

## Current Status

- Primary dataset format is WebDataset shards (`*.tar`) accessed via glob patterns in `settings.py`.
- Shard creation tool (`create_webdataset.py`) is production‑ready; use it to build shards from JSONs.
- JSON mappings (`clip_dataset_train.json`, `clip_dataset_valid.json`) are included as sources for shard creation.
- Training (`clipFineTuner.py`) uses an efficient WebDataset pipeline with mixed precision.
- Evaluation (`evaluate_clip.py`) computes Recall@5 and Recall@10 for both text‑to‑image and image‑to‑text.
- Configuration is centralized in `settings.py` (defaults are set for Google Colab; set `ON_COLAB=False` for local use).

## Repository Overview

- `clip_prep.py`: Build JSON mappings from raw Fashion‑Gen metadata/images.
- `create_webdataset.py`: Convert JSON to WebDataset tar shards.
- `dBManagement.py`: Thin wrapper around WebDataset with optimal loaders for CLIP.
- `openClipManagement.py`: Model creation, preprocessing, tokenization, and helpers.
- `clipFineTuner.py`: Training loop, checkpoints, mixed precision (AMP).
- `evaluate_clip.py`: Validation metrics (Recall@k) using the same loader pipeline.
- `settings.py`: All configuration (model, batch size, paths, shard patterns, checkpoints).

## Environment Setup

Option A — pip (recommended):

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Option B — conda:

```bash
conda env create -f environment.yml
conda activate clip-finetuning
```

Notes:
- Install a CUDA‑enabled PyTorch build if you have a GPU.
- On Windows, prefer smaller `NUM_WORKERS` (see `settings.py`).

## Data Preparation

Primary dataset representation is WebDataset shards (`*.tar`). Create shards from the included JSONs, or regenerate JSONs from raw Fashion‑Gen assets if needed.

### 1) Create WebDataset shards (primary)

```bash
# Validation shards
python create_webdataset.py \
  --json clip_dataset_valid.json \
  --output-dir webdataset_shards \
  --shard-size 1000

# Training shards
python create_webdataset.py \
  --json clip_dataset_train.json \
  --output-dir webdataset_shards \
  --shard-size 1000
```

The script prints how many shards were created and the canonical pattern, e.g.:
```
Shard pattern: clip_dataset_valid.{000000..000006}.tar
```
Use this pattern in `settings.py` or pass it via CLI (see below).

### 2) (Optional) Regenerate JSON mappings from raw assets

If you have the original Fashion‑Gen assets, you can recreate the JSON mapping files used to build shards.

`clip_prep.py` expects directories like `full_{split}_info_PAI/` (pickles) and `extracted_{split}_images/` (PNGs).

```bash
# Example invocation (see __main__ in clip_prep.py)
python clip_prep.py  # default in __main__ creates train JSON; edit as needed

# Or inside the script:
# create_clip_dataset("valid", "clip_dataset_valid.json")
# create_clip_dataset("train", "clip_dataset_train.json")
```

Each JSON entry includes:
- `image_path`: path to image
- `image_name`: file name
- `caption`: product description
- `category`: coarse category
- `product_id`: unique id

## Configuration

All knobs live in `settings.py`:
- `ON_COLAB`: set to `False` for local paths; `True` uses Google Drive paths.
- `BATCH_SIZE`, `NUM_WORKERS`: tune for your GPU/CPU (T4 defaults provided).
- `MODEL_NAME`, `MODEL_PRETRAINED`, `MODEL_CACHE_DIR`: OpenCLIP model/version and cache.
- `CHECKPOINT_DIR`, `EVAL_CHECKPOINT`, `CHECKPOINT_INTERVAL`.
- `MAX_STEPS`: limit training steps when experimenting locally.
- `TRAIN_DATASET_PATTERN`, `VALID_DATASET_PATTERN`, `TEST_DATASET_PATTERN`: shard patterns like `webdataset_shards/clip_dataset_valid.{000000..000006}.tar`.

Tip: After shard creation, update the `{000000..NNNNNN}` ranges to match your printed counts.

## Training

Make sure `settings.py` points to your training shards and checkpoint directory. Then:

```bash
python clipFineTuner.py
```

Behavior:
- Uses mixed precision (AMP) for speed/memory savings.
- Loads shards via an efficient WebDataset loader with batched tokenization.
- Logs progress every 10 steps; saves periodic checkpoints every `CHECKPOINT_INTERVAL`.
- Saves `final_checkpoint.pt` on completion in `CHECKPOINT_DIR`.

Common adjustments:
- Reduce `BATCH_SIZE` if you see CUDA OOM.
- Set `MAX_STEPS` (e.g., 500–2000) for quick iterations.

## Evaluation

Evaluate either the base pretrained model or a fine‑tuned checkpoint on validation shards.

Base model (no checkpoint):
```bash
python evaluate_clip.py
```

Fine‑tuned checkpoint (uses `settings.EVAL_CHECKPOINT`):
```bash
python evaluate_clip.py --use-checkpoint
```

Override dataset pattern explicitly:
```bash
python evaluate_clip.py \
  --dataset-path "webdataset_shards/clip_dataset_valid.{000000..000006}.tar"
```

Outputs:
- Pretty‑printed metrics in console.
- JSON written to `evaluation_results.json` (or `evaluation_results_checkpoint.json` when `--use-checkpoint`).

Metrics reported:
- Recall@5 and Recall@10 for text→image and image→text retrieval.

## Troubleshooting

- Paths on Colab vs. local: set `ON_COLAB` correctly and ensure shard paths exist.
- Many samples skipped during shard creation: check that `image_path` in JSON points to real files.
- Windows workers: try `NUM_WORKERS = 0–2` if you encounter DataLoader issues.
- CUDA OOM: lower `BATCH_SIZE`; optionally use a smaller model or gradient checkpointing (not enabled here).

## Notes on the Raw Dataset

- Original images are PNGs; shards store JPEGs (~95% quality) for efficiency.
- Provided JSONs cover both train (~260k) and valid (~32k) entries.
- WebDataset shards provide faster I/O and better shuffling at scale.

## Acknowledgments

- Built on top of `open-clip-torch` and `webdataset`.