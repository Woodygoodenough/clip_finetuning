# CLIP Fashion Fine-Tuning Toolkit

This repository packages the assets we use to fine-tune CLIP‑style vision–language models on a fashion catalogue. The codebase assumes an interactive workflow (Google Colab or a local notebook). The scripts can be imported and executed from cells; there is no standalone CLI entry point.

---

## Contents

| Path | Purpose |
| --- | --- |
| `clip_prep.py` | Build base JSON manifests (`clip_dataset_{split}.json`) from the Fashion‑Gen metadata pickles. |
| `caption_augmentation.py` / `caption_augmentation_multi.py` | Create augmented caption JSONL files. |
| `test.py` | Convert JSON/JSONL plus the base WebDataset shards into augmented shards (one shard per 1 000 samples). |
| `clipFineTuner.py` | Fine-tuning loop (import into a notebook, configure `ProjectConfig`, and call `training_prep(config).train()`). |
| `evaluate_clip.py` | Zero-shot / validation evaluation helper used by both the trainer and ad-hoc scripts. |
| `evaluations_basic/`, `evaluations_aug/`, `evaluations_resolution/` | Saved metrics from recent runs (baseline, augmented, and resolution experiments). |

---

## Data Assets

All canonical datasets live inside this repository so they can be mounted directly in Colab or opened locally:

- **Raw shards** (6 912‑sample evaluation slices and 260K training samples) in `webdataset_shards/`.
- **Augmented shards** (augmented captions applied to every sample) in `webdataset_shards_aug/`.
- **JSON manifests** and **JSONL augmentations** that back those shards (`clip_dataset_*.json`, `clip_dataset_*_aug*.jsonl`).

If you regenerate any artefact (e.g., new augmentations), keep shard indices aligned: one shard per 1 000 records, with filenames `clip_dataset_train[_aug].000000.tar`, etc.

---

## Environment

The repository targets **Python 3.10+**. Create a fresh environment and install only the packages listed below (the exported Conda environment is intentionally *not* checked in).

Minimal dependencies — see `requirements.txt`:

```
torch>=2.0.0
open-clip-torch>=2.20.0
webdataset>=1.0.0
pillow>=9.0.0
pandas>=1.5.0
matplotlib>=3.5.0
seaborn>=0.12.0
tqdm>=4.60.0
```

Suggested setup:

```bash
conda create -n clip-fashion python=3.10 -y
conda activate clip-fashion
pip install -r requirements.txt
```

> **GPU support** is optional for data prep, but strongly recommended for training and evaluation. Colab notebooks typically provision a T4 or A100; on local machines install the matching CUDA build of PyTorch.

---

## Typical Notebook Workflow (Colab / Jupyter)

1. **Mount or clone the repo** so the data directories (`webdataset_shards*`) are accessible.

2. **Inspect or regenerate manifests** (optional):
   ```python
   from clip_prep import create_clip_dataset
   create_clip_dataset(split="valid", output_file="clip_dataset_valid.json")
   ```

3. **Augment and convert to shards** (if you have new captions):
   ```python
   !python test.py --train-json clip_dataset_train.json \
                   --train-aug-jsonl clip_dataset_train_aug_full.jsonl \
                   --valid-json clip_dataset_valid.json \
                   --valid-aug-jsonl clip_dataset_valid_aug_0_70k.jsonl \
                   --input-root webdataset_shards \
                   --output-root webdataset_shards_aug
   ```

   > This script mirrors the existing shard layout; rerun it whenever you add more augmented captions.

4. **Configure and launch a run**:
   ```python
   from pathlib import Path
   from clipFineTuner import training_prep
   from config import ProjectConfig, ClipModelOptions, LossFunctionOptions, TrainingConfig

   config = ProjectConfig(
       clip_model=ClipModelOptions.ViT_SigLIP_2_16,
       loss_function=LossFunctionOptions.SIGLIP,
       on_colab=True,  # or False if you are local
       dataset_root=Path("webdataset_shards_aug"),
       training=TrainingConfig(
           batch_size=64,
           num_workers=2,
           learning_rate=1e-5,
           checkpoint_interval=500,
           max_steps=None,
       ),
       train_shards_pattern="clip_dataset_train_aug.{000000..000260}.tar",
       valid_shards_pattern="clip_dataset_valid_aug.{000000..000006}.tar",
       shardshuffle=False,
   )

   finetuner = training_prep(config)
   finetuner.train()
   ```

   Training and periodic evaluation statistics will be written to `evaluations/`, and the best checkpoints to `checkpoints/`.

5. **Analyse metrics**:
   - `evaluations_basic/` contains baseline metrics.
   - `evaluations_aug/` stores augmented-run metrics (`aug_evaluation_metrics.csv` summarises them).
   - Use a notebook (e.g., `test_eval.ipynb`) to plot recall curves or compare zero-shot vs fine-tuned performance.

---

## Running Zero-Shot Evaluations

Use `evaluate_clip.CLIPEvaluator` to score a pretrained checkpoint without training:

```python
from pathlib import Path
from config import ProjectConfig, ClipModelOptions, LossFunctionOptions, TrainingConfig
from openClipManagement import OpenClipManagment
from dBManagement import ClipDataset
from evaluate_clip import CLIPEvaluator, print_results

config = ProjectConfig(
    clip_model=ClipModelOptions.ViT_SigLIP_2_16,
    loss_function=LossFunctionOptions.SIGLIP,
    on_colab=True,
    dataset_root=Path("webdataset_shards_aug"),
    training=TrainingConfig(batch_size=128, num_workers=0),
    shardshuffle=False,
    valid_eval_shards_pattern="clip_dataset_valid_aug.{000000..000006}.tar",
)

clip_manager = OpenClipManagment(config=config)
dataset = ClipDataset(
    config=config,
    dataset_pattern=config.dataset_dir / config.valid_eval_shards_pattern,
    shardshuffle=False,
)

evaluator = CLIPEvaluator(config=config, clip_manager=clip_manager, evaluation_dataset=dataset)
print_results(evaluator.evaluate_with_loader())
```

---

## Notes & Conventions

- All scripts assume shard names follow the `{split}.{index:06d}.tar` convention (one shard per 1 000 samples).
- Augmented shards and metrics live alongside baseline artefacts; keep them in sync if you regenerate data.
- Because the workflow is notebook-driven, logging is routed to stdout; save key results manually if you need persistent reports.
- When running locally, set `shardshuffle=False` unless you have enough disk bandwidth to reshuffle on the fly.

Feel free to adapt the notebooks for your own experiments, but keep the dependency list minimal so the project remains easy to reproduce on Colab or a clean virtual environment.***