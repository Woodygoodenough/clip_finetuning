# OpenCLIP Fine-Tuning for Multi-Modal Retrieval on FashionGen

This repository contains the implementation for fine-tuning OpenCLIP models on the Fashion-Gen dataset for image-to-text and text-to-image retrieval tasks. The project compares three CLIP-style architectures (ViT-B/32, ViT-B/16, and ViT-B/16-SigLIP2-256) and two contrastive loss functions (InfoNCE and BCE) under single-GPU constraints.


## Overview

This project fine-tunes CLIP-style dual-encoder models for fashion product retrieval, achieving ~30% relative improvement over zero-shot baselines. The codebase is designed for notebook-style work, suitable for use in Google Colab or local Jupyter notebooks, making it accessible for experimentation and reproduction.

### Key Features

- **Three Model Variants**: ViT-B/32, ViT-B/16, and ViT-B/16-SigLIP2-256
- **Two Loss Functions**: InfoNCE (contrastive) and BCE (SigLIP-style)
- **WebDataset Support**: Efficient streaming data loading with shuffling
- **Mixed Precision Training**: Optimized for T4 GPUs with automatic mixed precision
- **Evaluation Metrics**: Recall@5 and Recall@10 for both text-to-image and image-to-text retrieval
- **Caption Augmentation**: Support for augmented datasets with multiple captions per image

## Model Options

The codebase supports three OpenCLIP model variants:

- **`ViT_B_32`**: Baseline with 32×32 patches at 224×224 resolution (49 image tokens)
- **`ViT_B_16`**: Higher spatial resolution with 16×16 patches at 224×224 (196 image tokens)
- **`ViT_SigLIP_2_16`**: SigLIP2 variant with 16×16 patches at 256×256, smaller embedding dimension (D=256), and MAP pooling

## Loss Functions

Two contrastive objectives are implemented:

- **`CONTRASTIVE`** (InfoNCE): Standard softmax-based CLIP loss with batch-coupled normalization
- **`SIGLIP`**: Pairwise binary cross-entropy loss that decouples negatives, providing more stable training in small-batch settings

## Code Structure

- **`config.py`**: Configuration dataclasses for models, training, and project settings
- **`clipFineTuner.py`**: Main training loop with support for both loss functions and checkpointing
- **`evaluate_clip.py`**: Evaluation module computing Recall@5 and Recall@10 metrics
- **`openClipManagement.py`**: Wrapper for loading and managing OpenCLIP models
- **`dBManagement.py`**: WebDataset loading and data pipeline
- **`constants.py`**: Shared constants for dataset patterns and evaluation metrics

## Usage

The script is designed for notebook-style work, like in Colab or local notebook.

### Example Usage

```python
from pathlib import Path

from config import ProjectConfig, TrainingConfig, ClipModelOptions, LossFunctionOptions

from clipFineTuner import training_prep

aug_root = Path("/path/to/webdataset_shards_aug")

config = ProjectConfig(
    clip_model=ClipModelOptions.ViT_SigLIP_2_16,
    loss_function=LossFunctionOptions.SIGLIP,
    on_colab=True,
    dataset_root=aug_root,
    training=TrainingConfig(
        batch_size=64,
        num_workers=2,
        learning_rate=1e-5,
        checkpoint_interval=500,
        max_steps=None,
        epochs=5,
    ),
    train_shards_pattern="clip_dataset_train_aug.{000000..000260}.tar",
    valid_shards_pattern="clip_dataset_valid_aug.{000000..000006}.tar",
    train_eval_shards_pattern="clip_dataset_train_aug.{000000..000006}.tar",
    valid_eval_shards_pattern="clip_dataset_valid_aug.{000000..000006}.tar",
    shardshuffle=False,
)

finetuner = training_prep(config)

finetuner.train()
```

### Training Configuration

The `TrainingConfig` supports:
- `batch_size`: Batch size (default: 64, optimized for T4 GPU)
- `num_workers`: Data loading workers
- `learning_rate`: Initial learning rate (default: 1e-5)
- `checkpoint_interval`: Steps between checkpoints
- `epochs`: Number of training epochs
- `max_steps`: Optional step limit (overrides epochs if set)

### Evaluation

The training loop automatically evaluates on the validation set at checkpoint intervals, computing:
- **Recall@5** and **Recall@10** for text-to-image retrieval
- **Recall@5** and **Recall@10** for image-to-text retrieval

Results are saved to JSON files in the evaluation directory.

### Output Directories

The training process creates and uses several directories (relative to `base_path`):

- **`checkpoints/`**: Model checkpoints saved at specified intervals
- **`evaluations/`**: Evaluation results in JSON format (Recall@5 and Recall@10 metrics)
- **`openclip_cache/`**: Cached OpenCLIP model weights

### Training Logs

The `training_log/` directory contains evaluation results from our training sessions, organized in `.csv` files.

## Dataset

We release two versions of the Fashion-Gen dataset in WebDataset format:

**Original:**  
https://drive.google.com/drive/folders/1AKAqHYFPGE3X4uvt7Qe6vTaa8Vsd1eIH

**Augmented:**  
https://drive.google.com/drive/folders/1u0ky29PSAIh1Hu02VuA9ufy8fBDt0spn

The augmented dataset contains three captions per image (original + 2 GPT-generated paraphrases), enabling training with increased linguistic diversity while preserving all visual attributes.

## Results Summary

Fine-tuning on Fashion-Gen yields substantial improvements:
- **ViT-B/32 (InfoNCE)**: 31.36% relative improvement in Recall@10
- **ViT-B/16 (InfoNCE)**: 30.26% relative improvement
- **ViT-B/16-SigLIP2 (BCE)**: 28.77% relative improvement (best single model)
- **ViT-B/16-SigLIP2 (BCE, Augmented)**: 31.77% relative improvement (best overall)

The SigLIP2 model with BCE loss and caption augmentation achieves the highest performance, demonstrating the effectiveness of both architectural improvements and data augmentation strategies.
