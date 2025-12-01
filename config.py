# %%
"""
we conly use dataclasses for simplicity and ease of use.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from constants import (
    TRAIN_SHARDS_PATTERN,
    VALID_SHARDS_PATTERN,
    TRAIN_EVAL_SHARDS_PATTERN,
    VALID_EVAL_SHARDS_PATTERN,
)


def torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


@dataclass
class ClipModelRegistry:
    name: str
    pretrained: str


@dataclass
class LossFunctionRegistry:
    name: str


class ClipModelOptions(Enum):
    ViT_B_32 = ClipModelRegistry(name="ViT-B-32", pretrained="laion2b_s34b_b79k")
    ViT_B_16 = ClipModelRegistry(name="ViT-B-16", pretrained="laion2b_s34b_b88k")
    ViT_SigLIP_2_16 = ClipModelRegistry(name="ViT-B-16-SigLIP2-256", pretrained="webli")

    @property
    def name_(self) -> str:
        return self.value.name

    @property
    def pretrained_(self) -> str:
        return self.value.pretrained


class LossFunctionOptions(Enum):
    CONTRASTIVE = LossFunctionRegistry(name="contrastive")
    SIGLIP = LossFunctionRegistry(name="siglip")


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_workers: int = 2
    learning_rate: float = 1e-5
    checkpoint_interval: int = 500
    # normalize_embeddings_in_siglip: bool = False
    max_steps: int | None = None
    epochs: int = 3


@dataclass(frozen=True)
class ProjectConfig:
    clip_model: ClipModelOptions | None = None
    loss_function: LossFunctionOptions | None = None
    on_colab: bool | None = None
    device: str = "cuda" if torch_cuda_available() else "cpu"
    should_load_checkpoint: bool = False
    checkpoint_path: Path | None = None
    dataset_root: Path | None = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    shardshuffle: bool | int = 100
    train_shards_pattern: str = TRAIN_SHARDS_PATTERN
    valid_shards_pattern: str = VALID_SHARDS_PATTERN
    train_eval_shards_pattern: str = TRAIN_EVAL_SHARDS_PATTERN
    valid_eval_shards_pattern: str = VALID_EVAL_SHARDS_PATTERN

    def __post_init__(self):
        if self.clip_model is None:
            raise ValueError("clip_model is not set")
        if self.loss_function is None:
            raise ValueError("loss_function is not set")
        if self.on_colab is None:
            raise ValueError("explicitly set on_colab to True or False")

    @property
    def base_path(self) -> Path:
        return (
            Path("/content/drive/MyDrive/6740 Group Project")
            if self.on_colab
            else Path("./")
        )

    @property
    def dataset_dir(self) -> Path:
        if self.dataset_root is not None:
            return Path(self.dataset_root)
        return self.base_path / "webdataset_shards"

    @property
    def checkpoint_dir(self) -> Path:
        return self.base_path / "checkpoints"

    @property
    def evaluation_dir(self) -> Path:
        return self.base_path / "evaluations"

    @property
    def model_cache_dir(self) -> Path:
        return self.base_path / "openclip_cache"
