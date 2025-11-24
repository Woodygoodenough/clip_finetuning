from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from constants import (
    TEST_SHARDS_PATTERN,
    TRAIN_SHARDS_PATTERN,
    VALID_SHARDS_PATTERN,
)


def torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


class RegistryEntry(BaseModel):
    """Single registered OpenCLIP model option."""

    name: str
    pretrained: str


class ModelRegistry(BaseModel):
    """Collection of available OpenCLIP model presets."""

    entries: Dict[str, RegistryEntry] = Field(default_factory=dict)
    default_key: str = "ViT_SigLIP_2_16"

    @property
    def default(self) -> RegistryEntry:
        return self.entries[self.default_key]


class DatasetConfig(BaseModel):
    train_shards_file: str = TRAIN_SHARDS_PATTERN
    valid_shards_file: str = VALID_SHARDS_PATTERN
    test_shards_file: str = TEST_SHARDS_PATTERN

    def pattern(self, base_path: Path, split: Literal["train", "valid", "test"]) -> str:
        shards_map = {
            "train": self.train_shards_file,
            "valid": self.valid_shards_file,
            "test": self.test_shards_file,
        }
        return str(base_path / shards_map[split])


class PathConfig(BaseModel):
    on_colab: bool = True
    drive_path: Path = Path("/content/drive/MyDrive/6740 Group Project")
    local_dataset_root: Path = Path("./webdataset_shards")
    checkpoint_dir: Optional[Path] = None
    evaluation_dir: Optional[Path] = None

    @model_validator(mode="after")
    def _derive_paths(self) -> "PathConfig":
        if self.checkpoint_dir is None:
            self.checkpoint_dir = (
                self.drive_path / "checkpoints"
                if self.on_colab
                else Path("./checkpoints")
            )
        if self.evaluation_dir is None:
            self.evaluation_dir = (
                self.drive_path / "evaluations"
                if self.on_colab
                else Path("./evaluations")
            )
        return self

    def dataset_base(self) -> Path:
        return self.drive_path if self.on_colab else self.local_dataset_root


class TrainingConfig(BaseModel):
    batch_size: int = 64
    num_workers: int = 2
    learning_rate: float = 1e-5
    checkpoint_interval: int = 1000
    max_steps: Optional[int] = None
    epochs: Optional[int] = None


class MetricsConfig(BaseModel):
    total_train: int = 260_490
    total_valid: int = 32_528


class Settings(BaseModel):
    paths: PathConfig = PathConfig()
    datasets: DatasetConfig = DatasetConfig()
    training: TrainingConfig = TrainingConfig()
    registry: ModelRegistry = ModelRegistry(
        entries={
            "ViT_B_32": RegistryEntry(name="ViT-B-32", pretrained="laion2b_s34b_b79k"),
            "ViT_B_16": RegistryEntry(name="ViT-B-16", pretrained="laion2b_s34b_b88k"),
            "ViT_SigLIP_2_16": RegistryEntry(
                name="ViT-B-16-SigLIP2-256", pretrained="webli"
            ),
        },
        default_key="ViT_SigLIP_2_16",
    )
    metrics: MetricsConfig = MetricsConfig()
    device: str = "cuda" if torch_cuda_available() else "cpu"


CONFIG = Settings()
