from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

from enum import Enum
from pydantic import BaseModel, Field, model_validator, ConfigDict

from constants import (
    TEST_SHARDS_PATTERN,
    TRAIN_SHARDS_PATTERN,
    VALID_SHARDS_PATTERN,
    TRAIN_EVAL_SHARDS_PATTERN,
    VALID_EVAL_SHARDS_PATTERN,
    TEST_EVAL_SHARDS_PATTERN,
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
    shardshuffle: bool | int = 100
    train_eval_shards_file: str = TRAIN_EVAL_SHARDS_PATTERN
    valid_eval_shards_file: str = VALID_EVAL_SHARDS_PATTERN
    test_eval_shards_file: str = TEST_EVAL_SHARDS_PATTERN

    def pattern(self, base_path: Path, split: Literal["train", "valid", "test"]) -> str:
        shards_map = {
            "train": base_path / self.train_shards_file,
            "valid": base_path / self.valid_shards_file,
            "test": base_path / self.test_shards_file,
        }
        return str(shards_map[split])

    def eval_pattern(
        self, base_path: Path, split: Literal["train", "valid", "test"]
    ) -> str:
        shards_map = {
            "train": base_path / self.train_eval_shards_file,
            "valid": base_path / self.valid_eval_shards_file,
            "test": base_path / self.test_eval_shards_file,
        }
        return str(shards_map[split])


class PathConfig(BaseModel):
    on_colab: bool = False
    drive_path: Path = Path("/content/drive/MyDrive/6740 Group Project")
    local_dataset_root: Path = Path("./webdataset_shards")
    colab_dataset_root: Path = Path("webdataset_shards")
    model_cache_dir: Path = Path("./openclip_cache")
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
        if self.on_colab:
            return self.drive_path / self.colab_dataset_root
        return self.local_dataset_root

    def set_on_colab(self, on_colab: bool):
        self.on_colab = on_colab
        if self.on_colab:
            self.checkpoint_dir = self.drive_path / "checkpoints"
            self.evaluation_dir = self.drive_path / "evaluations"
        else:
            self.checkpoint_dir = Path("./checkpoints")
            self.evaluation_dir = Path("./evaluations")
        return self


class TrainingConfig(BaseModel):
    batch_size: int = 64
    num_workers: int = 2
    learning_rate: float = 1e-5
    checkpoint_interval: int = 1000
    max_steps: Optional[int] = None
    epochs: int = 5


class MetricsConfig(BaseModel):
    total_train: int = 260_490
    total_valid: int = 32_528


class LossFunction(str, Enum):
    CONTRASTIVE = "contrastive"
    SIGLIP = "siglip"


class ProjectConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True)
    on_colab: bool = False
    paths: PathConfig = PathConfig(on_colab=on_colab)
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
    model_key: str = Field(alias="model")
    loss_function: LossFunction = LossFunction.SIGLIP
    checkpoint_path: Optional[Path] = None
    checkpoint_model_key: Optional[str] = Field(default=None, alias="checkpoint_model")

    @model_validator(mode="after")
    def _validate_model_key(self) -> "ProjectConfig":
        using_checkpoint = self.model_key == "from_checkpoint"

        if using_checkpoint:
            if self.checkpoint_path is None:
                raise ValueError(
                    "Provide `checkpoint_path` when model is set to 'from_checkpoint'."
                )
            if self.checkpoint_model_key is None:
                raise ValueError(
                    "Provide `checkpoint_model` when model is set to 'from_checkpoint'."
                )
            if self.checkpoint_model_key not in self.registry.entries:
                raise KeyError(
                    f"Unknown checkpoint model key '{self.checkpoint_model_key}'."
                )
        else:
            if (
                self.checkpoint_path is not None
                or self.checkpoint_model_key is not None
            ):
                raise ValueError("Checkpoint fields require model='from_checkpoint'.")

        # Ensure provided model key resolves to a known registry entry.
        self.get_model(self.model_key)
        return self

    def get_model(self, key: Optional[str] = None) -> RegistryEntry:
        model_key = key or self.model_key
        if model_key == "from_checkpoint":
            if not self.checkpoint_model_key:
                raise ValueError(
                    "checkpoint_model must be provided when model='from_checkpoint'."
                )
            model_key = self.checkpoint_model_key
        if model_key not in self.registry.entries:
            raise KeyError(f"Unknown model key '{model_key}'.")
        return self.registry.entries[model_key]

    @property
    def model(self) -> RegistryEntry:
        return self.get_model()

    @property
    def train_dataset_pattern(self) -> str:
        return self.datasets.pattern(self.paths.dataset_base(), "train")

    @property
    def valid_dataset_pattern(self) -> str:
        return self.datasets.pattern(self.paths.dataset_base(), "valid")

    @property
    def test_dataset_pattern(self) -> str:
        return self.datasets.pattern(self.paths.dataset_base(), "test")

    @property
    def train_eval_dataset_pattern(self) -> str:
        return self.datasets.eval_pattern(self.paths.dataset_base(), "train")

    @property
    def valid_eval_dataset_pattern(self) -> str:
        return self.datasets.eval_pattern(self.paths.dataset_base(), "valid")

    @property
    def test_eval_dataset_pattern(self) -> str:
        return self.datasets.eval_pattern(self.paths.dataset_base(), "test")

    @property
    def should_load_checkpoint(self) -> bool:
        return self.model_key == "from_checkpoint" and self.checkpoint_path is not None
