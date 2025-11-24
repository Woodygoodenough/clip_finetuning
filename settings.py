import torch
from enum import Enum
from dataclasses import dataclass

ON_COLAB = True
# GPU-optimized batch size for T4 (16GB)
# ViT-B-32: ~3-4GB per batch of 64
# Start with 64, can increase to 96-128 if memory allows
BATCH_SIZE = 64
DRIVE_PATH = "/content/drive/MyDrive/6740 Group Project"
## dataset sanity check
TOTAL_TRAIN = 260490
TOTAL_VALID = 32528

# use enum to specify specific model configurations


# namedtuple to hold pretrained and name
@dataclass
class ModelConfig:
    pretrained: str
    name: str


class OpenClipModel(Enum):
    ViT_B_32 = ModelConfig(pretrained="laion2b_s34b_b79k", name="ViT-B-32")
    ViT_B_16 = ModelConfig(pretrained="laion2b_s34b_b88k", name="ViT-B-16")
    ViT_SigLIP_2_16 = ModelConfig(pretrained="webli", name="ViT-B-16-SigLIP2-256")


MODEL_CHOSEN = OpenClipModel.ViT_SigLIP_2_16


class LossFunction(Enum):
    CONTRASTIVE = 1
    SIGLIP = 2


LOSS_FUNCTION = LossFunction.SIGLIP

MODEL_CACHE_DIR = "./openclip_cache"

## device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## computational resources
# Reduced for Colab T4 to avoid memory issues
# 2 workers is optimal for T4 GPU
NUM_WORKERS = 2

## hyperparameters, training
LEARNING_RATE = 1e-5

## checkpoint directory
if ON_COLAB:
    CHECKPOINT_DIR = f"{DRIVE_PATH}/checkpoints"
else:
    CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_INTERVAL = 1000
if ON_COLAB:
    MAX_STEPS = None
else:
    MAX_STEPS = 10


## evaluation directory
if ON_COLAB:
    EVAL_DIR = f"{DRIVE_PATH}/evaluations"
else:
    EVAL_DIR = "./evaluations"

## dataset
TRAIN_SHARDS_FILE = "clip_dataset_train.{000000..000260}.tar"
VALID_SHARDS_FILE = "clip_dataset_valid.{000000..000006}.tar"
TEST_SHARDS_FILE = "clip_dataset_valid.{000007..000032}.tar"
if ON_COLAB:

    TRAIN_DATASET_PATTERN = f"{DRIVE_PATH}/{TRAIN_SHARDS_FILE}"
    VALID_DATASET_PATTERN = f"{DRIVE_PATH}/{VALID_SHARDS_FILE}"
    TEST_DATASET_PATTERN = f"{DRIVE_PATH}/{TEST_SHARDS_FILE}"
else:
    LOCAL_PATH = "./webdataset_shards"
    TRAIN_DATASET_PATTERN = f"{LOCAL_PATH}/{TRAIN_SHARDS_FILE}"
    VALID_DATASET_PATTERN = f"{LOCAL_PATH}/{VALID_SHARDS_FILE}"
    TEST_DATASET_PATTERN = f"{LOCAL_PATH}/{TEST_SHARDS_FILE}"
