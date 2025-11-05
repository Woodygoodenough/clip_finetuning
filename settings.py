import torch

ON_COLAB = False
# GPU-optimized batch size for T4 (16GB)
# ViT-B-32: ~3-4GB per batch of 64
# Start with 64, can increase to 96-128 if memory allows
BATCH_SIZE = 64

## dataset sanity check
TOTAL_TRAIN = 260490
TOTAL_VALID = 32528

## model parameters
MODEL_NAME = "ViT-B-32"
MODEL_PRETRAINED = "laion2b_s34b_b79k"
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
    CHECKPOINT_DIR = "drive/MyDrive/6740 Group Project/checkpoints"
else:
    CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_INTERVAL = 1000
if ON_COLAB:
    MAX_STEPS = None
else:
    MAX_STEPS = 10

## dataset
if ON_COLAB:
    TRAIN_DATASET_PATTERN = (
        "drive/MyDrive/6740 Group Project/clip_dataset_train.{000000..000260}.tar"
    )
    VALID_DATASET_PATTERN = (
        "drive/MyDrive/6740 Group Project/clip_dataset_valid.{000000..000032}.tar"
    )
else:
    TRAIN_DATASET_PATTERN = (
        "./webdataset_shards/clip_dataset_train.{000000..000260}.tar"
    )
    VALID_DATASET_PATTERN = (
        "./webdataset_shards/clip_dataset_valid.{000000..000032}.tar"
    )
MAX_VALID_SAMPLES = 5000
