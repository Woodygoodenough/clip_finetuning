import torch

ON_COLAB = True
# GPU-optimized batch size for T4 (16GB)
# ViT-B-32: ~3-4GB per batch of 64
# Start with 64, can increase to 96-128 if memory allows
BATCH_SIZE = 64

## dataset sanity check
TOTAL_TRAIN = 260490
TOTAL_VALID = 32528

## model parameters
# MODEL_NAME = "ViT-B-32"
MODEL_NAME = "ViT-B-16"
# MODEL_PRETRAINED = "laion2b_s34b_b79k"
MODEL_PRETRAINED = "laion2b_s34b_b88k"
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
EVAL_CHECKPOINT = CHECKPOINT_DIR + "/final_checkpoint.pt"
CHECKPOINT_INTERVAL = 1000
if ON_COLAB:
    MAX_STEPS = None
else:
    MAX_STEPS = 10


## dataset
TRAIN_SHARDS_FILE = "clip_dataset_train.{000000..000260}.tar"
VALID_SHARDS_FILE = "clip_dataset_valid.{000000..000006}.tar"
TEST_SHARDS_FILE = "clip_dataset_valid.{000007..000032}.tar"
if ON_COLAB:
    DRIVE_PATH = "/content/drive/MyDrive/6740 Group Project"
    TRAIN_DATASET_PATTERN = f"{DRIVE_PATH}/{TRAIN_SHARDS_FILE}"
    VALID_DATASET_PATTERN = f"{DRIVE_PATH}/{VALID_SHARDS_FILE}"
    TEST_DATASET_PATTERN = f"{DRIVE_PATH}/{TEST_SHARDS_FILE}"
else:
    LOCAL_PATH = "./webdataset_shards"
    TRAIN_DATASET_PATTERN = f"{LOCAL_PATH}/{TRAIN_SHARDS_FILE}"
    VALID_DATASET_PATTERN = f"{LOCAL_PATH}/{VALID_SHARDS_FILE}"
    TEST_DATASET_PATTERN = f"{LOCAL_PATH}/{TEST_SHARDS_FILE}"
