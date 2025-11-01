import torch
BATCH_SIZE = 32

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
NUM_WORKERS = 4

## hyperparameters, training
LEARNING_RATE = 1e-5

## dataset
TRAIN_DATASET_PATTERN = "webdataset_shards/clip_dataset_train.{000000..000000}.tar"
VALID_DATASET_PATTERN = "webdataset_shards/clip_dataset_valid.{000000..000000}.tar"