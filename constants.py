from __future__ import annotations

# Dataset field names
IMAGE_KEY = "image_path"
CAPTION_KEY = "caption"
CATEGORY_KEY = "category"
PRODUCT_ID_KEY = "product_id"

# WebDataset conventions
WDS_IMAGE_EXT = "jpg"
WDS_TEXT_EXT = "txt"

# WebDataset shard patterns
TRAIN_SHARDS_PATTERN = "clip_dataset_train.{000000..000260}.tar"
VALID_SHARDS_PATTERN = "clip_dataset_valid.{000000..000006}.tar"
TEST_SHARDS_PATTERN = "clip_dataset_valid.{000007..000032}.tar"

# Metric identifiers
RECALL_AT_5 = "recall@5"
RECALL_AT_10 = "recall@10"
T2I_METRIC = "text_to_image"
I2T_METRIC = "image_to_text"

# Evaluation filenames
EVAL_RESULTS_PREFIX = "eval"

# Misc
DEFAULT_SHARD_SIZE = 1000
