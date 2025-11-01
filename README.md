# Fashion Dataset for CLIP Fine-tuning

This repository contains a fashion image-caption dataset derived from the Fashion-Gen dataset, prepared for fine-tuning CLIP (Contrastive Language-Image Pre-training) models.

## Dataset Overview

- **Total Images**: ~293,018
- **Training Images**: ~260,490 (88.9%)
- **Validation Images**: ~32,528 (11.1%)
- **Domain**: Fashion & Apparel
- **Format**: Image-caption pairs with product metadata

## Dataset Structure

### Image Directories

#### `extracted_train_images/`
Contains all training images in PNG format.

- **Content**: Product images from the training split
- **File Format**: PNG images
- **Naming Convention**: `{product_id}_{image_index}.png`
- **Count**: ~260,490 images
- **Usage**: Used for model training during fine-tuning

#### `extracted_valid_images/`
Contains all validation images in PNG format.

- **Content**: Product images from the validation split
- **File Format**: PNG images
- **Naming Convention**: `{product_id}_{image_index}.png`
- **Count**: ~32,528 images
- **Usage**: Used for model evaluation and validation during fine-tuning

### Metadata Directories

#### `full_train_info_PAI/`
Contains metadata files for training images stored as pickle files.

- **Content**: Python pickle files (`.pkl`) containing product metadata
- **Structure**: Each pickle file contains a dictionary with:
  - `img_name`: Image filename
  - `captions`: Product description/caption text
  - `super_cls_name`: Product category (e.g., "SWEATERS", "TOPS", "JEANS")
  - `product_id`: Unique product identifier
- **Usage**: Source metadata for creating training dataset mappings

#### `full_valid_info_PAI/`
Contains metadata files for validation images stored as pickle files.

- **Content**: Python pickle files (`.pkl`) containing product metadata
- **Structure**: Same as `full_train_info_PAI/`, but for validation split
- **Usage**: Source metadata for creating validation dataset mappings

### Processed Dataset Files

#### `clip_dataset_valid.json`
Pre-processed JSON file containing validation image-caption pairs ready for CLIP fine-tuning.

- **Format**: JSON array of objects
- **Structure**: Each entry contains:
  ```json
  {
    "image_path": "extracted_valid_images/{product_id}_{image_index}.png",
    "image_name": "{product_id}_{image_index}.png",
    "caption": "Detailed product description...",
    "category": "CATEGORY_NAME",
    "product_id": "product_id"
  }
  ```
- **Count**: 32,528 entries
- **Usage**: Direct input for CLIP fine-tuning scripts

#### `clip_dataset_train.json` (if generated)
Pre-processed JSON file containing training image-caption pairs.

- **Format**: Same structure as `clip_dataset_valid.json`
- **Count**: Expected ~260,490 entries (when generated)
- **Usage**: Direct input for CLIP fine-tuning scripts

## Data Fields

Each dataset entry contains the following fields:

- **`image_path`**: Relative path to the image file
- **`image_name`**: Filename of the image
- **`caption`**: Detailed product description including:
  - Material (e.g., cotton, denim, leather)
  - Style features (e.g., sleeve length, collar type)
  - Color information
  - Design details (e.g., graphics, patterns, hardware)
  - Fit and sizing information
- **`category`**: Product category (e.g., "SWEATERS", "TOPS", "JEANS", "JACKETS & COATS", "SANDALS")
- **`product_id`**: Unique identifier for the product

## Dataset Preparation

The dataset can be prepared using `clip_prep.py`, which:
1. Reads metadata from pickle files in `full_{split}_info_PAI/` directories
2. Matches metadata with images in `extracted_{split}_images/` directories
3. Creates JSON mapping files (`clip_dataset_{split}.json`) ready for CLIP fine-tuning

## Usage

This dataset is designed for:
- Fine-tuning CLIP models for fashion domain
- Text-to-image retrieval tasks
- Image-to-text retrieval tasks
- Fashion product search and recommendation systems

## WebDataset Conversion

For efficient training with OpenCLIP, the dataset can be converted to WebDataset format (tar shards).

### Converting JSON to WebDataset Shards

Use `create_webdataset.py` to convert JSON files to tar shards:

```bash
# Convert validation set
python create_webdataset.py \
    --json clip_dataset_valid.json \
    --output-dir webdataset_shards \
    --shard-size 1000

# Convert training set (if available)
python create_webdataset.py \
    --json clip_dataset_train.json \
    --output-dir webdataset_shards \
    --shard-size 1000
```

**Parameters:**
- `--json`: Path to JSON file with image-caption pairs
- `--output-dir`: Output directory for tar shards (default: `webdataset_shards`)
- `--shard-size`: Number of samples per shard (default: 1000)
- `--max-shards`: Limit number of shards for testing (optional)
- `--image-key`: JSON key for image path (default: `image_path`)
- `--caption-key`: JSON key for caption text (default: `caption`)

**Output Format:**
- Shards: `{dataset_name}.{000000..NNNNNN}.tar`
- Each shard contains pairs of: `{index:09d}.jpg` and `{index:09d}.txt`
- Images are converted to JPEG format (95% quality) for efficiency

### Using WebDataset Shards

```python
import webdataset as wds

# Create dataset from shards
dataset = (
    wds.WebDataset("webdataset_shards/clip_dataset_train.{000000..000260}.tar")
    .decode("pil")
    .to_tuple("jpg", "txt")
    .map_tuple(lambda img: img, lambda txt: txt.decode('utf-8'))
)

# Create dataloader
loader = wds.WebLoader(dataset, batch_size=32, num_workers=4)
```

See `example_webdataset_usage.py` for a complete example.

## Notes

- Images are stored as PNG files in the original directories
- Metadata is stored in pickle format and needs to be processed before use
- The validation split has been processed into JSON format
- Training split JSON can be generated using `clip_prep.py` if needed
- WebDataset shards are recommended for large-scale training as they provide faster I/O and better shuffling