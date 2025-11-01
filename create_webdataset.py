"""
Convert image-caption JSON dataset to WebDataset tar shards for OpenCLIP fine-tuning
"""
import json
import tarfile
import io
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse


def create_webdataset_shards(
    json_path: str,
    output_dir: str,
    shard_size: int = 1000,
    max_shards: int = None,
    image_key: str = "image_path",
    caption_key: str = "caption",
    dataset_name: str = None,
):
    """
    Convert JSON dataset to WebDataset tar shards.
    
    Args:
        json_path: Path to JSON file with image-caption pairs
        output_dir: Directory to save tar shards
        shard_size: Number of samples per shard
        max_shards: Maximum number of shards to create (None = all)
        image_key: Key in JSON for image path
        caption_key: Key in JSON for caption text
        dataset_name: Name prefix for shards (defaults to JSON filename without extension)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load JSON data
    print(f"Loading JSON from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} image-caption pairs")
    
    # Determine dataset name
    if dataset_name is None:
        dataset_name = Path(json_path).stem
    
    # Process data in shards
    num_shards = (len(data) + shard_size - 1) // shard_size
    if max_shards is not None:
        num_shards = min(num_shards, max_shards)
    
    print(f"Creating {num_shards} shards with ~{shard_size} samples each...")
    
    samples_processed = 0
    skipped = 0
    
    for shard_idx in tqdm(range(num_shards), desc="Creating shards"):
        shard_path = output_dir / f"{dataset_name}.{shard_idx:06d}.tar"
        
        with tarfile.open(shard_path, 'w') as tar:
            # Process samples for this shard
            start_idx = shard_idx * shard_size
            end_idx = min(start_idx + shard_size, len(data))
            
            if max_shards is not None and samples_processed >= max_shards * shard_size:
                break
            
            for sample_idx in range(start_idx, end_idx):
                if max_shards is not None and samples_processed >= max_shards * shard_size:
                    break
                
                sample = data[sample_idx]
                
                # Get image path and caption
                image_path = Path(sample[image_key])
                caption = sample[caption_key]
                
                # Check if image exists
                if not image_path.exists():
                    skipped += 1
                    continue
                
                # Read image
                try:
                    img = Image.open(image_path)
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save image to bytes
                    img_bytes = io.BytesIO()
                    # Save as JPEG for smaller size (WebDataset standard)
                    img.save(img_bytes, format='JPEG', quality=95)
                    img_bytes.seek(0)
                    
                    # Create tar info for image
                    image_name = f"{sample_idx:09d}.jpg"
                    tarinfo = tarfile.TarInfo(name=image_name)
                    tarinfo.size = len(img_bytes.getvalue())
                    tar.addfile(tarinfo, img_bytes)
                    
                    # Create tar info for text caption
                    text_name = f"{sample_idx:09d}.txt"
                    caption_bytes = caption.encode('utf-8')
                    tarinfo = tarfile.TarInfo(name=text_name)
                    tarinfo.size = len(caption_bytes)
                    tar.addfile(tarinfo, io.BytesIO(caption_bytes))
                    
                    samples_processed += 1
                    
                except Exception as e:
                    print(f"\nWarning: Skipping {image_path}: {e}")
                    skipped += 1
                    continue
    
    print(f"\nSuccessfully created {num_shards} shards in {output_dir}")
    print(f"   Processed: {samples_processed} samples")
    print(f"   Skipped: {skipped} samples")
    print(f"\nShard pattern: {dataset_name}.{{000000..{num_shards-1:06d}}}.tar")
    
    # Create shard list file
    shard_list_path = output_dir / f"{dataset_name}_shards.txt"
    with open(shard_list_path, 'w') as f:
        for shard_idx in range(num_shards):
            shard_path = output_dir / f"{dataset_name}.{shard_idx:06d}.tar"
            if shard_path.exists():
                f.write(f"{shard_path}\n")
    
    print(f"Shard list saved to: {shard_list_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON dataset to WebDataset tar shards"
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to JSON file with image-caption pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="webdataset_shards",
        help="Output directory for tar shards (default: webdataset_shards)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Number of samples per shard (default: 1000)",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Maximum number of shards to create (for testing, default: None)",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="image_path",
        help="JSON key for image path (default: image_path)",
    )
    parser.add_argument(
        "--caption-key",
        type=str,
        default="caption",
        help="JSON key for caption text (default: caption)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name prefix for shards (default: JSON filename)",
    )
    
    args = parser.parse_args()
    
    create_webdataset_shards(
        json_path=args.json,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        max_shards=args.max_shards,
        image_key=args.image_key,
        caption_key=args.caption_key,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()

