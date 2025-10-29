"""
CLIP Dataset Preparation Helper
Creates a simple mapping file for CLIP fine-tuning from Fashion-Gen dataset.
"""

import pickle
import json
from pathlib import Path
from tqdm import tqdm

def create_clip_dataset(split="valid", output_file=None):
    """Create a CLIP-ready dataset mapping."""
    
    # Paths
    meta_dir = Path(f"full_{split}_info_PAI")
    img_dir = Path(f"extracted_{split}_images")
    
    if not meta_dir.exists():
        print(f"Metadata directory not found: {meta_dir}")
        return
    
    if not img_dir.exists():
        print(f"Image directory not found: {img_dir}")
        return
    
    # Get all pickle files
    pickle_files = list(meta_dir.glob("*.pkl"))
    print(f"Processing {len(pickle_files)} items from {split} set...")
    
    # Create mapping
    image_caption_pairs = []
    
    for pkl_file in tqdm(pickle_files):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            img_name = data.get('img_name', '')
            caption = data.get('captions', '')
            category = data.get('super_cls_name', 'Unknown')
            
            # Check if corresponding image exists
            img_path = img_dir / img_name
            if img_path.exists() and caption.strip():
                image_caption_pairs.append({
                    'image_path': str(img_path),
                    'image_name': img_name,
                    'caption': caption,
                    'category': category,
                    'product_id': data.get('product_id', '')
                })
                
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
    
    # Save results
    if output_file is None:
        output_file = f"clip_dataset_{split}.json"
    
    with open(output_file, 'w') as f:
        json.dump(image_caption_pairs, f, indent=2)
    
    print(f"\nDataset created: {output_file}")
    print(f"Total image-caption pairs: {len(image_caption_pairs)}")
    
    # Show statistics
    categories = [item['category'] for item in image_caption_pairs]
    from collections import Counter
    category_counts = Counter(categories)
    
    print(f"\nCategory distribution:")
    for cat, count in category_counts.most_common(10):
        print(f"  {cat}: {count}")
    
    # Show sample pairs
    print(f"\nSample pairs:")
    for i, item in enumerate(image_caption_pairs[:3]):
        print(f"{i+1}. {item['image_name']} -> {item['caption'][:80]}...")
    
    return image_caption_pairs

def main():
    print("CLIP Dataset Preparation")
    print("=" * 30)
    
    # Create validation set (smaller, for testing)
    print("Creating validation dataset...")
    valid_data = create_clip_dataset("valid")
    
    # Optionally create training set (larger)
    print(f"\n{'='*30}")
    response = input("Create training dataset? (y/n): ").lower().strip()
    if response == 'y':
        print("Creating training dataset...")
        train_data = create_clip_dataset("train")

if __name__ == "__main__":
    main()
