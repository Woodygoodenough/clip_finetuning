"""
CLIP Model Evaluation Script
Evaluates fine-tuned CLIP model on validation set using Recall@5 and Recall@10 metrics.

Metrics:
- Text-to-Image Recall@k: For each text query, percentage where correct image is in top-k
- Image-to-Text Recall@k: For each image query, percentage where correct text is in top-k
"""

import torch
from typing import List, Tuple, Optional
import settings
import webdataset as wds
import logging
from pathlib import Path
from openClipManagement import OpenClipManagment
import numpy as np
import argparse
import json
from typing import Union
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CLIPEvaluator:
    """Evaluator for CLIP model with recall metrics"""
    
    def __init__(self, clip_manager: OpenClipManagment):
        """
        Initialize evaluator
        Args:
            clip_manager: CLIP manager instance
        """
        self.clip = clip_manager
        self.device = self.clip.device
        
        # Set model to eval mode
        self.clip.model.eval()
        
    def encode_images_batch(self, images: List[Union[str, Path]], batch_size: int = 32) -> torch.Tensor:
        """
        Encode images in batches for memory efficiency
        Args:
            images: List of image paths (str) or PIL Images
            batch_size: Batch size for encoding
        Returns:
            Tensor of image embeddings [N, dim]
        """
        all_embeds = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_embeds = self.clip.encode_image_from_pil(batch_images)
                all_embeds.append(batch_embeds)
        
        return torch.cat(all_embeds, dim=0)
    
    def encode_texts_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode texts in batches for memory efficiency
        Args:
            texts: List of texts
            batch_size: Batch size for encoding
        Returns:
            Tensor of text embeddings [N, dim]
        """
        all_embeds = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeds = self.clip.encode_text(batch_texts)
                all_embeds.append(batch_embeds)
        
        return torch.cat(all_embeds, dim=0)
    
    def compute_similarity_matrix(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix between images and texts
        Args:
            image_embeds: Image embeddings [N_images, dim]
            text_embeds: Text embeddings [N_texts, dim]
        Returns:
            Similarity matrix [N_images, N_texts]
        """
        # Normalize embeddings
        image_embeds = self.clip.normalize_tensor(image_embeds)
        text_embeds = self.clip.normalize_tensor(text_embeds)
        
        # Compute similarity matrix
        similarity = image_embeds @ text_embeds.T  # [N_images, N_texts]
        
        return similarity
    
    def recall_at_k(self, similarity_matrix: torch.Tensor, k: int = 5) -> Tuple[float, float]:
        """
        Compute Recall@k for both text-to-image and image-to-text retrieval
        
        Args:
            similarity_matrix: Similarity matrix [N_images, N_texts]
            k: Number of top results to consider
        
        Returns:
            (text_to_image_recall, image_to_text_recall)
        """
        n_images, n_texts = similarity_matrix.shape
        
        # Text-to-Image: For each text, find top-k images
        # Get top-k indices for each text (column) - columns are texts, rows are images
        # argsort returns the indices
        # dim=0 means keep other dimensions fixed
        # we sort with gpu if possible, then move a smaller tensor to cpu
        text_to_image_topk = torch.argsort(similarity_matrix, dim=0)[-k:].T.cpu().numpy()
        
        # Check if image index i is in top-k for text i
        text_to_image_correct = np.array([i in text_to_image_topk[i] for i in range(min(n_texts, n_images))])
        text_to_image_recall = text_to_image_correct.mean() if len(text_to_image_correct) > 0 else 0.0
        
        # Image-to-Text: For each image, find top-k texts
        image_to_text_topk = torch.argsort(similarity_matrix, dim=1)[:, -k:].cpu().numpy()
        
        # Check if text index i is in top-k for image i
        image_to_text_correct = np.array([i in image_to_text_topk[i] for i in range(min(n_images, n_texts))])
        image_to_text_recall = image_to_text_correct.mean() if len(image_to_text_correct) > 0 else 0.0
        
        return text_to_image_recall, image_to_text_recall
    
    def evaluate_webdataset(self, dataset_pattern: str, max_samples: Optional[int] = None) -> dict:
        """
        Evaluate model on WebDataset validation set
        
        Args:
            dataset_pattern: Pattern for WebDataset shards
            max_samples: Maximum number of samples to evaluate (None for all)
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Loading validation dataset from {dataset_pattern}")
        
        # Load dataset
        dataset = (
            wds.WebDataset(dataset_pattern, empty_check=False)
            .decode("pil")
            .to_tuple("jpg", "txt")
            .map_tuple(lambda img: img, lambda txt: txt.decode('utf-8') if isinstance(txt, bytes) else txt)
        )
        
        # Collect all image-text pairs
        images = []
        texts = []
        
        logger.info("Collecting validation samples...")
        for i, (img, txt) in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            images.append(img)
            texts.append(txt)
            if (i + 1) % 1000 == 0:
                logger.info(f"Collected {i + 1} samples...")
        
        logger.info(f"Total samples collected: {len(images)}")
        
        # Encode images and texts in batches
        logger.info("Encoding images...")
        image_embeds = self.encode_images_batch(images, batch_size=settings.BATCH_SIZE)
        
        logger.info("Encoding texts...")
        text_embeds = self.encode_texts_batch(texts, batch_size=settings.BATCH_SIZE)
        
        logger.info("Computing similarity matrix...")
        similarity_matrix = self.compute_similarity_matrix(image_embeds, text_embeds)
        
        # Compute Recall@5 and Recall@10
        logger.info("Computing Recall@5...")
        recall5_t2i, recall5_i2t = self.recall_at_k(similarity_matrix, k=5)
        
        logger.info("Computing Recall@10...")
        recall10_t2i, recall10_i2t = self.recall_at_k(similarity_matrix, k=10)
        
        results = {
            'num_samples': len(images),
            'recall@5': {
                'text_to_image': recall5_t2i,
                'image_to_text': recall5_i2t,
            },
            'recall@10': {
                'text_to_image': recall10_t2i,
                'image_to_text': recall10_i2t,
            }
        }
        
        return results
    


def print_results(results: dict):
    """Print evaluation results in a formatted way"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples evaluated: {results['num_samples']}")
    print("\nRecall@5:")
    print(f"  Text-to-Image: {results['recall@5']['text_to_image']:.4f} ({results['recall@5']['text_to_image']*100:.2f}%)")
    print(f"  Image-to-Text: {results['recall@5']['image_to_text']:.4f} ({results['recall@5']['image_to_text']*100:.2f}%)")
    print("\nRecall@10:")
    print(f"  Text-to-Image: {results['recall@10']['text_to_image']:.4f} ({results['recall@10']['text_to_image']*100:.2f}%)")
    print(f"  Image-to-Text: {results['recall@10']['image_to_text']:.4f} ({results['recall@10']['image_to_text']*100:.2f}%)")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CLIP model on validation set')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pt file). If not provided, uses base pretrained model')
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to dataset (WebDataset pattern). Defaults to settings.VALID_DATASET_PATTERN')
    
    args = parser.parse_args()
    
    # Initialize CLIP manager
    logger.info("Initializing CLIP manager...")
    
    # Initialize evaluator and load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        clip_manager = OpenClipManagment.from_checkpoint(args.checkpoint)
    else:
        logger.info("No checkpoint provided, using base pretrained model")
        clip_manager = OpenClipManagment()
    evaluator = CLIPEvaluator(clip_manager)
    
    # Determine dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = settings.VALID_DATASET_PATTERN
    
    # Evaluate
    results = evaluator.evaluate_webdataset(dataset_path, max_samples=settings.MAX_VALID_SAMPLES)
        
    # Print results
    print_results(results)
    
    # Save results to file
    if args.checkpoint:
        results_file = Path("evaluation_results_checkpoint.json")
    else:
        results_file = Path("evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

