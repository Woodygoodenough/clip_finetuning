"""
CLIP Model Evaluation Script
Evaluates fine-tuned CLIP model on validation set using Recall@5 and Recall@10 metrics.

Metrics:
- Text-to-Image Recall@k: For each text query, percentage where correct image is in top-k
- Image-to-Text Recall@k: For each image query, percentage where correct text is in top-k
"""

import torch
from typing import Tuple
import settings
import logging
from pathlib import Path
from openClipManagement import OpenClipManagment
import numpy as np
import argparse
import json
from dBManagement import ClipDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CLIPEvaluator:
    """Evaluator for CLIP model with recall metrics"""

    def __init__(
        self,
        clip_manager: OpenClipManagment,
        clip_dataset: ClipDataset = ClipDataset(
            dataset_pattern=settings.VALID_DATASET_PATTERN
        ),
    ):
        """
        Initialize evaluator
        Args:
            clip_manager: CLIP manager instance
            clip_dataset: ClipDataset instance
        """
        self.clip = clip_manager
        self.clip_dataset = clip_dataset
        self.device = self.clip.device

        # Set model to eval mode
        self.clip.model.eval()

    def compute_similarity_matrix(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
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

    def recall_at_k(
        self, similarity_matrix: torch.Tensor, k: int = 5
    ) -> Tuple[float, float]:
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
        text_to_image_topk = (
            torch.argsort(similarity_matrix, dim=0)[-k:].T.cpu().numpy()
        )

        # Check if image index i is in top-k for text i
        text_to_image_correct = np.array(
            [i in text_to_image_topk[i] for i in range(min(n_texts, n_images))]
        )
        text_to_image_recall = (
            text_to_image_correct.mean() if len(text_to_image_correct) > 0 else 0.0
        )

        # Image-to-Text: For each image, find top-k texts
        image_to_text_topk = (
            torch.argsort(similarity_matrix, dim=1)[:, -k:].cpu().numpy()
        )

        # Check if text index i is in top-k for image i
        image_to_text_correct = np.array(
            [i in image_to_text_topk[i] for i in range(min(n_images, n_texts))]
        )
        image_to_text_recall = (
            image_to_text_correct.mean() if len(image_to_text_correct) > 0 else 0.0
        )

        return text_to_image_recall, image_to_text_recall

    def evaluate_with_loader(self) -> dict:
        """
        Evaluate model on validation set using efficient WebDataset loader.

        This approach is optimal because:
        - Uses the same efficient pipeline as training
        - Processes data in batches directly from disk
        - More memory efficient (doesn't load all data at once)
        - Consistent with training data flow

        Args:
            max_samples: Maximum number of samples to evaluate (None for all)

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Creating efficient WebDataset loader for evaluation...")

        # Get loader using the same approach as training
        loader = self.clip.get_loader(self.clip_dataset)

        # Collect embeddings batch by batch
        image_embeds_list = []
        text_embeds_list = []
        num_samples = 0

        logger.info("Processing batches and computing embeddings...")
        with torch.no_grad():
            for batch_idx, (img_tensors, text_strings) in enumerate(loader):
                # Move images to device
                img_tensors = img_tensors.to(self.device)

                # Tokenize text batch (same as training)
                txt_tensors = self.clip.txt_tokenizer(text_strings).to(self.device)

                # Encode batch
                img_embeds = self.clip.model.encode_image(img_tensors)
                txt_embeds = self.clip.model.encode_text(txt_tensors)

                image_embeds_list.append(img_embeds.cpu())
                text_embeds_list.append(txt_embeds.cpu())

                num_samples += len(img_embeds)

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {num_samples} samples...")

        logger.info(f"Total samples processed: {num_samples}")

        # Concatenate all embeddings
        image_embeds = torch.cat(image_embeds_list, dim=0)
        text_embeds = torch.cat(text_embeds_list, dim=0)

        # Move to device for similarity computation
        image_embeds = image_embeds.to(self.device)
        text_embeds = text_embeds.to(self.device)

        logger.info("Computing similarity matrix...")
        similarity_matrix = self.compute_similarity_matrix(image_embeds, text_embeds)

        # Compute Recall@5 and Recall@10
        logger.info("Computing Recall@5...")
        recall5_t2i, recall5_i2t = self.recall_at_k(similarity_matrix, k=5)

        logger.info("Computing Recall@10...")
        recall10_t2i, recall10_i2t = self.recall_at_k(similarity_matrix, k=10)

        results = {
            "num_samples": num_samples,
            "recall@5": {
                "text_to_image": recall5_t2i,
                "image_to_text": recall5_i2t,
            },
            "recall@10": {
                "text_to_image": recall10_t2i,
                "image_to_text": recall10_i2t,
            },
        }

        return results


def print_results(results: dict):
    """Print evaluation results in a formatted way"""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of samples evaluated: {results['num_samples']}")
    print("\nRecall@5:")
    print(
        f"  Text-to-Image: {results['recall@5']['text_to_image']:.4f} ({results['recall@5']['text_to_image']*100:.2f}%)"
    )
    print(
        f"  Image-to-Text: {results['recall@5']['image_to_text']:.4f} ({results['recall@5']['image_to_text']*100:.2f}%)"
    )
    print("\nRecall@10:")
    print(
        f"  Text-to-Image: {results['recall@10']['text_to_image']:.4f} ({results['recall@10']['text_to_image']*100:.2f}%)"
    )
    print(
        f"  Image-to-Text: {results['recall@10']['image_to_text']:.4f} ({results['recall@10']['image_to_text']*100:.2f}%)"
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP model on validation set"
    )

    parser.add_argument(
        "-c",
        "--use-checkpoint",
        action="store_true",
        help="Use checkpoint in settings.EVAL_CHECKPOINT, if not provided, uses base pretrained model",
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset (WebDataset pattern). Defaults to settings.VALID_DATASET_PATTERN",
    )

    args = parser.parse_args()

    # Initialize CLIP manager
    logger.info("Initializing CLIP manager...")

    # Initialize evaluator and load checkpoint if provided
    if args.use_checkpoint:
        logger.info(f"Loading checkpoint from {settings.EVAL_CHECKPOINT}")
        clip_manager = OpenClipManagment.from_checkpoint(settings.EVAL_CHECKPOINT)
    else:
        logger.info("No checkpoint provided, using base pretrained model")
        clip_manager = OpenClipManagment()

    # Determine dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = settings.VALID_DATASET_PATTERN

    logger.info(f"Loading validation dataset from {dataset_path}")
    clip_dataset = ClipDataset(dataset_path)

    # Create evaluator
    evaluator = CLIPEvaluator(clip_manager, clip_dataset)

    # Evaluate using efficient WebDataset loader
    results = evaluator.evaluate_with_loader()

    # Print results
    print_results(results)

    # Save results to file
    if args.use_checkpoint:
        results_file = Path("evaluation_results_checkpoint.json")
    else:
        results_file = Path("evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")
