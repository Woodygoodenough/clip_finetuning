# %%
from tabnanny import check
from openClipManagement import OpenClipManagment
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler  # Mixed precision for T4 GPU
from typing import Optional, List
import logging
from pathlib import Path
from dBManagement import ClipDataset
import settings
import webdataset as wds
from evaluate_clip import CLIPEvaluator, print_results, save_results
from typing import Tuple
import torch.nn.functional as F
from typing import Callable

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Checkpoint settings


# %%
class CLIPFineTuner:
    def __init__(
        self,
        clip_manager: OpenClipManagment,
        clip_dataset: ClipDataset,
        learning_rate: float = settings.LEARNING_RATE,
    ):
        self.clip = clip_manager
        self.clip_dataset = clip_dataset
        self.optimizer = torch.optim.AdamW(
            self.clip.model.parameters(), lr=learning_rate
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = self.clip.device
        # Mixed precision scaler for T4 GPU optimization
        # Enables ~2x faster training with ~50% less memory usage
        self.scaler = GradScaler()

    def contrastive_loss(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(
            "For now, we want to try siglip loss. So we temporarily disable contrastive loss in case it is somehow wired in."
        )
        """Compute CLIP contrastive loss"""
        image_embeds = self.clip.normalize_tensor(image_embeds)
        text_embeds = self.clip.normalize_tensor(text_embeds)

        logits = (image_embeds @ text_embeds.T) * self.clip.model.logit_scale.exp()
        labels = torch.arange(len(logits), device=logits.device)

        loss_i = self.loss_fn(logits, labels)
        loss_t = self.loss_fn(logits.T, labels)
        return (loss_i + loss_t) / 2

    def siglip_loss(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        # L2-normalize (you already have a helper)
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        logits = image_embeds @ text_embeds.T * self.clip.model.logit_scale.exp()

        B = logits.size(0)
        # target matrix: 1 on diagonal, 0 elsewhere
        targets = torch.eye(B, device=logits.device)

        # binary cross-entropy with logits over all pairs
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        return loss

    def train_step(
        self, img_tensors: torch.Tensor, text_strings: List[str], loss_fn: Callable
    ) -> float:
        """
        Perform one training step.

        Args:
            img_tensors: [batch, 3, 224, 224] image tensors (already preprocessed)
            text_strings: List of text strings (to be tokenized as a batch)
        """
        self.clip.model.train()
        self.optimizer.zero_grad()

        # Move image tensors to device
        img_tensors = img_tensors.to(self.device)

        # Tokenize batch of text strings
        txt_tensors = self.clip.txt_tokenizer(text_strings).to(self.device)

        # Use mixed precision for faster training and less memory usage
        # This enables automatic mixed precision (FP16) which is ~2x faster on T4
        with autocast():
            img_embeds = self.clip.model.encode_image(img_tensors)
            txt_embeds = self.clip.model.encode_text(txt_tensors)
            loss = loss_fn(img_embeds, txt_embeds)

        # Scale loss and backward pass for mixed precision
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def train(self, loader: wds.WebLoader):
        checkpoint_path = Path(settings.CHECKPOINT_DIR)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory: {checkpoint_path.absolute()}")
        step = -1
        loss = None
        # we use recall@10 for both text-to-image and image-to-text, only save when both are improved
        best_recall_t2i = 0.0
        best_recall_i2t = 0.0
        if settings.LOSS_FUNCTION == settings.LossFunction.CONTRASTIVE:
            loss_fn = self.contrastive_loss
        elif settings.LOSS_FUNCTION == settings.LossFunction.SIGLIP:
            loss_fn = self.siglip_loss
        else:
            raise ValueError(f"Invalid loss function: {settings.LOSS_FUNCTION}")
        for step, (img_tensors, text_strings) in enumerate(loader):
            if settings.MAX_STEPS is not None and step >= settings.MAX_STEPS:
                logger.info(
                    f"Reached maximum step limit ({settings.MAX_STEPS}). Stopping training."
                )
                break
            # Skip empty batches
            if len(img_tensors) == 0 or len(text_strings) == 0:
                continue
            loss = self.train_step(img_tensors, text_strings, loss_fn)
            if step % 10 == 0:
                logger.info(f"Step {step}, Loss: {loss:.4f}")
            if step % settings.CHECKPOINT_INTERVAL == 0 and step > 0:
                # we evaluate the model at this step
                best_recall_t2i, best_recall_i2t = self.periodic_checkpoint_evaluation(
                    step, best_recall_t2i, best_recall_i2t
                )

        # Save final checkpoint after training completes
        if step >= 0:
            # we also evaluate the model at the final step
            self.periodic_checkpoint_evaluation(step, best_recall_t2i, best_recall_i2t)
        logger.info("Training completed!")

    def save_checkpoint(
        self, checkpoint_name: str, additional_info: Optional[dict] = None
    ):
        """Save fine-tuned model checkpoint
        Args:
            checkpoint_name: Name of the checkpoint
            additional_info: Optional dictionary with additional info (epoch, loss, etc.)
        """
        checkpoint = {
            "model_state_dict": self.clip.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        checkpoint_dir = Path(settings.CHECKPOINT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if additional_info:
            checkpoint.update(additional_info)
        torch.save(checkpoint, checkpoint_dir / checkpoint_name)
        logger.info(f"Checkpoint saved to {checkpoint_dir / checkpoint_name}")

    def periodic_checkpoint_evaluation(
        self, step: int, best_recall_t2i: float, best_recall_i2t: float
    ) -> Tuple[float, float]:
        # we evaluate the model and save the best checkpoint when both recall@10 are improved
        evaluator = CLIPEvaluator(self.clip)
        result = evaluator.evaluate_with_loader()
        print_results(result)
        result_name = f"eval_{settings.MODEL_CHOSEN.name}_step_{step}.json"
        save_results(result, result_name)
        logger.info(f"Evaluation results saved to {result_name}")
        result_t2i = result["recall@10"]["text_to_image"]
        result_i2t = result["recall@10"]["image_to_text"]
        if result_t2i > best_recall_t2i and result_i2t > best_recall_i2t:
            best_recall_t2i = result_t2i
            best_recall_i2t = result_i2t
            logger.info(
                f"New best recall@10: {best_recall_t2i:.4f} (text-to-image) and {best_recall_i2t:.4f} (image-to-text)"
            )
            checkpoint_name = f"ck_{settings.MODEL_CHOSEN.name}_step_{step}.pt"
            self.save_checkpoint(checkpoint_name)
            logger.info(f"Periodic checkpoint saved at step {step}")
        else:
            logger.info(
                f"Recall@10: {result_t2i:.4f} (text-to-image) and {result_i2t:.4f} (image-to-text) is not better than best recall@10: {best_recall_t2i:.4f} (text-to-image) and {best_recall_i2t:.4f} (image-to-text)"
            )
        return best_recall_t2i, best_recall_i2t


def training_prep():
    logger.info("Initializing CLIP manager and fine-tuner...")
    logger.info(f"Model: {settings.MODEL_CHOSEN}")
    logger.info(f"Loss function: {settings.LOSS_FUNCTION}")
    clip = OpenClipManagment()
    clip_dataset = ClipDataset(settings.TRAIN_DATASET_PATTERN)
    finetuner = CLIPFineTuner(clip, clip_dataset, learning_rate=settings.LEARNING_RATE)
    # Log GPU optimizations
    device_info = f"Device: {finetuner.device}"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        device_info += f" | GPU: {gpu_name} ({gpu_memory:.1f} GB)"
    logger.info(
        f"Training configuration: {device_info} | Batch Size: {settings.BATCH_SIZE} | Workers: {settings.NUM_WORKERS} | Mixed Precision: Enabled"
    )
    logger.info(f"Loading dataset from {settings.TRAIN_DATASET_PATTERN}")
    loader = finetuner.clip.get_loader(clip_dataset)

    max_steps_msg = (
        f" (limited to {settings.MAX_STEPS} steps)" if settings.MAX_STEPS else ""
    )
    logger.info(f"Starting training{max_steps_msg}...")

    return finetuner, loader


if __name__ == "__main__":
    try:
        finetuner, loader = training_prep()
        finetuner.train(loader)
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise
