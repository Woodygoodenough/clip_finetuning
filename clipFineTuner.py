# %%
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
        """Compute CLIP contrastive loss"""
        image_embeds = self.clip.normalize_tensor(image_embeds)
        text_embeds = self.clip.normalize_tensor(text_embeds)

        logits = (image_embeds @ text_embeds.T) * self.clip.model.logit_scale.exp()
        labels = torch.arange(len(logits), device=logits.device)

        loss_i = self.loss_fn(logits, labels)
        loss_t = self.loss_fn(logits.T, labels)
        return (loss_i + loss_t) / 2

    def train_step(self, img_tensors: torch.Tensor, text_strings: List[str]) -> float:
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
            loss = self.contrastive_loss(img_embeds, txt_embeds)

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
        for step, (img_tensors, text_strings) in enumerate(loader):
            if settings.MAX_STEPS is not None and step >= settings.MAX_STEPS:
                logger.info(
                    f"Reached maximum step limit ({settings.MAX_STEPS}). Stopping training."
                )
                break
            # Skip empty batches
            if len(img_tensors) == 0 or len(text_strings) == 0:
                continue
            loss = self.train_step(img_tensors, text_strings)
            if step % 10 == 0:
                logger.info(f"Step {step}, Loss: {loss:.4f}")
            if step % settings.CHECKPOINT_INTERVAL == 0:
                checkpoint_file = checkpoint_path / f"checkpoint_step_{step}.pt"
                self.save_checkpoint(
                    str(checkpoint_file), additional_info={"step": step, "loss": loss}
                )
                logger.info(f"Periodic checkpoint saved at step {step}")

        # Save final checkpoint after training completes
        if step >= 0:  # Only save if at least one step was processed
            final_checkpoint_file = checkpoint_path / "final_checkpoint.pt"
            self.save_checkpoint(
                str(final_checkpoint_file),
                additional_info={"step": step, "loss": loss, "final": True},
            )
            logger.info(f"Final checkpoint saved at step {step}")
        logger.info("Training completed!")

    def save_checkpoint(self, path: str, additional_info: Optional[dict] = None):
        """Save fine-tuned model checkpoint
        Args:
            path: Path to save checkpoint
            additional_info: Optional dictionary with additional info (epoch, loss, etc.)
        """
        checkpoint = {
            "model_state_dict": self.clip.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        if additional_info:
            checkpoint.update(additional_info)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")


def training_prep():
    logger.info("Initializing CLIP manager and fine-tuner...")
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
