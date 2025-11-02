# %%
from openClipManagement import OpenClipManagment
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler  # Mixed precision for T4 GPU
from typing import List, Tuple, Optional
import settings
import webdataset as wds
import open_clip
from PIL import Image
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Checkpoint settings

# %%
class CLIPFineTuner:
    def __init__(self, clip_manager: OpenClipManagment, learning_rate: float = settings.LEARNING_RATE):
        self.clip = clip_manager
        self.optimizer = torch.optim.AdamW(
            self.clip.model.parameters(), lr=learning_rate
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = self.clip.device
        # Mixed precision scaler for T4 GPU optimization
        # Enables ~2x faster training with ~50% less memory usage
        self.scaler = GradScaler()

    def contrastive_loss(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Compute CLIP contrastive loss"""
        image_embeds = self.clip.normalize_tensor(image_embeds)
        text_embeds = self.clip.normalize_tensor(text_embeds)

        logits = (image_embeds @ text_embeds.T) * self.clip.model.logit_scale.exp()
        labels = torch.arange(len(logits), device=logits.device)

        loss_i = self.loss_fn(logits, labels)
        loss_t = self.loss_fn(logits.T, labels)
        return (loss_i + loss_t) / 2

    def train_step(self, image_tensors: torch.Tensor, captions: List[str]) -> float:
        """Single training step with mixed precision for T4 GPU optimization
        Args:
            image_tensors: Preprocessed image tensor batch [batch_size, channels, height, width]
            captions: List of text captions
        Returns:
            Loss value as float
        """
        self.clip.model.train()
        self.optimizer.zero_grad()

        # Ensure tensors are on correct device
        image_tensors = image_tensors.to(self.device)
        
        # Use mixed precision for faster training and less memory usage
        # This enables automatic mixed precision (FP16) which is ~2x faster on T4
        with autocast():
            image_embeds = self.clip.model.encode_image(image_tensors)
            text_embeds = self.clip.encode_text(captions)
            loss = self.contrastive_loss(image_embeds, text_embeds)
        
        # Scale loss and backward pass for mixed precision
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def save_checkpoint(self, path: str, additional_info: Optional[dict] = None):
        """Save fine-tuned model checkpoint
        Args:
            path: Path to save checkpoint
            additional_info: Optional dictionary with additional info (epoch, loss, etc.)
        """
        checkpoint = {
            'model_state_dict': self.clip.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),  # Save scaler state for mixed precision
        }
        if additional_info:
            checkpoint.update(additional_info)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.clip.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logger.info(f"Checkpoint loaded from {path}")


# %%
# Dataset loading utilities

# Cache preprocess function for worker processes
_preprocess_cache = None

def get_preprocess():
    """Get or create preprocess function (cached for worker processes)"""
    global _preprocess_cache
    if _preprocess_cache is None:
        _, _preprocess_cache, _ = open_clip.create_model_and_transforms(
            settings.MODEL_NAME, pretrained=settings.MODEL_PRETRAINED, cache_dir=settings.MODEL_CACHE_DIR
        )
    return _preprocess_cache

def process_image(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor using CLIP preprocessing
    Args:
        img: PIL Image
    Returns:
        Preprocessed tensor
    """
    preprocess = get_preprocess()
    img_rgb = img.convert("RGB")
    return preprocess(img_rgb)

def process_text(txt) -> str:
    """Process text from bytes or string
    Args:
        txt: Bytes or string
    Returns:
        String
    """
    if isinstance(txt, bytes):
        return txt.decode('utf-8')
    return txt

def load_webdataset(dataset_pattern: str, batch_size: int = settings.BATCH_SIZE, 
                   num_workers: int = settings.NUM_WORKERS) -> wds.WebLoader:
    """Load WebDataset for training
    Args:
        dataset_pattern: Pattern for dataset shards (e.g., "path/to/shards.{000000..000010}.tar")
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
    Returns:
        WebLoader instance
    """
    dataset = (
        wds.WebDataset(dataset_pattern, empty_check=False)
        .decode("pil")
        .to_tuple("jpg", "txt")
        .map_tuple(process_image, process_text)
    )
    return wds.WebLoader(dataset, batch_size=batch_size, num_workers=num_workers)


if __name__ == '__main__':
    # Create checkpoint directory
    checkpoint_path = Path(settings.CHECKPOINT_DIR)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_path.absolute()}")
    
    try:
        # Initialize components
        logger.info("Initializing CLIP manager and fine-tuner...")
        clip_manager = OpenClipManagment()
        finetuner = CLIPFineTuner(clip_manager)
        
        # Log GPU optimizations
        device_info = f"Device: {finetuner.device}"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            device_info += f" | GPU: {gpu_name} ({gpu_memory:.1f} GB)"
        logger.info(f"Training configuration: {device_info} | Batch Size: {settings.BATCH_SIZE} | Workers: {settings.NUM_WORKERS} | Mixed Precision: Enabled")
        
        # Load dataset
        logger.info(f"Loading dataset from {settings.TRAIN_DATASET_PATTERN}")
        loader = load_webdataset(settings.TRAIN_DATASET_PATTERN)
        
        # Training loop
        max_steps_msg = f" (limited to {settings.MAX_STEPS} steps)" if settings.MAX_STEPS else ""
        logger.info(f"Starting training{max_steps_msg}...")
        step = 0
        
        for batch in loader:
            # Check if we've reached the step limit
            if settings.MAX_STEPS is not None and step >= settings.MAX_STEPS:
                logger.info(f"Reached maximum step limit ({settings.MAX_STEPS}). Stopping training.")
                break
            
            images, captions = batch[0], batch[1]
            
            # Skip empty batches
            if len(images) == 0 or len(captions) == 0:
                continue
            
            loss = finetuner.train_step(images, captions)
            step += 1
            
            # Logging
            if step % 10 == 0:
                logger.info(f"Step {step}, Loss: {loss:.4f}")
            else:
                print(f"Step {step}, Loss: {loss:.4f}")
            
            # Save periodic checkpoint
            if step % settings.CHECKPOINT_INTERVAL == 0:
                checkpoint_file = checkpoint_path / f"checkpoint_step_{step}.pt"
                finetuner.save_checkpoint(
                    str(checkpoint_file),
                    additional_info={'step': step, 'loss': loss}
                )
                logger.info(f"Periodic checkpoint saved at step {step}")
            
        
        # Save final checkpoint
        final_checkpoint_file = checkpoint_path / "final_checkpoint.pt"
        finetuner.save_checkpoint(
            str(final_checkpoint_file),
            additional_info={'step': step, 'loss': loss, 'final': True}
        )
        logger.info(f"Final checkpoint saved at step {step}")
        logger.info("Training completed!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save checkpoint before exiting
        if 'step' in locals() and step > 0 and 'finetuner' in locals():
            interrupt_checkpoint_file = checkpoint_path / f"interrupted_step_{step}.pt"
            finetuner.save_checkpoint(
                str(interrupt_checkpoint_file),
                additional_info={'step': step, 'loss': loss if 'loss' in locals() else None, 'interrupted': True}
            )
            logger.info(f"Checkpoint saved before interruption at step {step}")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise
