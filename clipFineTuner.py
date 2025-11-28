# %%
from openClipManagement import OpenClipManagment
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Callable, List
import logging
import sys
from dBManagement import ClipDataset
from config import ProjectConfig
from evaluate_clip import CLIPEvaluator, print_results, save_results
import torch.nn.functional as F
from pathlib import Path
from constants import (
    TRAIN_EVAL_SHARDS_PATTERN,
    VALID_EVAL_SHARDS_PATTERN,
    TRAIN_SHARDS_PATTERN,
)
from config import LossFunctionOptions
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# Checkpoint settings


# %%
class CLIPFineTuner:
    def __init__(
        self,
        *,
        clip_manager: OpenClipManagment,
        clip_dataset: ClipDataset,
        config: ProjectConfig,
    ):
        self.clip = clip_manager
        self.clip_dataset = clip_dataset
        self.config = config
        self.optimizer = torch.optim.AdamW(
            list(self.clip.model.parameters()) + [self.logit_bias],
            lr=self.config.training.learning_rate,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.logit_bias = nn.Parameter(torch.tensor(-np.log(1)))
        self.device = self.clip.device
        # Mixed precision scaler for T4 GPU optimization
        # Enables ~2x faster training with ~50% less memory usage
        self.scaler = GradScaler()
        self.best_eval_score: float = float("-inf")
        self.best_checkpoint_path: Path | None = None

    def contrastive_loss(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
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
        if self.config.training.normalize_embeddings_in_siglip:
            pass
            # in case somehow we still normalize, we comment it out here
            # image_embeds = F.normalize(image_embeds, dim=-1)
            # text_embeds = F.normalize(text_embeds, dim=-1)

        logits = image_embeds @ text_embeds.T * self.clip.model.logit_scale.exp()
        logits = logits + self.logit_bias
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

    def train(self):
        checkpoint_path = self.config.checkpoint_dir
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory: {checkpoint_path.absolute()}")

        if self.config.loss_function == LossFunctionOptions.CONTRASTIVE:
            loss_fn = self.contrastive_loss
        elif self.config.loss_function == LossFunctionOptions.SIGLIP:
            loss_fn = self.siglip_loss
        else:
            raise ValueError(f"Invalid loss function: {self.config.loss_function}")

        epochs = self.config.training.epochs
        max_steps = self.config.training.max_steps
        checkpoint_interval = self.config.training.checkpoint_interval

        global_step = -1
        stop_training = False

        for epoch in range(epochs):
            logger.info("=" * 50)
            logger.info(f"Epoch {epoch + 1} of {epochs}")
            logger.info("=" * 50)

            epoch_loader = self.clip.get_loader(self.clip_dataset)

            for batch_idx, (img_tensors, text_strings) in enumerate(epoch_loader):
                global_step += 1

                if max_steps is not None and global_step >= max_steps:
                    logger.info(
                        f"Reached maximum step limit ({max_steps}). Stopping training."
                    )
                    stop_training = True
                    break

                # Skip empty batches
                if len(img_tensors) == 0 or len(text_strings) == 0:
                    continue

                loss = self.train_step(img_tensors, text_strings, loss_fn)

                if global_step % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}, Batch {batch_idx}, Global Step {global_step}, Loss: {loss:.4f}"
                    )

                if (
                    checkpoint_interval
                    and global_step > 0
                    and global_step % checkpoint_interval == 0
                ):
                    self.periodic_checkpoint_evaluation(global_step)

            if stop_training:
                break

        # Save final checkpoint after training completes
        if global_step >= 0:
            self.periodic_checkpoint_evaluation(global_step)
        logger.info("Training completed!")

    def save_checkpoint(
        self, checkpoint_name: str, additional_info: dict | None = None
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
        checkpoint_dir = self.config.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if additional_info:
            checkpoint.update(additional_info)
        torch.save(checkpoint, checkpoint_dir / checkpoint_name)
        logger.info(f"Checkpoint saved to {checkpoint_dir / checkpoint_name}")

    def periodic_checkpoint_evaluation(self, step: int) -> None:
        logger.info(f"Running evaluation at global step {step}...")
        split_patterns = {
            "train": TRAIN_EVAL_SHARDS_PATTERN,
            "valid": VALID_EVAL_SHARDS_PATTERN,
        }

        evaluation_results = {}

        for split, pattern in split_patterns.items():
            logger.info(f"Evaluating {split} subset: {pattern}")
            dataset = ClipDataset(
                config=self.config,
                dataset_pattern=self.config.dataset_dir / pattern,
                # we don't shuffle the evaluation dataset
                shardshuffle=False,
            )
            evaluator = CLIPEvaluator(
                config=self.config,
                clip_manager=self.clip,
                evaluation_dataset=dataset,
            )
            split_result = evaluator.evaluate_with_loader()
            evaluation_results[split] = split_result
            print(f"\n*** {split.upper()} METRICS ***")
            print_results(split_result)

        results_payload = {
            "step": step,
            "loss_function": self.config.loss_function.name,
            "splits": evaluation_results,
        }
        result_name = f"eval_{self.config.clip_model.name_}_step_{step}.json"
        save_results(self.config, results_payload, result_name)
        logger.info(f"Evaluation results saved to {result_name}")

        valid_results = evaluation_results["valid"]
        eval_recall_t2i = valid_results["recall@10"]["text_to_image"]
        eval_recall_i2t = valid_results["recall@10"]["image_to_text"]
        eval_score = (eval_recall_t2i + eval_recall_i2t) / 2.0

        if eval_score > self.best_eval_score:
            logger.info(
                f"Validation recall@10 average improved from {self.best_eval_score:.4f} to {eval_score:.4f}. Saving checkpoint."
            )
            if self.best_checkpoint_path and self.best_checkpoint_path.exists():
                logger.info(
                    f"Removing previous best checkpoint: {self.best_checkpoint_path}"
                )
                self.best_checkpoint_path.unlink()
            checkpoint_name = f"ck_{self.config.clip_model.name_}_step_{step}.pt"
            self.save_checkpoint(
                checkpoint_name,
                additional_info={
                    "step": step,
                    "eval_recall@10_text_to_image": eval_recall_t2i,
                    "eval_recall@10_image_to_text": eval_recall_i2t,
                    "eval_loss": valid_results["loss"],
                    "eval_loss_type": valid_results["loss_type"],
                },
            )
            self.best_checkpoint_path = self.config.checkpoint_dir / checkpoint_name
            self.best_eval_score = eval_score
        else:
            logger.info(
                f"Validation recall@10 average {eval_score:.4f} did not exceed best {self.best_eval_score:.4f}; checkpoint not updated."
            )


def training_prep(config: ProjectConfig):
    logger.info("Initializing CLIP manager and fine-tuner...")
    logger.info(f"Model: {config.clip_model.name_} ({config.clip_model.pretrained_})")
    logger.info(f"Loss function: {config.loss_function.name}")
    clip = OpenClipManagment(config=config)
    clip_dataset = ClipDataset(
        config=config,
        dataset_pattern=config.dataset_dir / TRAIN_SHARDS_PATTERN,
        shardshuffle=config.shardshuffle,
    )
    finetuner = CLIPFineTuner(
        clip_manager=clip,
        clip_dataset=clip_dataset,
        config=config,
    )
    # Log GPU optimizations
    device_info = f"Device: {finetuner.device}"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        device_info += f" | GPU: {gpu_name} ({gpu_memory:.1f} GB)"
    logger.info(
        f"Training configuration: {device_info} | Batch Size: {config.training.batch_size} | Workers: {config.training.num_workers} | Mixed Precision: Enabled"
    )
    logger.info(f"Directories used:")
    logger.info(f"  - Dataset: {config.dataset_dir / TRAIN_SHARDS_PATTERN}")
    logger.info(f"  - Checkpoints: {config.checkpoint_dir}")
    logger.info(f"  - Evaluations: {config.evaluation_dir}")
    logger.info(f"  - Model Cache: {config.model_cache_dir}")
    logger.info(f"Loading dataset from {config.dataset_dir / TRAIN_SHARDS_PATTERN}")
    logger.info(
        f" (limited to {config.training.max_steps} steps)"
        if config.training.max_steps
        else ""
    )
    logger.info(f"Ready for fine-tuning. Please call .train() method....")

    return finetuner
