# %%
from __future__ import annotations
import open_clip
from typing import Callable, List
from PIL import Image
import torch
from config import ProjectConfig
from dBManagement import ClipDataset
import webdataset as wds


class OpenClipManagment:
    def __init__(self, *, config: ProjectConfig):
        self.config = config
        self.model: open_clip.model.CLIP
        self.img_preprocess: Callable
        self.txt_tokenizer: Callable
        self.model_option = self.config.clip_model
        self.model, self.img_preprocess, _ = open_clip.create_model_and_transforms(
            self.model_option.name_,
            pretrained=self.model_option.pretrained_,
            cache_dir=str(self.config.model_cache_dir),
        )
        self.txt_tokenizer = open_clip.get_tokenizer(self.model_option.name_)
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)

        if self.config.should_load_checkpoint:
            self._load_checkpoint_weights()

    def _load_checkpoint_weights(self) -> None:
        if self.config.checkpoint_path is None:
            raise ValueError(
                "checkpoint_path must be set when loading from checkpoint."
            )
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        self.model.load_state_dict(state_dict)

    def view_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        return {
            "Total Parameters": total_params,
            "Trainable Parameters": trainable_params,
        }

    def get_loader(
        self, dataset: ClipDataset, *, caption_mode: str = "random"
    ) -> wds.WebLoader:
        """
        Create an optimal WebLoader that batches strings before tokenization.

        This approach is better than per-sample tokenization because:
        - CLIP tokenizer is designed for batch processing
        - No dimension manipulation needed
        - More efficient (single tokenization call per batch)

        Returns loader that yields (img_tensors, text_strings) where:
        - img_tensors: [batch, 3, 224, 224] (already tensors)
        - text_strings: List[str] (strings to be tokenized in training loop)
        """
        return dataset.get_loader_with_strings(
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            img_transform=self.img_preprocess,
            caption_mode=caption_mode,
        )

    def get_loader_with_caption_lists(
        self, dataset: ClipDataset
    ) -> wds.WebLoader:
        return dataset.get_loader_with_caption_lists(
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            img_transform=self.img_preprocess,
        )

    def encode_img_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode PIL images directly"""
        img_tokens = [self.img_preprocess(img) for img in images]
        batch = torch.stack(img_tokens).to(self.device)
        return self.model.encode_image(batch)

    def encode_txt_batch(self, texts: List[str]) -> torch.Tensor:
        text_tokens = self.txt_tokenizer(texts).to(self.device)  # shape: [batch, seq]
        return self.model.encode_text(text_tokens)
