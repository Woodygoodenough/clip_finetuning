# %%
from __future__ import annotations
import open_clip
from typing import Callable, List
from PIL import Image
import torch
import settings
from dBManagement import ClipDataset
import webdataset as wds


class OpenClipManagment:
    def __init__(self):
        self.model: open_clip.model.CLIP
        self.img_preprocess: Callable
        self.txt_tokenizer: Callable
        self.model, self.img_preprocess, _ = open_clip.create_model_and_transforms(
            settings.MODEL_NAME,
            pretrained=settings.MODEL_PRETRAINED,
            cache_dir=settings.MODEL_CACHE_DIR,
        )
        self.txt_tokenizer = open_clip.get_tokenizer(settings.MODEL_NAME)
        self.device = torch.device(settings.DEVICE)
        self.model = self.model.to(self.device)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device(settings.DEVICE)
        )
        model = cls()
        model.model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def view_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        return {
            "Total Parameters": total_params,
            "Trainable Parameters": trainable_params,
        }

    def get_loader(self, dataset: ClipDataset) -> wds.WebLoader:
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
            batch_size=settings.BATCH_SIZE,
            num_workers=settings.NUM_WORKERS,
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

    def normalize_tensor(self, tensor: torch.Tensor):
        return tensor / tensor.norm(dim=-1, keepdim=True)
