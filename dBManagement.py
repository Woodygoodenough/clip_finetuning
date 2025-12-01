import webdataset as wds
from typing import Tuple, List
from PIL import Image
from typing import Callable
from pathlib import Path
from config import ProjectConfig
import random
from constants import (
    IMG_KEY,
    PRIMARY_CAPTION_KEY,
    AUGMENTED_CAPTION_KEY,
    AUGMENTED_CAPTION_KEY_2,
)


class ClipDataset:
    """Wrapper around WebDataset that returns tuples (image, text) when iterated directly"""

    def __init__(
        self,
        *,
        config: ProjectConfig,
        dataset_pattern: str | Path,
        shardshuffle: bool,
    ):
        self.config = config
        self._pattern = str(dataset_pattern)

        base_kwargs = dict(shardshuffle=shardshuffle)
        decoder = dict(
            decode=wds.autodecode.imagehandler("pil"),
            handler=wds.autodecode.basichandlers,
        )

        self._dataset_primary = (
            wds.WebDataset(self._pattern, **base_kwargs)
            .decode(decoder["decode"], decoder["handler"])
            .to_tuple(IMG_KEY, PRIMARY_CAPTION_KEY)
        )
        self._dataset_multi = (
            wds.WebDataset(self._pattern, **base_kwargs)
            .decode(decoder["decode"], decoder["handler"])
            .to_tuple(
                IMG_KEY,
                PRIMARY_CAPTION_KEY,
                AUGMENTED_CAPTION_KEY,
                AUGMENTED_CAPTION_KEY_2,
            )
        )

    def __iter__(self):
        """Allow direct iteration over the dataset - returns tuples (image, text)"""
        return iter(self._dataset_primary)

    def get_first_n_samples(self, *, n: int = 10) -> List[Tuple[Image.Image, str]]:
        """Get a list of n samples from the dataset"""
        return [
            sample_tuple
            for i, sample_tuple in enumerate(self._dataset_primary)
            if i < n
        ]

    def get_loader_with_strings(
        self,
        batch_size: int | None = None,
        num_workers: int | None = None,
        img_transform: Callable = None,
        caption_mode: str = "random",
    ) -> wds.WebLoader:
        """
        Get WebLoader that transforms images but keeps text as strings.

        This is optimal because:
        - Images are transformed per-sample (PIL -> tensor)
        - Text stays as strings and is batched naturally
        - Text tokenization happens on the batch (more efficient)

        Returns:
            WebLoader yielding (img_tensors, text_strings) where:
            - img_tensors: [batch, 3, 224, 224]
            - text_strings: List[str] of length batch_size
        """
        if img_transform is None:
            raise ValueError(
                "img_transform must be provided to convert PIL images to tensors."
            )
        if caption_mode not in {"random", "primary"}:
            raise ValueError(
                f"caption_mode must be 'random' or 'primary', got {caption_mode!r}"
            )

        batch_size = batch_size or self.config.training.batch_size
        num_workers = num_workers or self.config.training.num_workers

        def _transform_img_and_pick_caption(sample):
            img, *caps = sample
            texts = [
                cap.decode("utf-8") if isinstance(cap, bytes) else cap
                for cap in caps
                if cap
            ]
            if not texts:
                texts = [""]
            text = random.choice(texts)
            return img_transform(img), text

        def _transform_img_and_primary(sample):
            img, text = sample
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            return img_transform(img), text

        # Only transform images; keep text as strings for batch tokenization
        if caption_mode == "random":
            dataset = self._dataset_multi.map(_transform_img_and_pick_caption)
        else:
            dataset = self._dataset_primary.map(_transform_img_and_primary)

        return wds.WebLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
        )

    def get_loader_with_caption_lists(
        self,
        batch_size: int | None = None,
        num_workers: int | None = None,
        img_transform: Callable | None = None,
    ) -> wds.WebLoader:
        """
        Return loader yielding (img_tensor, list[str]) per sample.
        """
        if img_transform is None:
            raise ValueError(
                "img_transform must be provided to convert PIL images to tensors."
            )

        batch_size = batch_size or self.config.training.batch_size
        num_workers = num_workers or self.config.training.num_workers

        def _transform_img_and_list(sample):
            img, *caps = sample
            texts = []
            for cap in caps:
                if not cap:
                    continue
                text = cap.decode("utf-8") if isinstance(cap, bytes) else cap
                if text:
                    texts.append(text)
            if not texts:
                texts.append("")
            return img_transform(img), texts

        return wds.WebLoader(
            self._dataset_multi.map(_transform_img_and_list),
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
        )
