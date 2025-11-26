import webdataset as wds
from typing import Tuple, List
from PIL import Image
from typing import Callable
from pathlib import Path
from config import ProjectConfig
from constants import TRAIN_SHARDS_PATTERN, VALID_SHARDS_PATTERN, TEST_SHARDS_PATTERN

record_pair = Tuple[Image.Image, str]


def _identity(x):
    """Identity function for text (keeps strings as-is). For multiprocessing compatibility, we cannot use a lambda function."""
    return x


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
        self._dataset = (
            wds.WebDataset(self._pattern, shardshuffle=shardshuffle)
            .decode(wds.autodecode.imagehandler("pil"), wds.autodecode.basichandlers)
            .to_tuple("jpg", "txt")
        )

    def __iter__(self):
        """Allow direct iteration over the dataset - returns tuples (image, text)"""
        return iter(self._dataset)

    def get_first_n_samples(self, *, n: int = 10) -> List[Tuple[Image.Image, str]]:
        """Get a list of n samples from the dataset"""
        return [sample_tuple for i, sample_tuple in enumerate(self._dataset) if i < n]

    def get_loader_with_strings(
        self,
        batch_size: int | None = None,
        num_workers: int | None = None,
        img_transform: Callable = None,
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

        batch_size = batch_size or self.config.training.batch_size
        num_workers = num_workers or self.config.training.num_workers

        # Only transform images; keep text as strings for batch tokenization
        return wds.WebLoader(
            self._dataset.map_tuple(img_transform, _identity),  # Identity for text
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
        )
