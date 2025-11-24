import webdataset as wds
import settings
from typing import Tuple, List
from PIL import Image
from typing import Callable

record_pair = Tuple[Image.Image, str]


def _identity(x):
    """Identity function for text (keeps strings as-is). For multiprocessing compatibility, we cannot use a lambda function."""
    return x


class ClipDataset:
    """Wrapper around WebDataset that returns tuples (image, text) when iterated directly"""

    def __init__(
        self,
        dataset_pattern: str = settings.VALID_DATASET_PATTERN,
        shardshuffle: bool = False,
    ):
        self._pattern = dataset_pattern
        self._shardshuffle = shardshuffle
        self._dataset = (
            wds.WebDataset(dataset_pattern, shardshuffle=shardshuffle)
            .decode(wds.autodecode.imagehandler("pil"), wds.autodecode.basichandlers)
            .to_tuple("jpg", "txt")
        )

    def __iter__(self):
        """Allow direct iteration over the dataset - returns tuples (image, text)"""
        return iter(self._dataset)

    def get_first_n_samples(self, n: int = 10) -> List[Tuple[Image.Image, str]]:
        """Get a list of n samples from the dataset"""
        return [sample_tuple for i, sample_tuple in enumerate(self._dataset) if i < n]

    def get_loader_with_strings(
        self,
        batch_size: int = settings.BATCH_SIZE,
        num_workers: int = settings.NUM_WORKERS,
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

        # Only transform images; keep text as strings for batch tokenization
        return wds.WebLoader(
            self._dataset.map_tuple(img_transform, _identity),  # Identity for text
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
        )
