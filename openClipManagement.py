# %%
from __future__ import annotations
import open_clip
from typing import Callable, List, Union
from pathlib import Path
from PIL import Image
import torch
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
import settings


# %%
@dataclass
class Record:
    image_path: str
    image_name: str
    caption: str
    category: str
    product_id: int

    @classmethod
    def from_series(cls, series: pd.Series) -> Record:
        return cls(**series.to_dict())


@dataclass
class Records:
    records: List[Record]

    @classmethod
    def from_json(cls, path: str) -> Records:
        return cls(
            records=[
                Record.from_series(series)
                for _, series in pd.read_json(path).iterrows()
            ]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Records(records=self.records[index])
        return self.records[index]

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([record.__dict__ for record in self.records])

    def get_image_paths(self) -> List[Union[str, Path]]:
        return [record.image_path for record in self.records]


class OpenClipManagment:
    def __init__(self):
        self.model: open_clip.model.CLIP
        self.preprocess: Callable
        self.tokenizer: Callable
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            settings.MODEL_NAME, pretrained=settings.MODEL_PRETRAINED, cache_dir=settings.MODEL_CACHE_DIR
        )
        self.tokenizer = open_clip.get_tokenizer(settings.MODEL_NAME)
        self.device = torch.device(settings.DEVICE)
        self.model = self.model.to(self.device)

    def view_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        return {
            "Total Parameters": total_params,
            "Trainable Parameters": trainable_params,
        }

    def encode_image_from_pil(self, images: List[Image.Image]):
        """Encode PIL images directly"""
        img_tokens = []
        for img in images:
            img_token = img.convert("RGB") if hasattr(img, 'convert') else img
            img_tokens.append(self.preprocess(img_token))
        batch = torch.stack(img_tokens).to(self.device)
        return self.model.encode_image(batch)

    def encode_image(self, images: List[Union[str, Path]]):
        """Encode images from image paths or PIL images"""
        #get PIL images from image paths
        pil_images = [Image.open(img) for img in images]
        return self.encode_image_from_pil(pil_images)

    def encode_text(self, texts: List[str]):
        text_tokens = self.tokenizer(texts).to(self.device)  # shape: [batch, seq]
        return self.model.encode_text(text_tokens)

    def normalize_tensor(self, tensor: torch.Tensor):
        return tensor / tensor.norm(dim=-1, keepdim=True)

    def compare_similarity(self, text: str, image_path: str):
        with torch.no_grad():
            text_tensor = self.normalize_tensor(self.encode_text([text]))
            image_tensor = self.normalize_tensor(self.encode_image([image_path]))
            similarity = (text_tensor @ image_tensor.T).item()
            return similarity

    def text_image_retrieval(self, query: str, dataset: Records, top_k: int = 5):
        """Retrieve top-k most similar images for a text query"""
        with torch.no_grad():
            text_tensor = self.normalize_tensor(self.encode_text([query]))
            image_tensors = self.normalize_tensor(
                self.encode_image(dataset.get_image_paths())
            )
            similarities = (text_tensor @ image_tensors.T).squeeze(0)
            top_indices = similarities.argsort(descending=True)[:top_k]
            results = [
                (dataset[i].image_path, similarities[i].item()) for i in top_indices
            ]
            return results


# %%
# small demo
if __name__ == "__main__":
    ds = Records.from_json("clip_dataset_valid.json")
    oc = OpenClipManagment()
    results = oc.text_image_retrieval("black pants", ds[:50])
    ## show the images
    for image_path, similarity in results:
        image = Image.open(image_path)
        plt.imshow(image)
        plt.title(f"Similarity: {similarity:.2f}")
        plt.show()
