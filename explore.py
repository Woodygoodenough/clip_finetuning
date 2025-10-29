#%%
from __future__ import annotations
import open_clip
from typing import Callable, List, Union
from pathlib import Path
from PIL import Image
import torch
import pandas as pd
from dataclasses import dataclass

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
    def from_dataframe(cls, dataframe: pd.DataFrame) -> Records:
        return cls(records=[Record.from_series(series) for _, series in dataframe.iterrows()])

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, index: int) -> Record:
        return self.records[index]

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([record.__dict__ for record in self.records])


class Dataset:
    def __init__(self, records: Records):
        self.records = records

    @classmethod
    def from_csv(cls, path: str) -> Dataset:
        return cls(Records.from_csv(path))

    


class OpenClipManagment():
    def __init__(self):
        self.model: open_clip.model.CLIP
        self.preprocess: Callable
        self.tokenizer: Callable
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir="./openclip_cache")
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def view_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"Total Parameters": total_params, "Trainable Parameters": trainable_params}
    
    def encode_image(self, images: List[Union[str, Path]]):
        img_tokens= []
        for img in images:
            img_token = Image.open(img).convert("RGB")
            img_tokens.append(self.preprocess(img_token))
        batch = torch.stack(img_tokens).to(self.device)  # shape: [batch, 3, 224, 224]
        return self.model.encode_image(batch)

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
    
    def text_image_retrieval(self, query: str, image_paths: List[Union[str, Path]], top_k: int = 5):
        """Retrieve top-k most similar images for a text query"""
        with torch.no_grad():
            text_tensor = self.normalize_tensor(self.encode_text([query]))
            image_tensors = self.normalize_tensor(self.encode_image(image_paths))
            similarities = (text_tensor @ image_tensors.T).squeeze(0)
            top_indices = similarities.argsort(descending=True)[:top_k]
            results = [(image_paths[i], similarities[i].item()) for i in top_indices]
            return results
    
