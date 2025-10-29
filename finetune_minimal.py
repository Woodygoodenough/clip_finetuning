# %%
from explore import OpenClipManagment
import torch
import torch.nn as nn
from typing import List


# %%
class CLIPFineTuner:
    def __init__(self, clip_manager: OpenClipManagment, learning_rate: float = 1e-5):
        self.clip = clip_manager
        self.optimizer = torch.optim.AdamW(
            self.clip.model.parameters(), lr=learning_rate
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def contrastive_loss(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor):
        """Compute CLIP contrastive loss"""
        image_embeds = self.clip.normalize_tensor(image_embeds)
        text_embeds = self.clip.normalize_tensor(text_embeds)

        logits = (image_embeds @ text_embeds.T) * self.clip.model.logit_scale.exp()
        labels = torch.arange(len(logits), device=logits.device)

        loss_i = self.loss_fn(logits, labels)
        loss_t = self.loss_fn(logits.T, labels)
        return (loss_i + loss_t) / 2

    def train_step(self, image_paths: List[str], captions: List[str]):
        """Single training step"""
        self.clip.model.train()
        self.optimizer.zero_grad()

        image_embeds = self.clip.encode_image(image_paths)
        text_embeds = self.clip.encode_text(captions)

        loss = self.contrastive_loss(image_embeds, text_embeds)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_checkpoint(self, path: str):
        """Save fine-tuned model"""
        torch.save(self.clip.model.state_dict(), path)


# %%
# Example usage:
# clip_manager = OpenClipManagment()
# finetuner = CLIPFineTuner(clip_manager)
#
# # Training loop
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         loss = finetuner.train_step(batch['images'], batch['captions'])
#         print(f"Loss: {loss:.4f}")
