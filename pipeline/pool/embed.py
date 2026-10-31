"""埋め込み(CLIP 系)。画像と文章を同じ空間に写す。ベクトルは長さ 1 に正規化して返す。"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from PIL import Image


class Embedder(Protocol):
    name: str
    dim: int

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


class ClipEmbedder:
    """open_clip の ViT-B-32(LAION-2B の重み。モデルカードの表記は MIT、コードも MIT。2026-10-31 確認)。"""

    def __init__(self, model: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str = "cpu"):
        import open_clip
        import torch

        self._torch = torch
        self._device = device
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model)
        self.name = f"open_clip:{model}:{pretrained}"
        self.dim = int(self._model.visual.output_dim)

    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            batch = self._torch.stack([self._preprocess(im.convert("RGB")) for im in images[i:i + batch_size]]).to(self._device)
            with self._torch.no_grad():
                out.append(self._model.encode_image(batch).float().cpu().numpy())
        return _normalize(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        tokens = self._tokenizer(list(texts)).to(self._device)
        with self._torch.no_grad():
            return _normalize(self._model.encode_text(tokens).float().cpu().numpy())
