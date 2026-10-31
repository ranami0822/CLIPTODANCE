"""検出器(Mask R-CNN、COCO)による第二の候補源。CLIP-CLOP の patch maker(PixelLib の Mask R-CNN)と同じ種類のモデル。

SAM の「色の塊」と違い、人・動物・ボートなど意味のある物体だけが、クラス名つきで出る。COCO に木・雲・建物・手・顔だけの分類は
ない(SAM が補う)。風景画では、植え込みを動物と誤検出することがある(met-436441 で確認)ので、クラス名とスコアは
候補の手がかりであって、正解ではない。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

# 風景コラージュで使う物体のクラス(COCO)。ほかのクラス(couch・suitcase など)は、絵画ではほぼ誤検出
DEFAULT_LABELS = frozenset({"person", "horse", "cow", "sheep", "dog", "cat", "bird", "boat"})


@dataclass
class Detection:
    mask: np.ndarray    # bool (H, W)
    label: str
    score: float


class Detector(Protocol):
    name: str

    def detect(self, rgb: np.ndarray) -> list[Detection]: ...


class MaskRCNNDetector:
    """torchvision の Mask R-CNN(ResNet50-FPN v2、COCO の重み)。CPU で 1 枚約 4 秒(2048 ピクセルの絵、2026-10-31)。
    重みのライセンスは未確認。"""

    def __init__(self, score_thresh: float = 0.5, labels: frozenset[str] = DEFAULT_LABELS, device: str = "cpu"):
        import torch
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self._torch = torch
        self._model = maskrcnn_resnet50_fpn_v2(weights=weights).eval().to(device)
        self._cats = weights.meta["categories"]
        self._device = device
        self.score_thresh, self.labels = score_thresh, labels
        self.name = "torchvision-maskrcnn_resnet50_fpn_v2-coco"

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        t = self._torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self._device)
        with self._torch.no_grad():
            r = self._model([t])[0]
        out = []
        for k in range(len(r["scores"])):
            label = self._cats[int(r["labels"][k])]
            score = float(r["scores"][k])
            if score >= self.score_thresh and label in self.labels:
                out.append(Detection((r["masks"][k, 0] > 0.5).cpu().numpy(), label, score))
        return out
