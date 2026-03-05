from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]


class YoloRecognizer:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self._model_path = model_path
        self._device = device
        self._model: Any | None = None

    def load(self) -> None:
        from ultralytics import YOLO
        import torch

        # 检查GPU是否可用
        if self._device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU (CUDA) is not available. Please install CUDA or use CPU.")
        
        self._model = YOLO(self._model_path)

    def detect(self, bgr_image: Any, conf: float = 0.25) -> List[Detection]:
        if self._model is None:
            raise RuntimeError("YOLO model not loaded")
        results = self._model.predict(bgr_image, conf=conf, verbose=False, device=self._device)
        detections: List[Detection] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            xyxy = getattr(boxes, "xyxy", None)
            confs = getattr(boxes, "conf", None)
            clss = getattr(boxes, "cls", None)
            names = getattr(getattr(r, "names", {}), "get", None)
            if xyxy is None or confs is None or clss is None:
                continue
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [int(v) for v in xyxy[i].tolist()]
                c = float(confs[i].tolist())
                cls_id = int(clss[i].tolist())
                label = str(names(cls_id) if callable(names) else cls_id)
                detections.append(
                    Detection(label=label, confidence=c, box_xyxy=(x1, y1, x2, y2))
                )
        return detections

