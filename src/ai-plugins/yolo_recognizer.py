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
        import os
        import urllib3

        # 检查GPU是否可用
        if self._device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU (CUDA) is not available. Please install CUDA or use CPU.")
        
        # 禁用SSL证书验证
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        os.environ['PYTHONHTTPSVERIFY'] = '0'
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        
        # 尝试加载模型
        try:
            self._model = YOLO(self._model_path, verbose=False)
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            print("Trying to load model using torch.load...")
            try:
                import torch
                checkpoint = torch.load(self._model_path, map_location=self._device)
                self._model = YOLO(checkpoint['model'].state_dict())
                print("YOLO model loaded successfully using torch.load!")
            except Exception as e2:
                print(f"Failed to load YOLO model using torch.load: {e2}")
                raise e

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

