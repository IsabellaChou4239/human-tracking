"""Ultralytics adapters. Importing this module does not load models."""
from __future__ import annotations

import time
from typing import Any


class PersonDetector:
    def __init__(self, model_path: str) -> None:
        import torch
        from ultralytics import YOLO
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)

    def detect(self, image: Any) -> tuple[list[dict], dict, float]:
        start = time.perf_counter()
        result = self.model.predict(source=image, device=self.device, imgsz=640, conf=0.4, classes=[0], verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({"confidence": float(box.conf[0]), "box": (x1, y1, x2, y2),
                               "center": ((x1 + x2) // 2, (y1 + y2) // 2)})
        return detections, result.speed, (time.perf_counter() - start) * 1000


class PoseEstimator:
    def __init__(self, model_path: str) -> None:
        import torch
        from ultralytics import YOLO
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)

    def estimate(self, image: Any, box: tuple[int, int, int, int]) -> tuple[list[dict] | None, dict | None, float]:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        mx, my = int((x2 - x1) * .15), int((y2 - y1) * .10)
        left, top, right, bottom = max(0, x1 - mx), max(0, y1 - my), min(width, x2 + mx), min(height, y2 + my)
        crop = image[top:bottom, left:right]
        if crop.size == 0: return None, None, 0.0
        start = time.perf_counter(); result = self.model.predict(source=crop, device=self.device, imgsz=640, conf=.25, verbose=False)[0]
        elapsed = (time.perf_counter() - start) * 1000
        if result.keypoints is None or len(result.keypoints) == 0: return None, result.speed, elapsed
        xy, confidence = result.keypoints.xy[0], result.keypoints.conf[0] if result.keypoints.conf is not None else None
        points = [{"id": i, "x": int(point[0]) + left, "y": int(point[1]) + top,
                   "confidence": float(confidence[i]) if confidence is not None else 1.0}
                  for i, point in enumerate(xy)]
        return points, result.speed, elapsed
