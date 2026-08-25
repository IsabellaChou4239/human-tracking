"""Select one operator and bridge short detector dropouts with a simple motion model."""
from __future__ import annotations

import math


class OperatorTracker:
    def __init__(self, max_lost_frames: int = 90, max_predict_frames: int = 15, max_match_distance: float = 250) -> None:
        self.max_lost_frames, self.max_predict_frames, self.max_match_distance = max_lost_frames, max_predict_frames, max_match_distance
        self.last: dict | None = None; self.velocity = (0.0, 0.0); self.lost_frames = 0

    def update(self, detections: list[dict]) -> dict | None:
        if self.last is None:
            if not detections: return None
            self.last = max(detections, key=lambda item: item["confidence"]).copy(); self.last.update(status="DETECTED", lost_frames=0)
            return self.last.copy()
        previous = self.last["center"]; predicted = (previous[0] + self.velocity[0], previous[1] + self.velocity[1])
        candidate = min(detections, key=lambda item: math.dist(item["center"], predicted), default=None)
        if candidate is not None and (math.dist(candidate["center"], predicted) <= self.max_match_distance or (self.lost_frames > self.max_predict_frames and len(detections) == 1)):
            self.velocity = (0.9 * (candidate["center"][0] - previous[0]), 0.9 * (candidate["center"][1] - previous[1]))
            self.last = candidate.copy(); self.last.update(status="DETECTED", lost_frames=0); self.lost_frames = 0
            return self.last.copy()
        self.lost_frames += 1
        if self.lost_frames > self.max_lost_frames: self.last = None; return None
        if self.lost_frames > self.max_predict_frames: return None
        x1, y1, x2, y2 = self.last["box"]; cx, cy = map(int, predicted); w, h = x2 - x1, y2 - y1
        self.last = {"confidence": 0.0, "center": (cx, cy), "box": (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2), "status": "PREDICTED", "lost_frames": self.lost_frames}
        return self.last.copy()
