"""Orchestration independent of ROS bag and UI; therefore easy to smoke-test."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .localization import keypoints_to_3d, operator_center
from .tracking import OperatorTracker


@dataclass
class FrameResult:
    frame: int; timestamp: float | None; image: Any; detections: list[dict]; operator: dict | None; keypoints: list[dict] | None; center: dict | None


class OperatorPipeline:
    def __init__(self, config: PipelineConfig, detector: Any, pose_estimator: Any, tracker: OperatorTracker | None = None) -> None:
        self.config, self.detector, self.pose_estimator, self.tracker = config, detector, pose_estimator, tracker or OperatorTracker()
        self.depth = None; self.frame = 0
        config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._file = config.output_csv.open("w", newline=""); self._csv = csv.DictWriter(self._file, fieldnames=["frame", "timestamp", "x", "y", "z"]); self._csv.writeheader()

    def close(self) -> None: self._file.close()
    def accept_depth(self, image: Any) -> None: self.depth = image

    def accept_color(self, image: Any, timestamp: float | None) -> FrameResult:
        detections, _, _ = self.detector.detect(image); operator = self.tracker.update(detections); keypoints = center = None
        if operator and operator["status"] == "DETECTED":
            keypoints, _, _ = self.pose_estimator.estimate(image, operator["box"])
            if keypoints and self.depth is not None: center = operator_center(keypoints_to_3d(keypoints, self.depth, self.config.intrinsics))
        self.frame += 1
        if center: self._csv.writerow({"frame": self.frame, "timestamp": timestamp, **center}); self._file.flush()
        return FrameResult(self.frame, timestamp, image, detections, operator, keypoints, center)
