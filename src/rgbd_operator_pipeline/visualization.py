"""OpenCV display for live inspection of the RGB-D processing result."""
from __future__ import annotations

from typing import Any

import cv2

from .pipeline import FrameResult


_SKELETON = ((5, 6), (5, 11), (6, 12), (11, 12))


class StopDisplay(Exception):
    """Raised when the user closes the visualisation window or presses q/Esc."""


class LiveVisualizer:
    def __init__(self, scale: float = 0.5, window_name: str = "RGB-D operator localisation") -> None:
        self.scale, self.window_name = scale, window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def show(self, result: FrameResult) -> None:
        canvas = result.image.copy()
        for detection in result.detections:
            x1, y1, x2, y2 = detection["box"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (100, 100, 100), 1)
        if result.operator:
            x1, y1, x2, y2 = result.operator["box"]
            detected = result.operator["status"] == "DETECTED"
            color = (0, 220, 0) if detected else (0, 180, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, result.operator["status"], (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
        points = {point["id"]: (point["x"], point["y"]) for point in result.keypoints or [] if point["confidence"] >= .5}
        for start, end in _SKELETON:
            if start in points and end in points: cv2.line(canvas, points[start], points[end], (255, 180, 0), 2)
        for point in points.values(): cv2.circle(canvas, point, 4, (0, 0, 255), -1)
        label = f"frame {result.frame}"
        if result.center: label += " | 3D: ({x:.2f}, {y:.2f}, {z:.2f}) m".format(**result.center)
        else: label += " | no valid 3D centre"
        cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        if self.scale != 1: canvas = cv2.resize(canvas, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        if key in (27, ord("q")) or visible < 1: raise StopDisplay

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
