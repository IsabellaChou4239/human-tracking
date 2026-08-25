"""Convert 2-D pose points and raw depth into a 3-D body centre."""
from __future__ import annotations

import numpy as np
from .config import CameraIntrinsics


def keypoints_to_3d(keypoints: list[dict], depth: np.ndarray, camera: CameraIntrinsics) -> list[dict]:
    height, width = depth.shape[:2]; output = []
    for point in keypoints:
        u, v = point["x"], point["y"]
        if point["confidence"] < .5 or not (0 <= u < width and 0 <= v < height): continue
        z = float(depth[v, u]) * camera.depth_scale
        if not .3 <= z <= 4.0: continue
        output.append({"id": point["id"], "confidence": point["confidence"], "x": (u - camera.cx) * z / camera.fx, "y": (v - camera.cy) * z / camera.fy, "z": z})
    return output


def operator_center(keypoints: list[dict]) -> dict | None:
    torso = [[point[axis] for axis in ("x", "y", "z")] for point in keypoints if point["id"] in {5, 6, 11, 12} and point["confidence"] > .5]
    if len(torso) < 2: return None
    x, y, z = np.mean(torso, axis=0)
    return {"x": float(x), "y": float(y), "z": float(z)}
