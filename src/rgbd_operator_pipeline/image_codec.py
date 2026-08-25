"""Decode ROS sensor_msgs/Image messages into NumPy arrays."""
from __future__ import annotations

import struct
import numpy as np


def _string(data: bytes, offset: int) -> tuple[str, int]:
    if offset + 4 > len(data): raise ValueError("string length out of range")
    length = struct.unpack_from("<I", data, offset)[0]; offset += 4
    if offset + length > len(data): raise ValueError("string data out of range")
    return data[offset:offset + length].decode("utf-8", errors="replace"), offset + length


def decode_image_message(data: bytes) -> tuple[str, np.ndarray]:
    if len(data) < 12: raise ValueError("image message too short")
    offset = 12
    _, offset = _string(data, offset)  # Header.frame_id
    if offset + 8 > len(data): raise ValueError("image dimensions missing")
    height, width = struct.unpack_from("<II", data, offset); offset += 8
    encoding, offset = _string(data, offset)
    if offset + 5 > len(data): raise ValueError("image metadata missing")
    _, step = struct.unpack_from("<BI", data, offset); offset += 5
    if offset + 4 > len(data): raise ValueError("image payload length missing")
    size = struct.unpack_from("<I", data, offset)[0]; offset += 4
    payload = data[offset:offset + size]
    if len(payload) != size: raise ValueError("incomplete image payload")
    if encoding in {"bgr8", "rgb8"}:
        image = np.frombuffer(payload, np.uint8).reshape(height, step)[:, :width * 3].reshape(height, width, 3)
    elif encoding in {"16UC1", "mono16"}:
        image = np.frombuffer(payload, "<u2").reshape(height, step // 2)[:, :width]
    elif encoding == "32FC1":
        image = np.frombuffer(payload, "<f4").reshape(height, step // 4)[:, :width]
    else:
        raise ValueError(f"unsupported image encoding: {encoding}")
    return encoding, image
