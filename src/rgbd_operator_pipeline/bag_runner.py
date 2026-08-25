"""Connect the generic pipeline to a ROS bag V2 file."""
from __future__ import annotations

import warnings
from pathlib import Path
from .image_codec import decode_image_message
from .rosbag_v2 import OP_CHUNK, OP_CONNECTION, OP_MSG_DATA, field_text, field_u32, parse_fields, read_record, record_op, records_in_chunk, unpack_chunk


def _connection(fields: dict[str, bytes], data: bytes) -> tuple[int, dict] | None:
    connection_id = field_u32(fields, "conn")
    if connection_id is None: return None
    header = parse_fields(data)
    return connection_id, {"topic": field_text(fields, "topic") or field_text(header, "topic"), "type": field_text(header, "type")}


def _stream_kind(info: dict) -> str | None:
    topic, message_type = info["topic"].lower(), info["type"].lower()
    # Topics such as ``.../image/metadata`` are diagnostics, not image
    # payloads.  A topic name alone is therefore not enough to identify an
    # image stream.
    if "sensor_msgs/image" not in message_type: return None
    if "depth" in topic: return "depth"
    if "color" in topic: return "color"
    return None


def _timestamp(fields: dict[str, bytes]) -> float | None:
    raw = fields.get("time")
    if raw is None or len(raw) != 8: return None
    import struct
    seconds, nanoseconds = struct.unpack("<II", raw)
    return seconds + nanoseconds / 1_000_000_000


def run_bag(path: Path, pipeline, on_frame=None) -> int:
    """Process RGB-D frames. ``on_frame`` receives each FrameResult for optional UI."""
    connections: dict[int, dict] = {}
    with path.open("rb") as bag:
        if bag.readline() != b"#ROSBAG V2.0\n": raise ValueError("not a ROSBAG V2.0 file")
        while record := read_record(bag):
            operation = record_op(record.fields)
            if operation == OP_CONNECTION:
                item = _connection(record.fields, record.data)
                if item: connections[item[0]] = item[1]
            elif operation == OP_CHUNK:
                for inner in records_in_chunk(unpack_chunk(record.fields, record.data)):
                    if record_op(inner.fields) == OP_CONNECTION:
                        item = _connection(inner.fields, inner.data)
                        if item: connections[item[0]] = item[1]
                    elif record_op(inner.fields) == OP_MSG_DATA:
                        _consume(inner.fields, inner.data, connections, pipeline, on_frame)
            elif operation == OP_MSG_DATA:
                _consume(record.fields, record.data, connections, pipeline, on_frame)
    return pipeline.frame


def _consume(fields, data, connections, pipeline, on_frame) -> None:
    connection_id = field_u32(fields, "conn"); info = connections.get(connection_id)
    if info is None: return
    kind = _stream_kind(info)
    if kind is None: return
    try:
        _, image = decode_image_message(data)
    except ValueError as error:
        # A ROS bag can contain an incomplete/corrupt image record.  Keep the
        # remaining RGB-D stream usable instead of aborting the entire run.
        warnings.warn(
            f"Skipping unreadable {kind} image on {info['topic']}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if kind == "depth": pipeline.accept_depth(image); return
    result = pipeline.accept_color(image, _timestamp(fields))
    if on_frame: on_frame(result)
