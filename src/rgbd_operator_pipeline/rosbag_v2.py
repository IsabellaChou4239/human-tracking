"""Small, dependency-free reader for the parts of ROS bag V2 used here."""
from __future__ import annotations

import bz2
import io
import struct
from dataclasses import dataclass
from typing import BinaryIO, Iterator

OP_MSG_DATA, OP_CHUNK, OP_CONNECTION = 0x02, 0x05, 0x07
MAX_HEADER = 64 * 1024 * 1024


@dataclass(frozen=True)
class Record:
    position: int
    fields: dict[str, bytes]
    data: bytes


def u32(data: bytes) -> int:
    return struct.unpack("<I", data)[0]


def parse_fields(raw: bytes) -> dict[str, bytes]:
    fields, offset = {}, 0
    while offset + 4 <= len(raw):
        length = u32(raw[offset:offset + 4]); offset += 4
        if length <= 0 or offset + length > len(raw):
            raise ValueError("invalid bag header field length")
        field = raw[offset:offset + length]; offset += length
        if b"=" not in field:
            raise ValueError("invalid bag header field")
        key, value = field.split(b"=", 1)
        fields[key.decode("ascii", errors="replace")] = value
    if offset != len(raw):
        raise ValueError("bag header has trailing bytes")
    return fields


def read_record(stream: BinaryIO) -> Record | None:
    position, raw = stream.tell(), stream.read(4)
    if not raw:
        return None
    if len(raw) != 4:
        raise EOFError("incomplete record header length")
    header_length = u32(raw)
    if not 0 < header_length <= MAX_HEADER:
        raise ValueError(f"unreasonable header length at 0x{position:X}")
    header = stream.read(header_length)
    if len(header) != header_length:
        raise EOFError("incomplete record header")
    raw = stream.read(4)
    if len(raw) != 4:
        raise EOFError("incomplete record data length")
    data = stream.read(u32(raw))
    if len(data) != u32(raw):
        raise EOFError("incomplete record data")
    return Record(position, parse_fields(header), data)


def field_u32(fields: dict[str, bytes], name: str) -> int | None:
    value = fields.get(name)
    return u32(value) if value is not None and len(value) == 4 else None


def field_text(fields: dict[str, bytes], name: str) -> str:
    return fields.get(name, b"").decode("utf-8", errors="replace")


def record_op(fields: dict[str, bytes]) -> int | None:
    value = fields.get("op")
    return value[0] if value else None


def unpack_chunk(fields: dict[str, bytes], data: bytes) -> bytes:
    compression = field_text(fields, "compression")
    if compression == "none": return data
    if compression == "bz2": return bz2.decompress(data)
    raise RuntimeError(f"unsupported ROS bag compression: {compression or 'missing'}")


def records_in_chunk(data: bytes) -> Iterator[Record]:
    stream = io.BytesIO(data)
    while record := read_record(stream):
        yield record
