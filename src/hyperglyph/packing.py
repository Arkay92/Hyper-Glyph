"""Binary packing helpers for compact Hyper Glyph streams."""

from __future__ import annotations

import numpy as np


def pack_uint4(values: np.ndarray) -> bytes:
    """Pack unsigned 4-bit values, two values per byte."""
    arr = np.asarray(values, dtype=np.uint8).reshape(-1)
    if np.any(arr > 15):
        raise ValueError("uint4 values must be in [0, 15]")
    if arr.size % 2:
        arr = np.concatenate([arr, np.zeros(1, dtype=np.uint8)])
    packed = (arr[0::2] & 0x0F) | ((arr[1::2] & 0x0F) << 4)
    return packed.astype(np.uint8).tobytes()


def unpack_uint4(data: bytes, length: int) -> np.ndarray:
    """Unpack unsigned 4-bit values."""
    raw = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(raw.size * 2, dtype=np.uint8)
    out[0::2] = raw & 0x0F
    out[1::2] = (raw >> 4) & 0x0F
    return out[:length].copy()


def pack_int4(values: np.ndarray) -> bytes:
    """Pack signed 4-bit values in [-8, 7]."""
    arr = np.asarray(values, dtype=np.int8).reshape(-1)
    if np.any(arr < -8) or np.any(arr > 7):
        raise ValueError("int4 values must be in [-8, 7]")
    return pack_uint4((arr & 0x0F).astype(np.uint8))


def unpack_int4(data: bytes, length: int) -> np.ndarray:
    """Unpack signed 4-bit values in [-8, 7]."""
    unsigned = unpack_uint4(data, length).astype(np.int8)
    return np.where(unsigned >= 8, unsigned - 16, unsigned).astype(np.int8)


def choose_min_uint_dtype(max_value: int) -> str | np.dtype:
    """Choose the smallest unsigned dtype that can hold max_value."""
    if max_value <= 15:
        return "uint4"
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def varint_encode(values: list[int] | np.ndarray) -> bytes:
    """Encode non-negative integers as unsigned LEB128 varints."""
    encoded = bytearray()
    for value in np.asarray(values, dtype=np.uint64).reshape(-1):
        current = int(value)
        if current < 0:
            raise ValueError("varint values must be non-negative")
        while current >= 0x80:
            encoded.append((current & 0x7F) | 0x80)
            current >>= 7
        encoded.append(current)
    return bytes(encoded)


def varint_decode(data: bytes) -> list[int]:
    """Decode unsigned LEB128 varints."""
    values: list[int] = []
    shift = 0
    current = 0
    for byte in data:
        current |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            continue
        values.append(current)
        current = 0
        shift = 0
    if shift:
        raise ValueError("truncated varint stream")
    return values


def delta_encode(indices: list[int] | np.ndarray) -> list[int]:
    """Delta encode sorted integer indices."""
    values = [int(value) for value in np.asarray(indices, dtype=np.int64).reshape(-1)]
    if not values:
        return []
    deltas = [values[0]]
    deltas.extend(values[idx] - values[idx - 1] for idx in range(1, len(values)))
    return deltas


def delta_decode(deltas: list[int] | np.ndarray) -> list[int]:
    """Decode delta-encoded integer indices."""
    total = 0
    values: list[int] = []
    for delta in np.asarray(deltas, dtype=np.int64).reshape(-1):
        total += int(delta)
        values.append(total)
    return values
