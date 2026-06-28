"""Binary packing helpers for compact Hyper Glyph streams."""

from __future__ import annotations

import numpy as np


def pack_bits(values: np.ndarray, bits: int) -> bytes:
    """Pack unsigned integer values using a fixed number of bits per value."""
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    arr: np.ndarray = np.asarray(values, dtype=np.uint8).reshape(-1)
    max_value = (1 << bits) - 1
    if np.any(arr > max_value):
        raise ValueError(f"packed values must be in [0, {max_value}]")

    encoded = bytearray((arr.size * bits + 7) // 8)
    bit_offset = 0
    for value in arr:
        current = int(value)
        byte_index = bit_offset // 8
        shift = bit_offset % 8
        encoded[byte_index] |= (current << shift) & 0xFF
        spill = shift + bits - 8
        if spill > 0:
            encoded[byte_index + 1] |= current >> (bits - spill)
        bit_offset += bits
    return bytes(encoded)


def unpack_bits(data: bytes, bits: int, length: int) -> np.ndarray:
    """Unpack fixed-width unsigned integer values."""
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    raw: np.ndarray = np.frombuffer(data, dtype=np.uint8)
    out: np.ndarray = np.empty(length, dtype=np.uint8)
    mask = (1 << bits) - 1
    bit_offset = 0
    for index in range(length):
        byte_index = bit_offset // 8
        shift = bit_offset % 8
        value = int(raw[byte_index]) >> shift
        spill = shift + bits - 8
        if spill > 0 and byte_index + 1 < raw.size:
            value |= int(raw[byte_index + 1]) << (bits - spill)
        out[index] = value & mask
        bit_offset += bits
    return out


def pack_uint4(values: np.ndarray) -> bytes:
    """Pack unsigned 4-bit values, two values per byte."""
    return pack_bits(values, bits=4)


def unpack_uint4(data: bytes, length: int) -> np.ndarray:
    """Unpack unsigned 4-bit values."""
    return unpack_bits(data, bits=4, length=length)


def pack_int4(values: np.ndarray) -> bytes:
    """Pack signed 4-bit values in [-8, 7]."""
    arr: np.ndarray = np.asarray(values, dtype=np.int8).reshape(-1)
    if np.any(arr < -8) or np.any(arr > 7):
        raise ValueError("int4 values must be in [-8, 7]")
    return pack_uint4((arr & 0x0F).astype(np.uint8))


def unpack_int4(data: bytes, length: int) -> np.ndarray:
    """Unpack signed 4-bit values in [-8, 7]."""
    unsigned: np.ndarray = unpack_uint4(data, length).astype(np.int8)
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


def rle_encode_uint(values: np.ndarray) -> bytes:
    """Run-length encode uint assignments as varint(count), varint(value) pairs."""
    arr: np.ndarray = np.asarray(values, dtype=np.uint32).reshape(-1)
    if arr.size == 0:
        return b""
    encoded = bytearray()
    current = int(arr[0])
    count = 1
    for value in arr[1:]:
        item = int(value)
        if item == current:
            count += 1
            continue
        encoded.extend(varint_encode([count, current]))
        current = item
        count = 1
    encoded.extend(varint_encode([count, current]))
    return bytes(encoded)


def rle_decode_uint(data: bytes, length: int) -> np.ndarray:
    """Decode run-length encoded uint assignments."""
    pairs = varint_decode(data)
    if len(pairs) % 2:
        raise ValueError("invalid RLE stream")
    values: list[int] = []
    for index in range(0, len(pairs), 2):
        count = pairs[index]
        value = pairs[index + 1]
        values.extend([value] * count)
    if len(values) < length:
        raise ValueError("RLE stream shorter than expected length")
    return np.asarray(values[:length], dtype=np.uint32)
