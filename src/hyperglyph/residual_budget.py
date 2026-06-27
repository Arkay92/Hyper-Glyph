"""Adaptive sparse residual budgeting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .packing import delta_decode, delta_encode, varint_decode, varint_encode


@dataclass(slots=True)
class ResidualCandidate:
    """A candidate residual entry."""

    flat_index: int
    value: float
    abs_error: float


@dataclass(slots=True)
class EncodedResidualStream:
    """Encoded sparse residual stream."""

    index_bytes: bytes
    value_bytes: bytes
    scale: float
    count: int


def collect_residual_candidates(
    original: np.ndarray,
    reconstructed: np.ndarray,
    threshold: float | None = None,
) -> list[ResidualCandidate]:
    """Collect residual candidates sorted by descending absolute error."""
    diff = np.asarray(original, dtype=np.float32).reshape(-1) - np.asarray(
        reconstructed, dtype=np.float32
    ).reshape(-1)
    candidates: list[ResidualCandidate] = []
    for index, value in enumerate(diff):
        abs_error = float(abs(value))
        if threshold is not None and abs_error < threshold:
            continue
        candidates.append(ResidualCandidate(index, float(value), abs_error))
    candidates.sort(key=lambda item: item.abs_error, reverse=True)
    return candidates


def allocate_residual_budget(
    candidates: list[ResidualCandidate],
    byte_budget: int,
    max_k: int | None = None,
) -> list[ResidualCandidate]:
    """Allocate residual entries within a rough byte budget."""
    if byte_budget <= 0:
        return []
    limit = byte_budget // 3
    if max_k is not None:
        limit = min(limit, max_k)
    return candidates[: max(limit, 0)]


def quantize_residual_values_int8(values: list[float] | np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize residual values to int8 with one shared scale."""
    arr: np.ndarray = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return np.asarray([], dtype=np.int8), 1.0
    scale = float(np.max(np.abs(arr)) / 127.0)
    if scale == 0.0:
        scale = 1.0
    quantized = np.clip(np.rint(arr / scale), -127, 127).astype(np.int8)
    return quantized, scale


def encode_residual_stream(candidates: list[ResidualCandidate]) -> EncodedResidualStream:
    """Encode sparse residual indices and int8 values."""
    ordered = sorted(candidates, key=lambda item: item.flat_index)
    indices = [item.flat_index for item in ordered]
    values = [item.value for item in ordered]
    quantized, scale = quantize_residual_values_int8(values)
    return EncodedResidualStream(
        index_bytes=varint_encode(delta_encode(indices)),
        value_bytes=quantized.tobytes(),
        scale=scale,
        count=len(indices),
    )


def decode_residual_stream(stream: EncodedResidualStream) -> tuple[np.ndarray, np.ndarray]:
    """Decode sparse residual indices and dequantized values."""
    indices: np.ndarray = np.asarray(
        delta_decode(varint_decode(stream.index_bytes)), dtype=np.int64
    )
    quantized: np.ndarray = np.frombuffer(stream.value_bytes, dtype=np.int8).astype(np.float32)
    return indices, quantized * stream.scale
