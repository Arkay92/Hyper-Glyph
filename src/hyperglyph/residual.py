"""Sparse residual helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_topk_residual(
    original_block: np.ndarray, reconstructed_block: np.ndarray, k: int
) -> dict[str, Any]:
    """Return the indices and values of the top-k residual entries."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if original_block.shape != reconstructed_block.shape:
        raise ValueError("blocks must have the same shape")

    diff = np.asarray(original_block, dtype=np.float32) - np.asarray(
        reconstructed_block, dtype=np.float32
    )
    if k == 0:
        return {"indices": [], "values": []}

    if diff.size == 0:
        return {"indices": [], "values": []}

    flat = diff.reshape(-1)
    topk_idx = np.argsort(np.abs(flat))[-k:][::-1]
    return {
        "indices": [int(index) for index in topk_idx],
        "values": [float(flat[index]) for index in topk_idx],
    }


def apply_residual(block: np.ndarray, residual: dict[str, Any]) -> np.ndarray:
    """Apply sparse residual values to a block."""
    result = np.asarray(block, dtype=np.float32).reshape(-1).copy()
    for index, value in zip(residual.get("indices", []), residual.get("values", [])):
        result[int(index)] += float(value)
    return result.reshape(block.shape)


def serialize_residual(residual: dict[str, Any]) -> dict[str, Any]:
    """Serialize residual metadata for JSON compatibility."""
    return {
        "indices": list(residual.get("indices", [])),
        "values": list(residual.get("values", [])),
    }
