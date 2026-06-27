"""Sparse residual helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_topk_residual(
    original_block: np.ndarray,
    reconstructed_block: np.ndarray,
    k: int,
    dtype: str = "int8",
) -> dict[str, Any]:
    """Return the indices and values of the top-k residual entries."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if dtype not in {"float32", "int8"}:
        raise ValueError("dtype must be 'float32' or 'int8'")
    if original_block.shape != reconstructed_block.shape:
        raise ValueError("blocks must have the same shape")

    diff = np.asarray(original_block, dtype=np.float32) - np.asarray(
        reconstructed_block, dtype=np.float32
    )
    if k == 0:
        return {"indices": [], "values": [], "dtype": dtype}

    if diff.size == 0:
        return {"indices": [], "values": [], "dtype": dtype}

    flat = diff.reshape(-1)
    topk_idx = np.argsort(np.abs(flat))[-k:][::-1]
    values: np.ndarray = np.asarray([float(flat[index]) for index in topk_idx], dtype=np.float32)
    if dtype == "int8":
        scale = float(np.max(np.abs(values)) / 127.0) if values.size else 1.0
        if scale == 0.0:
            scale = 1.0
        quantized: np.ndarray = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
        return {
            "indices": [int(index) for index in topk_idx],
            "values": [int(value) for value in quantized],
            "scale": scale,
            "dtype": "int8",
        }
    return {
        "indices": [int(index) for index in topk_idx],
        "values": [float(value) for value in values],
        "dtype": "float32",
    }


def apply_residual(block: np.ndarray, residual: dict[str, Any]) -> np.ndarray:
    """Apply sparse residual values to a block."""
    result = np.asarray(block, dtype=np.float32).reshape(-1).copy()
    scale = float(residual.get("scale", 1.0))
    dtype = str(residual.get("dtype", "float32"))
    for index, value in zip(residual.get("indices", []), residual.get("values", [])):
        decoded_value = float(value) * scale if dtype == "int8" else float(value)
        result[int(index)] += decoded_value
    return result.reshape(block.shape)


def serialize_residual(residual: dict[str, Any]) -> dict[str, Any]:
    """Serialize residual metadata for JSON compatibility."""
    return {
        "indices": list(residual.get("indices", [])),
        "values": list(residual.get("values", [])),
        "scale": float(residual.get("scale", 1.0)),
        "dtype": str(residual.get("dtype", "float32")),
    }
