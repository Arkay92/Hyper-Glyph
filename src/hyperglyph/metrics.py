"""Metrics for compression quality."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def original_size_bytes(state_dict: Mapping[str, np.ndarray]) -> int:
    """Estimate the byte size of a state_dict."""
    total = 0
    for tensor in state_dict.values():
        total += np.asarray(tensor).nbytes
    return total


def baseline_size_bytes(state_dict: Mapping[str, Any], bytes_per_value: int) -> int:
    """Estimate a dense baseline size with a fixed number of bytes per value."""
    total = 0
    for tensor in state_dict.values():
        total += int(np.asarray(tensor).size) * bytes_per_value
    return total


def compressed_size_bytes(compressed_model: object) -> int:
    """Estimate the compressed size in bytes."""
    if isinstance(compressed_model, Mapping):
        return len(compressed_model.get("payload", b""))
    tensors = getattr(compressed_model, "tensors", None)
    if isinstance(tensors, Mapping):
        total = 0
        for tensor in tensors.values():
            total += compressed_tensor_size_bytes(tensor)
        return total
    return 0


def compressed_tensor_size_bytes(tensor: object) -> int:
    """Estimate the byte size of a compressed tensor payload."""
    prototype_matrix = np.asarray(getattr(tensor, "prototype_matrix", np.asarray([])))
    prototype_bytes = int(prototype_matrix.size) * 4
    prototype_id_bytes = len(getattr(tensor, "prototype_ids", [])) * 4
    scale_bytes = len(getattr(tensor, "scales", [])) * 4
    shape_bytes = len(getattr(tensor, "shape", ())) * 4
    residual_bytes = 0
    for residual in getattr(tensor, "residuals", []):
        indices = residual.get("indices", [])
        values = residual.get("values", [])
        residual_bytes += len(indices) * 2
        residual_bytes += len(values) if residual.get("dtype") == "int8" else len(values) * 4
        residual_bytes += 4
    return prototype_bytes + prototype_id_bytes + scale_bytes + shape_bytes + residual_bytes


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    """Compute compression ratio as original / compressed."""
    if compressed_bytes <= 0:
        return float("inf")
    return original_bytes / compressed_bytes


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute mean squared error."""
    original = np.asarray(original, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    return float(np.mean((original - reconstructed) ** 2))


def mae(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute mean absolute error."""
    original = np.asarray(original, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    return float(np.mean(np.abs(original - reconstructed)))


def max_abs_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute maximum absolute error."""
    original = np.asarray(original, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    return float(np.max(np.abs(original - reconstructed)))


def cosine_weight_similarity(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute cosine similarity between two arrays."""
    original = np.asarray(original, dtype=np.float32).ravel()
    reconstructed = np.asarray(reconstructed, dtype=np.float32).ravel()
    denom = np.linalg.norm(original) * np.linalg.norm(reconstructed)
    if denom == 0:
        return 0.0
    return float(np.dot(original, reconstructed) / denom)
