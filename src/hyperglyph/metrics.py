"""Metrics for compression quality."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def original_size_bytes(state_dict: Mapping[str, np.ndarray]) -> int:
    """Estimate the byte size of a state_dict."""
    total = 0
    for tensor in state_dict.values():
        total += np.asarray(tensor).nbytes
    return total


def compressed_size_bytes(compressed_model: object) -> int:
    """Estimate the compressed size in bytes."""
    if isinstance(compressed_model, Mapping):
        return len(compressed_model.get("payload", b""))
    return 0


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
