"""Global prototype codebook helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .blocks import split_array_blocks
from .prototypes import assign_prototypes, learn_prototypes


def collect_blocks_from_state_dict(
    state_dict: Mapping[str, Any],
    block_size: int,
    min_tensor_size: int = 1,
    sample_blocks: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, list[tuple[str, tuple[int, ...], int]]]:
    """Collect flattened blocks across all eligible tensors."""
    blocks: list[np.ndarray] = []
    metadata: list[tuple[str, tuple[int, ...], int]] = []
    for name, tensor in state_dict.items():
        array = np.asarray(tensor, dtype=np.float32)
        if array.size < min_tensor_size:
            continue
        tensor_blocks = split_array_blocks(array, block_size)
        start = len(blocks)
        blocks.extend(tensor_blocks)
        metadata.append((name, tuple(array.shape), start))
    if not blocks:
        return np.empty((0, block_size), dtype=np.float32), metadata
    matrix = np.stack(blocks, axis=0).astype(np.float32)
    if sample_blocks is not None and matrix.shape[0] > sample_blocks:
        rng = np.random.default_rng(seed)
        indices = rng.choice(matrix.shape[0], size=sample_blocks, replace=False)
        matrix = matrix[np.sort(indices)]
    return matrix, metadata


def learn_global_prototypes(
    blocks: np.ndarray,
    n_prototypes: int,
    seed: int = 42,
    normalize: bool = True,
) -> np.ndarray:
    """Learn a shared prototype table across model blocks."""
    blocks = np.asarray(blocks, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(blocks, axis=1, keepdims=True)
        blocks = blocks / np.maximum(norms, 1e-6)
    return learn_prototypes(blocks, n_prototypes, seed)


def assign_global_prototypes(blocks: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Assign blocks to global prototypes."""
    blocks = np.asarray(blocks, dtype=np.float32)
    norms = np.linalg.norm(blocks, axis=1, keepdims=True)
    normalized = blocks / np.maximum(norms, 1e-6)
    return assign_prototypes(normalized, prototypes)


def reconstruct_blocks_from_global_prototypes(
    assignments: np.ndarray, prototypes: np.ndarray, scales: np.ndarray | None = None
) -> np.ndarray:
    """Reconstruct blocks from a global prototype table."""
    reconstructed = np.asarray(prototypes, dtype=np.float32)[
        np.asarray(assignments, dtype=np.int32)
    ]
    if scales is not None:
        reconstructed = reconstructed * np.asarray(scales, dtype=np.float32).reshape(-1, 1)
    return reconstructed.astype(np.float32)
