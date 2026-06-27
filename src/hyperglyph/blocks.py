"""Helpers for splitting arrays into blocks."""

from __future__ import annotations

import numpy as np


def pad_tensor_to_blocks(array: np.ndarray, block_size: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Pad a flattened tensor so its size is divisible by block_size."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    flattened = array.reshape(-1)
    remainder = flattened.size % block_size
    if remainder == 0:
        return flattened.copy(), tuple(array.shape)
    padded_size = flattened.size + (block_size - remainder)
    padded = np.pad(flattened, (0, padded_size - flattened.size), mode="constant")
    return padded.astype(np.float32), tuple(array.shape)


def split_array_blocks(array: np.ndarray, block_size: int) -> list[np.ndarray]:
    """Split a flat or regular array into blocks of size block_size."""
    padded, _ = pad_tensor_to_blocks(array, block_size)
    flat_padded = padded.reshape(-1)
    return [
        flat_padded[index : index + block_size] for index in range(0, flat_padded.size, block_size)
    ]


def merge_array_blocks(
    blocks: list[np.ndarray],
    original_shape: tuple[int, ...],
    padded_shape: tuple[int, ...],
    block_size: int,
) -> np.ndarray:
    """Merge blocks back into their original shape."""
    flat = np.concatenate(blocks, axis=0)
    flat = flat[: int(np.prod(original_shape))]
    return flat.reshape(original_shape)


def flatten_tensor_for_blocks(array: np.ndarray) -> np.ndarray:
    """Flatten tensors to 1D for block processing."""
    return np.asarray(array, dtype=np.float32).reshape(-1)


def restore_tensor_shape(array: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore a flattened vector to the original shape."""
    return np.asarray(array, dtype=np.float32).reshape(original_shape)
