"""Prototype learning and reconstruction helpers."""

from __future__ import annotations

import numpy as np


def _init_prototypes(blocks: np.ndarray, n_prototypes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(blocks), size=min(n_prototypes, len(blocks)), replace=False)
    return blocks[indices].astype(np.float32).copy()


def learn_prototypes(blocks: np.ndarray, n_prototypes: int, seed: int) -> np.ndarray:
    """Learn deterministic prototypes from a set of blocks using mini k-means."""
    blocks = np.asarray(blocks, dtype=np.float32)
    if blocks.size == 0:
        return np.empty((0, blocks.shape[-1]), dtype=np.float32)
    if n_prototypes <= 0:
        raise ValueError("n_prototypes must be positive")

    prototypes = _init_prototypes(blocks, n_prototypes, seed)
    for _ in range(5):
        distances = np.linalg.norm(blocks[:, None, :] - prototypes[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)
        new_prototypes = np.empty_like(prototypes)
        for idx in range(len(prototypes)):
            members = blocks[assignments == idx]
            if len(members) == 0:
                new_prototypes[idx] = prototypes[idx]
            else:
                new_prototypes[idx] = members.mean(axis=0)
        if np.allclose(new_prototypes, prototypes):
            break
        prototypes = new_prototypes
    return prototypes


def assign_prototypes(blocks: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Assign each block to its nearest prototype."""
    blocks = np.asarray(blocks, dtype=np.float32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if len(prototypes) == 0:
        return np.zeros(len(blocks), dtype=np.int32)
    distances = np.linalg.norm(blocks[:, None, :] - prototypes[None, :, :], axis=2)
    return np.argmin(distances, axis=1).astype(np.int32)


def reconstruct_from_prototypes(
    assignments: np.ndarray, prototypes: np.ndarray, scales: np.ndarray | None = None
) -> np.ndarray:
    """Reconstruct blocks from prototype assignments."""
    assignments = np.asarray(assignments, dtype=np.int32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if len(prototypes) == 0:
        return np.empty((len(assignments), 0), dtype=np.float32)
    reconstructed = prototypes[assignments]
    if scales is not None:
        return reconstructed * np.asarray(scales, dtype=np.float32)[:, None]
    return reconstructed
