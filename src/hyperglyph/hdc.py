"""Simple deterministic hyperdimensional computing helpers."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def _hash_seed(name: str, index: int, seed: int) -> int:
    payload = f"{name}:{index}:{seed}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def make_role_vector(name: str, index: int, dim: int, seed: int) -> np.ndarray:
    """Create a deterministic bipolar role vector."""
    rng = np.random.default_rng(_hash_seed(name, index, seed))
    vector = rng.choice([-1.0, 1.0], size=dim)
    return vector.astype(np.float32)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two bipolar vectors via elementwise multiplication."""
    return a * b


def bind_many(vectors: Iterable[np.ndarray]) -> np.ndarray:
    """Bind a sequence of vectors together."""
    vectors = list(vectors)
    if not vectors:
        raise ValueError("at least one vector is required")
    result = np.ones_like(vectors[0], dtype=np.float32)
    for vector in vectors:
        result = result * vector
    return result


def bundle(vectors: Iterable[np.ndarray]) -> np.ndarray:
    """Bundle vectors by elementwise summation and sign."""
    vectors = list(vectors)
    if not vectors:
        raise ValueError("at least one vector is required")
    result = np.zeros_like(vectors[0], dtype=np.float32)
    for vector in vectors:
        result = result + vector
    return np.sign(result)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Compute Hamming distance between two bipolar vectors."""
    a = np.asarray(a)
    b = np.asarray(b)
    return int(np.count_nonzero(a != b))


def binarize(v: np.ndarray) -> np.ndarray:
    """Binarize a vector to {-1, 1}."""
    return np.sign(np.asarray(v, dtype=np.float32)).astype(np.int8)
