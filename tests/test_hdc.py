import numpy as np

from hyperglyph.hdc import (
    bind,
    cosine_similarity,
    hamming_distance,
    make_role_vector,
)


def test_role_vectors_are_deterministic() -> None:
    first = make_role_vector("name", 1, 16, 42)
    second = make_role_vector("name", 1, 16, 42)
    assert np.array_equal(first, second)


def test_bind_output_shape_is_correct() -> None:
    a = np.ones(16, dtype=np.float32)
    b = np.ones(16, dtype=np.float32)
    assert bind(a, b).shape == (16,)


def test_hamming_distance_works() -> None:
    a = np.array([1, -1, 1], dtype=np.int8)
    b = np.array([-1, -1, 1], dtype=np.int8)
    assert hamming_distance(a, b) == 1


def test_cosine_similarity_range_is_valid() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert -1.0 <= cosine_similarity(a, b) <= 1.0
