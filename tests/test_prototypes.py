import numpy as np

from hyperglyph.prototypes import assign_prototypes, learn_prototypes, reconstruct_from_prototypes


def test_prototype_learning_is_deterministic() -> None:
    blocks = np.array([[1.0, 1.0], [1.1, 1.1], [5.0, 5.0], [5.1, 5.1]], dtype=np.float32)
    prototypes_a = learn_prototypes(blocks, 2, 42)
    prototypes_b = learn_prototypes(blocks, 2, 42)
    assert np.allclose(prototypes_a, prototypes_b)


def test_assignments_length_matches_blocks() -> None:
    blocks = np.array([[1.0, 1.0], [5.0, 5.0]], dtype=np.float32)
    prototypes = np.array([[1.0, 1.0], [5.0, 5.0]], dtype=np.float32)
    assignments = assign_prototypes(blocks, prototypes)
    assert len(assignments) == len(blocks)


def test_reconstruction_shape_is_correct() -> None:
    assignments = np.array([0, 1], dtype=np.int32)
    prototypes = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    reconstructed = reconstruct_from_prototypes(assignments, prototypes)
    assert reconstructed.shape == (2, 2)
