import numpy as np

from hyperglyph.global_codebook import (
    assign_global_prototypes,
    collect_blocks_from_state_dict,
    learn_global_prototypes,
    reconstruct_blocks_from_global_prototypes,
)


def test_global_prototypes_learned_once_and_deterministic() -> None:
    state = {
        "a.weight": np.arange(64, dtype=np.float32).reshape(8, 8),
        "b.weight": np.arange(64, 128, dtype=np.float32).reshape(8, 8),
    }
    blocks, _ = collect_blocks_from_state_dict(state, block_size=8, min_tensor_size=4)
    p1 = learn_global_prototypes(blocks, 4, seed=7)
    p2 = learn_global_prototypes(blocks, 4, seed=7)
    assert p1.shape == (4, 8)
    assert np.allclose(p1, p2)


def test_assignments_and_reconstruction_shape() -> None:
    state = {"a.weight": np.arange(64, dtype=np.float32).reshape(8, 8)}
    blocks, _ = collect_blocks_from_state_dict(state, block_size=8, min_tensor_size=4)
    prototypes = learn_global_prototypes(blocks, 4, seed=7)
    assignments = assign_global_prototypes(blocks, prototypes)
    restored = reconstruct_blocks_from_global_prototypes(assignments, prototypes)
    assert assignments.dtype == np.int32
    assert restored.shape == blocks.shape
