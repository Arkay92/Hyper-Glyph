import numpy as np

from hyperglyph.residual_budget import (
    allocate_residual_budget,
    collect_residual_candidates,
    decode_residual_stream,
    encode_residual_stream,
)


def test_residual_budget_respects_byte_budget() -> None:
    original = np.array([0.0, 10.0, 1.0, -5.0], dtype=np.float32)
    reconstructed = np.zeros_like(original)
    candidates = collect_residual_candidates(original, reconstructed)
    selected = allocate_residual_budget(candidates, byte_budget=6)
    assert len(selected) <= 2


def test_top_errors_selected_first() -> None:
    original = np.array([0.0, 10.0, 1.0, -5.0], dtype=np.float32)
    candidates = collect_residual_candidates(original, np.zeros_like(original))
    selected = allocate_residual_budget(candidates, byte_budget=3)
    assert selected[0].flat_index == 1


def test_residual_decode_restores_selected_entries() -> None:
    original = np.array([0.0, 10.0, 1.0, -5.0], dtype=np.float32)
    candidates = allocate_residual_budget(
        collect_residual_candidates(original, np.zeros_like(original)),
        byte_budget=6,
    )
    stream = encode_residual_stream(candidates)
    indices, values = decode_residual_stream(stream)
    restored = np.zeros_like(original)
    restored[indices] += values
    assert restored[1] > 9.0
