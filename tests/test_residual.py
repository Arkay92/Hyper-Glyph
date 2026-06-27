import numpy as np

from hyperglyph.residual import apply_residual, compute_topk_residual


def test_topk_residual_captures_largest_errors() -> None:
    original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    reconstructed = np.array([1.1, 1.8, 3.2], dtype=np.float32)
    residual = compute_topk_residual(original, reconstructed, 2)
    assert len(residual["indices"]) == 2
    assert residual["indices"][0] == 2


def test_applying_residual_changes_reconstructed_block_correctly() -> None:
    block = np.array([1.0, 2.0], dtype=np.float32)
    residual = {"indices": [1], "values": [0.5]}
    updated = apply_residual(block, residual)
    assert updated[1] == 2.5


def test_k_zero_works() -> None:
    residual = compute_topk_residual(np.ones(4), np.zeros(4), 0)
    assert residual == {"indices": [], "values": [], "dtype": "int8"}


def test_int8_residual_quantization_round_trips_close_values() -> None:
    original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    reconstructed = np.array([1.1, 1.8, 3.2], dtype=np.float32)
    residual = compute_topk_residual(original, reconstructed, 2, dtype="int8")

    assert residual["dtype"] == "int8"
    assert all(isinstance(value, int) for value in residual["values"])

    updated = apply_residual(reconstructed, residual)
    assert np.max(np.abs(updated - original)) < 0.21
