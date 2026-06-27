import numpy as np

from hyperglyph.blocks import (
    flatten_tensor_for_blocks,
    merge_array_blocks,
    pad_tensor_to_blocks,
    restore_tensor_shape,
    split_array_blocks,
)


def test_split_and_merge_recovers_original_array() -> None:
    array = np.arange(32, dtype=np.float32).reshape(4, 8)
    blocks = split_array_blocks(array, 8)
    padded, padded_shape = pad_tensor_to_blocks(array, 8)
    merged = merge_array_blocks(blocks, array.shape, padded_shape, 8)
    assert np.allclose(merged, array)


def test_padding_works_for_non_divisible_shapes() -> None:
    array = np.arange(10, dtype=np.float32)
    padded, padded_shape = pad_tensor_to_blocks(array, 4)
    assert padded.size % 4 == 0
    assert padded_shape == (10,)


def test_1d_tensors_work() -> None:
    array = np.arange(8, dtype=np.float32)
    flat = flatten_tensor_for_blocks(array)
    restored = restore_tensor_shape(flat, array.shape)
    assert restored.shape == array.shape


def test_2d_tensors_work() -> None:
    array = np.arange(16, dtype=np.float32).reshape(4, 4)
    blocks = split_array_blocks(array, 4)
    assert len(blocks) == 4


def test_4d_tensors_flatten_and_restore() -> None:
    array = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
    flat = flatten_tensor_for_blocks(array)
    restored = restore_tensor_shape(flat, array.shape)
    assert restored.shape == array.shape
