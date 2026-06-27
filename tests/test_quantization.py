import numpy as np

from hyperglyph.quantization import (
    dequantize_int4_packed,
    dequantize_int8,
    quantize_int4_packed,
    quantize_int8,
)


def test_int8_quant_dequant_shape() -> None:
    array = np.linspace(-1, 1, 32, dtype=np.float32).reshape(4, 8)
    payload = quantize_int8(array, axis=0)
    restored = dequantize_int8(payload)
    assert restored.shape == array.shape


def test_int8_error_reasonable() -> None:
    array = np.linspace(-1, 1, 128, dtype=np.float32)
    restored = dequantize_int8(quantize_int8(array))
    assert float(np.mean((array - restored) ** 2)) < 1e-4


def test_int4_packed_roundtrip() -> None:
    array = np.linspace(-1, 1, 128, dtype=np.float32)
    payload = quantize_int4_packed(array)
    restored = dequantize_int4_packed(payload)
    assert restored.shape == array.shape
    assert float(np.mean((array - restored) ** 2)) < 1e-2


def test_zero_tensor_works() -> None:
    array = np.zeros((4, 4), dtype=np.float32)
    restored = dequantize_int4_packed(quantize_int4_packed(array, axis=0))
    assert np.allclose(restored, array)


def test_constant_tensor_works() -> None:
    array = np.full((4, 4), 3.0, dtype=np.float32)
    restored = dequantize_int4_packed(quantize_int4_packed(array, axis=0))
    assert np.allclose(restored, array)
