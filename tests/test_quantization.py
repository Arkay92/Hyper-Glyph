import numpy as np

from hyperglyph.quantization import (
    dequantize_int4_packed,
    dequantize_int8,
    dequantize_uint_packed,
    estimate_quantized_bytes,
    quantize_int4_packed,
    quantize_int8,
    quantize_uint_packed,
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


def test_uint_packed_quantization_supports_all_bit_widths() -> None:
    array = np.linspace(-1, 1, 128, dtype=np.float32).reshape(16, 8)
    for bits in range(1, 9):
        payload = quantize_uint_packed(array, bits=bits)
        restored = dequantize_uint_packed(payload)
        assert payload.bits == bits
        assert restored.shape == array.shape
        assert estimate_quantized_bytes(array, bits) == len(bytes(payload.values)) + 8


def test_uint_packed_error_improves_with_more_bits() -> None:
    array = np.linspace(-1, 1, 128, dtype=np.float32)
    restored_2bit = dequantize_uint_packed(quantize_uint_packed(array, bits=2))
    restored_8bit = dequantize_uint_packed(quantize_uint_packed(array, bits=8))
    mse_2bit = float(np.mean((array - restored_2bit) ** 2))
    mse_8bit = float(np.mean((array - restored_8bit) ** 2))
    assert mse_8bit < mse_2bit


def test_zero_tensor_works() -> None:
    array = np.zeros((4, 4), dtype=np.float32)
    restored = dequantize_int4_packed(quantize_int4_packed(array, axis=0))
    assert np.allclose(restored, array)


def test_constant_tensor_works() -> None:
    array = np.full((4, 4), 3.0, dtype=np.float32)
    restored = dequantize_int4_packed(quantize_int4_packed(array, axis=0))
    assert np.allclose(restored, array)
