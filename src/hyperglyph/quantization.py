"""Quantization helpers for compact compression and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .packing import pack_uint4, unpack_uint4


@dataclass(slots=True)
class QuantizedArray:
    """Quantized array payload and affine decode parameters."""

    values: np.ndarray | bytes
    scale: np.ndarray
    zero_point: np.ndarray
    shape: tuple[int, ...]
    axis: int | None
    bits: int


def symmetric_quantize(array: np.ndarray, bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrically quantize an array."""
    array = np.asarray(array, dtype=np.float32)
    qmax = (2 ** (bits - 1)) - 1
    scale = np.asarray(np.max(np.abs(array)) / max(qmax, 1), dtype=np.float32)
    if float(scale) == 0.0:
        scale = np.asarray(1.0, dtype=np.float32)
    quantized = np.clip(np.rint(array / scale), -qmax, qmax).astype(np.int8)
    return quantized, scale


def symmetric_dequantize(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize symmetric integer values."""
    return np.asarray(values, dtype=np.float32) * np.asarray(scale, dtype=np.float32)


def quantize_int8(array: np.ndarray, axis: int | None = None) -> QuantizedArray:
    """Affine quantize an array to uint8."""
    return _affine_quantize(array, bits=8, axis=axis)


def dequantize_int8(payload: QuantizedArray) -> np.ndarray:
    """Dequantize a uint8 payload."""
    return _affine_dequantize(payload)


def quantize_int4_packed(array: np.ndarray, axis: int | None = None) -> QuantizedArray:
    """Affine quantize an array to packed unsigned int4."""
    payload = _affine_quantize(array, bits=4, axis=axis)
    payload.values = pack_uint4(np.asarray(payload.values, dtype=np.uint8))
    return payload


def dequantize_int4_packed(payload: QuantizedArray) -> np.ndarray:
    """Dequantize a packed unsigned int4 payload."""
    length = int(np.prod(payload.shape))
    values = unpack_uint4(bytes(payload.values), length).reshape(payload.shape)
    unpacked = QuantizedArray(
        values=values,
        scale=payload.scale,
        zero_point=payload.zero_point,
        shape=payload.shape,
        axis=payload.axis,
        bits=payload.bits,
    )
    return _affine_dequantize(unpacked)


def estimate_quantized_bytes(array: np.ndarray, bits: int, axis: int | None = None) -> int:
    """Estimate bytes for affine quantized values plus scale/zero-point metadata."""
    array = np.asarray(array)
    value_bytes = (array.size * bits + 7) // 8
    n_scales = 1 if axis is None else int(array.shape[axis])
    return value_bytes + n_scales * 4


def _affine_quantize(array: np.ndarray, bits: int, axis: int | None) -> QuantizedArray:
    array = np.asarray(array, dtype=np.float32)
    qmax = (2**bits) - 1
    if axis is None or array.ndim == 0:
        minimum = np.asarray(np.min(array), dtype=np.float32)
        maximum = np.asarray(np.max(array), dtype=np.float32)
    else:
        minimum = np.min(array, axis=tuple(i for i in range(array.ndim) if i != axis))
        maximum = np.max(array, axis=tuple(i for i in range(array.ndim) if i != axis))
    scale = (maximum - minimum) / max(qmax, 1)
    scale = np.asarray(np.where(scale == 0, 1.0, scale), dtype=np.float32)
    zero_point = np.asarray(minimum, dtype=np.float32)
    shaped_scale = _reshape_param(scale, array.ndim, axis)
    shaped_zero = _reshape_param(zero_point, array.ndim, axis)
    values = np.clip(np.rint((array - shaped_zero) / shaped_scale), 0, qmax).astype(np.uint8)
    return QuantizedArray(
        values=values,
        scale=scale,
        zero_point=zero_point,
        shape=tuple(array.shape),
        axis=axis,
        bits=bits,
    )


def _affine_dequantize(payload: QuantizedArray) -> np.ndarray:
    values: np.ndarray = np.asarray(payload.values, dtype=np.float32).reshape(payload.shape)
    scale = _reshape_param(payload.scale, len(payload.shape), payload.axis)
    zero = _reshape_param(payload.zero_point, len(payload.shape), payload.axis)
    return values * scale + zero


def _reshape_param(param: np.ndarray, ndim: int, axis: int | None) -> np.ndarray:
    arr = np.asarray(param, dtype=np.float32)
    if axis is None or ndim == 0 or arr.ndim == 0:
        return arr
    shape = [1] * ndim
    shape[axis] = arr.shape[0]
    return arr.reshape(shape)
