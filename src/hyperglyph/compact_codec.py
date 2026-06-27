"""Compact Hyper Glyph codec mode."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .config import HyperGlyphConfig
from .metrics import mae, max_abs_error, mse
from .quantization import QuantizedArray, dequantize_int4_packed, quantize_int4_packed


@dataclass(slots=True)
class CompactCompressedModel:
    """Compact compressed model backed by binary streams."""

    metadata: dict[str, Any]
    streams: dict[str, bytes]
    format_version: str = "0.3"
    mode: str = "compact"
    archive_compression: str = "zstd"
    zstd_level: int = 9
    payload_breakdown: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class CompactCompressionReport:
    """Report for a compact compression run."""

    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    tensors_compressed: int
    tensors_skipped: int
    total_mse: float
    total_mae: float
    max_abs_error: float
    payload_breakdown: dict[str, int]


class CompactHyperGlyphCodec:
    """Compact byte-oriented codec."""

    def __init__(self, config: HyperGlyphConfig | None = None) -> None:
        self.config = config or HyperGlyphConfig(mode="compact")

    def compress_state_dict(self, state_dict: Mapping[str, Any]) -> CompactCompressedModel:
        """Compress a state_dict into compact binary streams."""
        values_stream = bytearray()
        scales_stream = bytearray()
        zeros_stream = bytearray()
        tensors: list[dict[str, Any]] = []
        skipped: list[str] = []

        for name, tensor in state_dict.items():
            array = np.asarray(tensor, dtype=np.float32)
            if not self._should_compress(name, array):
                skipped.append(name)
                continue
            tensor_meta = compress_tensor_prototype(
                name,
                array,
                self.config,
                values_stream,
                scales_stream,
                zeros_stream,
            )
            tensors.append(tensor_meta)

        streams = {
            "assignments": bytes(values_stream),
            "scales": bytes(scales_stream),
            "zero_points": bytes(zeros_stream),
            "prototypes": b"",
            "residual_indices": b"",
            "residual_values": b"",
        }
        metadata = {
            "format_version": "0.3",
            "codec_version": "0.3.0",
            "mode": "compact",
            "tensor_count": len(tensors),
            "skipped_tensors": skipped,
            "config": {
                "scale_mode": self.config.scale_mode,
                "scale_dtype": self.config.scale_dtype,
                "assignment_packing": "uint4",
                "target_ratio": self.config.target_ratio,
            },
            "tensors": tensors,
        }
        breakdown = payload_breakdown(metadata, streams)
        return CompactCompressedModel(
            metadata=metadata,
            streams=streams,
            archive_compression=self.config.archive_compression,
            zstd_level=self.config.zstd_level,
            payload_breakdown=breakdown,
        )

    def decompress_state_dict(
        self, compressed_model: CompactCompressedModel
    ) -> dict[str, np.ndarray]:
        """Decompress a compact model."""
        restored: dict[str, np.ndarray] = {}
        for tensor_meta in compressed_model.metadata.get("tensors", []):
            restored[tensor_meta["name"]] = decompress_tensor_prototype(
                tensor_meta, compressed_model.streams
            )
        return restored

    def report(
        self,
        compressed_model: CompactCompressedModel,
        original_state_dict: Mapping[str, Any] | None = None,
        restored_state_dict: Mapping[str, Any] | None = None,
    ) -> CompactCompressionReport:
        """Report compact size and reconstruction quality."""
        original_bytes = sum(
            np.asarray(tensor).nbytes for tensor in (original_state_dict or {}).values()
        )
        compressed_bytes = sum(compressed_model.payload_breakdown.values())
        ratio = original_bytes / compressed_bytes if compressed_bytes else float("inf")
        total_mse = 0.0
        total_mae = 0.0
        max_error = 0.0
        if original_state_dict is not None and restored_state_dict is not None:
            for name, restored in restored_state_dict.items():
                if name not in original_state_dict:
                    continue
                original = np.asarray(original_state_dict[name], dtype=np.float32)
                total_mse += mse(original, restored)
                total_mae += mae(original, restored)
                max_error = max(max_error, max_abs_error(original, restored))
        return CompactCompressionReport(
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=ratio,
            tensors_compressed=len(compressed_model.metadata.get("tensors", [])),
            tensors_skipped=len(compressed_model.metadata.get("skipped_tensors", [])),
            total_mse=total_mse,
            total_mae=total_mae,
            max_abs_error=max_error,
            payload_breakdown=dict(compressed_model.payload_breakdown),
        )

    def _should_compress(self, name: str, array: np.ndarray) -> bool:
        if array.size < self.config.min_tensor_size:
            return False
        if self.config.compress_bias:
            return True
        lowered = name.lower()
        return "bias" not in lowered and "norm" not in lowered


def compress_state_dict_compact(
    state_dict: Mapping[str, Any], config: HyperGlyphConfig | None = None
) -> CompactCompressedModel:
    """Compress a state_dict with the compact codec."""
    return CompactHyperGlyphCodec(config).compress_state_dict(state_dict)


def decompress_state_dict_compact(
    compressed_model: CompactCompressedModel,
) -> dict[str, np.ndarray]:
    """Decompress a compact compressed model."""
    return CompactHyperGlyphCodec().decompress_state_dict(compressed_model)


def choose_tensor_codec(array: np.ndarray, config: HyperGlyphConfig) -> str:
    """Choose the tensor codec for compact mode."""
    if config.choose_best_tensor_codec:
        return "packed_int4"
    return "packed_int4"


def compress_tensor_prototype(
    name: str,
    array: np.ndarray,
    config: HyperGlyphConfig,
    values_stream: bytearray,
    scales_stream: bytearray,
    zeros_stream: bytearray,
) -> dict[str, Any]:
    """Compress one tensor with packed int4 affine values."""
    axis = 0 if config.scale_mode == "per_channel" and array.ndim >= 2 else None
    quantized = quantize_int4_packed(array, axis=axis)
    values_offset = len(values_stream)
    packed_values = bytes(quantized.values)
    values_stream.extend(packed_values)

    scale_dtype = np.float16 if config.scale_dtype == "float16" else np.float32
    scales = np.asarray(quantized.scale, dtype=scale_dtype)
    zeros = np.asarray(quantized.zero_point, dtype=scale_dtype)
    scales_offset = len(scales_stream)
    zeros_offset = len(zeros_stream)
    scales_stream.extend(scales.tobytes())
    zeros_stream.extend(zeros.tobytes())

    return {
        "name": name,
        "shape": list(array.shape),
        "codec": choose_tensor_codec(array, config),
        "bits": 4,
        "axis": axis,
        "value_offset": values_offset,
        "value_length": len(packed_values),
        "scale_offset": scales_offset,
        "scale_length": scales.nbytes,
        "zero_offset": zeros_offset,
        "zero_length": zeros.nbytes,
        "scale_dtype": config.scale_dtype,
    }


def decompress_tensor_prototype(
    tensor_meta: Mapping[str, Any], streams: Mapping[str, bytes]
) -> np.ndarray:
    """Decompress one packed int4 tensor."""
    scale_dtype = np.float16 if tensor_meta["scale_dtype"] == "float16" else np.float32
    shape = tuple(int(value) for value in tensor_meta["shape"])
    value_start = int(tensor_meta["value_offset"])
    value_end = value_start + int(tensor_meta["value_length"])
    scale_start = int(tensor_meta["scale_offset"])
    scale_end = scale_start + int(tensor_meta["scale_length"])
    zero_start = int(tensor_meta["zero_offset"])
    zero_end = zero_start + int(tensor_meta["zero_length"])
    values = streams["assignments"][value_start:value_end]
    scales = np.frombuffer(
        streams["scales"][scale_start:scale_end],
        dtype=scale_dtype,
    ).astype(np.float32)
    zeros = np.frombuffer(
        streams["zero_points"][zero_start:zero_end],
        dtype=scale_dtype,
    ).astype(np.float32)
    payload = QuantizedArray(
        values=values,
        scale=scales,
        zero_point=zeros,
        shape=shape,
        axis=tensor_meta["axis"],
        bits=4,
    )
    return dequantize_int4_packed(payload)


def compress_tensor_low_rank(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Placeholder extension point for low-rank tensor compression."""
    raise NotImplementedError("low-rank fallback is planned for a future compact codec update")


def decompress_tensor_low_rank(*args: Any, **kwargs: Any) -> np.ndarray:
    """Placeholder extension point for low-rank tensor decompression."""
    raise NotImplementedError("low-rank fallback is planned for a future compact codec update")


def payload_breakdown(metadata: Mapping[str, Any], streams: Mapping[str, bytes]) -> dict[str, int]:
    """Return compact payload byte breakdown before archive container overhead."""
    return {
        "metadata_bytes": len(json.dumps(metadata, sort_keys=True).encode("utf-8")),
        "prototype_bytes": len(streams.get("prototypes", b"")),
        "assignment_bytes": len(streams.get("assignments", b"")),
        "scale_bytes": len(streams.get("scales", b"")) + len(streams.get("zero_points", b"")),
        "residual_index_bytes": len(streams.get("residual_indices", b"")),
        "residual_value_bytes": len(streams.get("residual_values", b"")),
    }
