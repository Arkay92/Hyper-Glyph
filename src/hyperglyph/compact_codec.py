"""Compact Hyper Glyph codec mode."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .blocks import restore_tensor_shape, split_array_blocks
from .config import HyperGlyphConfig
from .global_codebook import assign_global_prototypes, learn_global_prototypes
from .metrics import mae, max_abs_error, mse
from .packing import pack_uint4, rle_decode_uint, rle_encode_uint, unpack_uint4
from .quantization import QuantizedArray, dequantize_int4_packed, quantize_int4_packed


@dataclass(slots=True)
class CompactCompressedModel:
    """Compact compressed model backed by binary streams."""

    metadata: dict[str, Any]
    streams: dict[str, bytes]
    format_version: str = "0.4"
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
        if self.config.compact_tensor_codec in {"auto", "codebook"}:
            return self._compress_state_dict_codebook(state_dict)

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
            "format_version": "0.4",
            "codec_version": "0.4.0",
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

    def _compress_state_dict_codebook(
        self, state_dict: Mapping[str, Any]
    ) -> CompactCompressedModel:
        assignments_stream = bytearray()
        scales_stream = bytearray()
        tensors: list[dict[str, Any]] = []
        skipped: list[str] = []
        tensor_blocks: list[tuple[str, tuple[int, ...], list[np.ndarray]]] = []
        all_blocks: list[np.ndarray] = []

        for name, tensor in state_dict.items():
            array = np.asarray(tensor, dtype=np.float32)
            if not self._should_compress(name, array):
                skipped.append(name)
                continue
            blocks = split_array_blocks(array, self.config.block_size)
            tensor_blocks.append((name, tuple(array.shape), blocks))
            all_blocks.extend(blocks)

        if not all_blocks:
            return CompactCompressedModel(
                metadata={
                    "format_version": "0.4",
                    "codec_version": "0.4.0",
                    "mode": "compact",
                    "tensor_count": 0,
                    "skipped_tensors": skipped,
                    "config": {"compact_tensor_codec": "codebook"},
                    "tensors": [],
                },
                streams={
                    "assignments": b"",
                    "scales": b"",
                    "zero_points": b"",
                    "prototypes": b"",
                    "residual_indices": b"",
                    "residual_values": b"",
                },
                format_version="0.4",
                archive_compression=self.config.archive_compression,
                zstd_level=self.config.zstd_level,
            )

        block_matrix: np.ndarray = np.stack(all_blocks, axis=0).astype(np.float32)
        prototypes: np.ndarray = learn_global_prototypes(
            block_matrix,
            min(self.config.n_global_prototypes, len(all_blocks)),
            seed=self.config.seed,
            normalize=True,
        )
        proto_values: np.ndarray
        proto_scales: np.ndarray
        proto_values, proto_scales = _quantize_prototypes_int8(prototypes)
        prototypes_stream = proto_values.tobytes() + proto_scales.astype(np.float32).tobytes()
        proto_scale_offset = proto_values.nbytes

        cursor = 0
        for name, shape, blocks in tensor_blocks:
            count = len(blocks)
            tensor_matrix: np.ndarray = np.stack(blocks, axis=0).astype(np.float32)
            assignments: np.ndarray = assign_global_prototypes(tensor_matrix, prototypes)
            grouped_assignments: np.ndarray = _group_assignments(
                assignments, self.config.assignment_group_size
            )
            assignments = _expand_group_assignments(
                grouped_assignments, self.config.assignment_group_size, count
            )
            base: np.ndarray = prototypes[assignments]
            scales: np.ndarray = _calibrated_block_scales(tensor_matrix, base)
            scale_dtype = np.float16 if self.config.scale_dtype == "float16" else np.float32
            scale_values: np.ndarray = scales.astype(scale_dtype)
            assignment_payload, assignment_encoding = _encode_assignments(
                grouped_assignments,
                len(prototypes),
                self.config.assignment_encoding,
            )
            assignment_offset = len(assignments_stream)
            scale_offset = len(scales_stream)
            assignments_stream.extend(assignment_payload)
            scales_stream.extend(scale_values.tobytes())
            tensors.append(
                {
                    "name": name,
                    "shape": list(shape),
                    "codec": "codebook",
                    "block_size": self.config.block_size,
                    "block_count": count,
                    "assignment_count": int(grouped_assignments.size),
                    "assignment_offset": assignment_offset,
                    "assignment_length": len(assignment_payload),
                    "assignment_encoding": assignment_encoding,
                    "assignment_group_size": self.config.assignment_group_size,
                    "scale_offset": scale_offset,
                    "scale_length": scale_values.nbytes,
                    "scale_dtype": self.config.scale_dtype,
                    "global_block_offset": cursor,
                }
            )
            cursor += count

        streams = {
            "assignments": bytes(assignments_stream),
            "scales": bytes(scales_stream),
            "zero_points": b"",
            "prototypes": prototypes_stream,
            "residual_indices": b"",
            "residual_values": b"",
        }
        metadata = {
            "format_version": "0.4",
            "codec_version": "0.4.0",
            "mode": "compact",
            "tensor_count": len(tensors),
            "skipped_tensors": skipped,
            "config": {
                "compact_tensor_codec": "codebook",
                "scale_mode": "per_block",
                "scale_dtype": self.config.scale_dtype,
                "assignment_packing": self.config.assignment_encoding,
                "assignment_group_size": self.config.assignment_group_size,
                "n_global_prototypes": int(len(prototypes)),
                "prototype_bits": 8,
                "target_ratio": self.config.target_ratio,
            },
            "codebook": {
                "shape": list(prototypes.shape),
                "dtype": "int8",
                "scale_dtype": "float32",
                "value_offset": 0,
                "value_length": proto_values.nbytes,
                "scale_offset": proto_scale_offset,
                "scale_length": proto_scales.astype(np.float32).nbytes,
            },
            "tensors": tensors,
        }
        breakdown = payload_breakdown(metadata, streams)
        return CompactCompressedModel(
            metadata=metadata,
            streams=streams,
            format_version="0.4",
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
            if tensor_meta.get("codec") == "codebook":
                restored[tensor_meta["name"]] = decompress_tensor_codebook(
                    tensor_meta, compressed_model.metadata, compressed_model.streams
                )
            else:
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
    if config.compact_tensor_codec == "codebook":
        return "codebook"
    if config.choose_best_tensor_codec:
        return "packed_int4"
    return "packed_int4"


def _quantize_prototypes_int8(prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales: np.ndarray = np.max(np.abs(prototypes), axis=1) / 127.0
    scales = np.where(scales == 0, 1.0, scales).astype(np.float32)
    values: np.ndarray = np.clip(np.rint(prototypes / scales[:, None]), -127, 127).astype(np.int8)
    return values, scales


def _dequantize_prototypes_int8(
    metadata: Mapping[str, Any], streams: Mapping[str, bytes]
) -> np.ndarray:
    codebook = metadata["codebook"]
    shape = tuple(int(value) for value in codebook["shape"])
    proto_stream = streams["prototypes"]
    value_start = int(codebook["value_offset"])
    value_end = value_start + int(codebook["value_length"])
    scale_start = int(codebook["scale_offset"])
    scale_end = scale_start + int(codebook["scale_length"])
    values: np.ndarray = np.frombuffer(proto_stream[value_start:value_end], dtype=np.int8).reshape(
        shape
    )
    scales: np.ndarray = np.frombuffer(proto_stream[scale_start:scale_end], dtype=np.float32)
    return values.astype(np.float32) * scales[:, None]


def _calibrated_block_scales(blocks: np.ndarray, base: np.ndarray) -> np.ndarray:
    numerator: np.ndarray = np.sum(blocks * base, axis=1)
    denominator: np.ndarray = np.sum(base * base, axis=1) + 1e-8
    return (numerator / denominator).astype(np.float32)


def _group_assignments(assignments: np.ndarray, group_size: int) -> np.ndarray:
    if group_size <= 1:
        return np.asarray(assignments, dtype=np.int32)
    return np.asarray(assignments, dtype=np.int32)[::group_size].copy()


def _expand_group_assignments(
    assignments: np.ndarray, group_size: int, block_count: int
) -> np.ndarray:
    if group_size <= 1:
        return np.asarray(assignments, dtype=np.int32)
    expanded: np.ndarray = np.repeat(np.asarray(assignments, dtype=np.int32), group_size)
    return expanded[:block_count].astype(np.int32)


def _encode_assignments(
    assignments: np.ndarray, n_prototypes: int, encoding: str
) -> tuple[bytes, str]:
    values: np.ndarray = np.asarray(assignments, dtype=np.uint32)
    raw_payload = _raw_assignment_bytes(values, n_prototypes)
    raw_encoding = "uint4" if n_prototypes <= 16 else _raw_assignment_encoding(n_prototypes)
    if encoding == "raw":
        return raw_payload, raw_encoding
    if encoding == "uint4":
        return pack_uint4(values.astype(np.uint8)), "uint4"
    rle_payload = rle_encode_uint(values)
    if encoding == "rle" or (encoding == "auto" and len(rle_payload) < len(raw_payload)):
        return rle_payload, "rle"
    return raw_payload, raw_encoding


def _decode_assignments(payload: bytes, encoding: str, count: int, n_prototypes: int) -> np.ndarray:
    if encoding == "rle":
        return rle_decode_uint(payload, count).astype(np.int32)
    if encoding == "uint4":
        return unpack_uint4(payload, count).astype(np.int32)
    dtype = np.uint8 if n_prototypes <= 256 else np.uint16
    return np.frombuffer(payload, dtype=dtype, count=count).astype(np.int32)


def _raw_assignment_encoding(n_prototypes: int) -> str:
    if n_prototypes <= 256:
        return "uint8"
    if n_prototypes <= 65535:
        return "uint16"
    return "uint32"


def _raw_assignment_bytes(values: np.ndarray, n_prototypes: int) -> bytes:
    if n_prototypes <= 16:
        return pack_uint4(values.astype(np.uint8))
    if n_prototypes <= 256:
        return values.astype(np.uint8).tobytes()
    if n_prototypes <= 65535:
        return values.astype(np.uint16).tobytes()
    return values.astype(np.uint32).tobytes()


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
    scales: np.ndarray = np.frombuffer(
        streams["scales"][scale_start:scale_end],
        dtype=scale_dtype,
    ).astype(np.float32)
    zeros: np.ndarray = np.frombuffer(
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


def decompress_tensor_codebook(
    tensor_meta: Mapping[str, Any],
    metadata: Mapping[str, Any],
    streams: Mapping[str, bytes],
) -> np.ndarray:
    """Decompress one global-codebook tensor."""
    prototypes: np.ndarray = _dequantize_prototypes_int8(metadata, streams)
    n_prototypes = int(prototypes.shape[0])
    assignment_start = int(tensor_meta["assignment_offset"])
    assignment_end = assignment_start + int(tensor_meta["assignment_length"])
    assignment_payload = streams["assignments"][assignment_start:assignment_end]
    assignments: np.ndarray = _decode_assignments(
        assignment_payload,
        str(tensor_meta["assignment_encoding"]),
        int(tensor_meta["assignment_count"]),
        n_prototypes,
    )
    assignments = _expand_group_assignments(
        assignments,
        int(tensor_meta.get("assignment_group_size", 1)),
        int(tensor_meta["block_count"]),
    )
    scale_dtype = np.float16 if tensor_meta["scale_dtype"] == "float16" else np.float32
    scale_start = int(tensor_meta["scale_offset"])
    scale_end = scale_start + int(tensor_meta["scale_length"])
    scales: np.ndarray = np.frombuffer(
        streams["scales"][scale_start:scale_end], dtype=scale_dtype
    ).astype(np.float32)
    blocks: np.ndarray = prototypes[assignments] * scales[:, None]
    flat: np.ndarray = blocks.reshape(-1)
    shape = tuple(int(value) for value in tensor_meta["shape"])
    return restore_tensor_shape(flat[: int(np.prod(shape))], shape)


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
