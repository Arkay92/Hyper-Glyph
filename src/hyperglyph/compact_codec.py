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
from .quantization import (
    QuantizedArray,
    dequantize_int4_packed,
    dequantize_int8,
    quantize_int4_packed,
    quantize_int8,
)
from .residual_budget import (
    allocate_residual_budget,
    collect_residual_candidates,
    decode_residual_stream,
    encode_residual_stream,
)


@dataclass(slots=True)
class CompactCompressedModel:
    """Compact compressed model backed by binary streams."""

    metadata: dict[str, Any]
    streams: dict[str, bytes]
    format_version: str = "0.5"
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


@dataclass(slots=True)
class TensorCandidate:
    """A compressed tensor candidate used by codec search."""

    meta: dict[str, Any]
    streams: dict[str, bytes]
    restored: np.ndarray
    size_bytes: int
    error: float


class CompactHyperGlyphCodec:
    """Compact byte-oriented codec."""

    def __init__(self, config: HyperGlyphConfig | None = None) -> None:
        self.config = config or HyperGlyphConfig(mode="compact")

    def compress_state_dict(self, state_dict: Mapping[str, Any]) -> CompactCompressedModel:
        """Compress a state_dict into compact binary streams."""
        if self.config.compact_tensor_codec == "auto":
            return self._compress_state_dict_portfolio(state_dict)
        if self.config.compact_tensor_codec == "codebook":
            return self._compress_state_dict_codebook(state_dict)
        if self.config.compact_tensor_codec == "low_rank":
            return self._compress_state_dict_portfolio(state_dict, force_codec="low_rank")

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
            "format_version": "0.5",
            "codec_version": "0.7.0",
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

    def _compress_state_dict_portfolio(
        self, state_dict: Mapping[str, Any], force_codec: str | None = None
    ) -> CompactCompressedModel:
        streams: dict[str, bytearray] = {
            "assignments": bytearray(),
            "scales": bytearray(),
            "zero_points": bytearray(),
            "prototypes": bytearray(),
            "residual_indices": bytearray(),
            "residual_values": bytearray(),
            "low_rank": bytearray(),
            "raw_values": bytearray(),
            "sparse_indices": bytearray(),
            "sparse_values": bytearray(),
        }
        tensors: list[dict[str, Any]] = []
        skipped: list[str] = []

        for name, tensor in state_dict.items():
            array = np.asarray(tensor, dtype=np.float32)
            if not self._should_compress(name, array):
                skipped.append(name)
                raw_candidate = _candidate_raw_quantized(name, array)
                _append_candidate(raw_candidate, streams, tensors)
                continue

            candidates = _build_tensor_candidates(name, array, self.config)
            if force_codec is not None:
                candidates = [
                    candidate
                    for candidate in candidates
                    if str(candidate.meta["codec"]).startswith(force_codec)
                ]
            if not candidates:
                candidates = [_candidate_packed_int4(name, array, self.config)]
            selected = _select_candidate(array, candidates, self.config)
            _append_candidate(selected, streams, tensors)

        frozen_streams = {name: bytes(data) for name, data in streams.items()}
        metadata = {
            "format_version": "0.5",
            "codec_version": "0.7.0",
            "mode": "compact",
            "tensor_count": len(tensors),
            "skipped_tensors": skipped,
            "config": {
                "compact_tensor_codec": self.config.compact_tensor_codec,
                "scale_dtype": self.config.scale_dtype,
                "assignment_encoding": self.config.assignment_encoding,
                "assignment_group_size": self.config.assignment_group_size,
                "codec_error_multiplier": self.config.codec_error_multiplier,
            },
            "tensors": tensors,
        }
        breakdown = payload_breakdown(metadata, frozen_streams)
        return CompactCompressedModel(
            metadata=metadata,
            streams=frozen_streams,
            format_version="0.5",
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
                    "format_version": "0.5",
                    "codec_version": "0.7.0",
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
                format_version="0.5",
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
            "format_version": "0.5",
            "codec_version": "0.7.0",
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
            format_version="0.5",
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
            codec_name = str(tensor_meta.get("codec", ""))
            if codec_name.startswith("codebook"):
                restored[tensor_meta["name"]] = decompress_tensor_codebook(
                    tensor_meta, compressed_model.metadata, compressed_model.streams
                )
            elif codec_name.startswith("low_rank"):
                restored[tensor_meta["name"]] = decompress_tensor_low_rank(
                    tensor_meta, compressed_model.streams
                )
            elif codec_name == "raw_int8":
                restored[tensor_meta["name"]] = decompress_tensor_raw_int8(
                    tensor_meta, compressed_model.streams
                )
            elif codec_name == "sparse":
                restored[tensor_meta["name"]] = decompress_tensor_sparse(
                    tensor_meta, compressed_model.streams
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


def _build_tensor_candidates(
    name: str, array: np.ndarray, config: HyperGlyphConfig
) -> list[TensorCandidate]:
    candidates = [
        _candidate_packed_int4(name, array, config),
        _candidate_raw_quantized(name, array),
    ]
    block_count = (array.size + config.block_size - 1) // config.block_size
    if block_count <= config.auto_max_codebook_blocks:
        candidates.append(_candidate_block_codebook(name, array, config))
    if array.ndim == 2 and min(array.shape) >= 2 and array.size <= config.auto_max_svd_elements:
        candidates.extend(_candidate_low_rank(name, array, config))
    sparse = _candidate_sparse(name, array)
    if sparse is not None:
        candidates.append(sparse)
    return candidates


def _select_candidate(
    array: np.ndarray, candidates: list[TensorCandidate], config: HyperGlyphConfig
) -> TensorCandidate:
    int4_error = min(
        (candidate.error for candidate in candidates if candidate.meta["codec"] == "packed_int4"),
        default=min(candidate.error for candidate in candidates),
    )
    threshold = config.target_mse
    if threshold is None:
        threshold = int4_error * config.codec_error_multiplier
    eligible = [candidate for candidate in candidates if candidate.error <= threshold]
    if eligible:
        return min(eligible, key=lambda candidate: candidate.size_bytes)
    return min(candidates, key=lambda candidate: (candidate.error, candidate.size_bytes))


def _append_candidate(
    candidate: TensorCandidate,
    streams: dict[str, bytearray],
    tensors: list[dict[str, Any]],
) -> None:
    meta = dict(candidate.meta)
    offset_keys = {
        "assignments": ["value_offset", "assignment_offset"],
        "scales": ["scale_offset"],
        "zero_points": ["zero_offset"],
        "prototypes": ["prototype_offset"],
        "residual_indices": ["residual_index_offset"],
        "residual_values": ["residual_value_offset"],
        "low_rank": ["u_offset", "v_offset"],
        "raw_values": ["raw_offset", "value_offset"],
        "sparse_indices": ["sparse_index_offset"],
        "sparse_values": ["sparse_value_offset"],
    }
    for stream_name, payload in candidate.streams.items():
        base = len(streams[stream_name])
        for key in offset_keys.get(stream_name, []):
            if key in meta:
                meta[key] = int(meta[key]) + base
        streams[stream_name].extend(payload)
    tensors.append(meta)


def _candidate_packed_int4(
    name: str, array: np.ndarray, config: HyperGlyphConfig
) -> TensorCandidate:
    values_stream = bytearray()
    scales_stream = bytearray()
    zeros_stream = bytearray()
    meta = compress_tensor_prototype(
        name, array, config, values_stream, scales_stream, zeros_stream
    )
    meta["codec"] = "packed_int4"
    meta["value_stream"] = "raw_values"
    streams = {
        "raw_values": bytes(values_stream),
        "scales": bytes(scales_stream),
        "zero_points": bytes(zeros_stream),
    }
    restored = decompress_tensor_prototype(meta, streams)
    return TensorCandidate(
        meta=meta,
        streams=streams,
        restored=restored,
        size_bytes=_candidate_size(meta, streams),
        error=mse(array, restored),
    )


def _candidate_block_codebook(
    name: str, array: np.ndarray, config: HyperGlyphConfig
) -> TensorCandidate:
    blocks = split_array_blocks(array, config.block_size)
    block_matrix = np.stack(blocks, axis=0).astype(np.float32)
    prototypes = learn_global_prototypes(
        block_matrix,
        min(config.n_global_prototypes, len(blocks)),
        seed=config.seed,
        normalize=True,
    )
    assignments = assign_global_prototypes(block_matrix, prototypes)
    grouped = _group_assignments(assignments, config.assignment_group_size)
    expanded = _expand_group_assignments(grouped, config.assignment_group_size, len(blocks))
    base = prototypes[expanded]
    scales = _calibrated_block_scales(block_matrix, base)
    proto_values, proto_scales = _quantize_prototypes_int8(prototypes)
    prototype_stream = proto_values.tobytes() + proto_scales.astype(np.float32).tobytes()
    scale_dtype = np.float16 if config.scale_dtype == "float16" else np.float32
    scale_values: np.ndarray = scales.astype(scale_dtype)
    assignments_payload, assignment_encoding = _encode_assignments(
        grouped, len(prototypes), config.assignment_encoding
    )
    meta = {
        "name": name,
        "shape": list(array.shape),
        "codec": "codebook",
        "block_size": config.block_size,
        "block_count": len(blocks),
        "assignment_count": int(grouped.size),
        "assignment_offset": 0,
        "assignment_length": len(assignments_payload),
        "assignment_encoding": assignment_encoding,
        "assignment_group_size": config.assignment_group_size,
        "scale_offset": 0,
        "scale_length": scale_values.nbytes,
        "scale_dtype": config.scale_dtype,
        "prototype_offset": 0,
        "prototype_length": len(prototype_stream),
        "prototype_shape": list(prototypes.shape),
        "prototype_value_length": proto_values.nbytes,
        "prototype_scale_offset": proto_values.nbytes,
        "prototype_scale_length": proto_scales.astype(np.float32).nbytes,
    }
    streams = {
        "assignments": assignments_payload,
        "scales": scale_values.tobytes(),
        "prototypes": prototype_stream,
    }
    restored = decompress_tensor_codebook(meta, {"codebook": meta}, streams)
    residual_candidate = _add_residual_to_candidate(meta, streams, array, restored, config)
    if residual_candidate is not None:
        return residual_candidate
    return TensorCandidate(
        meta=meta,
        streams=streams,
        restored=restored,
        size_bytes=_candidate_size(meta, streams),
        error=mse(array, restored),
    )


def _candidate_low_rank(
    name: str, array: np.ndarray, config: HyperGlyphConfig
) -> list[TensorCandidate]:
    candidates: list[TensorCandidate] = []
    for rank in config.low_rank_values:
        if rank >= min(array.shape):
            continue
        try:
            u, singular, vh = np.linalg.svd(array, full_matrices=False)
        except np.linalg.LinAlgError:
            return candidates
        left = (u[:, :rank] * singular[:rank]).astype(np.float32)
        right = vh[:rank, :].astype(np.float32)
        q_left = quantize_int8(left)
        q_right = quantize_int8(right)
        left_restored = dequantize_int8(q_left)
        right_restored = dequantize_int8(q_right)
        restored = left_restored @ right_restored
        low_rank_bytes = bytes(q_left.values) + bytes(q_right.values)
        scales: np.ndarray = np.asarray(
            [
                float(np.asarray(q_left.scale).reshape(-1)[0]),
                float(np.asarray(q_left.zero_point).reshape(-1)[0]),
                float(np.asarray(q_right.scale).reshape(-1)[0]),
                float(np.asarray(q_right.zero_point).reshape(-1)[0]),
            ],
            dtype=np.float32,
        )
        meta = {
            "name": name,
            "shape": list(array.shape),
            "codec": "low_rank",
            "rank": rank,
            "u_shape": list(left.shape),
            "v_shape": list(right.shape),
            "u_offset": 0,
            "u_length": int(np.asarray(q_left.values).nbytes),
            "v_offset": int(np.asarray(q_left.values).nbytes),
            "v_length": int(np.asarray(q_right.values).nbytes),
            "scale_offset": 0,
            "scale_length": scales.nbytes,
        }
        streams = {"low_rank": low_rank_bytes, "scales": scales.tobytes()}
        candidate = TensorCandidate(
            meta=meta,
            streams=streams,
            restored=restored,
            size_bytes=_candidate_size(meta, streams),
            error=mse(array, restored),
        )
        residual_candidate = _add_residual_to_candidate(meta, streams, array, restored, config)
        candidates.append(residual_candidate or candidate)
    return candidates


def _candidate_raw_quantized(name: str, array: np.ndarray) -> TensorCandidate:
    payload = quantize_int8(array)
    values = bytes(payload.values)
    params: np.ndarray = np.asarray(
        [
            float(np.asarray(payload.scale).reshape(-1)[0]),
            float(np.asarray(payload.zero_point).reshape(-1)[0]),
        ],
        dtype=np.float32,
    )
    meta = {
        "name": name,
        "shape": list(array.shape),
        "codec": "raw_int8",
        "raw_offset": 0,
        "raw_length": len(values),
        "scale_offset": 0,
        "scale_length": params.nbytes,
    }
    streams = {"raw_values": values, "scales": params.tobytes()}
    restored = dequantize_int8(payload)
    return TensorCandidate(
        meta=meta,
        streams=streams,
        restored=restored,
        size_bytes=_candidate_size(meta, streams),
        error=mse(array, restored),
    )


def _candidate_sparse(name: str, array: np.ndarray) -> TensorCandidate | None:
    flat = array.reshape(-1)
    nonzero: np.ndarray = np.flatnonzero(np.abs(flat) > 1e-12).astype(np.int64)
    if nonzero.size > flat.size * 0.2:
        return None
    values = flat[nonzero].astype(np.float16)
    meta = {
        "name": name,
        "shape": list(array.shape),
        "codec": "sparse",
        "sparse_index_offset": 0,
        "sparse_index_length": nonzero.astype(np.uint32).nbytes,
        "sparse_value_offset": 0,
        "sparse_value_length": values.nbytes,
        "count": int(nonzero.size),
    }
    streams = {
        "sparse_indices": nonzero.astype(np.uint32).tobytes(),
        "sparse_values": values.tobytes(),
    }
    restored = np.zeros_like(array, dtype=np.float32).reshape(-1)
    restored[nonzero] = values.astype(np.float32)
    restored = restored.reshape(array.shape)
    return TensorCandidate(
        meta=meta,
        streams=streams,
        restored=restored,
        size_bytes=_candidate_size(meta, streams),
        error=mse(array, restored),
    )


def _add_residual_to_candidate(
    meta: dict[str, Any],
    streams: dict[str, bytes],
    original: np.ndarray,
    restored: np.ndarray,
    config: HyperGlyphConfig,
) -> TensorCandidate | None:
    if config.residual_budget_ratio <= 0:
        return None
    byte_budget = int(original.nbytes * config.residual_budget_ratio)
    candidates = allocate_residual_budget(
        collect_residual_candidates(original, restored, config.residual_threshold),
        byte_budget,
        max_k=max(config.residual_max_k * max(1, original.size // config.block_size), 0),
    )
    if not candidates:
        return None
    residual_stream = encode_residual_stream(candidates)
    indices, values = decode_residual_stream(residual_stream)
    repaired = restored.reshape(-1).copy()
    repaired[indices] += values
    repaired = repaired.reshape(original.shape)
    repaired_error = mse(original, repaired)
    if repaired_error >= mse(original, restored):
        return None
    residual_meta = dict(meta)
    residual_meta["codec"] = f"{meta['codec']}_residual"
    residual_meta["residual_index_offset"] = 0
    residual_meta["residual_index_length"] = len(residual_stream.index_bytes)
    residual_meta["residual_value_offset"] = 0
    residual_meta["residual_value_length"] = len(residual_stream.value_bytes)
    residual_meta["residual_scale"] = residual_stream.scale
    residual_meta["residual_count"] = residual_stream.count
    residual_streams = dict(streams)
    residual_streams["residual_indices"] = residual_stream.index_bytes
    residual_streams["residual_values"] = residual_stream.value_bytes
    return TensorCandidate(
        meta=residual_meta,
        streams=residual_streams,
        restored=repaired,
        size_bytes=_candidate_size(residual_meta, residual_streams),
        error=repaired_error,
    )


def _candidate_size(meta: Mapping[str, Any], streams: Mapping[str, bytes]) -> int:
    return len(json.dumps(meta, sort_keys=True).encode("utf-8")) + sum(
        len(payload) for payload in streams.values()
    )


def _quantize_prototypes_int8(prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales: np.ndarray = np.max(np.abs(prototypes), axis=1) / 127.0
    scales = np.where(scales == 0, 1.0, scales).astype(np.float32)
    values: np.ndarray = np.clip(np.rint(prototypes / scales[:, None]), -127, 127).astype(np.int8)
    return values, scales


def _dequantize_prototypes_int8(
    metadata: Mapping[str, Any], streams: Mapping[str, bytes]
) -> np.ndarray:
    codebook = metadata["codebook"]
    shape_values = codebook.get("prototype_shape", codebook.get("shape"))
    shape = tuple(int(value) for value in shape_values)
    proto_stream = streams["prototypes"]
    value_start = int(codebook.get("value_offset", codebook.get("prototype_offset", 0)))
    value_length = int(codebook.get("value_length", codebook.get("prototype_value_length", 0)))
    value_end = value_start + value_length
    scale_start = int(
        codebook.get("prototype_scale_offset", codebook.get("scale_offset", value_end))
    )
    scale_length = int(
        codebook.get(
            "prototype_scale_length",
            codebook.get("scale_length", int(codebook.get("prototype_length", 0)) - value_length),
        )
    )
    scale_end = scale_start + scale_length
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
    value_stream = str(tensor_meta.get("value_stream", "assignments"))
    values = streams[value_stream][value_start:value_end]
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
    restored = restore_tensor_shape(flat[: int(np.prod(shape))], shape)
    return _apply_tensor_residual(restored, tensor_meta, streams)


def decompress_tensor_low_rank(
    tensor_meta: Mapping[str, Any], streams: Mapping[str, bytes]
) -> np.ndarray:
    """Decompress one low-rank tensor."""
    u_shape = tuple(int(value) for value in tensor_meta["u_shape"])
    v_shape = tuple(int(value) for value in tensor_meta["v_shape"])
    u_start = int(tensor_meta["u_offset"])
    u_end = u_start + int(tensor_meta["u_length"])
    v_start = int(tensor_meta["v_offset"])
    v_end = v_start + int(tensor_meta["v_length"])
    scale_start = int(tensor_meta["scale_offset"])
    scale_end = scale_start + int(tensor_meta["scale_length"])
    params: np.ndarray = np.frombuffer(streams["scales"][scale_start:scale_end], dtype=np.float32)
    u_values: np.ndarray = np.frombuffer(
        streams["low_rank"][u_start:u_end], dtype=np.uint8
    ).reshape(u_shape)
    v_values: np.ndarray = np.frombuffer(
        streams["low_rank"][v_start:v_end], dtype=np.uint8
    ).reshape(v_shape)
    left = (u_values.astype(np.float32) * params[0]) + params[1]
    right = (v_values.astype(np.float32) * params[2]) + params[3]
    restored = left @ right
    return _apply_tensor_residual(restored.astype(np.float32), tensor_meta, streams)


def decompress_tensor_raw_int8(
    tensor_meta: Mapping[str, Any], streams: Mapping[str, bytes]
) -> np.ndarray:
    """Decompress one raw int8 tensor."""
    shape = tuple(int(value) for value in tensor_meta["shape"])
    start = int(tensor_meta["raw_offset"])
    end = start + int(tensor_meta["raw_length"])
    scale_start = int(tensor_meta["scale_offset"])
    scale_end = scale_start + int(tensor_meta["scale_length"])
    params: np.ndarray = np.frombuffer(streams["scales"][scale_start:scale_end], dtype=np.float32)
    values: np.ndarray = np.frombuffer(streams["raw_values"][start:end], dtype=np.uint8).reshape(
        shape
    )
    return (values.astype(np.float32) * params[0]) + params[1]


def decompress_tensor_sparse(
    tensor_meta: Mapping[str, Any], streams: Mapping[str, bytes]
) -> np.ndarray:
    """Decompress one sparse tensor."""
    shape = tuple(int(value) for value in tensor_meta["shape"])
    index_start = int(tensor_meta["sparse_index_offset"])
    index_end = index_start + int(tensor_meta["sparse_index_length"])
    value_start = int(tensor_meta["sparse_value_offset"])
    value_end = value_start + int(tensor_meta["sparse_value_length"])
    indices: np.ndarray = np.frombuffer(
        streams["sparse_indices"][index_start:index_end], dtype=np.uint32
    )
    values: np.ndarray = np.frombuffer(
        streams["sparse_values"][value_start:value_end], dtype=np.float16
    )
    restored: np.ndarray = np.zeros(int(np.prod(shape)), dtype=np.float32)
    restored[indices.astype(np.int64)] = values.astype(np.float32)
    return restored.reshape(shape)


def _apply_tensor_residual(
    restored: np.ndarray, tensor_meta: Mapping[str, Any], streams: Mapping[str, bytes]
) -> np.ndarray:
    if "residual_count" not in tensor_meta:
        return restored
    index_start = int(tensor_meta["residual_index_offset"])
    index_end = index_start + int(tensor_meta["residual_index_length"])
    value_start = int(tensor_meta["residual_value_offset"])
    value_end = value_start + int(tensor_meta["residual_value_length"])
    from .residual_budget import EncodedResidualStream

    stream = EncodedResidualStream(
        index_bytes=streams["residual_indices"][index_start:index_end],
        value_bytes=streams["residual_values"][value_start:value_end],
        scale=float(tensor_meta["residual_scale"]),
        count=int(tensor_meta["residual_count"]),
    )
    indices, values = decode_residual_stream(stream)
    repaired = restored.reshape(-1).copy()
    repaired[indices] += values
    return repaired.reshape(restored.shape)


def payload_breakdown(metadata: Mapping[str, Any], streams: Mapping[str, bytes]) -> dict[str, int]:
    """Return compact payload byte breakdown before archive container overhead."""
    return {
        "metadata_bytes": len(json.dumps(metadata, sort_keys=True).encode("utf-8")),
        "prototype_bytes": len(streams.get("prototypes", b"")),
        "assignment_bytes": len(streams.get("assignments", b"")),
        "scale_bytes": len(streams.get("scales", b"")) + len(streams.get("zero_points", b"")),
        "residual_index_bytes": len(streams.get("residual_indices", b"")),
        "residual_value_bytes": len(streams.get("residual_values", b"")),
        "low_rank_bytes": len(streams.get("low_rank", b"")),
        "raw_value_bytes": len(streams.get("raw_values", b"")),
        "sparse_index_bytes": len(streams.get("sparse_indices", b"")),
        "sparse_value_bytes": len(streams.get("sparse_values", b"")),
    }
