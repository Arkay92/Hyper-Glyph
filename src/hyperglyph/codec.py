"""Main compression codec."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .blocks import restore_tensor_shape, split_array_blocks
from .compact_codec import CompactCompressedModel, CompactHyperGlyphCodec
from .config import HyperGlyphConfig
from .metrics import (
    baseline_size_bytes,
    compressed_size_bytes,
    compression_ratio,
    mae,
    max_abs_error,
    mse,
    original_size_bytes,
)
from .prototypes import assign_prototypes, learn_prototypes, reconstruct_from_prototypes
from .residual import apply_residual, compute_topk_residual, serialize_residual


@dataclass(slots=True)
class CompressedTensor:
    """Compressed representation for a single tensor."""

    name: str
    shape: tuple[int, ...]
    block_size: int
    prototype_ids: list[int]
    scales: list[float]
    residuals: list[dict[str, Any]]
    prototype_matrix: np.ndarray
    seed: int
    codec_config: dict[str, Any]


@dataclass(slots=True)
class CompressedModel:
    """Compressed representation of a model state_dict."""

    tensors: dict[str, CompressedTensor]
    payload: bytes = field(default_factory=bytes)
    format_version: str = "0.2"


@dataclass(slots=True)
class CompressionReport:
    """Report about a compression run."""

    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    fp16_estimate_bytes: int
    int8_estimate_bytes: int
    fp16_compression_ratio: float
    int8_compression_ratio: float
    tensors_compressed: int
    tensors_skipped: int
    total_mse: float
    total_mae: float
    max_abs_error: float


class HyperGlyphCodec:
    """A simple experimental compression codec for weight tensors."""

    def __init__(self, config: HyperGlyphConfig | None = None) -> None:
        self.config = config or HyperGlyphConfig()

    def compress_array(self, name: str, array: np.ndarray) -> CompressedTensor:
        """Compress a single NumPy array."""
        array = np.asarray(array, dtype=np.float32)
        if array.size < self.config.min_tensor_size:
            raise ValueError("tensor too small to compress")

        blocks = split_array_blocks(array, self.config.block_size)
        if not blocks:
            raise ValueError("no blocks available")

        block_matrix = np.stack([np.asarray(block, dtype=np.float32) for block in blocks], axis=0)
        prototypes = learn_prototypes(block_matrix, self.config.n_prototypes, self.config.seed)
        assignments = assign_prototypes(block_matrix, prototypes)
        reconstructed_prototypes = reconstruct_from_prototypes(assignments, prototypes)

        prototype_ids: list[int] = [int(idx) for idx in assignments]
        scales = self._block_scales(array, blocks, reconstructed_prototypes)
        residuals: list[dict[str, Any]] = []
        for idx, block in enumerate(blocks):
            proto = reconstructed_prototypes[idx]
            scale = scales[idx]
            proto_scaled = proto * scale
            residual = compute_topk_residual(
                block,
                proto_scaled,
                self.config.residual_k,
                dtype=self.config.residual_dtype,
            )
            residuals.append(serialize_residual(residual))

        return CompressedTensor(
            name=name,
            shape=tuple(array.shape),
            block_size=self.config.block_size,
            prototype_ids=prototype_ids,
            scales=scales,
            residuals=residuals,
            prototype_matrix=prototypes,
            seed=self.config.seed,
            codec_config={
                "hdc_dim": self.config.hdc_dim,
                "block_size": self.config.block_size,
                "n_buckets": self.config.n_buckets,
                "n_prototypes": self.config.n_prototypes,
                "residual_k": self.config.residual_k,
                "residual_dtype": self.config.residual_dtype,
                "scale_mode": self.config.scale_mode,
                "seed": self.config.seed,
                "dtype": self.config.dtype,
                "device": self.config.device,
            },
        )

    def decompress_array(self, compressed: CompressedTensor) -> np.ndarray:
        """Decompress a single tensor."""
        if not compressed.prototype_matrix.size:
            return np.zeros(compressed.shape, dtype=np.float32)

        prototype_vectors: np.ndarray = compressed.prototype_matrix.astype(np.float32)
        reconstructed_blocks: list[np.ndarray] = []
        for idx, prototype_id in enumerate(compressed.prototype_ids):
            prototype = prototype_vectors[prototype_id]
            scale = compressed.scales[idx]
            block = prototype * scale
            block = apply_residual(block, compressed.residuals[idx])
            reconstructed_blocks.append(block)

        flat = np.concatenate(reconstructed_blocks, axis=0)
        return restore_tensor_shape(flat[: int(np.prod(compressed.shape))], compressed.shape)

    def compress_state_dict(self, state_dict: Mapping[str, Any]) -> CompressedModel:
        """Compress an entire state_dict."""
        if self.config.mode == "compact":
            return CompactHyperGlyphCodec(self.config).compress_state_dict(state_dict)  # type: ignore[return-value]
        compressed_tensors: dict[str, CompressedTensor] = {}
        for name, tensor in state_dict.items():
            if not self._should_compress(name, tensor):
                continue
            compressed_tensors[name] = self.compress_array(
                name, np.asarray(tensor, dtype=np.float32)
            )
        payload = json.dumps({"tensors": list(compressed_tensors)}).encode("utf-8")
        return CompressedModel(tensors=compressed_tensors, payload=payload)

    def decompress_state_dict(self, compressed_model: CompressedModel) -> dict[str, np.ndarray]:
        """Reconstruct a state_dict from compressed data."""
        if isinstance(compressed_model, CompactCompressedModel):
            return CompactHyperGlyphCodec(self.config).decompress_state_dict(compressed_model)
        restored: dict[str, np.ndarray] = {}
        for name, compressed in compressed_model.tensors.items():
            restored[name] = self.decompress_array(compressed)
        return restored

    def report(
        self,
        compressed_model: CompressedModel,
        original_state_dict: Mapping[str, Any] | None = None,
        restored_state_dict: Mapping[str, Any] | None = None,
    ) -> CompressionReport:
        """Create a report summarizing compression quality and size."""
        if isinstance(compressed_model, CompactCompressedModel):
            compact_report = CompactHyperGlyphCodec(self.config).report(
                compressed_model, original_state_dict, restored_state_dict
            )
            return CompressionReport(
                original_bytes=compact_report.original_bytes,
                compressed_bytes=compact_report.compressed_bytes,
                compression_ratio=compact_report.compression_ratio,
                fp16_estimate_bytes=baseline_size_bytes(
                    original_state_dict or {}, bytes_per_value=2
                ),
                int8_estimate_bytes=baseline_size_bytes(
                    original_state_dict or {}, bytes_per_value=1
                ),
                fp16_compression_ratio=compression_ratio(
                    compact_report.original_bytes,
                    baseline_size_bytes(original_state_dict or {}, bytes_per_value=2),
                ),
                int8_compression_ratio=compression_ratio(
                    compact_report.original_bytes,
                    baseline_size_bytes(original_state_dict or {}, bytes_per_value=1),
                ),
                tensors_compressed=compact_report.tensors_compressed,
                tensors_skipped=compact_report.tensors_skipped,
                total_mse=compact_report.total_mse,
                total_mae=compact_report.total_mae,
                max_abs_error=compact_report.max_abs_error,
            )
        original_bytes = original_size_bytes(original_state_dict or {})
        compressed_bytes = compressed_size_bytes(compressed_model)
        ratio = compression_ratio(original_bytes, compressed_bytes)
        fp16_bytes = baseline_size_bytes(original_state_dict or {}, bytes_per_value=2)
        int8_bytes = baseline_size_bytes(original_state_dict or {}, bytes_per_value=1)
        tensors_compressed = len(compressed_model.tensors)
        tensors_skipped = 0
        if original_state_dict is not None:
            tensors_skipped = sum(
                1 for name in original_state_dict if name not in compressed_model.tensors
            )

        total_mse = 0.0
        total_mae = 0.0
        max_error = 0.0
        if original_state_dict is not None and restored_state_dict is not None:
            for name in compressed_model.tensors:
                if name in original_state_dict and name in restored_state_dict:
                    original = np.asarray(original_state_dict[name], dtype=np.float32)
                    restored = np.asarray(restored_state_dict[name], dtype=np.float32)
                    total_mse += mse(original, restored)
                    total_mae += mae(original, restored)
                    max_error = max(max_error, max_abs_error(original, restored))
        return CompressionReport(
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=ratio,
            fp16_estimate_bytes=fp16_bytes,
            int8_estimate_bytes=int8_bytes,
            fp16_compression_ratio=compression_ratio(original_bytes, fp16_bytes),
            int8_compression_ratio=compression_ratio(original_bytes, int8_bytes),
            tensors_compressed=tensors_compressed,
            tensors_skipped=tensors_skipped,
            total_mse=total_mse,
            total_mae=total_mae,
            max_abs_error=max_error,
        )

    def _should_compress(self, name: str, tensor: Any) -> bool:
        if not hasattr(tensor, "shape"):
            return False
        if self.config.compress_bias:
            return int(np.prod(tensor.shape)) >= self.config.min_tensor_size
        return (
            "bias" not in name.lower() and int(np.prod(tensor.shape)) >= self.config.min_tensor_size
        )

    def _block_scales(
        self,
        array: np.ndarray,
        blocks: list[np.ndarray],
        reconstructed_prototypes: np.ndarray,
    ) -> list[float]:
        """Calculate block, tensor, or channel scale values for prototype decoding."""
        if self.config.scale_mode == "per_tensor":
            block_matrix = np.stack(blocks, axis=0).astype(np.float32)
            block_norm = float(np.linalg.norm(block_matrix))
            proto_norm = max(float(np.linalg.norm(reconstructed_prototypes)), 1e-6)
            return [block_norm / proto_norm for _ in blocks]

        if self.config.scale_mode == "per_channel":
            channel_scales = self._channel_scales(array, reconstructed_prototypes)
            channel_ids = self._block_channel_ids(array.shape, len(blocks))
            return [channel_scales[channel_id] for channel_id in channel_ids]

        scales: list[float] = []
        for idx, block in enumerate(blocks):
            block_norm = float(np.linalg.norm(block))
            proto_norm = max(float(np.linalg.norm(reconstructed_prototypes[idx])), 1e-6)
            scales.append(block_norm / proto_norm)
        return scales

    def _channel_scales(
        self, array: np.ndarray, reconstructed_prototypes: np.ndarray
    ) -> list[float]:
        if array.ndim == 0:
            return [1.0]
        channel_count = int(array.shape[0]) if array.ndim > 0 else 1
        channel_size = int(np.prod(array.shape[1:])) if array.ndim > 1 else 1
        flat_original = np.asarray(array, dtype=np.float32).reshape(-1)
        flat_reconstructed = reconstructed_prototypes.reshape(-1)[: flat_original.size]
        scales: list[float] = []
        for channel in range(channel_count):
            start = channel * channel_size
            end = min(start + channel_size, flat_original.size)
            original_norm = float(np.linalg.norm(flat_original[start:end]))
            proto_norm = max(float(np.linalg.norm(flat_reconstructed[start:end])), 1e-6)
            scales.append(original_norm / proto_norm)
        return scales

    def _block_channel_ids(self, shape: tuple[int, ...], block_count: int) -> list[int]:
        if not shape:
            return [0 for _ in range(block_count)]
        channel_count = int(shape[0])
        channel_size = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        ids: list[int] = []
        for block_index in range(block_count):
            flat_index = block_index * self.config.block_size
            channel_id = min(flat_index // max(channel_size, 1), channel_count - 1)
            ids.append(int(channel_id))
        return ids
