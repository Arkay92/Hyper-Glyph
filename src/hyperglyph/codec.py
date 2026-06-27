"""Main compression codec."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .blocks import restore_tensor_shape, split_array_blocks
from .config import HyperGlyphConfig
from .metrics import (
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
    format_version: str = "0.1"


@dataclass(slots=True)
class CompressionReport:
    """Report about a compression run."""

    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
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
        scales: list[float] = []
        residuals: list[dict[str, Any]] = []
        for idx, block in enumerate(blocks):
            proto = reconstructed_prototypes[idx]
            scale = float(np.linalg.norm(block) / max(np.linalg.norm(proto), 1e-6))
            scales.append(scale)
            proto_scaled = proto * scale
            residual = compute_topk_residual(block, proto_scaled, self.config.residual_k)
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
        original_bytes = original_size_bytes(original_state_dict or {})
        compressed_bytes = compressed_size_bytes(compressed_model)
        ratio = compression_ratio(original_bytes, compressed_bytes)
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
