"""Configuration dataclasses for Hyper Glyph."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HyperGlyphConfig:
    """Configuration for the Hyper Glyph compression codec."""

    mode: str = "compact"
    hdc_dim: int = 4096
    block_size: int = 16
    n_buckets: int = 16
    n_prototypes: int = 128
    residual_k: int = 8
    residual_dtype: str = "int8"
    scale_mode: str = "per_channel"
    seed: int = 42
    min_tensor_size: int = 256
    compress_bias: bool = False
    dtype: str = "float32"
    device: str = "cpu"
    global_prototypes: bool = True
    n_global_prototypes: int = 16
    prototype_bits: int = 8
    prototype_dtype: str = "int8"
    prototype_symmetric: bool = True
    prototype_scale_dtype: str = "float16"
    prototype_scale_mode: str = "per_prototype"
    assignment_packing: str = "auto"
    pack_uint4_assignments: bool = True
    scale_dtype: str = "float16"
    calibrate_scales: bool = True
    residual_mode: str = "budget"
    residual_bits: int = 8
    residual_budget_ratio: float = 0.03
    residual_max_k: int = 4
    residual_threshold: float | None = None
    residual_index_encoding: str = "delta_varint"
    archive_compression: str = "zstd"
    zstd_level: int = 9
    enable_low_rank_fallback: bool = True
    low_rank_values: tuple[int, ...] = (4, 8, 16, 32)
    low_rank_bits: int = 8
    choose_best_tensor_codec: bool = True
    target_mse: float | None = None
    target_ratio: float = 6.0
    benchmark_mode: bool = False
    compact_tensor_codec: str = "codebook"
    assignment_encoding: str = "auto"
    assignment_group_size: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"standard", "compact"}:
            raise ValueError("mode must be 'standard' or 'compact'")
        if self.hdc_dim <= 0:
            raise ValueError("hdc_dim must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.n_buckets <= 0:
            raise ValueError("n_buckets must be positive")
        if self.n_prototypes <= 0:
            raise ValueError("n_prototypes must be positive")
        if self.residual_k < 0:
            raise ValueError("residual_k must be non-negative")
        if self.residual_dtype not in {"float32", "int8"}:
            raise ValueError("residual_dtype must be 'float32' or 'int8'")
        if self.scale_mode in {"tensor", "channel", "block"}:
            self.scale_mode = f"per_{self.scale_mode}"
        if self.scale_mode not in {"none", "per_block", "per_tensor", "per_channel"}:
            raise ValueError(
                "scale_mode must be 'none', 'per_block', 'per_tensor', or 'per_channel'"
            )
        if self.min_tensor_size <= 0:
            raise ValueError("min_tensor_size must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.n_global_prototypes <= 0:
            raise ValueError("n_global_prototypes must be positive")
        if self.prototype_bits not in {4, 8}:
            raise ValueError("prototype_bits must be 4 or 8")
        if self.prototype_dtype not in {"int8", "int4"}:
            raise ValueError("prototype_dtype must be 'int8' or 'int4'")
        if self.scale_dtype not in {"float16", "float32"}:
            raise ValueError("scale_dtype must be 'float16' or 'float32'")
        if self.residual_mode not in {"none", "fixed", "budget"}:
            raise ValueError("residual_mode must be 'none', 'fixed', or 'budget'")
        if self.residual_bits not in {4, 8}:
            raise ValueError("residual_bits must be 4 or 8")
        if self.residual_budget_ratio < 0:
            raise ValueError("residual_budget_ratio must be non-negative")
        if self.residual_max_k < 0:
            raise ValueError("residual_max_k must be non-negative")
        if self.compact_tensor_codec not in {"auto", "codebook", "packed_int4"}:
            raise ValueError("compact_tensor_codec must be 'auto', 'codebook', or 'packed_int4'")
        if self.assignment_encoding not in {"auto", "raw", "uint4", "rle"}:
            raise ValueError("assignment_encoding must be 'auto', 'raw', 'uint4', or 'rle'")
        if self.assignment_group_size <= 0:
            raise ValueError("assignment_group_size must be positive")
