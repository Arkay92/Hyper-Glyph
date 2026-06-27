"""Configuration dataclasses for Hyper Glyph."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HyperGlyphConfig:
    """Configuration for the Hyper Glyph compression codec."""

    hdc_dim: int = 4096
    block_size: int = 16
    n_buckets: int = 16
    n_prototypes: int = 128
    residual_k: int = 8
    seed: int = 42
    min_tensor_size: int = 256
    compress_bias: bool = False
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
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
        if self.min_tensor_size <= 0:
            raise ValueError("min_tensor_size must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
