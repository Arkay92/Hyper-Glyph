"""Hyper Glyph package."""

from .benchmark import BaselineComparison, BenchmarkReport, benchmark_state_dict
from .codec import CompressedModel, CompressedTensor, CompressionReport, HyperGlyphCodec
from .compact_codec import CompactCompressedModel, CompactHyperGlyphCodec
from .config import HyperGlyphConfig
from .serialization import load_compressed, save_compressed
from .torch_adapter import compress_state_dict, decompress_state_dict

__all__ = [
    "HyperGlyphCodec",
    "CompactHyperGlyphCodec",
    "HyperGlyphConfig",
    "BaselineComparison",
    "BenchmarkReport",
    "CompressionReport",
    "CompressedModel",
    "CompactCompressedModel",
    "CompressedTensor",
    "compress_state_dict",
    "decompress_state_dict",
    "save_compressed",
    "load_compressed",
    "benchmark_state_dict",
]
__version__ = "0.4.0"
