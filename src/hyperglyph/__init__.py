"""Hyper Glyph package."""

from .benchmark import BaselineComparison, BenchmarkReport, benchmark_state_dict
from .codec import CompressedModel, CompressedTensor, CompressionReport, HyperGlyphCodec
from .compact_codec import CompactCompressedModel, CompactHyperGlyphCodec
from .config import HyperGlyphConfig
from .evaluation import (
    OPEN_MODEL_SUITE,
    STRONG_QUANTIZATION_LIBRARIES,
    AblationResult,
    InferenceComparison,
    PerplexityResult,
    QuantizationLibraryStatus,
    TensorError,
    ablation_configs,
    available_strong_quantization_libraries,
    compare_inference_after_decompression,
    evaluate_perplexity,
    run_ablation_study,
    tensor_error_analysis,
    tensor_error_markdown,
)
from .serialization import load_compressed, save_compressed
from .torch_adapter import compress_state_dict, decompress_state_dict

__all__ = [
    "HyperGlyphCodec",
    "CompactHyperGlyphCodec",
    "HyperGlyphConfig",
    "BaselineComparison",
    "BenchmarkReport",
    "CompressionReport",
    "TensorError",
    "AblationResult",
    "InferenceComparison",
    "PerplexityResult",
    "QuantizationLibraryStatus",
    "CompressedModel",
    "CompactCompressedModel",
    "CompressedTensor",
    "compress_state_dict",
    "decompress_state_dict",
    "save_compressed",
    "load_compressed",
    "benchmark_state_dict",
    "tensor_error_analysis",
    "tensor_error_markdown",
    "ablation_configs",
    "run_ablation_study",
    "compare_inference_after_decompression",
    "evaluate_perplexity",
    "available_strong_quantization_libraries",
    "OPEN_MODEL_SUITE",
    "STRONG_QUANTIZATION_LIBRARIES",
]
__version__ = "0.7.0"
