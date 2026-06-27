"""Benchmark reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .codec import CompressionReport, HyperGlyphCodec


@dataclass(slots=True)
class BaselineComparison:
    """A size comparison against a baseline representation."""

    name: str
    bytes: int
    ratio_vs_fp32: float
    mse: float | None = None
    mae: float | None = None
    max_abs_error: float | None = None


@dataclass(slots=True)
class BenchmarkReport:
    """A benchmark report with baseline and Hyper Glyph metrics."""

    compression: CompressionReport
    baselines: list[BaselineComparison]

    def to_markdown(self) -> str:
        """Export the benchmark as a markdown table."""
        lines = [
            "# Hyper Glyph Benchmark",
            "",
            "| Representation | Bytes | Ratio vs FP32 | MSE | MAE | Max abs error |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for baseline in self.baselines:
            lines.append(
                "| "
                f"{baseline.name} | "
                f"{baseline.bytes} | "
                f"{baseline.ratio_vs_fp32:.2f}x | "
                f"{_format_optional(baseline.mse)} | "
                f"{_format_optional(baseline.mae)} | "
                f"{_format_optional(baseline.max_abs_error)} |"
            )
        lines.extend(
            [
                "",
                "## Tensor Summary",
                "",
                f"- Tensors compressed: {self.compression.tensors_compressed}",
                f"- Tensors skipped: {self.compression.tensors_skipped}",
            ]
        )
        return "\n".join(lines) + "\n"


def benchmark_state_dict(
    state_dict: Mapping[str, Any],
    codec: HyperGlyphCodec | None = None,
) -> BenchmarkReport:
    """Compress a state dict and return baseline comparisons."""
    active_codec = codec or HyperGlyphCodec()
    compressed = active_codec.compress_state_dict(state_dict)
    restored = active_codec.decompress_state_dict(compressed)
    compression = active_codec.report(compressed, state_dict, restored)
    fp32_bytes = compression.original_bytes
    baselines = [
        BaselineComparison("FP32", fp32_bytes, 1.0, 0.0, 0.0, 0.0),
        BaselineComparison(
            "FP16 estimate",
            compression.fp16_estimate_bytes,
            _ratio(fp32_bytes, compression.fp16_estimate_bytes),
        ),
        BaselineComparison(
            "INT8 estimate",
            compression.int8_estimate_bytes,
            _ratio(fp32_bytes, compression.int8_estimate_bytes),
        ),
        BaselineComparison(
            "Hyper Glyph",
            compression.compressed_bytes,
            compression.compression_ratio,
            compression.total_mse,
            compression.total_mae,
            compression.max_abs_error,
        ),
    ]
    return BenchmarkReport(compression=compression, baselines=baselines)


def _ratio(original_bytes: int, compressed_bytes: int) -> float:
    if compressed_bytes <= 0:
        return float("inf")
    return original_bytes / compressed_bytes


def _format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    if value == 0.0:
        return "0"
    return f"{value:.6g}"
