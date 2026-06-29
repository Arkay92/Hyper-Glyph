"""Command-line interface for Hyper Glyph."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Sequence

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency path
    torch = None

from .benchmark import benchmark_state_dict
from .codec import HyperGlyphCodec
from .config import HyperGlyphConfig
from .evaluation import (
    available_strong_quantization_libraries,
    evaluate_perplexity,
    run_ablation_study,
    tensor_error_analysis,
    tensor_error_markdown,
)
from .serialization import load_compressed, save_compressed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="hyperglyph", description="Hyper Glyph compression CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser("compress")
    compress_parser.add_argument("input", help="Input torch state dict file (.pt)")
    compress_parser.add_argument("output", help="Output .hwz file")
    compress_parser.add_argument("--mode", choices=["standard", "compact"], default="compact")
    compress_parser.add_argument("--block-size", type=int, default=16)
    compress_parser.add_argument("--hdc-dim", type=int, default=4096)
    compress_parser.add_argument("--n-buckets", type=int, default=16)
    compress_parser.add_argument("--n-prototypes", type=int, default=128)
    compress_parser.add_argument("--residual-k", type=int, default=8)
    compress_parser.add_argument("--residual-dtype", choices=["float32", "int8"], default="int8")
    compress_parser.add_argument(
        "--scale-mode", choices=["block", "tensor", "channel"], default="block"
    )
    compress_parser.add_argument("--seed", type=int, default=42)
    compress_parser.add_argument("--compress-bias", action="store_true")
    compress_parser.add_argument("--min-tensor-size", type=int, default=256)
    compress_parser.add_argument("--target-ratio", type=float, default=6.0)

    decompress_parser = subparsers.add_parser("decompress")
    decompress_parser.add_argument("input", help="Input .hwz file")
    decompress_parser.add_argument("output", help="Output torch state dict file (.pt)")

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("input", help="Input .hwz file")

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("input", help="Input torch state dict file (.pt)")
    benchmark_parser.add_argument("--mode", choices=["standard", "compact"], default="compact")
    benchmark_parser.add_argument(
        "--markdown-output", help="Write benchmark report to a markdown file"
    )
    benchmark_parser.add_argument("--block-size", type=int, default=16)
    benchmark_parser.add_argument("--n-prototypes", type=int, default=128)
    benchmark_parser.add_argument("--residual-k", type=int, default=8)
    benchmark_parser.add_argument("--residual-dtype", choices=["float32", "int8"], default="int8")
    benchmark_parser.add_argument(
        "--scale-mode", choices=["block", "tensor", "channel"], default="block"
    )

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("original", help="Original torch state dict file (.pt)")
    analyze_parser.add_argument("restored", help="Restored torch state dict file (.pt)")
    analyze_parser.add_argument("--limit", type=int, default=25)
    analyze_parser.add_argument("--markdown-output", help="Write tensor error report")

    ablation_parser = subparsers.add_parser("ablation")
    ablation_parser.add_argument("input", help="Input torch state dict file (.pt)")
    ablation_parser.add_argument("--names", nargs="*", help="Subset of ablation row names")
    ablation_parser.add_argument("--markdown-output", help="Write ablation report")

    perplexity_parser = subparsers.add_parser("perplexity")
    perplexity_parser.add_argument("--model", required=True)
    perplexity_parser.add_argument("--dataset", default="wikitext")
    perplexity_parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    perplexity_parser.add_argument("--split", default="test")
    perplexity_parser.add_argument("--max-samples", type=int, default=64)
    perplexity_parser.add_argument("--sequence-length", type=int, default=512)
    perplexity_parser.add_argument("--decompressed", action="store_true")

    subparsers.add_parser("quant-libs")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "compress":
        if torch is None:
            raise SystemExit("PyTorch is required for the compress/decompress CLI commands")
        state_dict = torch.load(args.input, map_location="cpu")
        config = HyperGlyphConfig(
            hdc_dim=args.hdc_dim,
            block_size=args.block_size,
            n_buckets=args.n_buckets,
            n_prototypes=args.n_prototypes,
            residual_k=args.residual_k,
            residual_dtype=args.residual_dtype,
            scale_mode=args.scale_mode,
            seed=args.seed,
            compress_bias=args.compress_bias,
            min_tensor_size=args.min_tensor_size,
            mode=args.mode,
            target_ratio=args.target_ratio,
        )
        codec = HyperGlyphCodec(config)
        compressed_model = codec.compress_state_dict(state_dict)
        save_compressed(compressed_model, args.output)
        print(f"Saved compressed model to {args.output}")
        return 0

    if args.command == "decompress":
        if torch is None:
            raise SystemExit("PyTorch is required for the compress/decompress CLI commands")
        loaded_model = load_compressed(args.input)
        restored = HyperGlyphCodec().decompress_state_dict(loaded_model)
        torch.save(restored, args.output)
        print(f"Restored state dict to {args.output}")
        return 0

    if args.command == "inspect":
        inspected_model = load_compressed(args.input)
        breakdown = getattr(inspected_model, "payload_breakdown", {})
        print(
            json.dumps(
                {
                    "format_version": inspected_model.format_version,
                    "mode": getattr(inspected_model, "mode", "standard"),
                    "tensor_count": len(getattr(inspected_model, "tensors", {}))
                    or int(getattr(inspected_model, "metadata", {}).get("tensor_count", 0)),
                    "payload_breakdown": breakdown,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "benchmark":
        if torch is None:
            raise SystemExit("PyTorch is required for benchmark CLI commands")
        state_dict = torch.load(args.input, map_location="cpu")
        codec = HyperGlyphCodec(
            HyperGlyphConfig(
                block_size=args.block_size,
                n_prototypes=args.n_prototypes,
                residual_k=args.residual_k,
                residual_dtype=args.residual_dtype,
                scale_mode=args.scale_mode,
                mode=args.mode,
            )
        )
        report = benchmark_state_dict(state_dict, codec)
        markdown = report.to_markdown()
        if args.markdown_output:
            with open(args.markdown_output, "w", encoding="utf-8") as handle:
                handle.write(markdown)
        print(markdown)
        return 0

    if args.command == "analyze":
        if torch is None:
            raise SystemExit("PyTorch is required for analyze CLI commands")
        original = torch.load(args.original, map_location="cpu")
        restored = torch.load(args.restored, map_location="cpu")
        rows = tensor_error_analysis(original, restored)
        markdown = tensor_error_markdown(rows, limit=args.limit)
        if args.markdown_output:
            with open(args.markdown_output, "w", encoding="utf-8") as handle:
                handle.write(markdown)
        print(markdown)
        return 0

    if args.command == "ablation":
        if torch is None:
            raise SystemExit("PyTorch is required for ablation CLI commands")
        state_dict = torch.load(args.input, map_location="cpu")
        selected = set(args.names) if args.names else None
        rows = run_ablation_study(state_dict, names=selected)
        markdown_lines = [
            "| Ablation | Bytes | Ratio vs FP32 | MSE | MAE | Max Abs Error | Tensors |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            markdown_lines.append(
                f"| {row.name} | {row.compressed_bytes} | {row.ratio_vs_fp32:.2f}x | "
                f"{row.mse:.8g} | {row.mae:.8g} | {row.max_abs_error:.8g} | "
                f"{row.tensors_compressed} |"
            )
        markdown = "\n".join(markdown_lines) + "\n"
        if args.markdown_output:
            with open(args.markdown_output, "w", encoding="utf-8") as handle:
                handle.write(markdown)
        print(markdown)
        return 0

    if args.command == "perplexity":
        config = None
        if args.decompressed:
            config = HyperGlyphConfig(mode="compact", compact_tensor_codec="auto")
        result = evaluate_perplexity(
            model_name=args.model,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            max_samples=args.max_samples,
            sequence_length=args.sequence_length,
            compression_config=config,
        )
        print(json.dumps(asdict(result), indent=2))
        return 0

    if args.command == "quant-libs":
        print(
            json.dumps(
                [asdict(status) for status in available_strong_quantization_libraries()],
                indent=2,
            )
        )
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
