"""Command-line interface for Hyper Glyph."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency path
    torch = None

from .benchmark import benchmark_state_dict
from .codec import HyperGlyphCodec
from .config import HyperGlyphConfig
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

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
